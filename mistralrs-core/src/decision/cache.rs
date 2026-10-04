use std::collections::{BTreeMap, HashMap};

use candle_core::{DType, Device, Tensor};
use sha2::{Digest, Sha256};

const DEFAULT_CACHE_BUDGET: &str = "64MiB";
const FREE_MEMORY_PERCENT: usize = 90;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) enum Role {
    State,
    Action,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) struct CacheKey(Role, [u8; 32]);

impl CacheKey {
    pub fn new(role: Role, text: &str) -> Self {
        Self(role, Sha256::digest(text.as_bytes()).into())
    }
}

struct Entry {
    row: u32,
    stamp: u64,
}

pub(super) struct VectorCache {
    buffer: Option<Tensor>,
    entries: HashMap<CacheKey, Entry>,
    recency: BTreeMap<u64, CacheKey>,
    clock: u64,
    capacity: usize,
}

impl VectorCache {
    pub fn from_env(dim: usize, device: &Device) -> anyhow::Result<Self> {
        let spec =
            std::env::var("CLM_ACTION_CACHE").unwrap_or_else(|_| DEFAULT_CACHE_BUDGET.to_string());
        let memory = crate::MemoryUsage.query(device)?;
        let requested = parse_budget(&spec, memory.total())?;
        let bytes = if device.is_cpu() {
            requested
        } else {
            requested.min(memory.available() / 100 * FREE_MEMORY_PERCENT)
        };
        let cache = Self::new(bytes, dim, device)?;
        tracing::info!(
            rows = cache.capacity,
            bytes = cache.capacity * dim * DType::F32.size_in_bytes(),
            ?device,
            "Reserved CLM state/action vector cache"
        );
        Ok(cache)
    }

    pub fn new(bytes: usize, dim: usize, device: &Device) -> candle_core::Result<Self> {
        let capacity = (bytes / (dim * DType::F32.size_in_bytes())).min(u32::MAX as usize);
        Ok(Self {
            buffer: if capacity == 0 {
                None
            } else {
                Some(Tensor::zeros((capacity, dim), DType::F32, device)?)
            },
            entries: HashMap::new(),
            recency: BTreeMap::new(),
            clock: 0,
            capacity,
        })
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.recency.clear();
        self.clock = 0;
    }

    pub fn get(&mut self, keys: &[CacheKey]) -> candle_core::Result<Vec<Option<Tensor>>> {
        let mut result = vec![None; keys.len()];
        let Some(buffer) = &self.buffer else {
            return Ok(result);
        };
        let mut rows = Vec::new();
        let mut hits = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            if let Some(entry) = self.entries.get_mut(key) {
                self.recency.remove(&entry.stamp);
                self.clock += 1;
                entry.stamp = self.clock;
                self.recency.insert(entry.stamp, *key);
                rows.push(entry.row);
                hits.push(i);
            }
        }
        if !rows.is_empty() {
            let indices = Tensor::new(rows.as_slice(), buffer.device())?;
            // Gather before inserting misses so evictions cannot overwrite this request's hits.
            let vectors = buffer.index_select(&indices, 0)?;
            for (row, index) in hits.into_iter().enumerate() {
                result[index] = Some(vectors.narrow(0, row, 1)?);
            }
        }
        Ok(result)
    }

    pub fn insert(&mut self, keys: &[CacheKey], vectors: &Tensor) -> candle_core::Result<()> {
        let Some(buffer) = &self.buffer else {
            return Ok(());
        };
        if keys.is_empty() {
            return Ok(());
        }
        let start = keys.len().saturating_sub(self.capacity);
        let keys = &keys[start..];
        let vectors = vectors.narrow(0, start, keys.len())?.contiguous()?;
        let mut rows = Vec::with_capacity(keys.len());
        for key in keys {
            let row = if let Some(entry) = self.entries.remove(key) {
                self.recency.remove(&entry.stamp);
                entry.row
            } else if self.entries.len() == self.capacity {
                let (_, oldest) = self.recency.pop_first().expect("full cache has entries");
                self.entries.remove(&oldest).expect("LRU entry").row
            } else {
                u32::try_from(self.entries.len()).expect("bounded cache row")
            };
            self.clock += 1;
            self.entries.insert(
                *key,
                Entry {
                    row,
                    stamp: self.clock,
                },
            );
            self.recency.insert(self.clock, *key);
            rows.push(row);
        }
        let inserted = (|| {
            let indices = Tensor::new(rows.as_slice(), buffer.device())?
                .unsqueeze(1)?
                .broadcast_as(vectors.shape())?
                .contiguous()?;
            buffer.scatter_set(&indices, &vectors, 0)
        })();
        if let Err(error) = inserted {
            self.clear();
            return Err(error);
        }
        Ok(())
    }
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn parse_budget(spec: &str, total: usize) -> anyhow::Result<usize> {
    let spec = spec.trim();
    let split = spec
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(spec.len());
    let value: f64 = spec[..split].parse()?;
    let unit = spec[split..].trim().to_ascii_uppercase();
    anyhow::ensure!(
        value.is_finite() && value >= 0.0,
        "invalid CLM_ACTION_CACHE budget"
    );
    let multiplier = match unit.as_str() {
        "" => {
            anyhow::ensure!(
                value < 1.0,
                "CLM_ACTION_CACHE without a unit must be a fraction in [0, 1)"
            );
            total as f64
        }
        "B" => 1.0,
        "KB" => 1_000.0,
        "MB" => 1_000_000.0,
        "GB" => 1_000_000_000.0,
        "KIB" => 1024.0,
        "MIB" => 1024.0 * 1024.0,
        "GIB" => 1024.0 * 1024.0 * 1024.0,
        _ => anyhow::bail!("unknown CLM_ACTION_CACHE unit; use B, KB, MB, GB, KiB, MiB, or GiB"),
    };
    let bytes = value * multiplier;
    anyhow::ensure!(
        bytes.is_finite() && bytes < usize::MAX as f64,
        "CLM_ACTION_CACHE budget is too large"
    );
    Ok(bytes as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets() {
        assert_eq!(parse_budget("0", 1000).unwrap(), 0);
        assert_eq!(parse_budget("0.02", 1000).unwrap(), 20);
        assert_eq!(parse_budget("1 MiB", 1000).unwrap(), 1 << 20);
        for invalid in ["", "-1", "1", "NaN", "1TB", "1e99"] {
            assert!(parse_budget(invalid, 1000).is_err(), "{invalid}");
        }
    }

    fn eviction_and_ownership(device: Device) -> anyhow::Result<()> {
        let a = CacheKey::new(Role::State, "a");
        let b = CacheKey::new(Role::Action, "a");
        let c = CacheKey::new(Role::State, "c");
        let mut cache = VectorCache::new(2 * 2 * 4, 2, &device)?;
        cache.insert(&[a, b], &Tensor::new(&[[1_f32, 2.], [3., 4.]], &device)?)?;
        let saved = cache.get(&[a])?.remove(0).unwrap();
        cache.insert(&[c], &Tensor::new(&[[5_f32, 6.]], &device)?)?;
        let result = cache.get(&[a, b, c])?;
        assert!(result[0].is_some());
        assert!(result[1].is_none());
        assert_eq!(result[2].as_ref().unwrap().to_vec2::<f32>()?, [[5., 6.]]);
        cache.insert(
            &[a, b, c],
            &Tensor::new(&[[7_f32, 8.], [9., 10.], [11., 12.]], &device)?,
        )?;
        assert_eq!(saved.to_vec2::<f32>()?, [[1., 2.]]);
        let result = cache.get(&[a, b, c])?;
        assert!(result[0].is_none());
        assert_eq!(result[1].as_ref().unwrap().to_vec2::<f32>()?, [[9., 10.]]);
        assert_eq!(result[2].as_ref().unwrap().to_vec2::<f32>()?, [[11., 12.]]);
        cache.clear();
        assert!(cache.get(&[a, b, c])?.iter().all(Option::is_none));
        Ok(())
    }

    #[test]
    fn cpu_eviction_and_ownership() -> anyhow::Result<()> {
        eviction_and_ownership(Device::Cpu)
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_eviction_and_ownership() -> anyhow::Result<()> {
        eviction_and_ownership(Device::new_metal(0)?)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_eviction_and_ownership() -> anyhow::Result<()> {
        eviction_and_ownership(Device::new_cuda(0)?)
    }
}
