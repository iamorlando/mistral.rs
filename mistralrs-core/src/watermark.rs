#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashSet, fmt};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const KEY_BYTES: usize = 32;
const DEFAULT_NGRAM_LEN: usize = 5;
const MAX_NGRAM_LEN: usize = 32;
const DEFAULT_DEPTH: usize = 30;
const MAX_DEPTH: usize = 256;
const CONTEXT_HISTORY_SIZE: usize = 1024;
const HASH_DOMAIN: &[u8] = b"mistralrs-synthid-text-v1\0";

/// Opt-in SynthID-Text tournament sampling with an independent, secret deployment key.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct SynthIdTextWatermarkConfig {
    /// A random 32-byte key encoded as 64 hexadecimal characters.
    pub key: String,
    #[serde(default = "default_ngram_len")]
    pub ngram_len: usize,
    #[serde(default = "default_depth")]
    pub depth: usize,
}

fn default_ngram_len() -> usize {
    DEFAULT_NGRAM_LEN
}

fn default_depth() -> usize {
    DEFAULT_DEPTH
}

impl fmt::Debug for SynthIdTextWatermarkConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SynthIdTextWatermarkConfig")
            .field("key", &"[redacted]")
            .field("ngram_len", &self.ngram_len)
            .field("depth", &self.depth)
            .finish()
    }
}

impl SynthIdTextWatermarkConfig {
    pub fn new(key: String) -> anyhow::Result<Self> {
        let config = Self {
            key,
            ngram_len: DEFAULT_NGRAM_LEN,
            depth: DEFAULT_DEPTH,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.key.len() == KEY_BYTES * 2
                && self.key.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "watermark key must contain exactly 64 hexadecimal characters"
        );
        anyhow::ensure!(
            (2..=MAX_NGRAM_LEN).contains(&self.ngram_len),
            "watermark ngram_len must be between 2 and {MAX_NGRAM_LEN}"
        );
        anyhow::ensure!(
            (1..=MAX_DEPTH).contains(&self.depth),
            "watermark depth must be between 1 and {MAX_DEPTH}"
        );
        Ok(())
    }
}

/// Mean g-value evidence, without a calibrated classification or probability of authorship.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WatermarkDetection {
    pub tokens_scored: usize,
    pub mean_g_value: Option<f64>,
}

/// SynthID-Text with a versioned keyed SHA-256 g-function, independent of Claude and Gemini keys.
#[derive(Clone)]
pub struct SynthIdTextWatermark {
    hash_prefix: Sha256,
    context_len: usize,
    depth: usize,
}

impl SynthIdTextWatermark {
    pub fn new(config: &SynthIdTextWatermarkConfig) -> anyhow::Result<Self> {
        config.validate()?;
        let mut key = [0; KEY_BYTES];
        for (index, byte) in key.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&config.key[index * 2..index * 2 + 2], 16)?;
        }
        let mut hash_prefix = Sha256::new();
        hash_prefix.update(HASH_DOMAIN);
        hash_prefix.update(key);
        hash_prefix.update((config.ngram_len as u32).to_le_bytes());
        Ok(Self {
            hash_prefix,
            context_len: config.ngram_len - 1,
            depth: config.depth,
        })
    }

    fn context_hash(&self, context: &[u32]) -> Sha256 {
        let mut hash = self.hash_prefix.clone();
        for token in context {
            hash.update(token.to_le_bytes());
        }
        hash
    }

    fn g_values(hash: &Sha256, token: u32) -> [u8; KEY_BYTES] {
        let mut hash = hash.clone();
        hash.update(token.to_le_bytes());
        hash.finalize().into()
    }

    fn g_value(values: &[u8; KEY_BYTES], layer: usize) -> f64 {
        f64::from((values[layer / 8] >> (layer % 8)) & 1)
    }

    fn repeated_context(&self, context: &[u32], prompt_len: usize) -> bool {
        let current = &context[context.len() - self.context_len..];
        let start = prompt_len
            .max(self.context_len)
            .max(context.len().saturating_sub(CONTEXT_HISTORY_SIZE));
        (start..context.len()).any(|end| &context[end - self.context_len..end] == current)
    }

    pub(crate) fn apply(&self, probs: &mut [f32], context: &[u32], prompt_len: usize) {
        if context.len() < self.context_len || self.repeated_context(context, prompt_len) {
            return;
        }
        let hash = self.context_hash(&context[context.len() - self.context_len..]);
        let mut candidates: Vec<_> = probs
            .iter()
            .enumerate()
            .filter(|(_, prob)| **prob > 0.0)
            .map(|(token, prob)| (token, f64::from(*prob), Self::g_values(&hash, token as u32)))
            .collect();
        for layer in 0..self.depth {
            let total: f64 = candidates.iter().map(|(_, prob, _)| prob).sum();
            let g_mass = candidates
                .iter()
                .map(|(_, prob, values)| prob * Self::g_value(values, layer))
                .sum::<f64>()
                / total;
            for (_, prob, values) in &mut candidates {
                // The exact two-candidate tournament distribution avoids drawing 2^depth tokens.
                *prob = (*prob / total) * (1.0 + Self::g_value(values, layer) - g_mass);
            }
        }
        for (token, prob, _) in candidates {
            probs[token] = prob as f32;
        }
    }

    /// Score generated token IDs, excluding the prompt, EOS and all repeated contexts.
    pub fn detect(
        &self,
        tokens: &[u32],
        prompt_len: usize,
        eos_token_ids: &[u32],
    ) -> anyhow::Result<WatermarkDetection> {
        anyhow::ensure!(prompt_len <= tokens.len(), "prompt_len exceeds token count");
        let mut seen = HashSet::new();
        let mut tokens_scored = 0;
        let mut sum = 0.0;
        for position in prompt_len..tokens.len() {
            if eos_token_ids.contains(&tokens[position]) {
                break;
            }
            if position < self.context_len {
                continue;
            }
            let context = &tokens[position - self.context_len..position];
            if !seen.insert(context) {
                continue;
            }
            let values = Self::g_values(&self.context_hash(context), tokens[position]);
            sum += (0..self.depth)
                .map(|layer| Self::g_value(&values, layer))
                .sum::<f64>();
            tokens_scored += 1;
        }
        Ok(WatermarkDetection {
            tokens_scored,
            mean_g_value: (tokens_scored > 0)
                .then(|| sum / (tokens_scored as f64 * self.depth as f64)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{distr::Distribution, SeedableRng};
    use rand_isaac::Isaac64Rng;

    fn config() -> SynthIdTextWatermarkConfig {
        SynthIdTextWatermarkConfig::new(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f".into(),
        )
        .unwrap()
    }

    #[test]
    fn watermark_config_validation_and_redaction() {
        let mut config = config();
        assert!(!format!("{config:?}").contains(&config.key));
        for key in ["", "abc", &"g".repeat(64), &"\u{e9}".repeat(32)] {
            assert!(SynthIdTextWatermarkConfig::new(key.into()).is_err());
        }
        for depth in [0, MAX_DEPTH + 1] {
            config.depth = depth;
            assert!(config.validate().is_err());
        }
        config.depth = DEFAULT_DEPTH;
        for ngram_len in [0, 1, MAX_NGRAM_LEN + 1] {
            config.ngram_len = ngram_len;
            assert!(config.validate().is_err());
        }
        let decoded: SynthIdTextWatermarkConfig =
            serde_json::from_value(serde_json::json!({"key": config.key})).unwrap();
        assert_eq!(decoded.ngram_len, DEFAULT_NGRAM_LEN);
        assert_eq!(decoded.depth, DEFAULT_DEPTH);
        assert!(serde_json::from_value::<SynthIdTextWatermarkConfig>(
            serde_json::json!({"key": decoded.key, "unknown": true})
        )
        .is_err());
    }

    #[test]
    fn watermark_hash_matches_independent_sha256_fixture() {
        let watermark = SynthIdTextWatermark::new(&config()).unwrap();
        let values = SynthIdTextWatermark::g_values(&watermark.context_hash(&[1, 2, 3, 4]), 7);
        let hex: String = values.iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(
            hex,
            "20120032c8ac8d1bf7dcee4c4abca87dbb18a34b31639a53189be67bbf626edd"
        );
    }

    #[test]
    fn watermark_matches_enumerated_pairwise_tournament() {
        let mut config = config();
        config.depth = 1;
        let watermark = SynthIdTextWatermark::new(&config).unwrap();
        let context = [0, 2, 3, 4];
        let initial = [0.1f32, 0.2, 0.3, 0.4];
        let mut expected = [0.0; 4];
        let hash = watermark.context_hash(&context);
        let scores: Vec<_> = (0..4)
            .map(|token| {
                SynthIdTextWatermark::g_value(&SynthIdTextWatermark::g_values(&hash, token), 0)
            })
            .collect();
        assert!(scores.contains(&0.0) && scores.contains(&1.0));
        for a in 0..4 {
            for b in 0..4 {
                let mass = initial[a] * initial[b];
                if scores[a] == scores[b] {
                    expected[a] += mass / 2.0;
                    expected[b] += mass / 2.0;
                } else {
                    expected[if scores[a] > scores[b] { a } else { b }] += mass;
                }
            }
        }
        let mut actual = initial;
        watermark.apply(&mut actual, &context, context.len());
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn watermark_preserves_support_and_point_masses() {
        let watermark = SynthIdTextWatermark::new(&config()).unwrap();
        let mut probs = [0.0, 0.3, 0.0, 0.7];
        watermark.apply(&mut probs, &[1, 2, 3, 4], 4);
        assert_eq!(probs[0], 0.0);
        assert_eq!(probs[2], 0.0);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(probs.iter().all(|value| value.is_finite() && *value >= 0.0));
        let mut point_mass = [0.0, 1.0, 0.0];
        watermark.apply(&mut point_mass, &[1, 2, 3, 4], 4);
        assert_eq!(point_mass, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn watermark_context_history_is_replayable_and_prompt_aware() {
        let watermark = SynthIdTextWatermark::new(&config()).unwrap();
        let initial = vec![0.125; 8];
        for context in [&[1, 2, 3][..], &[1, 2, 3, 4, 1, 2, 3, 4][..]] {
            let mut probs = initial.clone();
            watermark.apply(&mut probs, context, 0);
            assert_eq!(probs, initial);
        }
        let context = [1, 2, 3, 4, 1, 2, 3, 4];
        let mut first = initial.clone();
        watermark.apply(&mut first, &context, context.len());
        assert_ne!(first, initial);
        let mut retry = initial.clone();
        watermark.clone().apply(&mut retry, &context, context.len());
        assert_eq!(first, retry);
        let mut unrelated = initial.clone();
        watermark.apply(&mut unrelated, &[9, 8, 7, 6], 4);
        let mut after_rollback = initial;
        watermark.apply(&mut after_rollback, &context, context.len());
        assert_eq!(first, after_rollback);
    }

    #[test]
    fn watermark_detector_excludes_prompt_repeats_and_eos() {
        let watermark = SynthIdTextWatermark::new(&config()).unwrap();
        assert!(watermark.detect(&[1], 2, &[]).is_err());
        for tokens in [&[][..], &[1, 2, 3, 4][..]] {
            let detection = watermark.detect(tokens, 0, &[]).unwrap();
            assert_eq!(detection.tokens_scored, 0);
            assert_eq!(detection.mean_g_value, None);
        }
        let detection = watermark.detect(&[1, 2, 3, 4, 7, 99, 8], 4, &[99]).unwrap();
        assert_eq!(detection.tokens_scored, 1);
        let values = SynthIdTextWatermark::g_values(&watermark.context_hash(&[1, 2, 3, 4]), 7);
        let expected = (0..DEFAULT_DEPTH)
            .map(|layer| SynthIdTextWatermark::g_value(&values, layer))
            .sum::<f64>()
            / DEFAULT_DEPTH as f64;
        assert_eq!(detection.mean_g_value, Some(expected));
        assert_eq!(
            watermark.detect(&[1; 100], 4, &[]).unwrap().tokens_scored,
            1
        );
    }

    #[test]
    fn watermark_signal_separates_marked_unmarked_and_wrong_key() {
        let watermark = SynthIdTextWatermark::new(&config()).unwrap();
        let wrong_config = SynthIdTextWatermarkConfig::new("ff".repeat(KEY_BYTES)).unwrap();
        let wrong_watermark = SynthIdTextWatermark::new(&wrong_config).unwrap();
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let mut marked = vec![1, 2, 3, 4];
        let mut unmarked = marked.clone();
        for _ in 0..1000 {
            let mut probs = vec![1.0 / 128.0; 128];
            let baseline = rand::distr::weighted::WeightedIndex::new(&probs).unwrap();
            watermark.apply(&mut probs, &marked, 4);
            let distribution = rand::distr::weighted::WeightedIndex::new(&probs).unwrap();
            marked.push(distribution.sample(&mut rng) as u32);
            unmarked.push(baseline.sample(&mut rng) as u32);
        }
        let marked_score = watermark
            .detect(&marked, 4, &[])
            .unwrap()
            .mean_g_value
            .unwrap();
        let unmarked_score = watermark
            .detect(&unmarked, 4, &[])
            .unwrap()
            .mean_g_value
            .unwrap();
        let wrong_score = wrong_watermark
            .detect(&marked, 4, &[])
            .unwrap()
            .mean_g_value
            .unwrap();
        assert!(marked_score > 0.60, "marked={marked_score}");
        assert!(
            (unmarked_score - 0.5).abs() < 0.02,
            "unmarked={unmarked_score}"
        );
        assert!((wrong_score - 0.5).abs() < 0.02, "wrong={wrong_score}");
    }

    #[test]
    fn watermark_preserves_distribution_over_independent_contexts() {
        let watermark = SynthIdTextWatermark::new(&config()).unwrap();
        let expected = [0.1, 0.2, 0.3, 0.4];
        let mut totals = [0.0; 4];
        for nonce in 0..10000 {
            let mut probs = expected;
            watermark.apply(&mut probs, &[nonce, 2, 3, 4], 4);
            for (total, prob) in totals.iter_mut().zip(probs) {
                *total += f64::from(prob);
            }
        }
        for (total, expected) in totals.into_iter().zip(expected) {
            assert!((total / 10000.0 - f64::from(expected)).abs() < 0.02);
        }
    }
}
