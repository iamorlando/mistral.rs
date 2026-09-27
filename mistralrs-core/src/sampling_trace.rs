use serde::{Deserialize, Serialize};

const DEFAULT_STEPS: usize = 32;
const DEFAULT_CANDIDATES: usize = 32;
const DEFAULT_LAYERS: usize = 8;
const MAX_STEPS: usize = 256;
const MAX_CANDIDATES: usize = 128;
const MAX_LAYERS: usize = 32;
const MAX_OBSERVATIONS: usize = 65_536;
const MAX_BYTES: usize = 8 * 1024 * 1024;
const FRAME_ALLOWANCE: usize = 256;
const SUMMARY_ALLOWANCE: usize = 1024;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct SamplingTraceConfig {
    pub max_steps: usize,
    pub max_candidates: usize,
    pub max_layers: usize,
}

impl Default for SamplingTraceConfig {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_STEPS,
            max_candidates: DEFAULT_CANDIDATES,
            max_layers: DEFAULT_LAYERS,
        }
    }
}

impl SamplingTraceConfig {
    pub fn validate(&self, logprobs: bool, n_choices: usize) -> anyhow::Result<()> {
        anyhow::ensure!(logprobs, "sampling_trace requires logprobs");
        anyhow::ensure!(n_choices == 1, "sampling_trace requires n=1");
        anyhow::ensure!(
            (1..=MAX_STEPS).contains(&self.max_steps)
                && (1..=MAX_CANDIDATES).contains(&self.max_candidates)
                && self.max_layers <= MAX_LAYERS,
            "sampling_trace limits: max_steps=1..256, max_candidates=1..128, max_layers=0..32"
        );
        anyhow::ensure!(
            self.max_steps * self.max_candidates * self.max_layers.max(1) <= MAX_OBSERVATIONS,
            "sampling_trace exceeds 65536 candidate-layer observations"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct SamplingTrace {
    pub steps: Vec<SamplingTraceStep>,
    pub truncated: bool,
    pub truncation_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct SamplingTraceStep {
    pub generated_index: usize,
    pub context_length: usize,
    pub attempt: usize,
    pub selected_token_id: u32,
    pub selection_rule: String,
    pub candidate_count: usize,
    pub candidates_truncated: bool,
    pub watermark: Option<TraceWatermark>,
    pub candidates: Vec<TraceCandidate>,
    pub layers: Vec<TraceLayer>,
    pub layers_truncated: bool,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct TraceCandidate {
    pub token_id: u32,
    pub text: Option<String>,
    pub input_logit: Option<f32>,
    pub processed_logit: Option<f32>,
    pub reporting_probability: f32,
    pub pre_watermark_probability: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_watermark_probability: Option<f64>,
    pub post_watermark_log_probability: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub membership: Option<TraceMembership>,
    pub selection_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selection_score_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inverse_rank: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cdf_lower: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cdf_upper: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct TraceMembership {
    pub kind: String,
    pub favored: bool,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct TraceWatermark {
    pub scheme: String,
    pub status: String,
    pub bias_delta: Option<f64>,
    pub payload_position: Option<usize>,
    pub payload_symbol: Option<u8>,
    pub key_position: Option<usize>,
    pub score_kind: Option<String>,
    pub inverse: Option<TraceInverse>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct TraceInverse {
    pub uniform: f64,
    pub threshold: f64,
    pub total_weight: f64,
}

/// Layer arrays follow the step's candidate order and retain full-support normalization.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct TraceLayer {
    pub index: usize,
    pub green_mass: f64,
    pub input_normalizer: f64,
    pub g_values: Vec<u8>,
    pub input_probabilities: Vec<f64>,
    pub probabilities: Vec<f64>,
}

pub(crate) struct TraceState {
    config: SamplingTraceConfig,
    trace: SamplingTrace,
    bytes: usize,
    summary_pending: bool,
}

impl TraceState {
    pub(crate) fn new(config: SamplingTraceConfig) -> Self {
        Self {
            config,
            trace: SamplingTrace::default(),
            bytes: SUMMARY_ALLOWANCE,
            summary_pending: false,
        }
    }

    pub(crate) fn capture(&mut self, generated_index: usize) -> Option<SamplingTraceConfig> {
        if generated_index >= self.config.max_steps && !self.trace.truncated {
            self.truncate("max_steps");
        }
        (!self.trace.truncated).then_some(self.config)
    }

    fn truncate(&mut self, reason: &str) {
        self.trace.truncated = true;
        self.trace.truncation_reason = Some(reason.into());
        self.summary_pending = true;
    }

    pub(crate) fn record(&mut self, step: SamplingTraceStep) -> candle_core::Result<()> {
        let mut size = JsonSize::default();
        serde_json::to_writer(&mut size, &step).map_err(candle_core::Error::wrap)?;
        let bytes = self
            .bytes
            .saturating_add(size.0)
            .saturating_add(FRAME_ALLOWANCE);
        if bytes > MAX_BYTES {
            self.truncate("max_bytes");
        } else {
            self.bytes = bytes;
            self.trace.steps.push(step);
        }
        Ok(())
    }

    pub(crate) fn take(&mut self, terminal: bool) -> Option<SamplingTrace> {
        if self.trace.steps.is_empty() && !self.summary_pending && !terminal {
            return None;
        }
        self.summary_pending = false;
        Some(SamplingTrace {
            steps: std::mem::take(&mut self.trace.steps),
            truncated: self.trace.truncated,
            truncation_reason: self.trace.truncation_reason.clone(),
        })
    }
}

#[derive(Default)]
struct JsonSize(usize);

impl std::io::Write for JsonSize {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self.0.saturating_add(bytes.len());
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn step(index: usize) -> SamplingTraceStep {
        SamplingTraceStep {
            generated_index: index,
            context_length: index + 1,
            attempt: 0,
            selected_token_id: 1,
            selection_rule: "categorical".into(),
            candidate_count: 1,
            candidates_truncated: false,
            watermark: None,
            candidates: Vec::new(),
            layers: Vec::new(),
            layers_truncated: false,
        }
    }

    #[test]
    fn sampling_trace_bounds_and_opt_in_validation() {
        let config = SamplingTraceConfig::default();
        assert!(config.validate(true, 1).is_ok());
        assert!(config.validate(false, 1).is_err());
        assert!(config.validate(true, 2).is_err());
        for (steps, candidates, layers) in [
            (0, 1, 0),
            (257, 1, 0),
            (1, 0, 0),
            (1, 129, 0),
            (1, 1, 33),
            (256, 128, 32),
            (usize::MAX, usize::MAX, usize::MAX),
        ] {
            assert!(SamplingTraceConfig {
                max_steps: steps,
                max_candidates: candidates,
                max_layers: layers
            }
            .validate(true, 1)
            .is_err());
        }
        assert!(SamplingTraceConfig {
            max_layers: 0,
            ..config
        }
        .validate(true, 1)
        .is_ok());
        assert!(serde_json::from_str::<SamplingTraceConfig>("{\"max_steps\":-1}").is_err());
        assert!(serde_json::from_str::<SamplingTraceConfig>("{\"unknown\":1}").is_err());
    }

    #[test]
    fn sampling_trace_drains_once_and_preserves_truncation_summary() {
        let mut state = TraceState::new(SamplingTraceConfig {
            max_steps: 2,
            ..Default::default()
        });
        assert!(state.capture(0).is_some());
        state.record(step(0)).unwrap();
        assert_eq!(state.take(false).unwrap().steps[0].generated_index, 0);
        assert!(state.take(false).is_none());
        assert!(state.capture(1).is_some());
        state.record(step(1)).unwrap();
        assert_eq!(state.take(false).unwrap().steps[0].generated_index, 1);
        assert!(state.capture(2).is_none());
        let summary = state.take(false).unwrap();
        assert!(summary.truncated);
        assert!(summary.steps.is_empty());
        assert_eq!(summary.truncation_reason.as_deref(), Some("max_steps"));
        assert!(state.take(false).is_none());
        assert_eq!(
            state.take(true).unwrap().truncation_reason.as_deref(),
            Some("max_steps")
        );
    }

    #[test]
    fn sampling_trace_byte_budget_stops_capture_without_retaining_oversized_rows() {
        let mut state = TraceState::new(SamplingTraceConfig::default());
        let mut oversized = step(0);
        oversized.selection_rule = "x".repeat(MAX_BYTES);
        state.record(oversized).unwrap();
        assert!(state.capture(1).is_none());
        let summary = state.take(true).unwrap();
        assert!(summary.steps.is_empty());
        assert_eq!(summary.truncation_reason.as_deref(), Some("max_bytes"));
        assert!(serde_json::to_vec(&summary).unwrap().len() < SUMMARY_ALLOWANCE);
    }
}
