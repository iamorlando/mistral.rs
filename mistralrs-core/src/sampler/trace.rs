use std::sync::{Arc, Mutex};

use candle_core::{Result, Tensor};
use llm_watermarking::trace::{SelectionScoreKind, TraceOptions, TraceStatus, TraceView};
use rand_isaac::Isaac64Rng;

use super::{argmax_f32, partial_sort_top_k, Logprobs, Sampler};
use crate::sampling_trace::{
    SamplingTraceConfig, SamplingTraceStep, TraceCandidate, TraceInverse, TraceLayer,
    TraceMembership, TraceWatermark,
};

pub(crate) struct TraceStepContext<'a> {
    pub context: &'a [u32],
    pub prompt_len: usize,
    pub options: SamplingTraceConfig,
}

impl Sampler {
    pub(crate) fn sample_traced(
        &self,
        logits: Tensor,
        step: TraceStepContext<'_>,
        rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<(Logprobs, SamplingTraceStep)> {
        let input = logits.to_vec1::<f32>()?;
        let mut processed = self.apply_penalties(input.clone(), step.context, step.prompt_len)?;
        for processor in &self.logits_processors {
            processed = processor.apply(&processed, step.context)?;
        }
        let processed_values = processed.to_vec1::<f32>()?;
        let reporting = match self.temperature {
            Some(temperature) => candle_nn::ops::softmax_last_dim(&(&processed / temperature)?)?,
            None => candle_nn::ops::softmax_last_dim(&processed)?,
        }
        .to_vec1::<f32>()?;
        let mut sampling = if self.temperature.is_none() {
            let mut weights = vec![0.0; reporting.len()];
            weights[argmax_f32(&processed_values)? as usize] = 1.0;
            weights
        } else {
            reporting.clone()
        };
        if self.watermark.is_some() {
            self.filter_top_kp_min_p(&mut sampling);
            Self::normalize_probs(&mut sampling)?;
        } else if self.temperature.is_some() {
            // The ordinary non-watermark sampler receives F32 filter thresholds.
            let mut filter = self.clone();
            filter.top_p = f64::from(self.top_p as f32);
            filter.min_p = f64::from(self.min_p as f32);
            filter.filter_top_kp_min_p(&mut sampling);
        }
        let before = sampling.clone();
        let library_trace = if let Some(watermark) = &self.watermark {
            if self.temperature.is_some() {
                let trace = watermark.resolve(sampling.len())?.apply_traced(
                    &mut sampling,
                    step.context,
                    step.prompt_len,
                    &TraceOptions {
                        max_layers: step.options.max_layers,
                    },
                )?;
                Self::normalize_probs(&mut sampling)?;
                Some(trace)
            } else {
                None
            }
        } else {
            None
        };
        let selected = if self.watermark.is_some() {
            // Preserve the ordinary path's RNG draw even for a keyed or greedy point mass.
            self.sample_multinomial(&sampling, &reporting, true, rng)?
        } else if self.temperature.is_none() {
            self.sample_argmax(processed, true)?
        } else {
            self.sample_top_kp_min_p(
                &reporting,
                self.top_k,
                self.top_p as f32,
                self.min_p as f32,
                true,
                rng,
            )?
        };
        let keyed = library_trace
            .as_ref()
            .is_some_and(|trace| trace.output_weights().is_none());
        let ranking_input = if self.temperature.is_none() {
            &reporting
        } else {
            &before
        };
        let ids = candidate_ids(
            selected.token,
            ranking_input,
            (!keyed).then_some(sampling.as_slice()),
            step.options.max_candidates,
        );
        let snapshot = library_trace
            .as_ref()
            .map(|trace| {
                trace
                    .snapshot(
                        Some(&ids),
                        &TraceView {
                            max_rows: step.options.max_candidates,
                            ..TraceView::default()
                        },
                    )
                    .map_err(candle_core::Error::wrap)
            })
            .transpose()?;
        let before_total: f64 = before.iter().map(|&p| f64::from(p)).sum();
        let after_total: f64 = sampling.iter().map(|&p| f64::from(p)).sum();
        let scheme = self.watermark.as_ref().map(|w| w.scheme());
        let mut candidates = Vec::with_capacity(ids.len());
        for (row, &token_id) in ids.iter().enumerate() {
            let index = token_id as usize;
            let after = (!keyed).then(|| f64::from(sampling[index]) / after_total);
            let score = snapshot
                .as_ref()
                .and_then(|s| s.selection_scores.as_ref())
                .map(|s| s[row]);
            let inverse = snapshot.as_ref().and_then(|s| s.inverse.as_ref());
            candidates.push(TraceCandidate {
                token_id,
                text: self
                    .tokenizer
                    .as_ref()
                    .map(|t| {
                        t.decode(&[token_id], false)
                            .map_err(|error| candle_core::Error::Msg(error.to_string()))
                    })
                    .transpose()?,
                input_logit: input[index].is_finite().then_some(input[index]),
                processed_logit: processed_values[index]
                    .is_finite()
                    .then_some(processed_values[index]),
                reporting_probability: reporting[index],
                pre_watermark_probability: f64::from(before[index]) / before_total,
                post_watermark_probability: after,
                post_watermark_log_probability: after.filter(|&p| p > 0.0).map(f64::ln),
                membership: snapshot
                    .as_ref()
                    .and_then(|s| s.favored_mask.as_ref())
                    .map(|mask| TraceMembership {
                        kind: if scheme == Some("mpac") {
                            "favored"
                        } else {
                            "green"
                        }
                        .into(),
                        favored: mask[row],
                    }),
                selection_score: score.filter(|s| s.is_finite()),
                selection_score_status: score
                    .filter(|s| !s.is_finite())
                    .map(|_| "negative_infinity".into()),
                inverse_rank: inverse.map(|i| i.ranks[row]),
                cdf_lower: inverse.map(|i| i.cdf_lower[row]),
                cdf_upper: inverse.map(|i| i.cdf_upper[row]),
            });
        }
        let watermark = scheme.map(|scheme| TraceWatermark {
            scheme: scheme.into(),
            status: snapshot
                .as_ref()
                .map(|s| match s.status {
                    TraceStatus::Applied => "applied",
                    TraceStatus::Warmup => "warmup",
                    TraceStatus::RepeatedContext => "repeated_context",
                })
                .unwrap_or("greedy")
                .into(),
            bias_delta: snapshot.as_ref().and_then(|s| s.bias_delta),
            payload_position: snapshot.as_ref().and_then(|s| s.payload_position),
            payload_symbol: snapshot.as_ref().and_then(|s| s.payload_symbol),
            key_position: snapshot.as_ref().and_then(|s| s.key_position),
            score_kind: snapshot.as_ref().and_then(|s| s.score_kind).map(|kind| {
                match kind {
                    SelectionScoreKind::NegativeExponentialCost => "negative_exponential_cost",
                    SelectionScoreKind::GumbelMax => "gumbel_max",
                    SelectionScoreKind::NegativeRank => "negative_rank",
                }
                .into()
            }),
            inverse: snapshot
                .as_ref()
                .and_then(|s| s.inverse.as_ref())
                .map(|i| TraceInverse {
                    uniform: i.uniform,
                    threshold: i.threshold,
                    total_weight: i.total_weight,
                }),
        });
        let layers_truncated = snapshot
            .as_ref()
            .is_some_and(|s| s.captured_layers < s.total_layers);
        let layers = snapshot
            .into_iter()
            .flat_map(|s| s.layers)
            .map(|l| TraceLayer {
                index: l.index,
                green_mass: l.green_mass,
                input_normalizer: l.input_normalizer,
                g_values: l.g_values,
                input_probabilities: l.input_probabilities,
                probabilities: l.probabilities,
            })
            .collect();
        let trace = SamplingTraceStep {
            generated_index: step.context.len() - step.prompt_len,
            context_length: step.context.len(),
            attempt: 0,
            selected_token_id: selected.token,
            selection_rule: if self.temperature.is_none() {
                "greedy"
            } else if keyed {
                "keyed_argmax"
            } else {
                "categorical"
            }
            .into(),
            candidate_count: before.len(),
            candidates_truncated: ids.len() < before.len(),
            watermark,
            candidates,
            layers,
            layers_truncated,
        };
        Ok((selected, trace))
    }
}

fn candidate_ids(selected: u32, before: &[f32], after: Option<&[f32]>, limit: usize) -> Vec<u32> {
    let before = partial_sort_top_k(&mut before.to_vec(), limit, false);
    let after = after
        .map(|p| partial_sort_top_k(&mut p.to_vec(), limit, false))
        .unwrap_or_default();
    let mut ids = vec![selected];
    for index in 0..limit {
        for entry in [before.get(index), after.get(index)].into_iter().flatten() {
            if ids.len() < limit && !ids.contains(&entry.0) {
                ids.push(entry.0);
            }
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use rand::{RngCore, SeedableRng};
    use std::collections::HashMap;

    const VOCAB: usize = 17;

    fn sampler(temperature: Option<f64>) -> Sampler {
        Sampler::new(
            temperature,
            4,
            None,
            Some(0.1),
            Some(0.2),
            Some(1.1),
            None,
            8,
            0.8,
            0.02,
            HashMap::from([(2, -0.5)]),
            vec![],
        )
        .unwrap()
    }

    fn parity(device: &Device) -> anyhow::Result<()> {
        let logits = Tensor::from_vec(
            (0..VOCAB).map(|i| (i as f32 * 0.7).sin()).collect(),
            VOCAB,
            device,
        )?;
        let configs = std::iter::once(None)
            .chain(crate::watermark::token_configs(VOCAB).into_iter().map(Some));
        for config in configs {
            for temperature in [None, Some(0.7)] {
                let sampler = sampler(temperature).with_watermark(config.as_ref())?;
                for context in [vec![1], vec![1, 2, 3, 4, 5], vec![1, 2, 3, 4, 1, 2, 3, 4]] {
                    for seed in 0..12 {
                        let ordinary_rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(seed)));
                        let traced_rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(seed)));
                        let ordinary = sampler.sample(
                            logits.clone(),
                            &context,
                            1,
                            true,
                            ordinary_rng.clone(),
                            false,
                            false,
                        )?;
                        let (traced, trace) = sampler.sample_traced(
                            logits.clone(),
                            TraceStepContext {
                                context: &context,
                                prompt_len: 1,
                                options: SamplingTraceConfig {
                                    max_candidates: 4,
                                    max_layers: 2,
                                    ..Default::default()
                                },
                            },
                            traced_rng.clone(),
                        )?;
                        assert_eq!(
                            serde_json::to_value(ordinary)?,
                            serde_json::to_value(&traced)?,
                            "{config:?}"
                        );
                        assert_eq!(
                            ordinary_rng.lock().unwrap().next_u64(),
                            traced_rng.lock().unwrap().next_u64()
                        );
                        assert_eq!(trace.candidates[0].token_id, traced.token);
                        assert!(trace.candidates.len() <= 4);
                        assert_eq!(trace.generated_index, context.len() - 1);
                        assert!(trace.layers.len() <= 2);
                        for layer in &trace.layers {
                            for row in 0..trace.candidates.len() {
                                let expected = layer.input_probabilities[row]
                                    * (1.0 + f64::from(layer.g_values[row]) - layer.green_mass);
                                assert!((layer.probabilities[row] - expected).abs() < 1e-12);
                            }
                        }
                        let serialized = serde_json::to_string(&trace)?;
                        assert!(!serialized.contains(&"42".repeat(32)));
                        if trace.selection_rule == "keyed_argmax" {
                            assert!(trace
                                .candidates
                                .iter()
                                .all(|c| c.post_watermark_probability.is_none()));
                            assert!(trace.watermark.as_ref().unwrap().score_kind.is_some());
                        } else if config.is_none() {
                            for candidate in trace.candidates {
                                assert_eq!(
                                    Some(candidate.pre_watermark_probability),
                                    candidate.post_watermark_probability
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn sampling_trace_preserves_draws_reporting_and_rng() -> anyhow::Result<()> {
        parity(&Device::Cpu)
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn sampling_trace_metal_host_path_parity() -> anyhow::Result<()> {
        parity(&Device::new_metal(0)?)
    }

    #[test]
    fn sampling_trace_uses_full_support_and_real_logit_stages() -> anyhow::Result<()> {
        let logits = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu)?;
        let sampler = Sampler::new(
            Some(0.5),
            2,
            None,
            None,
            None,
            None,
            None,
            0,
            1.0,
            1.0 - 1e-10,
            HashMap::from([(3, -1.0)]),
            vec![],
        )?;
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
        let (_, trace) = sampler.sample_traced(
            logits,
            TraceStepContext {
                context: &[1],
                prompt_len: 1,
                options: SamplingTraceConfig {
                    max_candidates: 2,
                    ..Default::default()
                },
            },
            rng,
        )?;
        assert!(
            trace
                .candidates
                .iter()
                .map(|c| c.pre_watermark_probability)
                .sum::<f64>()
                < 1.0
        );
        let row = trace.candidates.iter().find(|c| c.token_id == 3).unwrap();
        assert_eq!(row.input_logit, Some(4.0));
        assert_eq!(row.processed_logit, Some(3.0));
        Ok(())
    }

    #[test]
    fn sampling_trace_keyed_masked_scores_are_json_safe() -> anyhow::Result<()> {
        for config in crate::watermark::token_configs(4)
            .into_iter()
            .filter(|c| matches!(c.scheme(), "exponential" | "inverse_transform"))
        {
            let sampler = Sampler::new(
                Some(1.0),
                1,
                None,
                None,
                None,
                None,
                None,
                1,
                1.0,
                0.0,
                HashMap::new(),
                vec![],
            )?
            .with_watermark(Some(&config))?;
            let (_, trace) = sampler.sample_traced(
                Tensor::new(&[0.0f32, 1.0, 2.0, 3.0], &Device::Cpu)?,
                TraceStepContext {
                    context: &[1, 2, 3],
                    prompt_len: 2,
                    options: SamplingTraceConfig {
                        max_candidates: 4,
                        ..Default::default()
                    },
                },
                Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42))),
            )?;
            assert_eq!(trace.selected_token_id, 3);
            assert!(trace
                .candidates
                .iter()
                .any(
                    |c| c.selection_score_status.as_deref() == Some("negative_infinity")
                        && c.selection_score.is_none()
                ));
            let value = serde_json::to_value(trace)?;
            assert!(value["candidates"][0]
                .get("post_watermark_probability")
                .is_none());
        }
        Ok(())
    }
}
