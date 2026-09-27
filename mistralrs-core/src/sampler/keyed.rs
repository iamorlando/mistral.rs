use super::*;
use mistralrs_keyed_rng::{Purpose, RNG_VERSION};

pub(crate) struct KeyedSampleContext<'a> {
    pub tokens: &'a [u32],
    pub prompt_len: usize,
    pub return_logprobs: bool,
    pub uniform: f32,
}

#[cfg(feature = "metal")]
pub(crate) struct KeyedMetalContext {
    pub key: mistralrs_keyed_rng::SequenceKey,
    pub attempt: u32,
    pub return_logprobs: bool,
}

pub(crate) fn keyed_sampling_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        std::env::var("MISTRALRS_SAMPLING_RNG").is_ok_and(|value| value == RNG_VERSION)
    });
    *ENABLED
}

impl Sampler {
    #[cfg(feature = "metal")]
    pub(crate) fn keyed_metal_batch_compatible(&self, other: &Self) -> bool {
        self.can_sample_keyed_metal()
            && other.can_sample_keyed_metal()
            && self.logits_bias.is_empty()
            && other.logits_bias.is_empty()
            && self.temperature.is_none() == other.temperature.is_none()
            && self.top_k == other.top_k
            && self.top_p == other.top_p
            && self.min_p == other.min_p
    }

    #[cfg(feature = "metal")]
    pub(crate) fn keyed_metal_params(
        &self,
        key: mistralrs_keyed_rng::SequenceKey,
    ) -> mistralrs_keyed_rng::metal::LogitsSampling {
        mistralrs_keyed_rng::metal::LogitsSampling {
            key,
            attempt: 0,
            temperature: self.temperature.unwrap_or(1.0) as f32,
            frequency: self.frequency_penalty.unwrap_or(0.0),
            presence: self.presence_penalty.unwrap_or(0.0),
            repetition: self.repetition_penalty.unwrap_or(1.0),
            min_p: self.min_p as f32,
            greedy: self.temperature.is_none(),
        }
    }

    #[cfg(feature = "metal")]
    pub(crate) fn keyed_metal_filter(&self) -> mistralrs_keyed_rng::metal::Filter {
        mistralrs_keyed_rng::metal::Filter {
            top_k: self.top_k.max(0) as usize,
            top_p: self.top_p as f32,
            min_p: self.min_p as f32,
            greedy: self.temperature.is_none(),
        }
    }

    pub(crate) fn sample_keyed_cpu(
        &self,
        logits: Tensor,
        context: KeyedSampleContext<'_>,
    ) -> Result<Logprobs> {
        let KeyedSampleContext {
            tokens: context,
            prompt_len,
            return_logprobs,
            uniform,
        } = context;
        let mut logits = self.apply_penalties(logits.to_vec1()?, context, prompt_len)?;
        for processor in &self.logits_processors {
            logits = processor.apply(&logits, context)?;
        }
        let Some(temperature) = self.temperature else {
            return self.sample_argmax(logits, return_logprobs);
        };
        let reporting =
            candle_nn::ops::softmax_last_dim(&(&logits / temperature)?)?.to_vec1::<f32>()?;
        let mut sampling = reporting.clone();
        let order = if self.top_k > 0 || (self.top_p > 0.0 && self.top_p < 1.0) {
            let k = if self.top_k > 0 {
                self.top_k as usize
            } else {
                sampling.len()
            };
            let retained = partial_sort_top_k(&mut sampling, k, true);
            if self.top_p > 0.0 && self.top_p < 1.0 {
                let cutoff = top_p_cutoff(self.top_p as f32, retained.iter().map(|(_, p)| *p));
                let mut cumulative = 0.0;
                for &(token, probability) in &retained {
                    if cumulative >= cutoff {
                        sampling[token as usize] = 0.0;
                    } else {
                        cumulative += probability;
                    }
                }
            }
            retained
                .into_iter()
                .map(|(token, _)| token as usize)
                .collect::<Vec<_>>()
        } else {
            (0..sampling.len()).collect::<Vec<_>>()
        };
        if self.min_p > 0.0 && self.min_p < 1.0 {
            let threshold = self.min_p as f32 * reporting.iter().copied().fold(0.0, f32::max);
            for probability in &mut sampling {
                if *probability <= threshold {
                    *probability = 0.0;
                }
            }
        }
        Self::normalize_probs(&mut sampling)?;
        let mut cumulative = 0.0;
        let mut chosen = None;
        for token in order {
            let weight = sampling[token];
            if weight <= 0.0 {
                continue;
            }
            chosen = Some(token as u32);
            cumulative += weight;
            if cumulative > uniform {
                break;
            }
        }
        let token = chosen.ok_or_else(|| Error::Msg("empty keyed sampling distribution".into()))?;
        self.logprobs_from_probs(token, &reporting, return_logprobs)
    }

    #[cfg(feature = "metal")]
    pub(crate) fn can_sample_keyed_metal(&self) -> bool {
        self.logits_processors.is_empty()
            && self
                .dry_params
                .as_ref()
                .is_none_or(|params| params.multiplier == 0.0)
    }

    #[cfg(feature = "metal")]
    pub(crate) fn sample_keyed_metal(
        &self,
        logits: &Tensor,
        history: &mistralrs_keyed_rng::metal::DeviceHistory,
        context: KeyedMetalContext,
    ) -> Result<(mistralrs_keyed_rng::metal::Selection, Option<Tensor>)> {
        use mistralrs_keyed_rng::metal::{select, Filter, LogitsSampling};
        let direct = self.logits_bias.is_empty()
            && (self.temperature.is_none()
                || (self.top_k <= 0 && !(self.top_p > 0.0 && self.top_p < 1.0)));
        let direct = if direct {
            let selection = history.sample_logits(
                logits,
                LogitsSampling {
                    key: context.key,
                    attempt: context.attempt,
                    temperature: self.temperature.unwrap_or(1.0) as f32,
                    frequency: self.frequency_penalty.unwrap_or(0.0),
                    presence: self.presence_penalty.unwrap_or(0.0),
                    repetition: self.repetition_penalty.unwrap_or(1.0),
                    min_p: self.min_p as f32,
                    greedy: self.temperature.is_none(),
                },
            )?;
            if !context.return_logprobs {
                return Ok((selection, None));
            }
            Some(selection)
        } else {
            None
        };
        let mut logits = history.penalties(
            logits,
            self.frequency_penalty.unwrap_or(0.0),
            self.presence_penalty.unwrap_or(0.0),
            self.repetition_penalty.unwrap_or(1.0),
        )?;
        if !self.logits_bias.is_empty() {
            let mut bias = vec![0.0f32; logits.elem_count()];
            for (&token, &value) in &self.logits_bias {
                if let Some(slot) = bias.get_mut(token as usize) {
                    *slot = value;
                }
            }
            logits = (&logits + Tensor::new(bias, logits.device())?)?;
        }
        let temperature = self.temperature.unwrap_or(1.0);
        let probs = candle_nn::ops::softmax_last_dim(&(&logits / temperature)?)?;
        if let Some(selection) = direct {
            return Ok((selection, Some(probs)));
        }
        let sorted = self.temperature.is_some()
            && (self.top_k > 0 || (self.top_p > 0.0 && self.top_p < 1.0));
        let (weights, ids) = if sorted
            && self.top_k > 0
            && self.top_k as usize <= mistralrs_keyed_rng::metal::MAX_PARTIAL_TOP_K
        {
            mistralrs_keyed_rng::metal::top_candidates(&probs.unsqueeze(0)?, self.top_k as usize)?
        } else {
            mistralrs_keyed_rng::metal::candidates(&probs.unsqueeze(0)?, sorted)?
        };
        let event = history.event(context.key, Purpose::Generation, context.attempt)?;
        let selection = select(
            &weights,
            &ids,
            &weights,
            &event,
            Filter {
                top_k: self.top_k.max(0) as usize,
                top_p: self.top_p as f32,
                min_p: self.min_p as f32,
                greedy: self.temperature.is_none(),
            },
        )?;
        Ok((selection, Some(probs)))
    }
}

impl crate::sequence::Sequence {
    pub(crate) fn sample_keyed_attempt(
        &mut self,
        logits: Tensor,
        return_logprobs: bool,
        attempt: u32,
        max_model_len: usize,
    ) -> Result<Logprobs> {
        let key = self.keyed_sampling_key;
        #[cfg(feature = "metal")]
        {
            self.pending_keyed_selection = None;
            if logits.device().is_metal() && self.sampler().can_sample_keyed_metal() {
                let (selection, reporting) =
                    self.submit_keyed_metal(&logits, attempt, max_model_len, return_logprobs)?;
                let (token, logprob) = selection.readback()?[0];
                self.pending_keyed_selection = Some(selection);
                if return_logprobs {
                    return self.sampler().logprobs_from_probs(
                        token,
                        &reporting.expect("reporting requested").to_vec1::<f32>()?,
                        true,
                    );
                }
                return Ok(Logprobs {
                    token,
                    logprob,
                    bytes: None,
                    top_logprobs: None,
                });
            }
        }
        #[cfg(not(feature = "metal"))]
        let _ = max_model_len;
        let position = u32::try_from(self.generated_len()).map_err(Error::msg)?;
        self.sampler().sample_keyed_cpu(
            logits,
            KeyedSampleContext {
                tokens: self.committed_toks(),
                prompt_len: self.prompt_tokens(),
                return_logprobs,
                uniform: key.uniform(Purpose::Generation, position, attempt),
            },
        )
    }

    #[cfg(feature = "metal")]
    pub(crate) fn submit_keyed_metal(
        &mut self,
        logits: &Tensor,
        attempt: u32,
        max_model_len: usize,
        return_logprobs: bool,
    ) -> Result<(mistralrs_keyed_rng::metal::Selection, Option<Tensor>)> {
        self.prepare_keyed_metal_history(logits, max_model_len)?;
        let history = self.keyed_history.as_ref().unwrap();
        self.sampler().sample_keyed_metal(
            logits,
            history,
            KeyedMetalContext {
                key: self.keyed_sampling_key,
                attempt,
                return_logprobs,
            },
        )
    }

    #[cfg(feature = "metal")]
    pub(crate) fn prepare_keyed_metal_history(
        &mut self,
        logits: &Tensor,
        max_model_len: usize,
    ) -> Result<()> {
        self.pending_keyed_selection = None;
        let len = self.committed_toks().len();
        let vocab = logits.elem_count();
        if self
            .keyed_history
            .as_ref()
            .is_none_or(|history| !history.matches(len, vocab, logits.device()))
        {
            let capacity = self
                .prompt_tokens()
                .saturating_add(self.max_generation_len(max_model_len).max(1))
                .min(max_model_len.saturating_add(1));
            self.keyed_history = Some(mistralrs_keyed_rng::metal::DeviceHistory::new(
                self.committed_toks(),
                self.prompt_tokens(),
                capacity,
                vocab,
                logits.device(),
            )?);
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    pub(crate) fn commit_keyed_selection(&mut self, eos: Option<&[u32]>) -> Result<()> {
        if let Some(selection) = self.pending_keyed_selection.take() {
            self.commit_keyed_metal_selection(&selection, eos)?;
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    pub(crate) fn commit_keyed_metal_selection(
        &mut self,
        selection: &mistralrs_keyed_rng::metal::Selection,
        eos: Option<&[u32]>,
    ) -> Result<()> {
        let stops = if self
            .tool_call_state
            .as_ref()
            .is_some_and(|state| state.required_tool_call_unsatisfied())
        {
            Vec::new()
        } else {
            self.stop_tokens()
                .iter()
                .copied()
                .chain(eos.unwrap_or_default().iter().copied())
                .collect::<Vec<_>>()
        };
        self.keyed_history
            .as_mut()
            .unwrap()
            .commit_with_stop_tokens(selection, &stops)?;
        Ok(())
    }
}
