use super::*;
use mistralrs_keyed_rng::{Purpose, RNG_VERSION};

pub(crate) struct KeyedSampleContext<'a> {
    pub tokens: &'a [u32],
    pub prompt_len: usize,
    pub return_logprobs: bool,
    pub uniform: f32,
}

pub(crate) fn keyed_sampling_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        std::env::var("MISTRALRS_SAMPLING_RNG").is_ok_and(|value| value == RNG_VERSION)
    });
    *ENABLED
}

impl Sampler {
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
        self.filter_top_kp_min_p(&mut sampling);
        Self::normalize_probs(&mut sampling)?;
        let mut order = (0..sampling.len()).collect::<Vec<_>>();
        if self.top_k > 0 || (self.top_p > 0.0 && self.top_p < 1.0) {
            order.sort_unstable_by(|&a, &b| reporting[b].total_cmp(&reporting[a]).then(a.cmp(&b)));
        }
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
        event: &Tensor,
    ) -> Result<(mistralrs_keyed_rng::metal::Selection, Tensor)> {
        use mistralrs_keyed_rng::metal::{select, Filter};
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
        let sorted = self.temperature.is_some()
            && (self.top_k > 0 || (self.top_p > 0.0 && self.top_p < 1.0));
        let (weights, ids) = mistralrs_keyed_rng::metal::candidates(&probs.unsqueeze(0)?, sorted)?;
        let selection = select(
            &weights,
            &ids,
            &weights,
            event,
            Filter {
                top_k: self.top_k.max(0) as usize,
                top_p: self.top_p as f32,
                min_p: self.min_p as f32,
                greedy: self.temperature.is_none(),
            },
        )?;
        Ok((selection, probs))
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
                    self.submit_keyed_metal(&logits, attempt, max_model_len)?;
                let (token, logprob) = selection.readback()?[0];
                self.pending_keyed_selection = Some(selection);
                if return_logprobs {
                    return self.sampler().logprobs_from_probs(
                        token,
                        &reporting.to_vec1::<f32>()?,
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
    ) -> Result<(mistralrs_keyed_rng::metal::Selection, Tensor)> {
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
        let history = self.keyed_history.as_ref().unwrap();
        let event = history.event(self.keyed_sampling_key, Purpose::Generation, attempt)?;
        self.sampler().sample_keyed_metal(logits, history, &event)
    }

    #[cfg(feature = "metal")]
    pub(crate) fn commit_keyed_selection(&mut self, eos: Option<&[u32]>) -> Result<()> {
        if let Some(selection) = self.pending_keyed_selection.take() {
            let mut stops = if self
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
            if stops.is_empty() {
                stops.push(mistralrs_keyed_rng::metal::INVALID_TOKEN);
            }
            let stops = Tensor::new(stops.as_slice(), selection.tokens().device())?;
            self.keyed_history
                .as_mut()
                .unwrap()
                .commit(&selection, &stops)?;
        }
        Ok(())
    }
}
