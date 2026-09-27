#![cfg(feature = "metal")]

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_keyed_rng::{
    metal::{probe, select, DeviceHistory, Filter, LogitsSampling, INVALID_TOKEN},
    reference, Purpose, SequenceKey,
};

const CASES: usize = 4_119;
const DISTRIBUTION_ROWS: usize = 40_000;
const FUSED_STEPS: usize = 16;
const CDF_TOLERANCE: f64 = 2e-6;
const LOGPROB_TOLERANCE: f64 = 2e-5;
const BATCH_TEST_VOCAB: usize = 2051;

#[test]
fn batched_history_survives_reordering_splits_and_joining() -> Result<()> {
    let device = Device::new_metal(0)?;
    let schedule: &[&[usize]] = &[
        &[0, 1, 2],
        &[0, 1, 2],
        &[2, 0],
        &[1, 3],
        &[0, 3, 2],
        &[3, 2, 1, 0],
    ];
    for sampling in [
        filter(),
        Filter {
            greedy: true,
            ..filter()
        },
        Filter {
            top_k: 40,
            top_p: 0.9,
            ..filter()
        },
        Filter {
            min_p: 0.2,
            ..filter()
        },
    ] {
        let mut histories = (0..4)
            .map(|id| DeviceHistory::new(&[id], 1, 32, BATCH_TEST_VOCAB, &device).map(Some))
            .collect::<Result<Vec<_>>>()?;
        let mut contexts = (0..4).map(|id| vec![id as u32]).collect::<Vec<_>>();
        for order in schedule {
            let mut active = order
                .iter()
                .map(|&id| histories[id].take().unwrap())
                .collect::<Vec<_>>();
            let params = order
                .iter()
                .map(|&id| LogitsSampling {
                    key: SequenceKey::new(42 + id as u64),
                    attempt: 0,
                    temperature: 0.8,
                    frequency: 0.1,
                    presence: 0.3,
                    repetition: 1.1,
                    min_p: sampling.min_p,
                    greedy: sampling.greedy,
                })
                .collect::<Vec<_>>();
            let values = order
                .iter()
                .flat_map(|&id| {
                    (0..BATCH_TEST_VOCAB)
                        .map(move |i| ((i * 17 + id * 31) % 101) as f32 / 17.0 - 3.0)
                })
                .collect::<Vec<_>>();
            let logits =
                Tensor::from_vec(values.clone(), (order.len(), BATCH_TEST_VOCAB), &device)?;
            let selections = DeviceHistory::sample_batch(
                &logits,
                &mut active.iter_mut().collect::<Vec<_>>(),
                &params,
                sampling,
            )?;
            for (row, &id) in order.iter().enumerate() {
                active[row].commit_with_stop_tokens(&selections[row], &[])?;
                let (token, logprob) = selections[row].readback()?[0];
                let mut counts = vec![0u32; BATCH_TEST_VOCAB];
                for &t in &contexts[id][1..] {
                    counts[t as usize] += 1;
                }
                let scores = (0..BATCH_TEST_VOCAB)
                    .map(|i| {
                        let mut v = values[row * BATCH_TEST_VOCAB + i]
                            - 0.1 * counts[i] as f32
                            - if counts[i] > 0 { 0.3 } else { 0.0 };
                        if i == id || counts[i] > 0 {
                            v = if v >= 0.0 { v / 1.1 } else { v * 1.1 };
                        }
                        (v * 1.25) as f64
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights = scores.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
                let total: f64 = weights.iter().sum();
                assert!(
                    (logprob as f64 - (scores[token as usize] - max - total.ln())).abs()
                        < LOGPROB_TOLERANCE
                );
                if sampling.greedy {
                    assert_eq!(
                        token as usize,
                        scores.iter().position(|&v| v == max).unwrap()
                    );
                } else {
                    let mut ids = (0..BATCH_TEST_VOCAB).collect::<Vec<_>>();
                    if sampling.top_k > 0 {
                        ids.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
                        ids.truncate(sampling.top_k);
                    }
                    let cutoff =
                        ids.iter().map(|&i| weights[i]).sum::<f64>() * sampling.top_p as f64;
                    let mut mass = 0.0;
                    ids.retain(|&i| {
                        let keep = mass < cutoff && weights[i] > sampling.min_p as f64;
                        mass += weights[i];
                        keep
                    });
                    let retained = ids.iter().map(|&i| weights[i]).sum::<f64>();
                    let index = ids.iter().position(|&i| i == token as usize).unwrap();
                    let before = ids[..index].iter().map(|&i| weights[i]).sum::<f64>() / retained;
                    let after = before + weights[token as usize] / retained;
                    let uniform = params[row].key.uniform(
                        Purpose::Generation,
                        (contexts[id].len() - 1) as u32,
                        0,
                    ) as f64;
                    assert!(uniform >= before - CDF_TOLERANCE && uniform < after + CDF_TOLERANCE);
                }
                contexts[id].push(token);
                assert_eq!(active[row].tokens()?.to_vec1::<u32>()?, contexts[id]);
                assert_eq!(
                    active[row].state().to_vec1::<u32>()?,
                    vec![
                        contexts[id].len() as u32,
                        (contexts[id].len() - 1) as u32,
                        1
                    ]
                );
            }
            for (&id, history) in order.iter().zip(active) {
                histories[id] = Some(history);
            }
        }
    }
    Ok(())
}

#[test]
fn partial_topk_matches_stable_reference_across_blocks() -> Result<()> {
    let device = Device::new_metal(0)?;
    for width in [1, 33, 1023, 1024, 1025, 32771, 131072] {
        let rows = 3;
        let values = (0..rows * width)
            .map(|i| ((i * 3719) % 977) as f32)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(values.clone(), (rows, width), &device)?;
        for k in [1, 40, 128] {
            let (weights, ids) = mistralrs_keyed_rng::metal::top_candidates(&input, k)?;
            let weights = weights.to_vec2::<f32>()?;
            let ids = ids.to_vec2::<u32>()?;
            for row in 0..rows {
                let mut expected = (0..width).collect::<Vec<_>>();
                expected.sort_unstable_by(|&a, &b| {
                    values[row * width + b]
                        .total_cmp(&values[row * width + a])
                        .then(a.cmp(&b))
                });
                expected.truncate(k.min(width));
                assert_eq!(
                    ids[row],
                    expected.iter().map(|&i| i as u32).collect::<Vec<_>>()
                );
                assert_eq!(
                    weights[row],
                    expected
                        .iter()
                        .map(|&i| values[row * width + i])
                        .collect::<Vec<_>>()
                );
            }
        }
    }
    let invalid = Tensor::new(&[[0.5f32, f32::NAN, 0.2, f32::NAN]], &device)?;
    let (weights, ids) = mistralrs_keyed_rng::metal::top_candidates(&invalid, 2)?;
    assert_eq!(ids.to_vec2::<u32>()?[0], vec![1, 3]);
    assert!(select(
        &weights,
        &ids,
        &weights,
        &events(42, 1, 0, &device)?,
        filter()
    )?
    .readback()
    .is_err());
    Ok(())
}

#[test]
fn concurrent_shared_record_readbacks_wait_for_their_producer() -> Result<()> {
    let device = Device::new_metal(0)?;
    std::thread::scope(|scope| {
        let handles = (0..4)
            .map(|row| {
                let device = device.clone();
                scope.spawn(move || -> Result<()> {
                    let weights = Tensor::new(&[[0.0f32, 1.0]], &device)?;
                    let ids = Tensor::new(&[[10u32 + row, 20 + row]], &device)?;
                    let event = events(row as u64, 1, 0, &device)?;
                    for _ in 0..32 {
                        let selected = select(&weights, &ids, &weights, &event, filter())?;
                        assert_eq!(selected.readback()?[0].0, 20 + row);
                    }
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap()?;
        }
        Ok(())
    })
}

#[test]
#[ignore = "manual GPU kernel/dispatch profiling"]
fn benchmark_resident_logits() -> Result<()> {
    let device = Device::new_metal(0)?;
    let key = SequenceKey::new(42);
    let stops = Tensor::new(&[INVALID_TOKEN], &device)?;
    const STEPS: usize = 256;
    for vocab in [32768, 131072] {
        let logits = Tensor::new(
            (0..vocab)
                .map(|i| ((i * 17) % 101) as f32 / 17.0 - 3.0)
                .collect::<Vec<_>>(),
            &device,
        )?;
        for greedy in [true, false] {
            for compact in [false, true] {
                let mut history = DeviceHistory::new(&[0], 1, STEPS + 2, vocab, &device)?;
                device.synchronize()?;
                let start = std::time::Instant::now();
                for _ in 0..STEPS {
                    let selected = history.sample_logits(
                        &logits,
                        LogitsSampling {
                            key,
                            attempt: 0,
                            temperature: 0.8,
                            frequency: 0.0,
                            presence: 0.0,
                            repetition: 1.0,
                            min_p: 0.0,
                            greedy,
                        },
                    )?;
                    history.commit(&selected, &stops)?;
                    if compact {
                        std::hint::black_box(selected.readback()?);
                    }
                }
                device.synchronize()?;
                println!(
                    "resident vocab={vocab} greedy={greedy} compact={compact} us={:.3}",
                    start.elapsed().as_secs_f64() * 1e6 / STEPS as f64
                );
            }
        }
    }
    Ok(())
}

#[test]
fn fused_logits_parallel_boundaries_penalties_and_device_feedback() -> Result<()> {
    let device = Device::new_metal(0)?;
    let key = SequenceKey::new(42);
    let stops = Tensor::new(&[INVALID_TOKEN], &device)?;
    for vocab in [1, 31, 32, 33, 255, 256, 257, 1025, 32771] {
        let values = (0..vocab)
            .map(|i| ((i * 17) % 101) as f32 / 17.0 - 3.0)
            .collect::<Vec<_>>();
        let logits = Tensor::new(values.as_slice(), &device)?;
        for (greedy, min_p) in [(true, 0.0), (false, 0.0), (false, 0.2)] {
            let mut history = DeviceHistory::new(&[0], 1, FUSED_STEPS + 2, vocab, &device)?;
            let mut selections = Vec::new();
            for _ in 0..FUSED_STEPS {
                let selection = history.sample_logits(
                    &logits,
                    LogitsSampling {
                        key,
                        attempt: 0,
                        temperature: 0.8,
                        frequency: 0.1,
                        presence: 0.3,
                        repetition: 1.1,
                        min_p,
                        greedy,
                    },
                )?;
                history.commit(&selection, &stops)?;
                selections.push(selection);
            }
            let result = mistralrs_keyed_rng::metal::Selection::readback_batch(
                &selections.iter().collect::<Vec<_>>(),
            )?;
            let mut counts = vec![0u32; vocab];
            let mut committed = vec![0u32];
            for (position, result) in result.into_iter().enumerate() {
                let (token, logprob) = result?;
                let scores = values
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let mut v =
                            v - 0.1 * counts[i] as f32 - if counts[i] > 0 { 0.3 } else { 0.0 };
                        if i == 0 || counts[i] > 0 {
                            v = if v >= 0.0 { v / 1.1 } else { v * 1.1 };
                        }
                        (v * 1.25) as f64
                    })
                    .collect::<Vec<_>>();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights = scores
                    .iter()
                    .map(|&v| (v - maximum).exp())
                    .collect::<Vec<_>>();
                let mass: f64 = weights.iter().sum();
                assert!(
                    (logprob as f64 - (scores[token as usize] - maximum - mass.ln())).abs()
                        < LOGPROB_TOLERANCE
                );
                if greedy {
                    assert_eq!(
                        token as usize,
                        scores.iter().position(|&v| v == maximum).unwrap()
                    );
                } else {
                    let kept = weights
                        .iter()
                        .map(|&w| if w > min_p as f64 { w } else { 0.0 })
                        .collect::<Vec<_>>();
                    let retained: f64 = kept.iter().sum();
                    let before: f64 = kept[..token as usize].iter().sum::<f64>() / retained;
                    let after = before + kept[token as usize] / retained;
                    let uniform = key.uniform(Purpose::Generation, position as u32, 0) as f64;
                    assert!(uniform >= before - CDF_TOLERANCE && uniform < after + CDF_TOLERANCE,
                        "vocab={vocab} position={position} token={token} uniform={uniform} interval={before}..{after}");
                }
                counts[token as usize] += 1;
                committed.push(token);
            }
            assert_eq!(history.tokens()?.to_vec1::<u32>()?, committed);
        }
    }
    Ok(())
}

fn filter() -> Filter {
    Filter {
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        greedy: false,
    }
}

fn events(seed: u64, rows: usize, attempt: u32, device: &Device) -> Result<Tensor> {
    let key = SequenceKey::new(seed).words(Purpose::Generation);
    let values = (0..rows)
        .flat_map(|position| [key[0], key[1], position as u32, attempt])
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (rows, 4), device)
}

#[test]
fn exact_cpu_metal_bits_and_offsets() -> Result<()> {
    let device = Device::new_metal(0)?;
    let mut state = 0x9123_1452_8123_9231u64;
    let mut rows = vec![
        [0u32; 4],
        [u32::MAX; 4],
        [0x13198a2e, 0x03707344, 0x243f6a88, 0x85a308d3],
        [0, u32::MAX, 0x8000_0000, 0x7fff_ffff],
    ];
    while rows.len() < CASES {
        let row = std::array::from_fn(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u32
        });
        rows.push(row);
    }
    let mut padded = vec![0u32; 4];
    padded.extend(rows.iter().flatten().copied());
    let input = Tensor::from_vec(padded, (CASES + 1, 4), &device)?.narrow(0, 1, CASES)?;
    let result = probe(&input)?.to_vec2::<u32>()?;
    for (row, result) in rows.iter().zip(result) {
        assert_eq!(result, reference([row[2], row[3]], [row[0], row[1]]));
    }
    Ok(())
}

#[test]
fn categorical_distribution_replay_retries_and_batch_permutations() -> Result<()> {
    let device = Device::new_metal(0)?;
    let weights = Tensor::new(&[[0.6f32, 0.3, 0.1]], &device)?
        .broadcast_as((DISTRIBUTION_ROWS, 3))?
        .contiguous()?;
    let ids = Tensor::new(&[[17u32, 90, 30_000_001]], &device)?
        .broadcast_as((DISTRIBUTION_ROWS, 3))?
        .contiguous()?;
    let first_events = events(42, DISTRIBUTION_ROWS, 0, &device)?;
    let first = select(&weights, &ids, &weights, &first_events, filter())?.readback()?;
    let second = select(
        &weights,
        &ids,
        &weights,
        &events(42, DISTRIBUTION_ROWS, 1, &device)?,
        filter(),
    )?
    .readback()?;
    let mut counts = [0; 3];
    let mut conditional = [0; 3];
    for ((token, logprob), (retry, _)) in first.iter().zip(&second) {
        let index = [17, 90, 30_000_001]
            .iter()
            .position(|v| v == token)
            .unwrap();
        counts[index] += 1;
        assert!((*logprob - [0.6f32, 0.3, 0.1][index].ln()).abs() < 1e-5);
        if *token == 17 {
            conditional[[17, 90, 30_000_001]
                .iter()
                .position(|v| v == retry)
                .unwrap()] += 1;
        }
    }
    for (counts, total) in [
        (counts, DISTRIBUTION_ROWS),
        (conditional, conditional.iter().sum()),
    ] {
        for (count, expected) in counts.iter().zip([0.6, 0.3, 0.1]) {
            assert!((*count as f64 / total as f64 - expected).abs() < 0.015);
        }
    }
    for row in [100, 9, 2_015, 0] {
        let actual = select(
            &weights.narrow(0, row, 1)?,
            &ids.narrow(0, row, 1)?,
            &weights.narrow(0, row, 1)?,
            &first_events.narrow(0, row, 1)?,
            filter(),
        )?
        .readback()?;
        assert_eq!(actual[0], first[row]);
    }
    Ok(())
}

#[test]
fn filters_exclusions_reporting_and_invalid_mass() -> Result<()> {
    let device = Device::new_metal(0)?;
    let weights = Tensor::new(&[[0.6f32, 0.3, 0.1, 0.0]], &device)?;
    let reporting = Tensor::new(&[[0.2f32, 0.5, 0.1, 0.2]], &device)?;
    let ids = Tensor::new(&[[11u32, 12, 13, 14]], &device)?;
    let event = events(9, 1, 0, &device)?;
    let selected = select(
        &weights,
        &ids,
        &reporting,
        &event,
        Filter {
            top_k: 2,
            top_p: 0.5,
            ..filter()
        },
    )?
    .readback()?;
    assert_eq!(selected[0].0, 11);
    assert!((selected[0].1 - 0.2f32.ln()).abs() < 1e-5);
    assert_eq!(
        select(
            &weights,
            &ids,
            &reporting,
            &event,
            Filter {
                min_p: 0.9,
                ..filter()
            }
        )?
        .readback()?[0]
            .0,
        11
    );
    for row in [
        [0.0f32; 4],
        [f32::NAN, 0., 0., 0.],
        [-1., 1., 0., 0.],
        [f32::INFINITY, 0., 0., 0.],
    ] {
        let weights = Tensor::new(&[row], &device)?;
        assert!(select(&weights, &ids, &reporting, &event, filter())?
            .readback()
            .is_err());
    }
    let ties = Tensor::new(&[[0.5f32, 0.5, 0., 0.]], &device)?;
    let reversed_ids = Tensor::new(&[[12u32, 11, 13, 14]], &device)?;
    assert_eq!(
        select(
            &ties,
            &reversed_ids,
            &reporting,
            &event,
            Filter {
                greedy: true,
                ..filter()
            }
        )?
        .readback()?[0]
            .0,
        11
    );
    let scores = Tensor::new(&[[-10.0f32, -4.0, -2.0, f32::NEG_INFINITY]], &device)?;
    assert_eq!(
        select(
            &scores,
            &ids,
            &reporting,
            &event,
            Filter {
                greedy: true,
                top_k: 1,
                ..filter()
            }
        )?
        .readback()?[0]
            .0,
        13
    );
    Ok(())
}

#[test]
fn queued_device_feedback_penalties_eos_and_diagnostics() -> Result<()> {
    let device = Device::new_metal(0)?;
    let mut history = DeviceHistory::new(&[0, 1, 1], 3, 32, 3, &device)?;
    let key = SequenceKey::new(99);
    let stops = Tensor::new(&[2u32], &device)?;
    let weights = Tensor::new(&[[0.0f32, 1.0, 0.0]], &device)?;
    let ids = Tensor::new(&[[0u32, 1, 2]], &device)?;
    for i in 0..8 {
        let event = history.event(key, Purpose::Generation, 0)?;
        let diagnostic = history.event(key, Purpose::Diagnostics, 0)?;
        let _diagnostic_bits = probe(&diagnostic)?;
        let selection = select(&weights, &ids, &weights, &event, filter())?;
        history.commit(&selection, &stops)?;
        let next = history.next_input(4 + i, &device).unwrap();
        assert_eq!(next.dims(), &[1, 1]);
    }
    let logits = Tensor::new(&[10.0f32, 10., -10.], &device)?;
    let penalized = history.penalties(&logits, 0.5, 1.0, 2.0)?;
    assert_eq!(penalized.to_vec1::<f32>()?, vec![5., 2.5, -10.]);
    assert_eq!(history.state().to_vec1::<u32>()?, vec![11, 8, 1]);
    assert_eq!(
        history.tokens()?.to_vec1::<u32>()?,
        [vec![0, 1, 1], vec![1; 8]].concat()
    );
    let event = history.event(key, Purpose::Generation, 1)?;
    let got = probe(&event)?.to_vec2::<u32>()?[0][2];
    assert_eq!(got, key.uniform(Purpose::Generation, 8, 1).to_bits());
    let eos = Tensor::new(&[[0.0f32, 0.0, 1.0]], &device)?;
    let selection = select(&eos, &ids, &eos, &event, filter())?;
    history.commit(&selection, &stops)?;
    assert_eq!(history.state().to_vec1::<u32>()?, vec![12, 9, 0]);
    Ok(())
}

#[test]
fn rejects_cross_device_and_bad_shapes() -> Result<()> {
    let first = Device::new_metal(0)?;
    let second = Device::new_metal(0)?;
    let weights = Tensor::ones((1, 3), DType::F32, &first)?;
    let ids = Tensor::new(&[[0u32, 1, 2]], &second)?;
    assert!(select(
        &weights,
        &ids,
        &weights,
        &events(1, 1, 0, &first)?,
        filter()
    )
    .is_err());
    assert!(probe(&Tensor::zeros((1, 3), DType::U32, &first)?).is_err());
    Ok(())
}

#[test]
fn vocabulary_sized_candidates_are_stable_and_keep_token_ids() -> Result<()> {
    let device = Device::new_metal(0)?;
    for width in [1, 3, 127, 1_029, 131_071] {
        let values = (0..width)
            .map(|i| ((i * 31) % 113) as f32)
            .collect::<Vec<_>>();
        let probs = Tensor::new(values.as_slice(), &device)?.unsqueeze(0)?;
        let (sorted, ids) = mistralrs_keyed_rng::metal::candidates(&probs, true)?;
        let mut expected = (0..width as u32).collect::<Vec<_>>();
        expected.sort_by(|&a, &b| {
            values[b as usize]
                .total_cmp(&values[a as usize])
                .then(a.cmp(&b))
        });
        assert_eq!(ids.to_vec2::<u32>()?[0], expected);
        assert_eq!(
            sorted.to_vec2::<f32>()?[0],
            expected
                .iter()
                .map(|&id| values[id as usize])
                .collect::<Vec<_>>()
        );
    }
    let invalid = Tensor::new(&[[0.5f32, f32::NAN, 0.2, f32::NAN]], &device)?;
    let (weights, ids) = mistralrs_keyed_rng::metal::candidates(&invalid, true)?;
    assert_eq!(ids.to_vec2::<u32>()?[0], vec![1, 3, 0, 2]);
    assert!(select(
        &weights,
        &ids,
        &weights,
        &events(42, 1, 0, &device)?,
        filter()
    )?
    .readback()
    .is_err());
    Ok(())
}
