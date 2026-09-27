use symcoe_core::rng::prng::{
    engines::threefry::cpu::threefry::{threefry2x32, uniform_f32_from_u32, Threefry2x32Key},
    types::PrngKey,
};

#[cfg(feature = "metal")]
pub mod metal;

pub const RNG_VERSION: &str = "keyed-threefry2x32-v1";
const STREAM_VERSION: u64 = 1;
const PROMPT_REBASE_DOMAIN: u64 = 0x7265_6261_7365_0001;

#[derive(Clone, Copy, Debug)]
#[repr(u64)]
pub enum Purpose {
    Generation = 0,
    Draft = 1,
    Acceptance = 2,
    Correction = 3,
    Diagnostics = 4,
}

#[derive(Clone, Copy, Debug)]
pub struct SequenceKey(PrngKey);

impl SequenceKey {
    pub fn new(seed: u64) -> Self {
        Self(PrngKey::new(seed).fold_in(STREAM_VERSION))
    }

    pub fn words(self, purpose: Purpose) -> [u32; 2] {
        let (hi, lo) = self.0.fold_in(purpose as u64).as_u32_pair();
        [hi, lo]
    }

    pub fn next_prompt(self) -> Self {
        Self(self.0.fold_in(PROMPT_REBASE_DOMAIN))
    }

    pub fn uniform(self, purpose: Purpose, position: u32, attempt: u32) -> f32 {
        let [hi, lo] = self.words(purpose);
        uniform_f32_from_u32(threefry2x32(position, attempt, Threefry2x32Key::from_words(hi, lo)).0)
    }
}

pub fn reference(counter: [u32; 2], key: [u32; 2]) -> [u32; 4] {
    let (a, b) = threefry2x32(
        counter[0],
        counter[1],
        Threefry2x32Key::from_words(key[0], key[1]),
    );
    [
        a,
        b,
        uniform_f32_from_u32(a).to_bits(),
        uniform_f32_from_u32(b).to_bits(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random123_known_answers() {
        // https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
        assert_eq!(&reference([0, 0], [0, 0])[..2], &[0x6b200159, 0x99ba4efe]);
        assert_eq!(
            &reference([u32::MAX; 2], [u32::MAX; 2])[..2],
            &[0x1cb996fc, 0xbb002be7]
        );
        assert_eq!(
            &reference([0x243f6a88, 0x85a308d3], [0x13198a2e, 0x03707344])[..2],
            &[0xc4923a9c, 0x483df7a0]
        );
    }

    #[test]
    fn streams_replay_without_mutation() {
        let key = SequenceKey::new(42);
        let expected = key.uniform(Purpose::Generation, 17, 0);
        assert_ne!(expected, key.uniform(Purpose::Generation, 17, 1));
        assert_ne!(expected, key.uniform(Purpose::Draft, 17, 0));
        assert_ne!(expected, key.uniform(Purpose::Acceptance, 17, 0));
        assert_ne!(expected, key.uniform(Purpose::Correction, 17, 0));
        assert_ne!(expected, key.uniform(Purpose::Diagnostics, 17, 0));
        assert_eq!(expected, key.uniform(Purpose::Generation, 17, 0));
        assert_ne!(
            key.words(Purpose::Generation),
            key.next_prompt().words(Purpose::Generation)
        );
    }
}
