
/// A small, fast, deterministic PRNG (xorshift32).
/// Not cryptographically secure — intended only for ML weight initialization
/// and similar non-adversarial randomness.
#[derive(Debug, Clone)]
pub struct Prng {
    state: u32,
}
 
impl Prng {
    /// Seed must be non-zero; xorshift32 produces all-zero output forever
    /// if seeded with 0, so we substitute a fallback seed in that case.
    pub fn new(seed: u32) -> Self {
        Prng {
            state: if seed == 0 { 0xDEAD_BEEF } else { seed },
        }
    }
 
    /// Returns the next raw u32 in the sequence. Equivalent to C's
    /// `prng_rand()` (as opposed to `prng_randf()`, which is `randf` below).
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }
 
    /// Returns a float in [0.0, 1.0), analogous to prng_randf().
    pub fn randf(&mut self) -> f32 {
        // Divide by u32::MAX + 1 so the result never reaches 1.0
        (self.next_u32() as f32) / (u32::MAX as f32 + 1.0)
    }
}
 