
#[derive(Debug, Clone)]
pub struct Prng {
    state: u32,
}
 
impl Prng {
    pub fn new(seed: u32) -> Self {
        Prng {
            state: if seed == 0 { 0xDEAD_BEEF } else { seed },
        }
    }
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }
 
    pub fn randf(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32 + 1.0)
    }
}
 
