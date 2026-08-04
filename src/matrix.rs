use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::rng::Prng;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: u32,
    pub cols: u32,
    pub data: Vec<f32>,
}

#[derive(Debug)]
pub enum MatrixError {
    ZeroDimension,
    SizeOverflow,
    ShapeMismatch,
    // Inner io::Error is never read back out (only ever printed via
    // Debug), which is fine — it's there for diagnostic context, not to
    // be matched on.
    #[allow(dead_code)]
    Io(std::io::Error),
}

impl From<std::io::Error> for MatrixError {
    fn from(e: std::io::Error) -> Self {
        MatrixError::Io(e)
    }
}

/// Internal helper: out += a * b, assuming out, a, b already have compatible
/// shapes and out is pre-zeroed by the caller. Not part of the public API.
fn mat_mul_nn(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for i in 0..out.rows as usize {
        for k in 0..a.cols as usize {
            for j in 0..out.cols as usize {
                out.data[j + i * out.cols as usize] +=
                    a.data[k + i * a.cols as usize] * b.data[j + k * b.cols as usize];
            }
        }
    }
}

/// Internal helper: out += a * b^T, assuming out, a, b already have compatible
/// shapes and out is pre-zeroed by the caller. Not part of the public API.
fn mat_mul_nt(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for i in 0..out.rows as usize {
        for j in 0..out.cols as usize {
            for k in 0..a.cols as usize {
                out.data[j + i * out.cols as usize] +=
                    a.data[k + i * a.cols as usize] * b.data[k + j * b.cols as usize];
            }
        }
    }
}

/// Internal helper: out += a^T * b, assuming out, a, b already have compatible
/// shapes and out is pre-zeroed by the caller. Not part of the public API.
fn mat_mul_tn(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for k in 0..a.rows as usize {
        for i in 0..out.rows as usize {
            for j in 0..out.cols as usize {
                out.data[j + i * out.cols as usize] +=
                    a.data[i + k * a.cols as usize] * b.data[j + k * b.cols as usize];
            }
        }
    }
}

/// Internal helper: out += a^T * b^T, assuming out, a, b already have compatible
/// shapes and out is pre-zeroed by the caller. Not part of the public API.
fn mat_mul_tt(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for i in 0..out.rows as usize {
        for j in 0..out.cols as usize {
            for k in 0..a.rows as usize {
                out.data[j + i * out.cols as usize] +=
                    a.data[i + k * a.cols as usize] * b.data[k + j * b.cols as usize];
            }
        }
    }
}

impl Matrix {
    pub fn create(rows: u32, cols: u32) -> Result<Self, MatrixError> {
        if rows == 0 || cols == 0 {
            return Err(MatrixError::ZeroDimension);
        }

        let len = (rows as usize)
            .checked_mul(cols as usize)
            .ok_or(MatrixError::SizeOverflow)?;

        Ok(Matrix {
            rows,
            cols,
            data: vec![0.0; len],
        })
    }

    pub fn load(rows: u32, cols: u32, filename: &str) -> Result<Self, MatrixError> {
        let mut mat = Matrix::create(rows, cols)?;

        let mut f = File::open(filename)?;
        let file_len = f.seek(SeekFrom::End(0))?;
        f.seek(SeekFrom::Start(0))?;

        // Bytes available in mat.data, as in the C version (sizeof(f32) * rows * cols)
        let max_bytes = mat.data.len() * std::mem::size_of::<f32>();
        let read_len = file_len.min(max_bytes as u64) as usize;

        // Reinterpret mat.data (Vec<f32>) as a byte buffer for reading raw bytes,
        // mirroring fread(mat->data, 1, size, f)
        let byte_buf = unsafe {
            std::slice::from_raw_parts_mut(mat.data.as_mut_ptr() as *mut u8, max_bytes)
        };

        f.read_exact(&mut byte_buf[..read_len])?;

        Ok(mat)
    }

    /// Not a translation of any C function — the write counterpart to
    /// `load`, added for model checkpointing. Writes the same flat,
    /// headerless raw-f32 format `load` reads, so a saved parameter file
    /// is byte-compatible with the same tooling used for the MNIST data.
    pub fn save(&self, filename: &str) -> Result<(), MatrixError> {
        let mut f = File::create(filename)?;

        // Reinterpret the f32 buffer as raw bytes for writing — the
        // write-side mirror of the same reinterpretation `load` does when
        // reading bytes in.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<f32>(),
            )
        };

        f.write_all(bytes)?;
        Ok(())
    }

    /// Equivalent to `mat_copy`. Now has a real caller:
    /// `ModelContext::load_parameters` uses this to copy freshly-loaded
    /// weight data into an already-shaped parameter matrix.
    pub fn copy_from(&mut self, src: &Matrix) -> Result<(), MatrixError> {
        if self.rows != src.rows || self.cols != src.cols {
            return Err(MatrixError::ShapeMismatch);
        }

        self.data.copy_from_slice(&src.data);
        Ok(())
    }

    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    pub fn fill(&mut self, x: f32) {
        self.data.fill(x);
    }

    pub fn fill_rand(&mut self, prng: &mut Prng, lower: f32, upper: f32) {
        for v in self.data.iter_mut() {
            *v = prng.randf() * (upper - lower) + lower;
        }
    }

    pub fn scale(&mut self, scale: f32) {
        for v in self.data.iter_mut() {
            *v *= scale;
        }
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Elementwise self += other. Not a direct translation of any single C
    /// function — added because Rust's borrow checker won't allow calling
    /// `self.add(self, other)` the way C's `mat_add(a->grad, a->grad,
    /// cur->grad)` safely could (pointer self-aliasing is fine there since
    /// it's a simple per-index read-then-write). Assumes shapes already
    /// match (guaranteed by how this is used elsewhere in this codebase).
    pub fn add_assign(&mut self, other: &Matrix) {
        for i in 0..self.data.len() {
            self.data[i] += other.data[i];
        }
    }

    /// Elementwise self -= other. Same rationale as `add_assign` above,
    /// covering C's `mat_sub(cur->val, cur->val, cur->grad)` and
    /// `mat_sub(b->grad, b->grad, cur->grad)` self-aliasing patterns.
    pub fn sub_assign(&mut self, other: &Matrix) {
        for i in 0..self.data.len() {
            self.data[i] -= other.data[i];
        }
    }

    /// Applies one Adam optimizer step to `self` (a parameter's value),
    /// given its running first-moment estimate `m`, second-moment
    /// estimate `v`, the current gradient, and the step number `t`
    /// (starting at 1, incrementing once per optimizer step across the
    /// whole training run — not per epoch). Not a translation of any C
    /// function; this is the new optimizer replacing plain SGD.
    #[allow(clippy::too_many_arguments)]
    pub fn adam_step(
        &mut self,
        m: &mut Matrix,
        v: &mut Matrix,
        grad: &Matrix,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        t: i32,
    ) {
        let bias_correction1 = 1.0 - beta1.powi(t);
        let bias_correction2 = 1.0 - beta2.powi(t);

        for i in 0..self.data.len() {
            // Exponential moving averages of the gradient and its square.
            m.data[i] = beta1 * m.data[i] + (1.0 - beta1) * grad.data[i];
            v.data[i] = beta2 * v.data[i] + (1.0 - beta2) * grad.data[i] * grad.data[i];

            // Bias correction: without this, m and v start too close to
            // zero (since they're initialized at zero), which would make
            // the very first few steps too small.
            let m_hat = m.data[i] / bias_correction1;
            let v_hat = v.data[i] / bias_correction2;

            self.data[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
        }
    }

    pub fn argmax(&self) -> u64 {
        let mut max_i: usize = 0;
        for (i, &v) in self.data.iter().enumerate() {
            if v > self.data[max_i] {
                max_i = i;
            }
        }
        max_i as u64
    }

    pub fn add(&mut self, a: &Matrix, b: &Matrix) -> bool {
        if a.rows != b.rows || a.cols != b.cols {
            return false;
        }
        if self.rows != a.rows || self.cols != a.cols {
            return false;
        }

        for i in 0..self.data.len() {
            self.data[i] = a.data[i] + b.data[i];
        }

        false
    }

    pub fn sub(&mut self, a: &Matrix, b: &Matrix) -> bool {
        if a.rows != b.rows || a.cols != b.cols {
            return false;
        }
        if self.rows != a.rows || self.cols != a.cols {
            return false;
        }

        for i in 0..self.data.len() {
            self.data[i] = a.data[i] - b.data[i];
        }

        false
    }

    pub fn mul(
        &mut self,
        a: &Matrix,
        b: &Matrix,
        zero_out: bool,
        transpose_a: bool,
        transpose_b: bool,
    ) -> bool {
        let a_rows = if transpose_a { a.cols } else { a.rows };
        let a_cols = if transpose_a { a.rows } else { a.cols };
        let b_rows = if transpose_b { b.cols } else { b.rows };
        let b_cols = if transpose_b { b.rows } else { b.cols };

        if a_cols != b_rows {
            return false;
        }
        if self.rows != a_rows || self.cols != b_cols {
            return false;
        }

        if zero_out {
            self.clear();
        }

        match (transpose_a, transpose_b) {
            (false, false) => mat_mul_nn(self, a, b),
            (false, true) => mat_mul_nt(self, a, b),
            (true, false) => mat_mul_tn(self, a, b),
            (true, true) => mat_mul_tt(self, a, b),
        }

        true
    }

    pub fn relu(&mut self, input: &Matrix) -> bool {
        if self.rows != input.rows || self.cols != input.cols {
            return false;
        }

        for i in 0..self.data.len() {
            self.data[i] = input.data[i].max(0.0);
        }

        true
    }

    pub fn softmax(&mut self, input: &Matrix) -> bool {
        if self.rows != input.rows || self.cols != input.cols {
            return false;
        }

        // Subtract the max value before exponentiating. Mathematically
        // this changes nothing (softmax(x) == softmax(x - c) for any
        // constant c), but it keeps every exponent <= 0, which prevents
        // exp() from overflowing to infinity on large logits — the
        // overflow that was causing NaN to appear and then propagate
        // through the whole network permanently.
        let max = input.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0f32;
        for i in 0..self.data.len() {
            self.data[i] = (input.data[i] - max).exp();
            sum += self.data[i];
        }
        self.scale(1.0 / sum);

        true
    }

    pub fn cross_entropy(&mut self, p: &Matrix, q: &Matrix) -> bool {
        if p.rows != q.rows || p.cols != q.cols {
            return false;
        }
        if self.rows != p.rows || self.cols != p.cols {
            return false;
        }

        for i in 0..self.data.len() {
            self.data[i] = if p.data[i] == 0.0 {
                0.0
            } else {
                p.data[i] * -q.data[i].ln()
            };
        }

        true
    }

    pub fn relu_add_grad(&mut self, input: &Matrix, grad: &Matrix) -> bool {
        if self.rows != input.rows || self.cols != input.cols {
            return false;
        }
        if self.rows != grad.rows || self.cols != grad.cols {
            return false;
        }

        for i in 0..self.data.len() {
            self.data[i] += if input.data[i] > 0.0 { grad.data[i] } else { 0.0 };
        }

        true
    }

    pub fn softmax_add_grad(&mut self, softmax_out: &Matrix, grad: &Matrix) -> bool {
        if softmax_out.rows != 1 && softmax_out.cols != 1 {
            return false;
        }

        let size = softmax_out.rows.max(softmax_out.cols);
        // size comes from an already-valid Matrix, so it's guaranteed
        // non-zero and can't overflow — this can't actually fail.
        let mut jacobian = Matrix::create(size, size)
            .expect("jacobian size derived from an existing valid matrix");

        for i in 0..size as usize {
            for j in 0..size as usize {
                let kronecker_delta = if i == j { 1.0 } else { 0.0 };
                jacobian.data[j + i * size as usize] =
                    softmax_out.data[i] * (kronecker_delta - softmax_out.data[j]);
            }
        }

        self.mul(&jacobian, grad, false, false, false);
        // jacobian is dropped automatically here — no manual scratch-arena release needed.

        true
    }

    pub fn cross_entropy_add_grad(
        p_grad: Option<&mut Matrix>,
        q_grad: Option<&mut Matrix>,
        p: &Matrix,
        q: &Matrix,
        grad: &Matrix,
    ) -> bool {
        if p.rows != q.rows || p.cols != q.cols {
            return false;
        }
        let size = p.data.len();

        if let Some(p_grad) = p_grad {
            if p_grad.rows != p.rows || p_grad.cols != p.cols {
                return false;
            }
            for i in 0..size {
                p_grad.data[i] += -q.data[i].ln() * grad.data[i];
            }
        }

        if let Some(q_grad) = q_grad {
            if q_grad.rows != q.rows || q_grad.cols != q.cols {
                return false;
            }
            for i in 0..size {
                q_grad.data[i] += -p.data[i] / q.data[i] * grad.data[i];
            }
        }

        true
    }
}