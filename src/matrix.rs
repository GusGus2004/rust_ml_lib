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
    #[allow(dead_code)]
    Io(std::io::Error),
}

impl From<std::io::Error> for MatrixError {
    fn from(e: std::io::Error) -> Self {
        MatrixError::Io(e)
    }
}

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

        let max_bytes = mat.data.len() * std::mem::size_of::<f32>();
        let read_len = file_len.min(max_bytes as u64) as usize;

        let byte_buf = unsafe {
            std::slice::from_raw_parts_mut(mat.data.as_mut_ptr() as *mut u8, max_bytes)
        };

        f.read_exact(&mut byte_buf[..read_len])?;

        Ok(mat)
    }

    pub fn save(&self, filename: &str) -> Result<(), MatrixError> {
        let mut f = File::create(filename)?;

        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<f32>(),
            )
        };

        f.write_all(bytes)?;
        Ok(())
    }

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

    pub fn add_assign(&mut self, other: &Matrix) {
        for i in 0..self.data.len() {
            self.data[i] += other.data[i];
        }
    }

    pub fn sub_assign(&mut self, other: &Matrix) {
        for i in 0..self.data.len() {
            self.data[i] -= other.data[i];
        }
    }

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
            m.data[i] = beta1 * m.data[i] + (1.0 - beta1) * grad.data[i];
            v.data[i] = beta2 * v.data[i] + (1.0 - beta2) * grad.data[i] * grad.data[i];

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

#[cfg(test)]
mod tests {
    use super::*;

    fn m(rows: u32, cols: u32, data: Vec<f32>) -> Matrix {
        Matrix { rows, cols, data }
    }

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    fn assert_close_slice(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                approx_eq(a, e, tol),
                "index {}: expected {}, got {}",
                i,
                e,
                a
            );
        }
    }

    #[test]
    fn create_gives_zeroed_matrix() {
        let mat = Matrix::create(2, 3).unwrap();
        assert_eq!(mat.rows, 2);
        assert_eq!(mat.cols, 3);
        assert_eq!(mat.data, vec![0.0; 6]);
    }

    #[test]
    fn create_rejects_zero_dimensions() {
        assert!(matches!(Matrix::create(0, 3), Err(MatrixError::ZeroDimension)));
        assert!(matches!(Matrix::create(3, 0), Err(MatrixError::ZeroDimension)));
    }

    #[test]
    fn clear_zeroes_everything() {
        let mut mat = m(1, 3, vec![1.0, 2.0, 3.0]);
        mat.clear();
        assert_eq!(mat.data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn fill_sets_every_element() {
        let mut mat = Matrix::create(2, 2).unwrap();
        mat.fill(7.0);
        assert_eq!(mat.data, vec![7.0, 7.0, 7.0, 7.0]);
    }

    #[test]
    fn scale_multiplies_every_element() {
        let mut mat = m(1, 3, vec![1.0, -2.0, 3.0]);
        mat.scale(2.0);
        assert_eq!(mat.data, vec![2.0, -4.0, 6.0]);
    }

    #[test]
    fn sum_adds_every_element() {
        let mat = m(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(mat.sum(), 10.0);
    }

    #[test]
    fn fill_rand_stays_in_bounds_and_is_deterministic() {
        let mut prng_a = Prng::new(42);
        let mut mat_a = Matrix::create(10, 10).unwrap();
        mat_a.fill_rand(&mut prng_a, -1.0, 1.0);

        assert!(mat_a.data.iter().all(|&v| v >= -1.0 && v < 1.0));

        let mut prng_b = Prng::new(42);
        let mut mat_b = Matrix::create(10, 10).unwrap();
        mat_b.fill_rand(&mut prng_b, -1.0, 1.0);

        assert_eq!(mat_a.data, mat_b.data);
    }

    #[test]
    fn argmax_finds_the_maximum() {
        let mat = m(1, 8, vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        assert_eq!(mat.argmax(), 5);
    }

    #[test]
    fn argmax_breaks_ties_toward_first_occurrence() {
        let mat = m(1, 3, vec![5.0, 5.0, 3.0]);
        assert_eq!(mat.argmax(), 0);
    }

    #[test]
    fn add_computes_elementwise_sum() {
        let a = m(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = m(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let mut out = Matrix::create(2, 2).unwrap();
        out.add(&a, &b);
        assert_eq!(out.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn sub_computes_elementwise_difference() {
        let a = m(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = m(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let mut out = Matrix::create(2, 2).unwrap();
        out.sub(&a, &b);
        assert_eq!(out.data, vec![-4.0, -4.0, -4.0, -4.0]);
    }

    #[test]
    fn add_leaves_output_untouched_on_operand_shape_mismatch() {
        let a = m(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = m(1, 2, vec![5.0, 6.0]);
        let mut out = m(2, 2, vec![9.0, 9.0, 9.0, 9.0]);
        out.add(&a, &b);
        assert_eq!(out.data, vec![9.0, 9.0, 9.0, 9.0]);
    }

    #[test]
    fn add_assign_accumulates_in_place() {
        let mut a = m(1, 3, vec![1.0, 1.0, 1.0]);
        let b = m(1, 3, vec![2.0, 3.0, 4.0]);
        a.add_assign(&b);
        assert_eq!(a.data, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn sub_assign_subtracts_in_place() {
        let mut a = m(1, 3, vec![1.0, 1.0, 1.0]);
        let b = m(1, 3, vec![2.0, 3.0, 4.0]);
        a.sub_assign(&b);
        assert_eq!(a.data, vec![-1.0, -2.0, -3.0]);
    }

    #[test]
    fn mul_all_transpose_variants_agree() {
        let expected = vec![58.0, 64.0, 139.0, 154.0];

        let a = m(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = m(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        let mut out_nn = Matrix::create(2, 2).unwrap();
        out_nn.mul(&a, &b, true, false, false);
        assert_eq!(out_nn.data, expected);

        let b_t = m(2, 3, vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0]);
        let mut out_nt = Matrix::create(2, 2).unwrap();
        out_nt.mul(&a, &b_t, true, false, true);
        assert_eq!(out_nt.data, expected);

        let a_t = m(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let mut out_tn = Matrix::create(2, 2).unwrap();
        out_tn.mul(&a_t, &b, true, true, false);
        assert_eq!(out_tn.data, expected);

        let mut out_tt = Matrix::create(2, 2).unwrap();
        out_tt.mul(&a_t, &b_t, true, true, true);
        assert_eq!(out_tt.data, expected);
    }

    #[test]
    fn mul_with_zero_out_false_accumulates() {
        let a = m(1, 2, vec![1.0, 2.0]);
        let b = m(2, 1, vec![3.0, 4.0]);
        let mut out = m(1, 1, vec![100.0]);
        out.mul(&a, &b, false, false, false);
        assert_eq!(out.data, vec![111.0]);
    }

    #[test]
    fn mul_rejects_incompatible_shapes() {
        let a = m(2, 3, vec![0.0; 6]);
        let b = m(2, 2, vec![0.0; 4]);
        let mut out = Matrix::create(2, 2).unwrap();
        assert!(!out.mul(&a, &b, true, false, false));
    }

    #[test]
    fn relu_zeroes_negatives_and_keeps_positives() {
        let input = m(1, 5, vec![-3.0, -1.0, 0.0, 2.0, 5.0]);
        let mut out = Matrix::create(1, 5).unwrap();
        out.relu(&input);
        assert_eq!(out.data, vec![0.0, 0.0, 0.0, 2.0, 5.0]);
    }

    #[test]
    fn softmax_of_equal_inputs_is_uniform_and_sums_to_one() {
        let input = m(1, 2, vec![0.0, 0.0]);
        let mut out = Matrix::create(1, 2).unwrap();
        out.softmax(&input);
        assert_close_slice(&out.data, &[0.5, 0.5], 1e-5);
        assert!(approx_eq(out.sum(), 1.0, 1e-5));
    }

    #[test]
    fn softmax_does_not_overflow_on_large_logits() {
        let input = m(1, 2, vec![1000.0, 1000.0]);
        let mut out = Matrix::create(1, 2).unwrap();
        out.softmax(&input);
        assert!(out.data.iter().all(|v| v.is_finite()));
        assert_close_slice(&out.data, &[0.5, 0.5], 1e-5);
    }

    #[test]
    fn cross_entropy_matches_hand_computed_value() {
        let p = m(1, 2, vec![1.0, 0.0]);
        let q = m(1, 2, vec![0.6, 0.4]);
        let mut out = Matrix::create(1, 2).unwrap();
        out.cross_entropy(&p, &q);
        assert!(approx_eq(out.data[0], 0.5108256, 1e-4));
        assert_eq!(out.data[1], 0.0);
    }

    #[test]
    fn relu_add_grad_blocks_gradient_for_negative_inputs() {
        let input = m(1, 3, vec![-1.0, 2.0, -3.0]);
        let grad = m(1, 3, vec![5.0, 6.0, 7.0]);
        let mut out = m(1, 3, vec![1.0, 1.0, 1.0]);
        out.relu_add_grad(&input, &grad);
        assert_eq!(out.data, vec![1.0, 7.0, 1.0]);
    }

    #[test]
    fn softmax_add_grad_matches_hand_computed_jacobian() {
        let softmax_out = m(2, 1, vec![0.5, 0.5]);
        let grad = m(2, 1, vec![1.0, 0.0]);
        let mut out = Matrix::create(2, 1).unwrap();
        out.softmax_add_grad(&softmax_out, &grad);
        assert_close_slice(&out.data, &[0.25, -0.25], 1e-5);
    }

    #[test]
    fn cross_entropy_add_grad_matches_hand_computed_values() {
        let p = m(1, 2, vec![1.0, 0.0]);
        let q = m(1, 2, vec![0.6, 0.4]);
        let grad = m(1, 2, vec![2.0, 3.0]);

        let mut p_grad = Matrix::create(1, 2).unwrap();
        let mut q_grad = Matrix::create(1, 2).unwrap();

        Matrix::cross_entropy_add_grad(Some(&mut p_grad), Some(&mut q_grad), &p, &q, &grad);

        assert!(approx_eq(p_grad.data[0], -0.6f32.ln() * 2.0, 1e-4));
        assert!(approx_eq(p_grad.data[1], -0.4f32.ln() * 3.0, 1e-4));

        assert!(approx_eq(q_grad.data[0], -1.0 / 0.6 * 2.0, 1e-4));
        assert_eq!(q_grad.data[1], 0.0);
    }

    #[test]
    fn cross_entropy_add_grad_skips_none_gracefully() {
        let p = m(1, 2, vec![1.0, 0.0]);
        let q = m(1, 2, vec![0.6, 0.4]);
        let grad = m(1, 2, vec![2.0, 3.0]);
        let mut q_grad = Matrix::create(1, 2).unwrap();

        let ok = Matrix::cross_entropy_add_grad(None, Some(&mut q_grad), &p, &q, &grad);
        assert!(ok);
        assert!(approx_eq(q_grad.data[0], -1.0 / 0.6 * 2.0, 1e-4));
    }

    #[test]
    fn adam_step_matches_hand_computed_first_step() {
        let mut val = m(1, 1, vec![10.0]);
        let mut mm = m(1, 1, vec![0.0]);
        let mut v = m(1, 1, vec![0.0]);
        let grad = m(1, 1, vec![4.0]);

        val.adam_step(&mut mm, &mut v, &grad, 0.1, 0.9, 0.999, 1e-8, 1);

        assert!(approx_eq(mm.data[0], 0.4, 1e-6));
        assert!(approx_eq(v.data[0], 0.016, 1e-6));
        assert!(approx_eq(val.data[0], 9.9, 1e-5));
    }

    #[test]
    fn copy_from_copies_matching_shapes() {
        let src = m(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let mut dst = Matrix::create(2, 2).unwrap();
        dst.copy_from(&src).unwrap();
        assert_eq!(dst.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn copy_from_rejects_shape_mismatch() {
        let src = Matrix::create(3, 3).unwrap();
        let mut dst = Matrix::create(2, 2).unwrap();
        assert!(matches!(dst.copy_from(&src), Err(MatrixError::ShapeMismatch)));
    }

    #[test]
    fn save_then_load_round_trips_exactly() {
        let path = std::env::temp_dir().join("rust_mll_test_save_load.mat");
        let path_str = path.to_str().unwrap();

        let original = m(2, 3, vec![1.5, -2.25, 3.0, 0.0, -4.75, 100.0]);
        original.save(path_str).unwrap();

        let loaded = Matrix::load(2, 3, path_str).unwrap();
        assert_eq!(loaded.data, original.data);

        let _ = std::fs::remove_file(path);
    }
}