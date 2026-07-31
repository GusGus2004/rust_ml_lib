// Auto-Differentiation 

use std::cell::RefCell;

use crate::matrix::{Matrix, MatrixError};
use crate::rng::Prng;


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelVarFlags(pub u32);

impl ModelVarFlags {
    pub const NONE: ModelVarFlags = ModelVarFlags(0);
    pub const REQUIRES_GRAD: ModelVarFlags = ModelVarFlags(1 << 0);
    pub const PARAMETER: ModelVarFlags = ModelVarFlags(1 << 1);
    pub const INPUT: ModelVarFlags = ModelVarFlags(1 << 2);
    pub const OUTPUT: ModelVarFlags = ModelVarFlags(1 << 3);
    pub const DESIRED_OUTPUT: ModelVarFlags = ModelVarFlags(1 << 4);
    pub const COST: ModelVarFlags = ModelVarFlags(1 << 5);

    pub fn contains(self, other: ModelVarFlags) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for ModelVarFlags {
    type Output = ModelVarFlags;
    fn bitor(self, rhs: ModelVarFlags) -> ModelVarFlags {
        ModelVarFlags(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for ModelVarFlags {
    fn bitor_assign(&mut self, rhs: ModelVarFlags) {
        self.0 |= rhs.0;
    }
}

/// The operation that produced a graph node.
///
/// NOTE: the C original computes arity (0/1/2 inputs) from where an op sits
/// in the enum's declaration order (`_MV_OP_UNARY_START`/`_MV_OP_BINARY_START`
/// sentinels + a comparison macro). That's fragile — reordering the C enum
/// silently breaks it. Here `num_inputs` is an explicit match instead, so
/// reordering variants can't change behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVarOp {
    // NOTE: Null and Sub aren't constructed by anything yet (no model
    // built so far uses subtraction, and Null is just a sentinel default,
    // same as in the C original). Both are dead_code-suppressed rather
    // than removed since they're part of the translated op set and will
    // get used once a model calling mv_sub exists.
    #[allow(dead_code)]
    Null,
    Create,
    Relu,
    Softmax,
    Add,
    #[allow(dead_code)]
    Sub,
    MatMul,
    CrossEntropy,
}

impl ModelVarOp {
    pub fn num_inputs(self) -> usize {
        match self {
            ModelVarOp::Null | ModelVarOp::Create => 0,
            ModelVarOp::Relu | ModelVarOp::Softmax => 1,
            ModelVarOp::Add | ModelVarOp::Sub | ModelVarOp::MatMul | ModelVarOp::CrossEntropy => 2,
        }
    }
}

pub const MODEL_VAR_MAX_INPUTS: usize = 2;

/// A single node in the computation graph.
///
/// `inputs` holds indices into the owning `ModelContext.vars` arena rather
/// than raw pointers (see the design note above `ModelContext`).
///
/// DESIGN NOTE: `val`/`grad` are wrapped in `RefCell` rather than being
/// plain `Option<Matrix>`. The backward pass needs to read one node's
/// matrix while writing another's — e.g. reading B's value while
/// accumulating into A's gradient — where A and B are different elements
/// of the same `Vec<ModelVar>`. Rust's borrow checker can't statically
/// prove two different indices don't alias, so plain indexing won't allow
/// borrowing two elements at once (one mutably). `RefCell` moves that
/// check to runtime, which is what makes this graph-walking code provably
/// correct without `unsafe`.
#[derive(Debug, Clone)]
pub struct ModelVar {
    pub flags: ModelVarFlags,
    pub val: Option<RefCell<Matrix>>,
    pub grad: Option<RefCell<Matrix>>,
    pub op: ModelVarOp,
    pub inputs: [Option<usize>; MODEL_VAR_MAX_INPUTS],
}

/// An ordered execution sequence: indices into `ModelContext.vars`, meant to
/// be run in order (a topological sort of the graph).
#[derive(Debug, Clone, Default)]
pub struct ModelProgram {
    pub vars: Vec<usize>,
}

impl ModelProgram {
    /// Equivalent to `model_prog_compute`. Deviation from the C signature:
    /// needs `model: &ModelContext` as well as `&self`, since `self.vars`
    /// here only holds indices — resolving them to actual ModelVar data
    /// requires the owning context. In C, model_program's pointers already
    /// point directly at the data, so model_prog_compute only needed the
    /// program itself.
    pub fn compute(&self, model: &ModelContext) {
        for &cur_idx in &self.vars {
            let op = model.vars[cur_idx].op;
            let inputs = model.vars[cur_idx].inputs;

            match op {
                ModelVarOp::Null | ModelVarOp::Create => {}

                ModelVarOp::Relu => {
                    let a_idx = inputs[0].unwrap();
                    let a_val = model.vars[a_idx].val.as_ref().unwrap().borrow();
                    let mut cur_val = model.vars[cur_idx].val.as_ref().unwrap().borrow_mut();
                    cur_val.relu(&a_val);
                }
                ModelVarOp::Softmax => {
                    let a_idx = inputs[0].unwrap();
                    let a_val = model.vars[a_idx].val.as_ref().unwrap().borrow();
                    let mut cur_val = model.vars[cur_idx].val.as_ref().unwrap().borrow_mut();
                    cur_val.softmax(&a_val);
                }

                ModelVarOp::Add => {
                    let a_idx = inputs[0].unwrap();
                    let b_idx = inputs[1].unwrap();
                    let a_val = model.vars[a_idx].val.as_ref().unwrap().borrow();
                    let b_val = model.vars[b_idx].val.as_ref().unwrap().borrow();
                    let mut cur_val = model.vars[cur_idx].val.as_ref().unwrap().borrow_mut();
                    cur_val.add(&a_val, &b_val);
                }
                ModelVarOp::Sub => {
                    let a_idx = inputs[0].unwrap();
                    let b_idx = inputs[1].unwrap();
                    let a_val = model.vars[a_idx].val.as_ref().unwrap().borrow();
                    let b_val = model.vars[b_idx].val.as_ref().unwrap().borrow();
                    let mut cur_val = model.vars[cur_idx].val.as_ref().unwrap().borrow_mut();
                    cur_val.sub(&a_val, &b_val);
                }
                ModelVarOp::MatMul => {
                    let a_idx = inputs[0].unwrap();
                    let b_idx = inputs[1].unwrap();
                    let a_val = model.vars[a_idx].val.as_ref().unwrap().borrow();
                    let b_val = model.vars[b_idx].val.as_ref().unwrap().borrow();
                    let mut cur_val = model.vars[cur_idx].val.as_ref().unwrap().borrow_mut();
                    // zero_out = true, matching mat_mul(cur->val, a->val, b->val, 1, 0, 0)
                    cur_val.mul(&a_val, &b_val, true, false, false);
                }
                ModelVarOp::CrossEntropy => {
                    let a_idx = inputs[0].unwrap();
                    let b_idx = inputs[1].unwrap();
                    let a_val = model.vars[a_idx].val.as_ref().unwrap().borrow();
                    let b_val = model.vars[b_idx].val.as_ref().unwrap().borrow();
                    let mut cur_val = model.vars[cur_idx].val.as_ref().unwrap().borrow_mut();
                    cur_val.cross_entropy(&a_val, &b_val);
                }
            }
        }
    }

    /// Equivalent to `model_prog_compute_grads`. Same `&ModelContext`
    /// deviation as `compute`, for the same reason.
    pub fn compute_grads(&self, model: &ModelContext) {
        // Clear grad buffers for every non-parameter var that requires
        // grad. Parameters are deliberately NOT cleared here — their
        // gradients accumulate across a whole batch, reset only once per
        // batch in `ModelContext::train`.
        for &idx in &self.vars {
            let flags = model.vars[idx].flags;
            if !flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                continue;
            }
            if flags.contains(ModelVarFlags::PARAMETER) {
                continue;
            }
            model.vars[idx].grad.as_ref().unwrap().borrow_mut().clear();
        }

        // Seed the final node's gradient to 1.0 (d(loss)/d(loss) = 1).
        let last_idx = *self.vars.last().expect("program should not be empty");
        model.vars[last_idx]
            .grad
            .as_ref()
            .unwrap()
            .borrow_mut()
            .fill(1.0);

        // Walk in reverse, applying the chain rule per node.
        for &cur_idx in self.vars.iter().rev() {
            let flags = model.vars[cur_idx].flags;
            if !flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                continue;
            }

            let op = model.vars[cur_idx].op;
            let inputs = model.vars[cur_idx].inputs;
            let num_inputs = op.num_inputs();

            if num_inputs == 1 {
                let a_idx = inputs[0].unwrap();
                if !model.vars[a_idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                    continue;
                }
            }

            if num_inputs == 2 {
                let a_idx = inputs[0].unwrap();
                let b_idx = inputs[1].unwrap();
                let a_requires = model.vars[a_idx].flags.contains(ModelVarFlags::REQUIRES_GRAD);
                let b_requires = model.vars[b_idx].flags.contains(ModelVarFlags::REQUIRES_GRAD);
                if !a_requires && !b_requires {
                    continue;
                }
            }

            match op {
                ModelVarOp::Null | ModelVarOp::Create => {}

                ModelVarOp::Relu => {
                    let a_idx = inputs[0].unwrap();
                    let a_val = model.vars[a_idx].val.as_ref().unwrap().borrow();
                    let cur_grad = model.vars[cur_idx].grad.as_ref().unwrap().borrow();
                    let mut a_grad = model.vars[a_idx].grad.as_ref().unwrap().borrow_mut();
                    a_grad.relu_add_grad(&a_val, &cur_grad);
                }
                ModelVarOp::Softmax => {
                    let a_idx = inputs[0].unwrap();
                    let cur_val = model.vars[cur_idx].val.as_ref().unwrap().borrow();
                    let cur_grad = model.vars[cur_idx].grad.as_ref().unwrap().borrow();
                    let mut a_grad = model.vars[a_idx].grad.as_ref().unwrap().borrow_mut();
                    a_grad.softmax_add_grad(&cur_val, &cur_grad);
                }

                ModelVarOp::Add => {
                    let a_idx = inputs[0].unwrap();
                    let b_idx = inputs[1].unwrap();
                    let cur_grad = model.vars[cur_idx].grad.as_ref().unwrap().borrow();

                    if model.vars[a_idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        model.vars[a_idx]
                            .grad
                            .as_ref()
                            .unwrap()
                            .borrow_mut()
                            .add_assign(&cur_grad);
                    }
                    if model.vars[b_idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        model.vars[b_idx]
                            .grad
                            .as_ref()
                            .unwrap()
                            .borrow_mut()
                            .add_assign(&cur_grad);
                    }
                }
                ModelVarOp::Sub => {
                    let a_idx = inputs[0].unwrap();
                    let b_idx = inputs[1].unwrap();
                    let cur_grad = model.vars[cur_idx].grad.as_ref().unwrap().borrow();

                    if model.vars[a_idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        model.vars[a_idx]
                            .grad
                            .as_ref()
                            .unwrap()
                            .borrow_mut()
                            .add_assign(&cur_grad);
                    }
                    if model.vars[b_idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        model.vars[b_idx]
                            .grad
                            .as_ref()
                            .unwrap()
                            .borrow_mut()
                            .sub_assign(&cur_grad);
                    }
                }
                ModelVarOp::MatMul => {
                    let a_idx = inputs[0].unwrap();
                    let b_idx = inputs[1].unwrap();
                    let cur_grad = model.vars[cur_idx].grad.as_ref().unwrap().borrow();

                    if model.vars[a_idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        let b_val = model.vars[b_idx].val.as_ref().unwrap().borrow();
                        let mut a_grad = model.vars[a_idx].grad.as_ref().unwrap().borrow_mut();
                        // mat_mul(a->grad, cur->grad, b->val, 0, 0, 1)
                        a_grad.mul(&cur_grad, &b_val, false, false, true);
                    }
                    if model.vars[b_idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        let a_val = model.vars[a_idx].val.as_ref().unwrap().borrow();
                        let mut b_grad = model.vars[b_idx].grad.as_ref().unwrap().borrow_mut();
                        // mat_mul(b->grad, a->val, cur->grad, 0, 1, 0)
                        b_grad.mul(&a_val, &cur_grad, false, true, false);
                    }
                }
                ModelVarOp::CrossEntropy => {
                    let p_idx = inputs[0].unwrap();
                    let q_idx = inputs[1].unwrap();

                    let cur_grad = model.vars[cur_idx].grad.as_ref().unwrap().borrow();
                    let p_val = model.vars[p_idx].val.as_ref().unwrap().borrow();
                    let q_val = model.vars[q_idx].val.as_ref().unwrap().borrow();

                    // p_grad/q_grad may not exist at all (e.g. the desired-
                    // output node never gets a grad buffer, matching the C
                    // original's NULL-safe handling inside
                    // mat_cross_entropy_add_grad).
                    let mut p_grad_ref = model.vars[p_idx].grad.as_ref().map(|c| c.borrow_mut());
                    let mut q_grad_ref = model.vars[q_idx].grad.as_ref().map(|c| c.borrow_mut());

                    Matrix::cross_entropy_add_grad(
                        p_grad_ref.as_deref_mut(),
                        q_grad_ref.as_deref_mut(),
                        &p_val,
                        &q_val,
                        &cur_grad,
                    );
                }
            }
        }
    }
}

/// The full computation graph for one model.
///
/// DESIGN NOTE: `vars` is the arena that actually owns every `ModelVar`.
/// The C struct doesn't have this field explicitly because that storage
/// lives in a `mem_arena` managed elsewhere; in Rust, something has to own
/// the nodes outright so they can be referenced by index without lifetime
/// issues, and `ModelContext` is the natural owner.
#[derive(Debug, Clone)]
pub struct ModelContext {
    pub vars: Vec<ModelVar>,
    pub input: Option<usize>,
    pub output: Option<usize>,
    pub desired_output: Option<usize>,
    pub cost: Option<usize>,
    pub forward_prog: ModelProgram,
    pub cost_prog: ModelProgram,
    /// Not present in the C struct — the C version relies on a hidden
    /// global `prng_randf()`/`prng_rand()`. Since our `Prng` (see rng.rs)
    /// is explicit, non-global state, ModelContext owns one so functions
    /// like mv_create (weight init) and train (batch shuffling) have
    /// somewhere to draw randomness from.
    pub prng: Prng,
}

impl ModelContext {
    /// Equivalent to `model_create`. Takes an explicit seed since there's
    /// no hidden global RNG state to fall back on — the C original doesn't
    /// take a seed here because `prng_randf()`/`prng_rand()` seed
    /// themselves elsewhere, invisibly to this function.
    pub fn new(seed: u32) -> Self {
        ModelContext {
            vars: Vec::new(),
            input: None,
            output: None,
            desired_output: None,
            cost: None,
            forward_prog: ModelProgram::default(),
            cost_prog: ModelProgram::default(),
            prng: Prng::new(seed),
        }
    }

    /// Equivalent to the C struct's `num_vars` field — kept as a method
    /// rather than a stored field so it can never drift out of sync with
    /// the actual contents of `vars`. Not called yet anywhere in this
    /// codebase; kept as public API for whoever builds on top of this.
    #[allow(dead_code)]
    pub fn num_vars(&self) -> usize {
        self.vars.len()
    }

    /// Equivalent to `mv_create`.
    pub fn mv_create(
        &mut self,
        rows: u32,
        cols: u32,
        flags: ModelVarFlags,
    ) -> Result<usize, MatrixError> {
        let index = self.vars.len();

        let val = Some(RefCell::new(Matrix::create(rows, cols)?));
        let grad = if flags.contains(ModelVarFlags::REQUIRES_GRAD) {
            Some(RefCell::new(Matrix::create(rows, cols)?))
        } else {
            None
        };

        self.vars.push(ModelVar {
            flags,
            val,
            grad,
            op: ModelVarOp::Create,
            inputs: [None, None],
        });

        if flags.contains(ModelVarFlags::INPUT) {
            self.input = Some(index);
        }
        if flags.contains(ModelVarFlags::OUTPUT) {
            self.output = Some(index);
        }
        if flags.contains(ModelVarFlags::DESIRED_OUTPUT) {
            self.desired_output = Some(index);
        }
        if flags.contains(ModelVarFlags::COST) {
            self.cost = Some(index);
        }

        Ok(index)
    }

    /// Equivalent to `_mv_unary_impl`. Private (no `pub`) — the leading
    /// underscore in the C name signals "private by convention" there,
    /// but in Rust a leading underscore instead means "I know this is
    /// unused, don't warn me." Real privacy comes from omitting `pub`.
    fn mv_unary_impl(
        &mut self,
        input: usize,
        rows: u32,
        cols: u32,
        mut flags: ModelVarFlags,
        op: ModelVarOp,
    ) -> Result<usize, MatrixError> {
        if self.vars[input].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
            flags |= ModelVarFlags::REQUIRES_GRAD;
        }

        let out = self.mv_create(rows, cols, flags)?;
        self.vars[out].op = op;
        self.vars[out].inputs[0] = Some(input);

        Ok(out)
    }

    /// Equivalent to `_mv_binary_impl`.
    fn mv_binary_impl(
        &mut self,
        a: usize,
        b: usize,
        rows: u32,
        cols: u32,
        mut flags: ModelVarFlags,
        op: ModelVarOp,
    ) -> Result<usize, MatrixError> {
        if self.vars[a].flags.contains(ModelVarFlags::REQUIRES_GRAD)
            || self.vars[b].flags.contains(ModelVarFlags::REQUIRES_GRAD)
        {
            flags |= ModelVarFlags::REQUIRES_GRAD;
        }

        let out = self.mv_create(rows, cols, flags)?;
        self.vars[out].op = op;
        self.vars[out].inputs[0] = Some(a);
        self.vars[out].inputs[1] = Some(b);

        Ok(out)
    }

    /// Equivalent to `mv_relu`.
    pub fn mv_relu(&mut self, input: usize, flags: ModelVarFlags) -> Result<usize, MatrixError> {
        let (rows, cols) = {
            let val = self.vars[input].val.as_ref().unwrap().borrow();
            (val.rows, val.cols)
        };
        self.mv_unary_impl(input, rows, cols, flags, ModelVarOp::Relu)
    }

    /// Equivalent to `mv_softmax`.
    pub fn mv_softmax(&mut self, input: usize, flags: ModelVarFlags) -> Result<usize, MatrixError> {
        let (rows, cols) = {
            let val = self.vars[input].val.as_ref().unwrap().borrow();
            (val.rows, val.cols)
        };
        self.mv_unary_impl(input, rows, cols, flags, ModelVarOp::Softmax)
    }

    /// Equivalent to `mv_add`. DEVIATION: the C version returns `NULL` on a
    /// shape mismatch; here that becomes `Err(MatrixError::ShapeMismatch)`,
    /// reusing the error variant already used elsewhere in this crate
    /// rather than introducing a second way to signal the same failure.
    pub fn mv_add(&mut self, a: usize, b: usize, flags: ModelVarFlags) -> Result<usize, MatrixError> {
        let (a_rows, a_cols, b_rows, b_cols) = {
            let a_val = self.vars[a].val.as_ref().unwrap().borrow();
            let b_val = self.vars[b].val.as_ref().unwrap().borrow();
            (a_val.rows, a_val.cols, b_val.rows, b_val.cols)
        };

        if a_rows != b_rows || a_cols != b_cols {
            return Err(MatrixError::ShapeMismatch);
        }

        self.mv_binary_impl(a, b, a_rows, a_cols, flags, ModelVarOp::Add)
    }

    /// Equivalent to `mv_sub`. Not called by `create_mnist_model` (which
    /// only uses Add/MatMul/Relu/Softmax/CrossEntropy) — kept for when a
    /// model needs subtraction.
    #[allow(dead_code)]
    pub fn mv_sub(&mut self, a: usize, b: usize, flags: ModelVarFlags) -> Result<usize, MatrixError> {
        let (a_rows, a_cols, b_rows, b_cols) = {
            let a_val = self.vars[a].val.as_ref().unwrap().borrow();
            let b_val = self.vars[b].val.as_ref().unwrap().borrow();
            (a_val.rows, a_val.cols, b_val.rows, b_val.cols)
        };

        if a_rows != b_rows || a_cols != b_cols {
            return Err(MatrixError::ShapeMismatch);
        }

        self.mv_binary_impl(a, b, a_rows, a_cols, flags, ModelVarOp::Sub)
    }

    /// Equivalent to `mv_matmul`.
    pub fn mv_matmul(&mut self, a: usize, b: usize, flags: ModelVarFlags) -> Result<usize, MatrixError> {
        let (a_rows, a_cols, b_rows, b_cols) = {
            let a_val = self.vars[a].val.as_ref().unwrap().borrow();
            let b_val = self.vars[b].val.as_ref().unwrap().borrow();
            (a_val.rows, a_val.cols, b_val.rows, b_val.cols)
        };

        if a_cols != b_rows {
            return Err(MatrixError::ShapeMismatch);
        }

        self.mv_binary_impl(a, b, a_rows, b_cols, flags, ModelVarOp::MatMul)
    }

    /// Equivalent to `mv_cross_entropy`.
    pub fn mv_cross_entropy(
        &mut self,
        p: usize,
        q: usize,
        flags: ModelVarFlags,
    ) -> Result<usize, MatrixError> {
        let (p_rows, p_cols, q_rows, q_cols) = {
            let p_val = self.vars[p].val.as_ref().unwrap().borrow();
            let q_val = self.vars[q].val.as_ref().unwrap().borrow();
            (p_val.rows, p_val.cols, q_val.rows, q_val.cols)
        };

        if p_rows != q_rows || p_cols != q_cols {
            return Err(MatrixError::ShapeMismatch);
        }

        self.mv_binary_impl(p, q, p_rows, p_cols, flags, ModelVarOp::CrossEntropy)
    }

    /// Equivalent to `model_prog_create`. DEVIATION: takes `&self` (not
    /// `&mut`) and doesn't need a scratch arena — `Vec` grows itself, so
    /// the C version's defensive `stack_size < model->num_vars` bounds
    /// checks (guarding a fixed-size preallocated buffer) have no
    /// equivalent need here.
    pub fn build_program(&self, out_var: usize) -> ModelProgram {
        let num_vars = self.vars.len();
        let mut visited = vec![false; num_vars];
        let mut stack: Vec<usize> = Vec::with_capacity(num_vars);
        let mut out: Vec<usize> = Vec::with_capacity(num_vars);

        stack.push(out_var);

        while let Some(cur) = stack.pop() {
            if cur >= num_vars {
                continue;
            }

            if visited[cur] {
                if out.len() < num_vars {
                    out.push(cur);
                }
                continue;
            }

            visited[cur] = true;
            stack.push(cur);

            let num_inputs = self.vars[cur].op.num_inputs();
            for i in 0..num_inputs {
                let input = match self.vars[cur].inputs[i] {
                    Some(idx) => idx,
                    None => continue,
                };

                if input >= num_vars || visited[input] {
                    continue;
                }

                if let Some(pos) = stack.iter().position(|&x| x == input) {
                    stack.remove(pos);
                }

                stack.push(input);
            }
        }

        ModelProgram { vars: out }
    }

    /// Equivalent to `model_compile`.
    pub fn compile(&mut self) {
        if let Some(output_idx) = self.output {
            self.forward_prog = self.build_program(output_idx);
        }
        if let Some(cost_idx) = self.cost {
            self.cost_prog = self.build_program(cost_idx);
        }
    }

    /// Equivalent to `model_feedforward`.
    pub fn feedforward(&mut self) {
        self.forward_prog.compute(self);
    }

    /// Equivalent to `model_train`.
    pub fn train(&mut self, desc: &ModelTrainingDesc) {
        let train_images = desc.train_images;
        let train_labels = desc.train_labels;
        let test_images = desc.test_images;
        let test_labels = desc.test_labels;

        let num_examples = train_images.rows as usize;
        let input_size = train_images.cols as usize;
        let output_size = train_labels.cols as usize;
        let num_tests = test_images.rows as usize;

        let num_batches = num_examples / desc.batch_size as usize;

        let mut training_order: Vec<usize> = (0..num_examples).collect();

        let input_idx = self.input.expect("no ModelVar was created with INPUT flag set");
        let desired_output_idx = self
            .desired_output
            .expect("no ModelVar was created with DESIRED_OUTPUT flag set");
        let output_idx = self.output.expect("no ModelVar was created with OUTPUT flag set");
        let cost_idx = self.cost.expect("no ModelVar was created with COST flag set");

        for epoch in 0..desc.epochs {
            for _ in 0..num_examples {
                let a = (self.prng.next_u32() as usize) % num_examples;
                let b = (self.prng.next_u32() as usize) % num_examples;
                training_order.swap(a, b);
            }

            for batch in 0..num_batches {
                for &idx in &self.cost_prog.vars {
                    if self.vars[idx].flags.contains(ModelVarFlags::PARAMETER) {
                        self.vars[idx].grad.as_ref().unwrap().borrow_mut().clear();
                    }
                }

                let mut avg_cost = 0.0f32;
                for i in 0..desc.batch_size as usize {
                    let order_index = batch * desc.batch_size as usize + i;
                    let index = training_order[order_index];

                    self.vars[input_idx]
                        .val
                        .as_ref()
                        .unwrap()
                        .borrow_mut()
                        .data[..input_size]
                        .copy_from_slice(&train_images.data[index * input_size..(index + 1) * input_size]);

                    self.vars[desired_output_idx]
                        .val
                        .as_ref()
                        .unwrap()
                        .borrow_mut()
                        .data[..output_size]
                        .copy_from_slice(&train_labels.data[index * output_size..(index + 1) * output_size]);

                    self.cost_prog.compute(self);
                    self.cost_prog.compute_grads(self);

                    avg_cost += self.vars[cost_idx].val.as_ref().unwrap().borrow().sum();
                }
                avg_cost /= desc.batch_size as f32;

                for &idx in &self.cost_prog.vars {
                    if !self.vars[idx].flags.contains(ModelVarFlags::PARAMETER) {
                        continue;
                    }

                    self.vars[idx]
                        .grad
                        .as_ref()
                        .unwrap()
                        .borrow_mut()
                        .scale(desc.learning_rate / desc.batch_size as f32);

                    let grad_snapshot = self.vars[idx].grad.as_ref().unwrap().borrow();
                    self.vars[idx]
                        .val
                        .as_ref()
                        .unwrap()
                        .borrow_mut()
                        .sub_assign(&grad_snapshot);
                }

                print!(
                    "Epoch {:2} / {:2}, Batch {:4} / {:4}, Average Cost: {:.4}\r",
                    epoch + 1,
                    desc.epochs,
                    batch + 1,
                    num_batches,
                    avg_cost
                );
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            println!();

            let mut num_correct = 0u32;
            let mut avg_cost = 0.0f32;
            for i in 0..num_tests {
                self.vars[input_idx]
                    .val
                    .as_ref()
                    .unwrap()
                    .borrow_mut()
                    .data[..input_size]
                    .copy_from_slice(&test_images.data[i * input_size..(i + 1) * input_size]);

                self.vars[desired_output_idx]
                    .val
                    .as_ref()
                    .unwrap()
                    .borrow_mut()
                    .data[..output_size]
                    .copy_from_slice(&test_labels.data[i * output_size..(i + 1) * output_size]);

                self.cost_prog.compute(self);

                avg_cost += self.vars[cost_idx].val.as_ref().unwrap().borrow().sum();

                let predicted = self.vars[output_idx].val.as_ref().unwrap().borrow().argmax();
                let desired = self.vars[desired_output_idx].val.as_ref().unwrap().borrow().argmax();
                if predicted == desired {
                    num_correct += 1;
                }
            }

            avg_cost /= num_tests as f32;
            println!(
                "Test Completed. Accuracy: {:5} / {:5} ({:.1}%), Average Cost: {:.4}",
                num_correct,
                num_tests,
                num_correct as f32 / num_tests as f32 * 100.0,
                avg_cost
            );
        }
    }
}

/// Plain training configuration — no graph structure, just data and hyperparameters.
///
/// DEVIATION: holds borrowed references, not owned `Matrix` values. `main()`
/// reuses `test_images` for a feedforward call *after* `model_train`
/// returns, so if this struct owned (and thus consumed) it, that data
/// would be moved away and unusable afterward. Borrowing with a lifetime
/// matches the C version's behavior (a pointer, not a copy) without an
/// unnecessary clone of the dataset.
#[derive(Debug, Clone)]
pub struct ModelTrainingDesc<'a> {
    pub train_images: &'a Matrix,
    pub train_labels: &'a Matrix,
    pub test_images: &'a Matrix,
    pub test_labels: &'a Matrix,
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f32,
}