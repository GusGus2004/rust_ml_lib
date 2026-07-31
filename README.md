# rust_ml_lib

A NN library and autodiff engine, built from scratch in Rust.

This started as a translation exercise: taking a small machine learning
library originally written in C — matrices, a reverse-mode automatic
differentiation engine, and a training loop — and reimplementing it in
idiomatic Rust. No external ML crates, just the standard library.

Thank you very much to Magicalbat for his inspiration on the original C ML library along with his Youtube too:
https://github.com/Magicalbat/videos/tree/main/machine-learning

The end-to-end project trains a small multilayer perceptron to recognise
handwritten digits from the MNIST dataset, entirely with code in this repo.

## What's implemented

- **A row-major `Matrix` type** with creation, disk loading, elementwise
  ops, matrix multiplication (with all four transpose combinations),
  ReLU, softmax, and cross-entropy — and the gradient
  version of every one of those operations.
- **A reverse-mode automatic differentiation engine** — a computation
  graph (`ModelContext`) where each node knows its operation and its
  inputs, can be compiled into a topologically-sorted order,
  run forward to produce values, and run backward to produce gradients
  automatically via the chain rule. This is the same core idea behind
  PyTorch/TensorFlow's autograd, at a much much smaller scale.
- **A small, seedable PRNG** (xorshift32) for reproducible weight
  initialization and batch shuffling.
- **A training loop** with mini-batch gradient descent - so far plain SGD
- **A working demo**: a 784 → 16 → 16 → 10 MLP with a residual
  connection, trained on MNIST, reaching roughly 94–95% test accuracy
  after 10 epochs.

## Project structure

```
src/
├── matrix.rs   Matrix type: creation, I/O, linear algebra, activations,
│               loss functions, and their gradients
├── rng.rs      Small seeded PRNG (xorshift32) used for weight init
│               and batch shuffling
├── model.rs    The autodiff engine: computation graph, forward/backward
│               passes, graph-building helpers, the training loop
└── main.rs     Builds the MNIST model, loads data, trains it, and
                shows before/after predictions

mnist.py        One-off script: downloads raw MNIST files and converts
                them into the flat f32 binary format the Rust code reads
```

## Getting started

### 1. Get the data

This project doesn't vendor the MNIST dataset — generate it locally:

```bash
pip install numpy
python mnist.py
```

This downloads the raw MNIST files and writes four files into the
project root: `train_images.mat`, `train_labels.mat`, `test_images.mat`,
`test_labels.mat`. These are gitignored — regenerate them any time with
the same command.

### 2. Run it

```bash
cargo run --release
```

`--release` matters a lot here — this does a large number of scalar
floating-point loops, and a debug build can be an order of magnitude
slower.

You should see: an ANSI-colored rendering of a sample test digit, a
pre-training prediction (should look close to uniform/random), live
training progress per epoch, and a post-training prediction that should
strongly favor the correct digit.

## Design notes

A few deliberate departures from a literal, line-for-line C translation,
made in the interest of idiomatic and provably correct Rust:

- **Ownership instead of a manual arena.** The C original threads a
  `mem_arena` through every allocation. Here, `Vec`/`Matrix` just own
  their memory and Rust drops it automatically — no manual
  allocate/release bookkeeping.
- **An index-based graph instead of a graph of pointers.** Nodes in the
  computation graph reference each other by position in a `Vec`, not by
  pointer — the standard Rust pattern for graph-like structures, since
  Rust's ownership rules make a literal web of internal pointers
  impractical.
- **`RefCell` around each node's matrix data.** The backward pass
  legitimately needs to read one node's value while mutating another's
  gradient, where the borrow checker can't statically prove two graph
  indices never alias. `RefCell` moves that specific check to runtime,
  which is what makes the graph-walking code correct without `unsafe`.

## Roadmap — where this is going

Roughly in order of "natural next step" to "bigger undertaking":

- [x] Translate the core matrix library and its gradients
- [x] Translate the autodiff engine and MNIST training loop
- [ ] **Replace plain SGD with the Adam optimizer**
- [ ] **Numerical gradient checking** — a test that compares the
      analytic gradients against a finite-difference approximation, to
      catch subtle backprop bugs directly rather than inferring
      correctness from accuracy numbers alone
- [ ] **Model checkpointing** — save/load trained weights to disk, so a
      trained model doesn't have to be retrained from scratch every run
- [ ] **True batched matrix operations** — right now a "batch" is a Rust
      loop over individual examples; reshaping so a whole batch is one
      larger matrix multiply would be the standard next step for
      throughput
- [ ] **More layer/op types** — convolution, and a higher-level
      "Dense layer" helper that bundles weight + bias + activation into
      one call instead of wiring three graph nodes by hand each time
- [ ] **A general test suite** — unit tests for `Matrix` operations, not
      just an end-to-end MNIST run
