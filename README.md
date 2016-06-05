# `basic_dsp`

[![Build Status](https://api.travis-ci.org/liebharc/basic_dsp.png)](https://travis-ci.org/liebharc/basic_dsp)

Digital signal processing based on 1xN (one times N) or Nx1 vectors in real or complex number space.
Vectors come with basic arithmetic, convolution, Fourier transformation and interpolation operations.

[Documentation](https://liebharc.github.io/basic_dsp/basic_dsp/)

[Example](https://github.com/liebharc/basic_dsp/blob/master/examples/modulation.rs)

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
basic_dsp = "^0.2.1"
```

and this to your crate root:

```rust
extern crate basic_dsp;
```

On CPUs with SSE instructions (should be all somewhat recent Intel or AMD desktop CPUs) performance can be improved by compiling the crate with the respective `rustc` options. See `rustc.sh` or `rustc.bat` for an example.

## Vector flavors
This crate brings vectors in different flavors.

1. Single precision (`f32`) or double precision (`f64`) floating point numbers. This can be used to trade precision against performance. If in doubt then it's likely better to start with double precision floating numbers.
2. Specialized or generic vectors. The specialized vectors track the data types inside the vector in Rusts type system and therefore prevent certain errors. The generic vectors will instead throw exceptions at runtime if an operation is undefined. If in doubt then it's recommended to start with the specialized vector types.

## Design principals
The main design goals are:

1. Prevent some typical errors by making use of the Rust type system, e.g. by preventing that an operation is called which isn't defined for a vector in real number space.
2. Provide an interface to other programming languages which might be not as performant as Rust.
3. Optimize for performance in terms of execution speed.
4. Avoid memory allocations. This means that memory used in order to optimize the performance of certain operations will not be freed until the vector will be dropped/destroyed. It's therefore recommended to reuse vectors once created, but to drop/destroy them as soon as the input data into the DSP processing chain changes in size.

## Contributions
Welcome!

## Stability
This project started as small pet project to learn more about DSP, CPU architecture and Rust. Since learning involves making mistakes, don't expect things to be flawless or even close to flawless. In fact issues are expected in all areas (including correctness, stability, performance) and while the crate should be useful already all results should be treated with caution at the same time.