# `basic_dsp`

[![Build Status](https://travis-ci.org/liebharc/basic_dsp.svg?branch=master)](https://travis-ci.org/liebharc/basic_dsp)
[![Crates.io](https://img.shields.io/crates/v/basic_dsp.svg)](https://crates.io/crates/basic_dsp)

Digital signal processing based on real or complex vectors in time or frequency domain. Vectors come with basic arithmetic, convolution, Fourier transformation and interpolation operations.

[Documentation](https://liebharc.github.io/basic_dsp/basic_dsp_vector/index.html)

[Examples](https://github.com/liebharc/basic_dsp/blob/master/examples/)

[Changelog](https://github.com/liebharc/basic_dsp/blob/master/Changelog.md)

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
basic_dsp = "*"
```

and this to your crate root:

```rust
extern crate basic_dsp;
```

See also [advanced building options](https://github.com/liebharc/basic_dsp/blob/master/building.md).

## Vector flavors
This crate brings vectors in different flavors.

1. Single precision (`f32`) or double precision (`f64`) floating point numbers. This can be used to trade precision against performance. If in doubt then it's likely better to start with double precision floating numbers.
2. Vectors track the meta data about the domain (Time/Frequency) and number space (real/complex) inside the vector in Rust's type system and therefore prevent certain errors. There is also vector type available which tracks meta data at runtime but if it's used then `self.len()` needs to be checked for error handling too and therefore the advice to not use this type unless absolutely required.

## 1.0 goal

1. Wait for SIMD to stabilize in Rust and incorporate this solution
2. Incorporate feedback to the crates API

## Long term goal
The long term vision for this lib is to allow GNU Octave/MATLAB scripts for DSP operations on large arrays/vectors to be relatively easily be rewritten in Rust. The Rust code should then perform much faster than the original code. At the same time a C interface should be available so that other programming languages can make use of this lib too. "Relatively easily be rewritten" implies that the API will not look like GNU Octave/MATLAB and that there is a learning curve involved from changing to this lib, however all necessary vector operations should be provided in some form and definitions should be close to GNU Octave/MATLAB. GNU Octave/MATLAB toolboxes are excluded from this goal since they are rather application specific and therefore should get an own Rust lib. There are already libs available for matrix operations so the aim for this lib will likely be to support matrix operations for matrices of large vectors and to integrated well this those and not to implement its own matrix calculations.

This is a very ambitious goal and it's likely that this lib will not make it there. Contributions are therefore highly appreciated.

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
