#![feature(box_syntax)]
#![feature(cfg_target_feature)]
//! Basic digital signal processing (DSP) operations
//!
//! Digital signal processing based on real or complex vectors in time or frequency domain. Vectors are expected to typically have a size which is at least in the order
//! of magnitude of a couple of thousand elements. This crate tries to balance between a clear API and performance in terms of processing speed.
//! This project started as small pet project to learn more about DSP, CPU architecture and Rust. Since learning
//! involves making mistakes, don't expect things to be flawless or even close to flawless.
//!
//! This library isn't suited - from my point of view - for game programming. If you are looking for vector types to do
//! 2D or 3D graphics calculations then you unfortunately have to continue with your search. However there seem to be 
//! a lot of suitable crates on `crates.io` for you.
//!
//! The vector types don't distinguish between 1xN or Nx1. This is a difference to other conventions such as in MATLAB or GNU Octave.
//! The reason for this decision is that it seems to be more practical to ignore the shape of the vector.
//!
//! Right now the library uses pretty aggressive parallelization. So this means that it will keep all CPU cores busy
//! even if the performance gain is minimal e.g. because the multi core overhead is nearly as large as the performance boost
//! of multiple cores. In future there will be likely an option which tells the library how it should balance betweeen processing time
//! and CPU utilization. The library also avoids to allocate and free memory and it allocates memory for temporary allocation.
//! so the library is likely not suitable for devices which are tight on memory. On normal desktop computers there is usually plenty of
//! memory available so that the optimization focus is on decreasing the processing time for every (common) operation and to spent little time with memory allocations.  
extern crate simd;
extern crate num_cpus;
extern crate crossbeam;
extern crate num;
extern crate rustfft;
mod vector_types;
mod multicore_support;
mod simd_extensions;
mod complex_extensions;
pub mod window_functions;
pub mod conv_types;
pub mod interop_facade;
pub mod combined_ops;
pub use vector_types::
    {
        DataVectorDomain,
        DataVector,
        VecResult,
        VoidResult,
        ErrorReason,
        GenericVectorOperations,
        RealVectorOperations,
        ComplexVectorOperations,
        TimeDomainOperations,
        FrequencyDomainOperations,
        SymmetricFrequencyDomainOperations,
        SymmetricTimeDomainOperations,
        GenericDataVector,
        ComplexFreqVector,
        ComplexTimeVector,
        RealTimeVector,
        RealFreqVector,
        DataVector32, 
        RealTimeVector32,
        ComplexTimeVector32, 
        RealFreqVector32,
        ComplexFreqVector32,
        DataVector64, 
        RealTimeVector64,
        ComplexTimeVector64, 
        RealFreqVector64,
        ComplexFreqVector64,
        Operation,
        Statistics,
        RededicateVector,
        Scale,
        Offset,
        DotProduct,
        StatisticsOperations,
        Convolution,
        VectorConvolution,
        FrequencyMultiplication,
        CrossCorrelation,
        Interpolation,
        RealInterpolation,
        PaddingOption
    };
 pub use multicore_support::MultiCoreSettings;
 use num::traits::Float;   
 
 /// A real floating pointer number intended to abstract over `f32` and `f64`.
 pub trait RealNumber : Float + Copy + Clone + Send + Sync { }
 impl<T> RealNumber for T
  where T: Float + Copy + Clone + Send + Sync {}