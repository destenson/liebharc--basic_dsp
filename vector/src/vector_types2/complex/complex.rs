use RealNumber;
use multicore_support::*;
use simd_extensions::Simd;
use num::Complex;
use super::super::{
    array_to_complex_mut,
    Vector, DspVec, ToSliceMut,
    Domain, ComplexNumberSpace,
};

/// Operations on complex types.
///
/// # Failures
///
/// If one of the methods is called on real data then `self.len()` will be set to `0`.
/// To avoid this it's recommended to use the `to_real_time_vec`, `to_real_freq_vec`
/// `to_complex_time_vec` and `to_complex_freq_vec` constructor methods since
/// the resulting types will already check at compile time (using the type system) that the data is complex.
pub trait ComplexOps<T>
    where T: RealNumber {
	/// Multiplies each vector element with `exp(j*(a*idx*self.delta() + b))`
    /// where `a` and `b` are arguments and `idx` is the index of the data points
    /// in the vector ranging from `0 to self.points() - 1`. `j` is the imaginary number and
    /// `exp` the exponential function.
    ///
    /// This method can be used to perform a frequency shift in time domain.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::f64;
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::vector_types2::*;
    /// # fn main() {
    /// let mut vector = vec!(1.0, 2.0, 3.0, 4.0).to_complex_time_vec();
    /// vector.multiply_complex_exponential(2.0, 3.0);
    /// let actual = &vector[..];
    /// let expected = &[-1.2722325, -1.838865, 4.6866837, -1.7421241];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!(f64::abs(actual[i] - expected[i]) < 1e-4);
    /// }
    /// # }
    /// ```
    /// ```
    /// use basic_dsp_vector::{ComplexTimeVector32, ComplexVectorOps, DataVec, InterleavedIndex};
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = vector.multiply_complex_exponential(2.0, 3.0).expect("Ignoring error handling in examples");
    /// let expected = [-1.2722325, -1.838865, 4.6866837, -1.7421241];
    /// let result = result.interleaved(0..);
    /// for i in 0..expected.len() {
    ///     assert!((result[i] - expected[i]).abs() < 1e-4);
    /// }
    /// ```
    fn multiply_complex_exponential(&mut self, a: T, b: T);

	/// Calculates the complex conjugate of the vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::vector_types2::*;
    /// # fn main() {
    /// let mut vector = vec!(1.0, 2.0, 3.0, 4.0).to_complex_time_vec();
    /// vector.conj();
    /// assert_eq!([1.0, -2.0, 3.0, -4.0], vector[..]);
    /// # }
    /// ```
    fn conj(&mut self);
}

macro_rules! assert_complex {
    ($self_: ident) => {
        if !$self_.is_complex() {
            $self_.number_space.to_real();
            $self_.valid_len = 0;
        }
    }
}

impl<S, T, N, D> ComplexOps<T> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: Domain {
    fn multiply_complex_exponential(&mut self, a: T, b: T) {
        assert_complex!(self);
        let a = a * self.delta();
        let data_length = self.len();
        let scalar_length = data_length % T::Reg::len();
        let vectorization_length = data_length - scalar_length;
        let mut array = self.data.to_slice_mut();
        Chunk::execute_with_range(
            Complexity::Small, &self.multicore_settings,
            &mut array[0..vectorization_length], T::Reg::len(),
            (a, b),
            move |array, range, args| {
            let two = T::one() + T::one();
            let (a, b) = args;
            let mut exponential =
                Complex::<T>::from_polar(&T::one(), &b)
                * Complex::<T>::from_polar(&T::one(), &(a * T::from(range.start).unwrap() as T / two));
            let increment = Complex::<T>::from_polar(&T::one(), &a);
            let array = array_to_complex_mut(array);
            for complex in array {
                *complex = (*complex) * exponential;
                exponential = exponential * increment;
            }
        });
        let two = T::one() + T::one();
        let mut exponential =
            Complex::<T>::from_polar(&T::one(), &b)
            * Complex::<T>::from_polar(&T::one(), &(a * T::from(vectorization_length).unwrap() as T / two));
        let increment = Complex::<T>::from_polar(&T::one(), &a);
        let array = array_to_complex_mut(&mut array[vectorization_length..data_length]);
        for complex in array {
            *complex = (*complex) * exponential;
            exponential = exponential * increment;
        }
    }

    fn conj(&mut self) {
        assert_complex!(self);
        let factor = T::Reg::from_complex(Complex::<T>::new(T::one(), -T::one()));
        self.simd_complex_operation(|x,y| x * y, |x,_| x.conj(), factor, Complexity::Small)
    }
}
