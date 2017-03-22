use RealNumber;
use super::super::{DspVec, Buffer, ComplexOps, ScaleOps, FrequencyDomainOperations,
                   TimeToFrequencyDomainOperations, RededicateForceOps, ToSliceMut, Owner,
                   PaddingOption, VoidResult, Vector, FromVector, MetaData, ComplexNumberSpace,
                   TimeDomain, ElementaryOps, ToFreqResult, InsertZerosOpsBuffered, DataDomain,
                   ErrorReason, ReorganizeDataOps, ToComplexVector, FrequencyToTimeDomainOperations,
                   ToDspVector, ToTimeResult, RededicateOps};

/// Cross-correlation of data vectors. See also https://en.wikipedia.org/wiki/Cross-correlation
///
/// The correlation is calculated in two steps. This is done to give you more control
/// over two things:
///
/// 1. Should the correlation use zero padding or not? This is done by calling either
///    `prepare_argument`
///    or `prepare_argument_padded`.
/// 2. The lifetime of the argument. The argument needs to be transformed for the correlation and
///    depending on the application that might be just fine, or a clone needs to be created or
///    it's okay to use one argument for multiple correlations.
///
/// To get the same behavior like GNU Octave or MATLAB `prepare_argument_padded` needs to be
/// called before doing the correlation. See also the example section for how to do this.
/// # Example
///
/// ```no_run
/// use std::f32;
/// use basic_dsp_vector::*;
/// let mut vector = vec!(1.0, 1.0, 2.0, 2.0, 3.0, 3.0).to_complex_time_vec();
/// let argument = vec!(3.0, 3.0, 2.0, 2.0, 1.0, 1.0).to_complex_time_vec();
/// let mut buffer = SingleBuffer::new();
/// let argument = argument.prepare_argument_padded(&mut buffer);
/// vector.correlate(&mut buffer, &argument).expect("Ignoring error handling in examples");
/// let expected = &[2.0, 0.0, 8.0, 0.0, 20.0, 0.0, 24.0, 0.0, 18.0, 0.0];
/// for i in 0..vector.len() {
///     assert!(f32::abs(vector[i] - expected[i]) < 1e-4);
/// }
/// ```
/// # Unstable
/// This functionality has been recently added in order to find out if the definitions are
/// consistent. However the actual implementation is lacking tests.
/// # Failures
/// `TransRes` may report the following `ErrorReason` members:
///
/// 1. `VectorMustBeComplex`: if `self` is in real number space.
/// 3. `VectorMetaDataMustAgree`: in case `self` and `function` are not
///    in the same number space and same domain.
pub trait CrossCorrelationArgumentOps<S, T>: ToFreqResult
    where S: ToSliceMut<T>,
          T: RealNumber
{
    /// Prepares an argument to be used for convolution. Preparing an argument includes two steps:
    ///
    /// 1. Calculate the plain FFT
    /// 2. Calculate the complex conjugate
    fn prepare_argument<B>(self, buffer: &mut B) -> Self::FreqResult where B: Buffer<S, T>;

    /// Prepares an argument to be used for convolution. The argument is zero padded to
    /// length of `2 * self.points() - 1`
    /// and then the same operations are performed as described for `prepare_argument`.
    fn prepare_argument_padded<B>(self, buffer: &mut B) -> Self::FreqResult where B: Buffer<S, T>;
}

/// A trait to calculate the cross correlation.
pub trait CrossCorrelationOps<S, T, A>
    where S: ToSliceMut<T>,
          T: RealNumber
{
    /// Calculates the correlation between `self` and `other`. `other`
    /// needs to be a time vector which
    /// went through one of the prepare functions `prepare_argument` or `prepare_argument_padded`.
    /// See also the trait description for more details.
    fn correlate<B>(&mut self, buffer: &mut B, other: &A) -> VoidResult where B: Buffer<S, T>;
}

impl<S, T, N, D> CrossCorrelationArgumentOps<S, T> for DspVec<S, T, N, D>
	where DspVec<S, T, N, D>: ToFreqResult
        + TimeToFrequencyDomainOperations<S, T>
        + ScaleOps<T>
		+ ReorganizeDataOps<T> + Clone,
	  <DspVec<S, T, N, D> as ToFreqResult>::FreqResult: RededicateForceOps<DspVec<S, T, N, D>>
        + FrequencyDomainOperations<S, T> + ComplexOps<T> + Vector<T>
        + ElementaryOps<<DspVec<S, T, N, D> as ToFreqResult>::FreqResult>
        + FromVector<T, Output=S>,
	  S: ToSliceMut<T> + Owner,
	  T: RealNumber,
	  N: ComplexNumberSpace,
	  D: TimeDomain {

	fn prepare_argument<B>(self, buffer: &mut B) -> Self::FreqResult
	 	where B: Buffer<S, T> {
		let mut result = self.plain_fft(buffer);
		result.conj();
		result
	}

	fn prepare_argument_padded<B>(mut self, buffer: &mut B) -> Self::FreqResult
		where B: Buffer<S, T> {
		let points = self.points();
		self.zero_pad_b(buffer, 2 * points - 1, PaddingOption::Surround);
		let mut result = self.plain_fft(buffer);
		result.conj();
		result
	}
}

impl<S, T, N, D> CrossCorrelationOps<S, T, <DspVec<S, T, N, D> as ToFreqResult>::FreqResult>
    for DspVec<S, T, N, D>
    where DspVec<S, T, N, D>: ToFreqResult
        + TimeToFrequencyDomainOperations<S, T>
        + RededicateOps<<<DspVec<S, T, N, D> as ToFreqResult>::FreqResult as ToTimeResult>::TimeResult>
        + ScaleOps<T>
		+ ReorganizeDataOps<T> + Clone,
	  <DspVec<S, T, N, D> as ToFreqResult>::FreqResult:
          FrequencyDomainOperations<S, T> + ComplexOps<T> + Vector<T>
        + ElementaryOps<<DspVec<S, T, N, D> as ToFreqResult>::FreqResult>
        + FromVector<T, Output=S> + FrequencyToTimeDomainOperations<S, T>,
	  S: ToSliceMut<T> + Owner + ToDspVector<T> + ToComplexVector<S, T>,
	  T: RealNumber,
	  N: ComplexNumberSpace,
	  D: TimeDomain
{
    fn correlate<B>(
            &mut self,
            buffer: &mut B,
            other: &<DspVec<S, T, N, D> as ToFreqResult>::FreqResult) -> VoidResult
	 	where B: Buffer<S, T> {
		if self.domain() != DataDomain::Time
		   || !self.is_complex() {
            self.valid_len = 0;
            self.number_space.to_complex();
            self.domain.to_freq();
            return Err(ErrorReason::InputMustBeInTimeDomain);
        }
		let points = other.points();
		self.zero_pad_b(buffer, points, PaddingOption::Surround);
        let complex = self.take_ownership(buffer.construct_new(0));
        let mut complex = complex.plain_fft(buffer);
        try!(complex.mul(other));
		let complex = complex.plain_ifft(buffer);
        let mut complex = Self::rededicate_from(complex);
		self.swap_data(&mut complex);
		let p = self.points();
		self.scale(T::one() / T::from(p).unwrap());
        self.swap_halves();
		Ok(())
	}
}

#[cfg(test)]
mod tests {
    use super::super::super::*;

    #[test]
    fn time_correlation_test() {
        let mut a = vec![0.0800, 0.0, 0.1876, 0.1170, 0.4601, 0.4132, 0.7700, 0.7500, 0.9723,
                         0.9698, 0.9723, 0.9698, 0.7700, 0.7500, 0.4601, 0.4132, 0.1876, 0.1170,
                         0.0800, 0.0]
            .to_complex_time_vec();
        let b = vec![0.1000, -0.6366, 0.3000, 0.0, 0.5000, 0.2122, 0.7000, 0.0, 0.9000, -0.1273,
                     0.9000, 0.0, 0.7000, 0.0909, 0.5000, 0.0, 0.3000, -0.0707, 0.1000, 0.0]
            .to_complex_time_vec();
        let c: &[f32] = &[0.0080, 0.0000, 0.0428, 0.0174, 0.1340, 0.0897, 0.3356, 0.2827, 0.7192,
                          0.6479, 1.3058, 1.1946, 2.0175, 1.8757, 2.7047, 2.5665, 3.2186, 3.0874,
                          3.4409, 3.2994, 3.2291, 3.1287, 2.5801, 2.7264, 1.7085, 2.1882, 0.8637,
                          1.6369, 0.2319, 1.1420, -0.0878, 0.7078, -0.1208, 0.3523, -0.0317,
                          0.1311, 0.0080, 0.0509];
        let mut buffer = SingleBuffer::new();
        let b = b.prepare_argument_padded(&mut buffer);
        a.correlate(&mut buffer, &b).unwrap();
        let res = &a[..];
        let tol = 0.1;
        for i in 0..c.len() {
            if (res[i] - c[i]).abs() > tol {
                panic!("assertion failed: {:?} != {:?} at index {}", res, c, i);
            }
        }
    }

    #[test]
    fn time_correlation_test2() {
        let mut a = vec![1.0, 1.0, 2.0, 1.0, 3.0, 1.0].to_complex_time_vec();
        let b = vec![4.0, 1.0, 5.0, 1.0, 6.0, 1.0].to_complex_time_vec();
        let c: &[f32] = &[7.0, 5.0, 19.0, 8.0, 35.0, 9.0, 25.0, 4.0, 13.0, 1.0];
        let mut buffer = SingleBuffer::new();
        let b = b.prepare_argument_padded(&mut buffer);
        a.correlate(&mut buffer, &b).unwrap();
        let res = &a[..];
        let tol = 0.1;
        for i in 0..c.len() {
            if (res[i] - c[i]).abs() > tol {
                panic!("assertion failed: {:?} != {:?} at index {}", res, c, i);
            }
        }
    }
}
