use RealNumber;
use conv_types::*;
use num::Complex;
use multicore_support::*;
use super::super::{array_to_complex, array_to_complex_mut, VoidResult, ToSliceMut, MetaData,
                   DspVec, NumberSpace, TimeDomain, FrequencyDomain, DataDomain, Vector,
                   ComplexNumberSpace, Buffer, ErrorReason};

/// Provides a convolution operations.
pub trait Convolution<'a, S, T, C: ?Sized + 'a>
    where S: ToSliceMut<T>,
          T: RealNumber
{
    /// Convolves `self` with the convolution function `impulse_response`.
    /// For performance consider to
    /// to use `FrequencyMultiplication` instead of this operation depending on `len`.
    ///
    /// An optimized convolution algorithm is used if  `1.0 / ratio`
    /// is an integer (inside a `1e-6` tolerance)
    /// and `len` is smaller than a threshold (`202` right now).
    /// # Failures
    /// TransRes may report the following `ErrorReason` members:
    ///
    /// 1. `VectorMustBeComplex`: if `self` is in real number space but
    ///    `impulse_response` is in complex number space.
    /// 2. `VectorMustBeInTimeDomain`: if `self` is in frequency domain.
    fn convolve<B>(&mut self, buffer: &mut B, impulse_response: &C, ratio: T, len: usize)
        where B: Buffer<S, T>;
}

/// Provides a convolution operation for types which at some point are slice based.
pub trait ConvolutionOps<S, T, A>
    where S: ToSliceMut<T>,
          T: RealNumber
{
    /// Convolves `self` with the convolution function `impulse_response`.
    /// For performance it's recommended
    /// to use multiply both vectors in frequency domain instead of this operation.
    /// # Failures
    /// TransRes may report the following `ErrorReason` members:
    ///
    /// 1. `VectorMustBeInTimeDomain`: if `self` is in frequency domain.
    /// 2. `VectorMetaDataMustAgree`: in case `self` and `impulse_response`
    ///    are not in the same number space and same domain.
    /// 3. `InvalidArgumentLength`: if `self.points() < impulse_response.points()`.
    fn convolve_vector<B>(&mut self, buffer: &mut B, impulse_response: &A) -> VoidResult
        where B: Buffer<S, T>;
}

/// Provides a frequency response multiplication operations.
pub trait FrequencyMultiplication<'a, S, T, C: ?Sized + 'a>
    where S: ToSliceMut<T>,
          T: RealNumber
{
    /// Multiplies `self` with the frequency response function `frequency_response`.
    ///
    /// In order to multiply a vector with another vector in frequency response use `mul`.
    /// # Assumptions
    /// The operation assumes that the vector contains a full spectrum centered at 0 Hz.
    /// If half a spectrum
    /// or a FFT shifted spectrum is provided the operation will come back with invalid results.
    /// # Failures
    /// TransRes may report the following `ErrorReason` members:
    ///
    /// 1. `VectorMustBeComplex`: if `self` is in real number space but `frequency_response`
    ///    is in complex number space.
    /// 2. `VectorMustBeInFreqDomain`: if `self` is in time domain.
    fn multiply_frequency_response(&mut self, frequency_response: &C, ratio: T);
}

macro_rules! assert_complex {
    ($self_: ident) => {
        if !$self_.is_complex() {
            $self_.valid_len = 0;
			return;
        }
    }
}

macro_rules! assert_time {
    ($self_: ident) => {
        if $self_.domain() != DataDomain::Time {
            $self_.valid_len = 0;
			return;
        }
    }
}

impl<'a, S, T, N, D> Convolution<'a, S, T, RealImpulseResponse<T>> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber + 'a,
          N: NumberSpace,
          D: TimeDomain
{
    fn convolve<B>(&mut self,
                   buffer: &mut B,
                   function: &RealImpulseResponse<T>,
                   ratio: T,
                   len: usize)
        where B: Buffer<S, T>
    {
        assert_time!(self);
        if !self.is_complex() {
            let ratio_inv = T::one() / ratio;
            if len <= 202 && self.len() > 2000 &&
               (ratio_inv.round() - ratio_inv).abs() < T::from(1e-6).unwrap() &&
               ratio > T::from(0.5).unwrap() {
                let ratio: usize = ratio.abs()
                    .round()
                    .to_usize()
                    .expect("Converting ratio to usize failed, is the interpolation factor \
                             perhaps really huge?");
                let points = (2 * len + 1) * ratio;
                let mut imp_resp = DspVec {
                    data: buffer.construct_new(points),
                    delta: self.delta(),
                    domain: self.domain.clone(),
                    number_space: self.number_space.clone(),
                    valid_len: self.valid_len,
                    multicore_settings: MultiCoreSettings::default(),
                };

                let mut i = 0;
                let mut j = -(T::from(len).unwrap());
                while i < imp_resp.len() {
                    let value = function.calc(j * ratio_inv);
                    imp_resp[i] = value;
                    i += 1;
                    j = j + T::one();
                }

                self.convolve_vector(buffer, &imp_resp)
                    .expect("Meta data should agree since we constructed the argument from this \
                             vector");
            }

            self.convolve_function_priv(buffer,
                                        ratio,
                                        len,
                                        |data| data,
                                        |temp| temp,
                                        |x| function.calc(x));
        } else {
            let ratio_inv = T::one() / ratio;
            if len <= 202 && self.len() > 2000 &&
               (ratio_inv.round() - ratio_inv).abs() < T::from(1e-6).unwrap() &&
               ratio > T::from(0.5).unwrap() {
                let ratio: usize = ratio.abs()
                    .round()
                    .to_usize()
                    .expect("Converting ratio to usize failed, is the interpolation factor \
                             perhaps really huge?");
                let points = (2 * len + 1) * ratio;
                let mut imp_resp = DspVec {
                    data: buffer.construct_new(2 * points),
                    delta: self.delta(),
                    domain: self.domain.clone(),
                    number_space: self.number_space.clone(),
                    valid_len: 2 * points,
                    multicore_settings: MultiCoreSettings::default(),
                };

                let mut i = 0;
                let mut j = -(T::from(len).unwrap());
                while i < imp_resp.len() {
                    let value = function.calc(j * ratio_inv);
                    imp_resp[i] = value;
                    i += 2;
                    j = j + T::one();
                }

                self.convolve_vector(buffer, &imp_resp)
                    .expect("Meta data should agree since we constructed the argument from this \
                             vector");
            }

            self.convolve_function_priv(buffer,
                                        ratio,
                                        len,
                                        |data| array_to_complex(data),
                                        |temp| array_to_complex_mut(temp),
                                        |x| Complex::<T>::new(function.calc(x), T::zero()));
        }
    }
}

impl<'a, S, T, N, D> Convolution<'a, S, T, ComplexImpulseResponse<T>> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber + 'a,
          N: ComplexNumberSpace,
          D: TimeDomain
{
    fn convolve<B>(&mut self,
                   buffer: &mut B,
                   function: &ComplexImpulseResponse<T>,
                   ratio: T,
                   len: usize)
        where B: Buffer<S, T>
    {
        assert_complex!(self);
        assert_time!(self);

        let ratio_inv = T::one() / ratio;
        if len <= 202 && self.len() > 2000 &&
           (ratio_inv.round() - ratio_inv).abs() < T::from(1e-6).unwrap() &&
           ratio > T::from(0.5).unwrap() {
            let ratio: usize = ratio.abs()
                .round()
                .to_usize()
                .expect("Converting ratio to usize failed, is the interpolation factor perhaps \
                         really huge?");
            let points = (2 * len + 1) * ratio;
            let mut imp_resp = DspVec {
                data: buffer.construct_new(2 * points),
                delta: self.delta(),
                domain: self.domain.clone(),
                number_space: self.number_space.clone(),
                valid_len: self.valid_len,
                multicore_settings: MultiCoreSettings::default(),
            };

            let mut i = 0;
            let mut j = -T::from(len).unwrap();
            while i < imp_resp.len() {
                let value = function.calc(j * ratio_inv);
                imp_resp[i] = value.re;
                i += 2;
                imp_resp[i] = value.im;
                i += 1;
                j = j + T::one();
            }

            self.convolve_vector(buffer, &imp_resp)
                .expect("Meta data should agree since we constructed the argument from this \
                         vector");
        }

        self.convolve_function_priv(buffer,
                                    ratio,
                                    len,
                                    |data| array_to_complex(data),
                                    |temp| array_to_complex_mut(temp),
                                    |x| function.calc(x));
    }
}

macro_rules! assert_meta_data {
    ($self_: ident, $other: ident) => {
         {
            let delta_ratio = $self_.delta / $other.delta;
            if $self_.is_complex() != $other.is_complex() ||
                $self_.domain != $other.domain ||
                delta_ratio > T::from(1.1).unwrap() || delta_ratio < T::from(0.9).unwrap() {
                return Err(ErrorReason::InputMetaDataMustAgree);
            }
         }
    }
}

impl<S, T, N, D> ConvolutionOps<S, T, DspVec<S, T, N, D>> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: NumberSpace,
          D: TimeDomain
{
    fn convolve_vector<B>(&mut self, buffer: &mut B, impulse_response: &Self) -> VoidResult
        where B: Buffer<S, T>
    {
        assert_meta_data!(self, impulse_response);
        if self.domain() != DataDomain::Time {
            return Err(ErrorReason::InputMustBeInTimeDomain);
        }

        if self.points() < impulse_response.points() {
            return Err(ErrorReason::InvalidArgumentLength);
        }

        // The values in this condition are nothing more than a
        // ... guess. The reasoning is basically this:
        // For the SIMD operation we need to clone `vector` several
        // times and this only is worthwhile if `vector.len() << self.len()`
        // where `<<` means "significant smaller".
        if self.len() > 1000 && impulse_response.len() <= 202 {
            self.convolve_vector_simd(buffer, impulse_response);
        } else {
            self.convolve_vector_scalar(buffer, impulse_response);
        }

        Ok(())
    }
}

impl<'a, S, T, N, D> FrequencyMultiplication<'a, S, T, ComplexFrequencyResponse<T>>
    for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber + 'a,
          N: ComplexNumberSpace,
          D: FrequencyDomain
{
    fn multiply_frequency_response(&mut self,
                                   frequency_response: &ComplexFrequencyResponse<T>,
                                   ratio: T) {
        if !self.is_complex() || self.domain() != DataDomain::Frequency {
            self.valid_len = 0;
            return;
        }
        self.multiply_function_priv(frequency_response.is_symmetric(),
                                    ratio,
                                    |array| array_to_complex_mut(array),
                                    frequency_response,
                                    |f, x| f.calc(x));
    }
}

impl<'a, S, T, N, D> FrequencyMultiplication<'a, S, T, RealFrequencyResponse<T>>
    for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber + 'a,
          N: NumberSpace,
          D: FrequencyDomain
{
    fn multiply_frequency_response(&mut self,
                                   frequency_response: &RealFrequencyResponse<T>,
                                   ratio: T) {
        if self.domain() != DataDomain::Frequency {
            self.valid_len = 0;
            return;
        }
        if self.is_complex() {
            self.multiply_function_priv(frequency_response.is_symmetric(),
                                        ratio,
                                        |array| array_to_complex_mut(array),
                                        frequency_response,
                                        |f, x| Complex::<T>::new(f.calc(x), T::zero()))
        } else {
            self.multiply_function_priv(frequency_response.is_symmetric(),
                                        ratio,
                                        |array| array,
                                        frequency_response,
                                        |f, x| f.calc(x))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{WrappingIterator, ReverseWrappingIterator};
    use super::super::super::*;
    use conv_types::*;
    use RealNumber;
    use num::complex::Complex32;
    use std::fmt::Debug;

    fn assert_eq_tol<T>(left: &[T], right: &[T], tol: T)
        where T: RealNumber + Debug
    {
        assert_eq!(left.len(), right.len());
        for i in 0..left.len() {
            if (left[i] - right[i]).abs() > tol {
                panic!("assertion failed: {:?} != {:?}", left, right);
            }
        }
    }

    #[test]
    fn convolve_complex_freq_and_freq32() {
        let mut vector = vec!(1.0; 10).to_complex_freq_vec();
        let rc: RaisedCosineFunction<f32> = RaisedCosineFunction::new(1.0);
        vector.multiply_frequency_response(&rc as &RealFrequencyResponse<f32>, 2.0);
        let expected = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0];
        assert_eq_tol(&vector[..], &expected, 1e-4);
    }

    #[test]
    fn convolve_complex_freq_and_freq_even32() {
        let mut vector = vec!(1.0; 12).to_complex_freq_vec();
        let rc: RaisedCosineFunction<f32> = RaisedCosineFunction::new(1.0);
        vector.multiply_frequency_response(&rc as &RealFrequencyResponse<f32>, 2.0);
        let expected = [0.0, 0.0, 0.5, 0.5, 1.5, 1.5, 2.0, 2.0, 1.5, 1.5, 0.5, 0.5];
        assert_eq_tol(&vector[..], &expected, 1e-4);
    }

    #[test]
    fn convolve_real_time_and_time32() {
        let mut vector = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0].to_real_time_vec();
        let rc: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
        let mut buffer = SingleBuffer::new();
        vector.convolve(&mut buffer, &rc as &RealImpulseResponse<f32>, 0.2, 5);
        let expected = [0.0,
                        0.2171850639713355,
                        0.4840621929215732,
                        0.7430526238101408,
                        0.9312114164253432,
                        1.0,
                        0.9312114164253432,
                        0.7430526238101408,
                        0.4840621929215732,
                        0.2171850639713355];
        assert_eq_tol(&vector[..], &expected, 1e-4);
    }

    #[test]
    fn convolve_complex_time_and_time32() {
        let len = 11;
        let mut time = vec!(0.0; 2 * len).to_complex_time_vec();
        time[len] = 1.0;
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.convolve(&mut buffer,
                      &sinc as &RealImpulseResponse<f32>,
                      0.5,
                      len / 2);
        let res = time.magnitude();
        let expected = [0.12732396,
                        0.000000027827534,
                        0.21220659,
                        0.000000027827534,
                        0.63661975,
                        1.0,
                        0.63661975,
                        0.000000027827534,
                        0.21220659,
                        0.000000027827534,
                        0.12732396];
        assert_eq_tol(&res[..], &expected, 1e-4);
    }

    #[test]
    fn compare_conv_freq_mul() {
        let len = 11;
        let mut time = vec!(0.0; 2 * len).to_complex_time_vec();
        time[len] = 1.0;
        let mut buffer = SingleBuffer::new();
        let mut freq = time.clone().fft(&mut buffer);
        let sinc: SincFunction<f32> = SincFunction::new();
        let ratio = 0.5;
        freq.multiply_frequency_response(&sinc as &RealFrequencyResponse<f32>, 1.0 / ratio);
        time.convolve(&mut buffer, &sinc as &RealImpulseResponse<f32>, 0.5, len);
        let ifft = freq.ifft(&mut buffer).magnitude();
        let time = time.magnitude();
        assert_eq!(ifft.is_complex(), time.is_complex());
        assert_eq!(ifft.domain(), time.domain());
        assert_eq_tol(&ifft[..], &time[..], 0.2);
    }

    #[test]
    fn invalid_length_parameter() {
        let len = 20;
        let mut time = vec!(0.0; 2 * len).to_complex_time_vec();
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.convolve(&mut buffer,
                      &sinc as &RealImpulseResponse<f32>,
                      0.5,
                      10 * len);
        // As long as we don't panic we are happy with the error handling here
    }

    #[test]
    fn convolve_complex_vectors32() {
        const LEN: usize = 11;
        let mut time = vec!(Complex32::new(0.0, 0.0); LEN).to_complex_time_vec();
        time[LEN] = 1.0;
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut real = [0.0; LEN];
        {
            let mut v = -5.0;
            for a in &mut real {
                *a = (&sinc as &RealImpulseResponse<f32>).calc(v * 0.5);
                v += 1.0;
            }
        }
        let imag = &[0.0; LEN];
        let argument = (&real[..]).interleave_to_complex_time_vec(&&imag[..]).unwrap();
        assert_eq!(time.points(), argument.points());
        let mut buffer = SingleBuffer::new();
        time.convolve_vector(&mut buffer, &argument).unwrap();
        assert_eq!(time.points(), LEN);
        let result = time.magnitude();
        assert_eq!(result.points(), LEN);
        let expected = [0.12732396,
                        0.000000027827534,
                        0.21220659,
                        0.000000027827534,
                        0.63661975,
                        1.0,
                        0.63661975,
                        0.000000027827534,
                        0.21220659,
                        0.000000027827534,
                        0.12732396];
        assert_eq_tol(&result[..], &expected, 1e-4);
    }

    #[test]
    fn wrapping_iterator() {
        let array = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut iter = WrappingIterator::new(&array, -3, 8);
        assert_eq!(iter.next().unwrap(), 4.0);
        assert_eq!(iter.next().unwrap(), 5.0);
        assert_eq!(iter.next().unwrap(), 1.0);
        assert_eq!(iter.next().unwrap(), 2.0);
        assert_eq!(iter.next().unwrap(), 3.0);
        assert_eq!(iter.next().unwrap(), 4.0);
        assert_eq!(iter.next().unwrap(), 5.0);
        assert_eq!(iter.next().unwrap(), 1.0);
    }

    #[test]
    fn wrapping_rev_iterator() {
        let array = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut iter = ReverseWrappingIterator::new(&array, 2, 5);
        assert_eq!(iter.next().unwrap(), 2.0);
        assert_eq!(iter.next().unwrap(), 1.0);
        assert_eq!(iter.next().unwrap(), 5.0);
        assert_eq!(iter.next().unwrap(), 4.0);
        assert_eq!(iter.next().unwrap(), 3.0);
    }

    #[test]
    fn vector_conv_vs_freq_multiplication() {
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].to_complex_time_vec();
        let b = vec![15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0].to_complex_time_vec();
        let mut buffer = SingleBuffer::new();
        let mut conv = a.clone();
        conv.convolve_vector(&mut buffer, &b).unwrap();
        let mut a = a.fft(&mut buffer);
        let b = b.fft(&mut buffer);
        a.mul(&b).unwrap();
        let mut mul = a.ifft(&mut buffer);
        mul.reverse();
        mul.swap_halves();
        assert_eq_tol(&mul[..], &conv[..], 1e-4);
    }

    #[test]
    fn shift_left_by_1_as_conv() {
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].to_real_time_vec();
        let b = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0].to_real_time_vec();
        let mut buffer = SingleBuffer::new();
        let mut a = a.to_complex().unwrap();
        let b = b.to_complex().unwrap();
        a.convolve_vector(&mut buffer, &b).unwrap();
        let a = a.magnitude();
        let exp = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq_tol(&a[..], &exp, 1e-4);
    }

    #[test]
    fn shift_left_by_1_as_conv_shorter() {
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].to_real_time_vec();
        let mut buffer = SingleBuffer::new();
        let b = vec![0.0, 0.0, 1.0].to_real_time_vec();
        let mut a = a.to_complex().unwrap();
        let b = b.to_complex().unwrap();
        a.convolve_vector(&mut buffer, &b).unwrap();
        let a = a.magnitude();
        let exp = [9.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq_tol(&a[..], &exp, 1e-4);
    }

    #[test]
    fn vector_conv_vs_freq_multiplication_pure_real_data() {
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].to_real_time_vec();
        let b = vec![15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0].to_real_time_vec();
        let mut buffer = SingleBuffer::new();
        let a = a.to_complex().unwrap();
        let b = b.to_complex().unwrap();
        let mut conv = a.clone();
        conv.convolve_vector(&mut buffer, &b).unwrap();
        let mut a = a.fft(&mut buffer);
        let b = b.fft(&mut buffer);
        a.mul(&b).unwrap();
        let mul = a.ifft(&mut buffer);
        let mut mul = mul.magnitude();
        mul.reverse();
        mul.swap_halves();
        let conv = conv.magnitude();
        assert_eq_tol(&mul[..], &conv[..], 1e-4);
    }

    #[test]
    fn vector_conv_vs_freq_multiplication_pure_real_data_odd() {
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].to_real_time_vec();
        let b = vec![15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0].to_real_time_vec();
        let mut buffer = SingleBuffer::new();
        let a = a.to_complex().unwrap();
        let b = b.to_complex().unwrap();
        let mut conv = a.clone();
        conv.convolve_vector(&mut buffer, &b).unwrap();
        let mut a = a.fft(&mut buffer);
        let b = b.fft(&mut buffer);
        a.mul(&b).unwrap();
        let mul = a.ifft(&mut buffer);
        let mut mul = mul.magnitude();
        mul.reverse();
        mul.swap_halves();
        let conv = conv.magnitude();
        assert_eq_tol(&mul[..], &conv[..], 1e-4);
    }
}
