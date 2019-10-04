use super::super::{
    Buffer, BufferBorrow, ComplexNumberSpace, DataDomain, Domain, DspVec, ErrorReason,
    FloatIndexMut, FrequencyDomain, FromVectorFloat, InsertZerosOps, MetaData, NumberSpace,
    PaddingOption, PosEq, ResizeOps, TimeDomain, TimeToFrequencyDomainOperations, ToComplexVector,
    ToSliceMut, Vector, VoidResult,
};
use crate::conv_types::*;
use crate::gpu_support::GpuSupport;
use crate::inline_vector::InlineVector;
use crate::multicore_support::*;
use crate::numbers::*;
use crate::simd_extensions::*;
use crate::{array_to_complex, array_to_complex_mut};
use rustfft::FFTplanner;

/// Provides a convolution operations.
pub trait Convolution<'a, S, T, C: 'a>
where
    S: ToSliceMut<T>,
    T: RealNumber,
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
    /// 1. `VectorMustBeComplex`: if `self` is in real number space overlap_discard
    ///    `impulse_response` is in complex number space.
    /// 2. `VectorMustBeInTimeDomain`: if `self` is in frequency domain.
    fn convolve<B>(&mut self, buffer: &mut B, impulse_response: C, ratio: T, len: usize)
    where
        B: for<'b> Buffer<'b, S, T>;
}

/// Provides a convolution operation for types which at some point are slice based.
pub trait ConvolutionOps<A, S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    /// Convolves `self` with the convolution function `impulse_response`.
    /// For performance it's recommended
    /// to use multiply both vectors in frequency domain instead of this operation.
    ///
    /// # Failures
    /// TransRes may report the following `ErrorReason` members:
    ///
    /// 1. `VectorMustBeInTimeDomain`: if `self` is in frequency domain.
    /// 2. `VectorMetaDataMustAgree`: in case `self` and `impulse_response`
    ///    are not in the same number space and same domain.
    /// 3. `InvalidArgumentLength`: if `self.points() < impulse_response.points()`.
    fn convolve_signal<B>(&mut self, buffer: &mut B, impulse_response: &A) -> VoidResult
    where
        B: for<'a> Buffer<'a, S, T>;
}

/// Provides a frequency response multiplication operations.
pub trait FrequencyMultiplication<'a, S, T, C: 'a>
where
    S: ToSliceMut<T>,
    T: RealNumber,
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
    fn multiply_frequency_response(&mut self, frequency_response: C, ratio: T);
}

macro_rules! assert_complex {
    ($self_: ident) => {
        if !$self_.is_complex() {
            $self_.mark_vector_as_invalid();
            return;
        }
    };
}

macro_rules! assert_time {
    ($self_: ident) => {
        if $self_.domain() != DataDomain::Time {
            $self_.mark_vector_as_invalid();
            return;
        }
    };
}

fn would_benefit_from_simd<T: RealNumber>(vec_len: usize, imp_len: usize, ratio: T) -> bool {
    let ratio_inv = T::one() / ratio;
    imp_len <= 202
        && vec_len > 2000
        && (ratio_inv.round() - ratio_inv).abs() < T::from(1e-6).unwrap()
        && ratio > T::from(0.5).unwrap()
}

fn can_use_simd<T: RealNumber>(points: usize) -> bool {
    points <= InlineVector::<T>::max_capacity()
}

fn calc_points<T: RealNumber>(imp_len: usize, ratio: T, is_complex: bool) -> usize {
    let step = if is_complex { 2 } else { 1 };
    let ratio_usize: usize = ratio
        .abs()
        .round()
        .to_usize()
        .expect("Converting ratio to usize failed, interpolation factor is likely too large");
    step * (2 * imp_len + 1) * ratio_usize
}

impl<'a, S, T, N, D> Convolution<'a, S, T, &'a dyn RealImpulseResponse<T>> for DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: NumberSpace,
    D: TimeDomain,
    DspVec<S, T, N, D>: TimeToFrequencyDomainOperations<S, T>
        + Clone
        + ConvolutionOps<DspVec<InlineVector<T>, T, N, D>, S, T, N, D>,
{
    fn convolve<B>(
        &mut self,
        buffer: &mut B,
        function: &dyn RealImpulseResponse<T>,
        ratio: T,
        len: usize,
    ) where
        B: for<'b> Buffer<'b, S, T>,
    {
        assert_time!(self);
        let ratio_inv = T::one() / ratio;
        let points = calc_points(len, ratio, self.is_complex());
        if would_benefit_from_simd(self.len(), len, ratio) && can_use_simd::<T>(points) {
            // Create a vector from the impulse response and use `convolve_signal` to get
            // SIMD support
            let mut imp_resp = DspVec {
                data: InlineVector::of_size(T::zero(), points),
                delta: self.delta(),
                domain: self.domain.clone(),
                number_space: self.number_space.clone(),
                valid_len: points,
                multicore_settings: MultiCoreSettings::default(),
            };

            let mut i = 0;
            let mut j = -(T::from(len).unwrap());
            while i < imp_resp.len() {
                let value = function.calc(j * ratio_inv);
                *imp_resp.data_mut(i) = value;
                i += 2;
                j = j + T::one();
            }

            self.convolve_signal(buffer, &imp_resp).expect(
                "Meta data should agree since we constructed the argument from this \
                 vector",
            );
        } else if self.is_complex() {
            self.convolve_function_priv(
                buffer,
                ratio,
                len,
                |data| array_to_complex(data),
                |temp| array_to_complex_mut(temp),
                |x| Complex::<T>::new(function.calc(x), T::zero()),
            );
        } else {
            self.convolve_function_priv(
                buffer,
                ratio,
                len,
                |data| data,
                |temp| temp,
                |x| function.calc(x),
            );
        }
    }
}

impl<'a, S, T, N, D> Convolution<'a, S, T, &'a dyn ComplexImpulseResponse<T>> for DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: TimeDomain,
    DspVec<S, T, N, D>: TimeToFrequencyDomainOperations<S, T>
        + Clone
        + ConvolutionOps<DspVec<InlineVector<T>, T, N, D>, S, T, N, D>,
{
    fn convolve<B>(
        &mut self,
        buffer: &mut B,
        function: &dyn ComplexImpulseResponse<T>,
        ratio: T,
        len: usize,
    ) where
        B: for<'b> Buffer<'b, S, T>,
    {
        assert_complex!(self);
        assert_time!(self);

        let ratio_inv = T::one() / ratio;
        let points = calc_points(len, ratio, self.is_complex());
        if would_benefit_from_simd(self.len(), len, ratio) && can_use_simd::<T>(points) {
            let mut imp_resp = DspVec {
                data: InlineVector::of_size(T::zero(), 2 * points),
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
                *imp_resp.data_mut(i) = value.re;
                i += 2;
                *imp_resp.data_mut(i) = value.im;
                i += 1;
                j = j + T::one();
            }

            self.convolve_signal(buffer, &imp_resp).expect(
                "Meta data should agree since we constructed the argument from this \
                 vector",
            );
        } else {
            self.convolve_function_priv(
                buffer,
                ratio,
                len,
                |data| array_to_complex(data),
                |temp| array_to_complex_mut(temp),
                |x| function.calc(x),
            );
        }
    }
}

macro_rules! assert_meta_data {
    ($self_: ident, $other: ident) => {{
        let delta_ratio = $self_.delta / $other.delta;
        if $self_.is_complex() != $other.is_complex()
            || $self_.domain() != $other.domain()
            || delta_ratio > T::from(1.1).unwrap()
            || delta_ratio < T::from(0.9).unwrap()
        {
            return Err(ErrorReason::InputMetaDataMustAgree);
        }
    }};
}

fn next_power_of_two(value: usize) -> usize {
    let mut count = 0;
    let mut n = value;
    if n != 0 && (n & (n - 1)) == 0 {
        return n;
    }
    while n != 0 {
        n >>= 1;
        count += 1;
    }

    1 << count
}

fn array_to_real<T>(array: &[Complex<T>]) -> &[T] {
    super::super::super::transmute_slice(array)
}

fn array_to_real_mut<T>(array: &mut [Complex<T>]) -> &mut [T] {
    super::super::super::transmute_slice_mut(array)
}

impl<S, T, N, D> DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: NumberSpace,
    D: TimeDomain,
    DspVec<S, T, N, D>: TimeToFrequencyDomainOperations<S, T> + Clone,
{
    /// Overlap-discard or overlap-save implementation:
    /// https://en.wikipedia.org/wiki/Overlap%E2%80%93save_method
    ///
    /// With this method the convolution is performed in freq domain
    fn overlap_discard<B, SO, NO, DO>(
        &mut self,
        buffer: &mut B,
        impulse_response: &DspVec<SO, T, NO, DO>,
        fft_len: usize,
    ) -> VoidResult
    where
        B: for<'a> Buffer<'a, S, T>,
        SO: ToSliceMut<T>,
        NO: NumberSpace,
        DO: Domain,
    {
        if !self.is_complex() {
            return Err(ErrorReason::InputMustBeComplex);
        }

        assert_meta_data!(self, impulse_response);

        let h_time = impulse_response;
        let imp_len = h_time.points();
        let x_len = self.points();
        let overlap = imp_len - 1;
        let min_fft_len = next_power_of_two(4 * overlap);
        let fft_len = if fft_len > min_fft_len {
            fft_len
        } else {
            min_fft_len
        };

        let fft = {
            let mut forward_planner = FFTplanner::new(false);
            forward_planner.plan_fft(fft_len)
        };
        let ifft = {
            let mut reverse_planner = FFTplanner::new(true);
            reverse_planner.plan_fft(fft_len)
        };
        let step_size = fft_len - overlap;
        let h_time_padded_len = 2 * fft_len;
        let h_freq_len = 2 * fft_len;
        let x_freq_len = 2 * fft_len;
        let tmp_len = 2 * fft_len;
        let remainder_len = x_len - x_len % fft_len;
        let mut array =
            buffer.borrow(h_time_padded_len + h_freq_len + x_freq_len + tmp_len + remainder_len);
        {
            let array = array.to_slice_mut();
            let (h_freq, array) = array.split_at_mut(h_freq_len);
            let (x_freq, array) = array.split_at_mut(x_freq_len);
            let (tmp, array) = array.split_at_mut(tmp_len);
            let (h_time_padded, array) = array.split_at_mut(h_time_padded_len);
            let end = array;
            let h_time = h_time.data.to_slice();
            (&mut h_time_padded[0..2 * imp_len]).copy_from_slice(&h_time[0..2 * imp_len]);

            let mut h_time_padded = (&mut h_time_padded[..]).to_complex_time_vec();
            h_time_padded
                .resize(2 * imp_len)
                .expect("Shrinking a vector should always succeed");
            h_time_padded
                .zero_pad(fft_len, PaddingOption::End)
                .expect("zero padding should always have enough space");
            let (h_time_padded, _) = h_time_padded.getf();

            let x_time = self.data.to_slice_mut();
            let h_time: &[Complex<T>] = array_to_complex(&h_time[..]);
            let h_time_padded: &mut [Complex<T>] =
                array_to_complex_mut(&mut h_time_padded[0..2 * fft_len]);
            let x_time: &mut [Complex<T>] = array_to_complex_mut(&mut x_time[0..2 * x_len]);
            let h_freq: &mut [Complex<T>] = array_to_complex_mut(&mut h_freq[0..2 * fft_len]);
            let x_freq: &mut [Complex<T>] = array_to_complex_mut(&mut x_freq[0..2 * fft_len]);
            let tmp: &mut [Complex<T>] = array_to_complex_mut(&mut tmp[0..2 * fft_len]);
            let end: &mut [Complex<T>] = array_to_complex_mut(&mut end[0..remainder_len]);
            fft.process(h_time_padded, h_freq);
            let mut position = 0;
            {
                // (1) Scalar convolution of the beginning
                for num in &mut tmp[0..imp_len / 2] {
                    *num = Self::convolve_iteration(
                        x_time,
                        h_time,
                        position,
                        ((imp_len + 1) / 2) as isize,
                        imp_len,
                    );
                    position += 1;
                }
                // (2) Scalar convolution of the end/tail
                position = (x_time.len() - end.len()) as isize;
                for num in &mut end[0..remainder_len / 2] {
                    *num = Self::convolve_iteration(
                        x_time,
                        h_time,
                        position,
                        ((imp_len + 1) / 2) as isize,
                        imp_len,
                    );
                    position += 1;
                }
            }

            let position: usize = if T::has_gpu_support() {
                // GPU implementation of overlap_discard `thinks` in interleaved arrays
                // while this implementation `thinks` in complex arrays. Therefore a factor
                // of 2 needs to be taken into account in the method call.
                let position = T::overlap_discard(
                    array_to_real_mut(x_time),
                    array_to_real_mut(tmp),
                    array_to_real_mut(x_freq),
                    array_to_real(h_freq),
                    2 * imp_len,
                    2 * step_size,
                ) / 2;
                position as usize
            } else {
                // `fft.process` will invalidate `x_time` we therefore need to remember the overlap region
                // and restore it.
                let mut overlap_buffer = InlineVector::of_size(Complex::<T>::zero(), overlap);
                let mut position = 0;
                let scaling = T::from(fft_len).unwrap();
                {
                    let range = position..fft_len + position;
                    // (3) The first iteration is different since it copies over the results which have been calculated for the beginning
                    (&mut overlap_buffer[..])
                        .copy_from_slice(&x_time[position + step_size..position + fft_len]);
                    fft.process(&mut x_time[range], x_freq);
                    // Copy over the results of the scalar convolution (1)
                    (&mut x_time[0..imp_len / 2]).copy_from_slice(&tmp[0..imp_len / 2]);
                    for (n, v) in (&mut x_freq[..]).iter_mut().zip(h_freq.iter()) {
                        *n = *n * *v / scaling;
                    }
                    ifft.process(x_freq, tmp);
                    position += step_size;
                }

                while position + fft_len < x_len {
                    let range = position..fft_len + position;
                    (&mut x_time[position..position + overlap])
                        .copy_from_slice(&overlap_buffer[..]);
                    (&mut overlap_buffer[..])
                        .copy_from_slice(&x_time[position + step_size..position + fft_len]);
                    fft.process(&mut x_time[range], x_freq);
                    // (4) Same as (3) except that the results of the previous iteration gets copied over
                    (&mut x_time[position - step_size + imp_len / 2..position + imp_len / 2])
                        .copy_from_slice(&tmp[imp_len - 1..fft_len]);
                    for (n, v) in (&mut x_freq[..]).iter_mut().zip(h_freq.iter()) {
                        *n = *n * *v / scaling;
                    }
                    ifft.process(x_freq, tmp);
                    position += step_size;
                }
                position
            };

            // (5) now store the result of the last iteration
            (&mut x_time[position - step_size + imp_len / 2..position + imp_len / 2])
                .copy_from_slice(&tmp[imp_len - 1..fft_len]);

            // (6) Copy over the end result which was calculated at the beginning (2)
            (&mut x_time[x_len - remainder_len / 2..]).copy_from_slice(&end[..]);
        }
        Ok(())
    }
}

impl<S, SO, T, N, D, NO, DO> ConvolutionOps<DspVec<SO, T, NO, DO>, S, T, NO, DO>
    for DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    SO: ToSliceMut<T>,
    T: RealNumber,
    N: NumberSpace,
    D: TimeDomain,
    DspVec<S, T, N, D>: TimeToFrequencyDomainOperations<S, T> + Clone,
    DspVec<SO, T, N, D>: TimeToFrequencyDomainOperations<SO, T> + Clone,
    NO: PosEq<N> + NumberSpace,
    DO: TimeDomain,
{
    fn convolve_signal<B>(
        &mut self,
        buffer: &mut B,
        impulse_response: &DspVec<SO, T, NO, DO>,
    ) -> VoidResult
    where
        B: for<'a> Buffer<'a, S, T>,
    {
        assert_meta_data!(self, impulse_response);
        if self.domain() != DataDomain::Time {
            return Err(ErrorReason::InputMustBeInTimeDomain);
        }

        if self.points() < impulse_response.points() {
            return Err(ErrorReason::InvalidArgumentLength);
        }

        // The values in this condition are nothing more than a guess.
        // The reasoning is basically this:
        // For the SIMD operation we need to clone `vector` several
        // times and this only is worthwhile if `vector.len() << self.len()`
        // where `<<` means "significant smaller".
        if self.len() > 1000 && impulse_response.len() <= 202 && impulse_response.len() > 11 {
            sel_reg!(self.convolve_signal_simd::<T>(buffer, impulse_response));
            return Ok(());
        }

        if self.len() > 10000 && T::has_gpu_support() {
            let is_complex = self.is_complex();
            let source_len = self.len();
            let imp_resp_len = impulse_response.len();
            let mut target = buffer.borrow(source_len);
            let range_done = {
                let source = self.data.to_slice();
                let imp_resp = impulse_response.data.to_slice();
                let target = target.to_slice_mut();
                T::gpu_convolve_vector(
                    is_complex,
                    &source[0..source_len],
                    &mut target[0..source_len],
                    &imp_resp[0..imp_resp_len],
                )
            };
            match range_done {
                None => (), // The GPU code can't handle this go to the next cases
                Some(range) => {
                    self.convolve_vector_range(target.to_slice_mut(), impulse_response, range);
                    target.trade(&mut self.data);
                    return Ok(());
                }
            };
        }

        if self.len() > 10000
            && impulse_response.len() > 15
            // Overlap-discard creates a lot of copies of the size of impulse_response
            // and that only seems to be worthwhile if the vector is long enough
            && self.len() > 10 * impulse_response.len() && self.is_complex()
        {
            let fft_len = next_power_of_two(impulse_response.len());
            return self.overlap_discard(buffer, impulse_response, fft_len);
        }

        self.convolve_signal_scalar(buffer, impulse_response);
        Ok(())
    }
}

impl<'a, S, T, N, D> FrequencyMultiplication<'a, S, T, &'a dyn ComplexFrequencyResponse<T>>
    for DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: FrequencyDomain,
{
    fn multiply_frequency_response(
        &mut self,
        frequency_response: &dyn ComplexFrequencyResponse<T>,
        ratio: T,
    ) {
        if !self.is_complex() || self.domain() != DataDomain::Frequency {
            self.mark_vector_as_invalid();
            return;
        }
        self.multiply_function_priv(
            frequency_response.is_symmetric(),
            false,
            ratio,
            |array| array_to_complex_mut(array),
            frequency_response,
            |f, x| f.calc(x),
        );
    }
}

impl<'a, S, T, N, D> FrequencyMultiplication<'a, S, T, &'a dyn RealFrequencyResponse<T>>
    for DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: NumberSpace,
    D: FrequencyDomain,
{
    fn multiply_frequency_response(
        &mut self,
        frequency_response: &dyn RealFrequencyResponse<T>,
        ratio: T,
    ) {
        if self.domain() != DataDomain::Frequency {
            self.mark_vector_as_invalid();
            return;
        }
        if self.is_complex() {
            self.multiply_function_priv(
                frequency_response.is_symmetric(),
                false,
                ratio,
                |array| array_to_complex_mut(array),
                frequency_response,
                |f, x| Complex::<T>::new(f.calc(x), T::zero()),
            )
        } else {
            self.multiply_function_priv(
                frequency_response.is_symmetric(),
                false,
                ratio,
                |array| array,
                frequency_response,
                |f, x| f.calc(x),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::*;
    use super::super::{ReverseWrappingIterator, WrappingIterator};
    use crate::conv_types::*;
    use num_complex::Complex32;
    use std::fmt::Debug;

    fn assert_eq_tol<T>(left: &[T], right: &[T], tol: T)
    where
        T: RealNumber + Debug,
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
        let mut vector = vec![1.0; 10].to_complex_freq_vec();
        let rc: RaisedCosineFunction<f32> = RaisedCosineFunction::new(1.0);
        vector.multiply_frequency_response(&rc as &dyn RealFrequencyResponse<f32>, 2.0);
        let expected = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0];
        assert_eq_tol(vector.data(..), &expected, 1e-4);
    }

    #[test]
    fn convolve_complex_freq_and_freq_even32() {
        let mut vector = vec![1.0; 12].to_complex_freq_vec();
        let rc: RaisedCosineFunction<f32> = RaisedCosineFunction::new(1.0);
        vector.multiply_frequency_response(&rc as &dyn RealFrequencyResponse<f32>, 2.0);
        let expected = [0.0, 0.0, 0.5, 0.5, 1.5, 1.5, 2.0, 2.0, 1.5, 1.5, 0.5, 0.5];
        assert_eq_tol(vector.data(..), &expected, 1e-4);
    }

    #[test]
    fn convolve_real_time_and_time32() {
        let mut vector = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0].to_real_time_vec();
        let rc: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
        let mut buffer = SingleBuffer::new();
        vector.convolve(&mut buffer, &rc, 0.2, 5);
        let expected = [
            0.0,
            0.2171850639713355,
            0.4840621929215732,
            0.7430526238101408,
            0.9312114164253432,
            1.0,
            0.9312114164253432,
            0.7430526238101408,
            0.4840621929215732,
            0.2171850639713355,
        ];
        assert_eq_tol(vector.data(..), &expected, 1e-4);
    }

    #[test]
    fn convolve_complex_time_and_time32() {
        let res = {
            let len = 11;
            let mut time = vec![0.0; 2 * len].to_complex_time_vec();
            *time.data_mut(len) = 1.0;
            let sinc: SincFunction<f32> = SincFunction::new();
            let mut buffer = SingleBuffer::new();
            time.convolve(
                &mut buffer,
                &sinc as &dyn RealImpulseResponse<f32>,
                0.5,
                len / 2,
            );
            time.magnitude()
        };

        let expected = [
            0.12732396,
            0.000000027827534,
            0.21220659,
            0.000000027827534,
            0.63661975,
            1.0,
            0.63661975,
            0.000000027827534,
            0.21220659,
            0.000000027827534,
            0.12732396,
        ];
        assert_eq_tol(res.data(..), &expected, 1e-4);
    }

    #[test]
    fn compare_conv_freq_mul() {
        let len = 11;
        let mut time = vec![0.0; 2 * len].to_complex_time_vec();
        *time.data_mut(len) = 1.0;
        let mut buffer = SingleBuffer::new();
        let mut freq = time.clone().fft(&mut buffer);
        let sinc: SincFunction<f32> = SincFunction::new();
        let ratio = 0.5;
        freq.multiply_frequency_response(&sinc as &dyn RealFrequencyResponse<f32>, 1.0 / ratio);
        time.convolve(&mut buffer, &sinc as &dyn RealImpulseResponse<f32>, 0.5, len);
        let ifft = freq.ifft(&mut buffer).magnitude();
        let time = time.magnitude();
        assert_eq!(ifft.is_complex(), time.is_complex());
        assert_eq!(ifft.domain(), time.domain());
        assert_eq_tol(ifft.data(..), time.data(..), 0.2);
    }

    #[test]
    fn invalid_length_parameter() {
        let len = 20;
        let mut time = vec![0.0; 2 * len].to_complex_time_vec();
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.convolve(
            &mut buffer,
            &sinc as &dyn RealImpulseResponse<f32>,
            0.5,
            10 * len,
        );
        // As long as we don't panic we are happy with the error handling here
    }

    #[test]
    fn convolve_complex_vectors32() {
        const LEN: usize = 11;
        let mut time = vec![Complex32::new(0.0, 0.0); LEN].to_complex_time_vec();
        *time.data_mut(LEN) = 1.0;
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut real = [0.0; LEN];
        {
            let mut v = -5.0;
            for a in &mut real {
                *a = (&sinc as &dyn RealImpulseResponse<f32>).calc(v * 0.5);
                v += 1.0;
            }
        }
        let imag = &[0.0; LEN];
        let argument = (&real[..])
            .interleave_to_complex_time_vec(&&imag[..])
            .unwrap();
        assert_eq!(time.points(), argument.points());
        let mut buffer = SingleBuffer::new();
        time.convolve_signal(&mut buffer, &argument).unwrap();
        assert_eq!(time.points(), LEN);
        let result = time.magnitude();
        assert_eq!(result.points(), LEN);
        let expected = [
            0.12732396,
            0.000000027827534,
            0.21220659,
            0.000000027827534,
            0.63661975,
            1.0,
            0.63661975,
            0.000000027827534,
            0.21220659,
            0.000000027827534,
            0.12732396,
        ];
        assert_eq_tol(result.data(..), &expected, 1e-4);
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
        conv.convolve_signal(&mut buffer, &b).unwrap();
        let mut a = a.fft(&mut buffer);
        let b = b.fft(&mut buffer);
        a.mul(&b).unwrap();
        let mut mul = a.ifft(&mut buffer);
        mul.reverse();
        mul.swap_halves();
        assert_eq_tol(mul.data(..), conv.data(..), 1e-4);
    }

    #[test]
    fn shift_left_by_1_as_conv() {
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].to_real_time_vec();
        let b = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0].to_real_time_vec();
        let mut buffer = SingleBuffer::new();
        let mut a = a.to_complex().unwrap();
        let b = b.to_complex().unwrap();
        a.convolve_signal(&mut buffer, &b).unwrap();
        let a = a.magnitude();
        let exp = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq_tol(a.data(..), &exp, 1e-4);
    }

    #[test]
    fn shift_left_by_1_as_conv_shorter() {
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].to_real_time_vec();
        let mut buffer = SingleBuffer::new();
        let b = vec![0.0, 0.0, 1.0].to_real_time_vec();
        let mut a = a.to_complex().unwrap();
        let b = b.to_complex().unwrap();
        a.convolve_signal(&mut buffer, &b).unwrap();
        let a = a.magnitude();
        let exp = [9.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq_tol(a.data(..), &exp, 1e-4);
    }

    #[test]
    fn vector_conv_vs_freq_multiplication_pure_real_data_even() {
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].to_real_time_vec();
        let b = vec![15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0].to_real_time_vec();
        let mut buffer = SingleBuffer::new();
        let a = a.to_complex().unwrap();
        let b = b.to_complex().unwrap();
        let mut conv = a.clone();
        conv.convolve_signal(&mut buffer, &b).unwrap();
        let mut a = a.fft(&mut buffer);
        let b = b.fft(&mut buffer);
        a.mul(&b).unwrap();
        let mul = a.ifft(&mut buffer);
        let mut mul = mul.magnitude();
        mul.reverse();
        mul.swap_halves();
        let conv = conv.magnitude();
        assert_eq_tol(mul.data(..), conv.data(..), 1e-4);
    }

    #[test]
    fn vector_conv_vs_freq_multiplication_pure_real_data_odd() {
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].to_real_time_vec();
        let b = vec![15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0].to_real_time_vec();
        let mut buffer = SingleBuffer::new();
        let a = a.to_complex().unwrap();
        let b = b.to_complex().unwrap();
        let mut conv = a.clone();
        conv.convolve_signal(&mut buffer, &b).unwrap();
        let mut a = a.fft(&mut buffer);
        let b = b.fft(&mut buffer);
        a.mul(&b).unwrap();
        let mul = a.ifft(&mut buffer);
        let mut mul = mul.magnitude();
        mul.reverse();
        mul.swap_halves();
        let conv = conv.magnitude();
        assert_eq_tol(mul.data(..), conv.data(..), 1e-4);
    }

    #[test]
    fn overlap_discard_test() {
        let a: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let b: Vec<f32> = vec![0.1, 0.2, 0.3, 0.5, 0.1, 0.2];
        let a = a.to_real_time_vec();
        let b = b.to_real_time_vec();
        let mut buffer = SingleBuffer::new();
        let a = a.to_complex().unwrap();
        let b = b.to_complex().unwrap();
        let mut conv = a.clone();
        conv.convolve_signal(&mut buffer, &b).unwrap();
        let mut overlap_discard = a;
        overlap_discard.overlap_discard(&mut buffer, &b, 0).unwrap();
        assert_eq_tol(overlap_discard.data(..), conv.data(..), 1e-4);
    }

    /// This test triggered an index out of range error in the past
    #[test]
    fn index_out_of_range_error_test() {
        let data: Vec<f32> = vec![0.0; 5000];
        let mut vec = data.clone().to_real_time_vec();
        let mut buffer = SingleBuffer::new();
        let sinc = SincFunction::new();
        vec.convolve(&mut buffer, &sinc as &dyn RealImpulseResponse<f32>, 1.0, 12);
        assert_eq_tol(&data[..], vec.data(..), 1e-4);
    }
}
