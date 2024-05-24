mod freq;
pub use self::freq::*;
mod time;
pub use self::time::*;
mod time_to_freq;
pub use self::time_to_freq::*;
mod freq_to_time;
pub use self::freq_to_time::*;
mod correlation;
pub use self::correlation::*;
mod convolution;
pub use self::convolution::*;
mod interpolation;
pub use self::interpolation::*;
mod real_interpolation;
pub use self::real_interpolation::*;

use super::{
    Buffer, BufferBorrow, Domain, DspVec, ErrorReason, GetMetaData, MetaData, NumberSpace,
    ToSliceMut, Vector, VoidResult,
};
use crate::gpu_support::GpuSupport;
use crate::inline_vector::InlineVector;
use crate::multicore_support::*;
use crate::numbers::*;
use crate::simd_extensions::*;
use crate::{array_to_complex, array_to_complex_mut};
use rustfft::{FftDirection, FftPlanner};
use std::fmt::Debug;
use std::ops::*;

fn fft<S, T, N, D, B>(vec: &mut DspVec<S, T, N, D>, buffer: &mut B, direction: FftDirection)
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
    B: for<'a> Buffer<'a, S, T>,
{
    let len = vec.len();
    let has_no_gpu_support = !T::has_gpu_support();
    let has_small_size = len < 10000;
    let is_unsupported_gpu_fft_len = !T::is_supported_fft_len(true, len);
    let use_rust_fft = has_no_gpu_support || has_small_size || is_unsupported_gpu_fft_len;
    if use_rust_fft {
        let points = len / 2; // By two since vector is always complex
        let fft = {
            let mut planner = FftPlanner::new();
            planner.plan_fft(points, direction)
        };
        let mut temp = buffer.borrow(2 * fft.get_inplace_scratch_len()); // By two since the FFT is done with complex numbers
        let scratch = temp.to_slice_mut();
        let signal = vec.data.to_slice_mut();
        let rbw = (T::from(points).unwrap()) * vec.delta;
        vec.delta = rbw;
        let signal = array_to_complex_mut(&mut signal[0..len]);
        let scratch = array_to_complex_mut(scratch);
        fft.process_with_scratch(signal, scratch);
    } else {
        let signal = vec.data.to_slice_mut();
        T::fft(true, &mut signal[0..len], direction);
    }
}

/// Transform a value on the x-axis the same way as a fft shift transforms the
/// x axis of a spectrum.
fn fft_swap_x<T: RealNumber>(is_fft_shifted: bool, x_value: T, x_max: T) -> T {
    if !is_fft_shifted {
        x_value / x_max
    } else if x_value <= T::zero() {
        let x_value = x_value / x_max;
        T::one() + x_value
    } else {
        let x_value = x_max - x_value + T::one();
        -x_value / x_max
    }
}

/// Creates shifted and reversed copies of the given data vector.
/// This function is especially designed for convolutions.
fn create_shifted_copies<O, T: RealNumber, Reg: SimdGeneric<T>>(
    vec: &O,
) -> InlineVector<InlineVector<Reg>>
where
    O: Vector<T>,
{
    let step = if vec.is_complex() { 2 } else { 1 };
    let number_of_shifts = Reg::LEN / step;
    let mut shifted_copies = InlineVector::with_capacity(number_of_shifts);
    let mut i = 0;
    let data = vec.data(..);
    while i < number_of_shifts {
        let mut data = data.iter().rev();

        // In general (number_of_shifts - i) indicates which prepared vector we need to use
        // if we later calculate end % number_of_shifts. Some examples:
        // number_of_shifts: 4, end: 13 -> mod: 1. The code will round end to the next
        // SIMD register
        // which ends at 16. In order to get back to 13 we therefore have to ignore 3 numbers.
        // Ignoring is done by shifting and inserting zeros. So in this example the correct
        // shift is 3
        // which equals number_of_shifts(4) - mod(1).
        // Now mod: 0 is a special case. This is because if we round up to the next SIMD
        // register then
        // we still don't need to add any offset and so for the case 0, 0 is the right shift.
        let shift = ((number_of_shifts - i) % number_of_shifts) * step;
        let min_len = vec.len() + shift;
        let len = (min_len + Reg::LEN - 1) / Reg::LEN;
        let mut copy: InlineVector<Reg> = InlineVector::with_capacity(len);

        let mut j = len * Reg::LEN;
        let mut k = 0;
        let mut current = InlineVector::of_size(T::zero(), Reg::LEN);
        while j > 0 {
            j -= step;
            if j < shift || j >= min_len {
                // Insert zeros
                current[k] = T::zero();
                k += 1;
                if k >= current.len() {
                    copy.push(Reg::load(&current[..], 0));
                    k = 0;
                }
                if step > 1 {
                    current[k] = T::zero();
                    k += 1;
                    if k >= current.len() {
                        copy.push(Reg::load(&current[..], 0));
                        k = 0;
                    }
                }
            } else if step > 1 {
                // Push complex number into vector
                let im = *data.next().unwrap();
                let re = *data.next().unwrap();
                current[k] = re;
                k += 1;
                if k >= current.len() {
                    copy.push(Reg::load(&current[..], 0));
                    k = 0;
                }
                current[k] = im;
                k += 1;
                if k >= current.len() {
                    copy.push(Reg::load(&current[..], 0));
                    k = 0;
                }
            } else {
                // Push real number into vector
                current[k] = *data.next().unwrap();
                k += 1;
                if k >= current.len() {
                    copy.push(Reg::load(&current[..], 0));
                    k = 0;
                }
            }
        }

        assert_eq!(k, 0);
        assert_eq!(copy.len(), len);
        shifted_copies.push(copy);
        i += 1;
    }
    shifted_copies
}

impl<S, T, N, D> DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn convolve_function_priv<B, TT, C, CMut, F>(
        &mut self,
        buffer: &mut B,
        ratio: T,
        conv_len: usize,
        convert: C,
        convert_mut: CMut,
        fun: F,
    ) where
        B: for<'a> Buffer<'a, S, T>,
        C: Fn(&[T]) -> &[TT],
        CMut: Fn(&mut [T]) -> &mut [TT],
        F: Fn(T) -> TT,
        TT: Zero + Mul<Output = TT> + Copy + Add<Output = TT>,
    {
        let len = self.len();
        let mut temp = buffer.borrow(len);
        {
            let data = self.data.to_slice();
            let temp = temp.to_slice_mut();
            let complex = convert(&data[0..len]);
            let dest = convert_mut(&mut temp[0..len]);
            let len = complex.len();
            let conv_len = if conv_len > len { len } else { conv_len };
            let sconv_len = conv_len as isize;
            for (i, num) in dest.iter_mut().enumerate() {
                let iter =
                    WrappingIterator::new(complex, i as isize - sconv_len - 1, 2 * conv_len + 1);
                let mut sum = TT::zero();
                let mut j = -(T::from(conv_len).unwrap());
                for c in iter {
                    sum = sum + c * fun(-j * ratio);
                    j = j + T::one();
                }
                *num = sum;
            }
        }

        temp.trade(&mut self.data);
    }

    /// Only calculate the convolution in the inverse range from the given range.
    /// This is intended to be used together with convolution implementations which
    /// are faster but don't handle the beginning and end of a vector correctly.
    fn convolve_vector_range<O, NO, DO>(
        &mut self,
        target: &mut [T],
        vector: &O,
        range: Range<usize>,
    ) where
        O: Vector<T> + GetMetaData<T, NO, DO>,
        NO: NumberSpace,
        DO: Domain,
    {
        let points = self.points();
        let other_points = vector.points();
        let (other_start, other_end, full_conv_len, conv_len) = if other_points > points {
            let center = other_points / 2;
            let conv_len = points / 2;
            (center - conv_len, center + conv_len, points, conv_len)
        } else {
            (
                0,
                other_points,
                other_points,
                other_points - other_points / 2,
            )
        };
        let len = self.len();
        if self.is_complex() {
            let other = vector.data(..);
            let data = self.data.to_slice();
            let other = array_to_complex(&other[0..vector.len()]);
            let complex = array_to_complex(&data[0..len]);
            let dest = array_to_complex_mut(&mut target[0..len]);
            let other_iter = &other[other_start..other_end];
            let conv_len = conv_len as isize;
            for (i, dest) in dest.iter_mut().enumerate().take(range.start / 2) {
                *dest = Self::convolve_iteration(
                    complex,
                    other_iter,
                    i as isize,
                    conv_len,
                    full_conv_len,
                );
            }
        } else {
            let other = vector.data(..);
            let data = self.data.to_slice();
            let other = &other[0..vector.len()];
            let data = &data[0..len];
            let dest = &mut target[0..len];
            let other_iter = &other[other_start..other_end];
            let conv_len = conv_len as isize;
            for (i, dest) in dest.iter_mut().enumerate().take(range.start) {
                *dest =
                    Self::convolve_iteration(data, other_iter, i as isize, conv_len, full_conv_len);
            }
        }
    }

    fn convolve_signal_scalar<B, O, NO, DO>(&mut self, buffer: &mut B, vector: &O)
    where
        B: for<'a> Buffer<'a, S, T>,
        O: Vector<T> + GetMetaData<T, NO, DO>,
        NO: NumberSpace,
        DO: Domain,
    {
        let points = self.points();
        let other_points = vector.points();
        let (other_start, other_end, full_conv_len, conv_len) = if other_points > points {
            let center = other_points / 2;
            let conv_len = points / 2;
            (center - conv_len, center + conv_len, points, conv_len)
        } else {
            (
                0,
                other_points,
                other_points,
                other_points - other_points / 2,
            )
        };
        if self.is_complex() {
            let len = self.len();
            let mut temp = buffer.borrow(len);
            {
                let other = vector.data(..);
                let data = self.data.to_slice();
                let temp = temp.to_slice_mut();
                let other = array_to_complex(&other[0..vector.len()]);
                let complex = array_to_complex(&data[0..len]);
                let dest = array_to_complex_mut(&mut temp[0..len]);
                let other_iter = &other[other_start..other_end];
                let conv_len = conv_len as isize;
                Chunk::execute_with_range(
                    Complexity::Large,
                    &self.multicore_settings,
                    dest,
                    1,
                    (complex, other_iter),
                    move |dest, range, (complex, other_iter)| {
                        for (num, i) in dest.iter_mut().zip(range) {
                            *num = Self::convolve_iteration(
                                complex,
                                other_iter,
                                i as isize,
                                conv_len,
                                full_conv_len,
                            );
                        }
                    },
                );
            }
            temp.trade(&mut self.data);
        } else {
            let len = self.len();
            let mut temp = buffer.borrow(len);
            {
                let other = vector.data(..);
                let data = self.data.to_slice();
                let temp = temp.to_slice_mut();
                let other = &other[0..vector.len()];
                let data = &data[0..len];
                let dest = &mut temp[0..len];
                let other_iter = &other[other_start..other_end];
                let conv_len = conv_len as isize;
                Chunk::execute_with_range(
                    Complexity::Large,
                    &self.multicore_settings,
                    dest,
                    1,
                    (data, other_iter),
                    move |dest, range, (data, other_iter)| {
                        for (num, i) in dest.iter_mut().zip(range) {
                            *num = Self::convolve_iteration(
                                data,
                                other_iter,
                                i as isize,
                                conv_len,
                                full_conv_len,
                            );
                        }
                    },
                );
            }
            temp.trade(&mut self.data);
        }
    }

    /// Convolves a vector of vectors (in this lib also considered a matrix) with a vector
    /// of impulse responses and stores the result in `target`.
    pub fn convolve_mat(
        matrix: &[&Self],
        impulse_response: &[&Self],
        target: &mut [T],
    ) -> VoidResult {
        // Since this function mainly exists to be used by the matrix lib
        // we just decide to ignore invalid calls.
        if impulse_response.len() != matrix.len() || impulse_response.is_empty() {
            return Err(ErrorReason::InvalidArgumentLength);
        }

        let expected_len = matrix[0].len();
        for v in &matrix[..] {
            if v.len() != expected_len {
                return Err(ErrorReason::InvalidArgumentLength);
            }
        }

        let expected_len = impulse_response[0].len();
        for v in &impulse_response[..] {
            if v.len() != expected_len {
                return Err(ErrorReason::InvalidArgumentLength);
            }
        }

        let points = matrix[0].points();
        let other_points = impulse_response[0].points();
        let (other_start, other_end, full_conv_len, conv_len) = if other_points > points {
            let center = other_points / 2;
            let conv_len = points / 2;
            (center - conv_len, center + conv_len, points, conv_len)
        } else {
            (
                0,
                other_points,
                other_points,
                other_points - other_points / 2,
            )
        };
        if matrix[0].is_complex() {
            let len = matrix[0].len();
            let others: InlineVector<&[T]> =
                impulse_response.iter().map(|v| v.data.to_slice()).collect();
            let data_vecs: InlineVector<&[T]> = matrix.iter().map(|v| v.data.to_slice()).collect();
            let others: InlineVector<&[Complex<T>]> = others
                .iter()
                .map(|o| {
                    let c = array_to_complex(&o[0..impulse_response[0].len()]);
                    &c[other_start..other_end]
                })
                .collect();
            let data_vecs: InlineVector<&[Complex<T>]> = data_vecs
                .iter()
                .map(|o| array_to_complex(&o[0..impulse_response[0].len()]))
                .collect();
            let dest = array_to_complex_mut(&mut target[0..len]);
            let conv_len = conv_len as isize;
            for (num, i) in dest.iter_mut().zip(0..) {
                *num = Self::convolve_mat_iteration(
                    &data_vecs[..],
                    &others[..],
                    i,
                    conv_len,
                    full_conv_len,
                );
            }
        } else {
            let len = matrix[0].len();
            let others: InlineVector<&[T]> =
                impulse_response.iter().map(|v| v.data.to_slice()).collect();
            let data_vecs: InlineVector<&[T]> = matrix.iter().map(|v| v.data.to_slice()).collect();
            let others: InlineVector<&[T]> =
                others.iter().map(|o| &o[other_start..other_end]).collect();
            let data_vecs: InlineVector<&[T]> = data_vecs.iter().map(|o| &o[0..len]).collect();
            let dest = &mut target[0..len];
            let conv_len = conv_len as isize;
            for (num, i) in dest.iter_mut().zip(0..) {
                *num = Self::convolve_mat_iteration(
                    &data_vecs[..],
                    &others[..],
                    i,
                    conv_len,
                    full_conv_len,
                );
            }
        }

        Ok(())
    }

    #[inline]
    fn convolve_iteration<TT>(
        data: &[TT],
        other_iter: &[TT],
        i: isize,
        conv_len: isize,
        full_conv_len: usize,
    ) -> TT
    where
        TT: Zero + Clone + Copy + Add<Output = TT> + Mul<Output = TT>,
    {
        let data_iter = ReverseWrappingIterator::new(data, i + conv_len, full_conv_len);
        let mut sum = TT::zero();
        let iteration = data_iter.zip(other_iter);
        for (this, other) in iteration {
            sum = sum + this * (*other);
        }
        sum
    }

    #[inline]
    fn convolve_mat_iteration<TT>(
        matrix: &[&[TT]],
        imp_resp: &[&[TT]],
        i: isize,
        conv_len: isize,
        full_conv_len: usize,
    ) -> TT
    where
        TT: Zero + Clone + Copy + Add<Output = TT> + Mul<Output = TT> + Debug,
    {
        let mut sum = TT::zero();
        for (data, other_iter) in matrix.iter().zip(imp_resp.iter()) {
            let data_iter = ReverseWrappingIterator::new(data, i + conv_len, full_conv_len);
            let iteration = data_iter.zip(*other_iter);
            for (this, other) in iteration {
                sum = sum + this * (*other);
            }
        }

        sum
    }

    fn convolve_signal_simd<Reg: SimdGeneric<T>, B, O>(
        &mut self,
        _: RegType<Reg>,
        buffer: &mut B,
        vector: &O,
    ) where
        B: for<'a> Buffer<'a, S, T>,
        O: Vector<T>,
    {
        if self.is_complex() {
            self.convolve_signal_simd_impl::<Reg, _, _, _, _, _, _, _>(
                buffer,
                vector,
                |x| array_to_complex(x),
                |x| array_to_complex_mut(x),
                |x, y| x.mul_complex(y),
                |x| x.sum_complex(),
            )
        } else {
            self.convolve_signal_simd_impl::<Reg, _, _, _, _, _, _, _>(
                buffer,
                vector,
                |x| x,
                |x| x,
                |x, y| x * y,
                |x| x.sum_real(),
            )
        }
    }

    /// SIMD optimizatino to convolve two vectors.
    /// Most of the parameters (convert, convert_mut, simd_mul, simd_sum) are used to have
    /// a single function for both real valued and complex valued vectors.
    fn convolve_signal_simd_impl<Reg, B, TT, O, C, CMut, RMul, RSum>(
        &mut self,
        buffer: &mut B,
        vector: &O,
        convert: C,
        convert_mut: CMut,
        simd_mul: RMul,
        simd_sum: RSum,
    ) where
        B: for<'a> Buffer<'a, S, T>,
        O: Vector<T>,
        TT: Zero + Clone + Copy + Add<Output = TT> + Mul<Output = TT> + Send + Sync,
        C: Fn(&[T]) -> &[TT],
        CMut: Fn(&mut [T]) -> &mut [TT],
        RMul: Fn(Reg, Reg) -> Reg + Sync,
        RSum: Fn(Reg) -> TT + Sync,
        Reg: SimdGeneric<T>,
    {
        let points = self.points();
        let other_points = vector.points();
        assert!(other_points < points);
        let (full_conv_len, conv_len) = (other_points, other_points - other_points / 2);
        let len = self.len();
        let mut temp = buffer.borrow(len);
        {
            let other = vector.data(..);
            let data = self.data.to_slice();
            let temp = temp.to_slice_mut();
            let points = self.points();
            let other = convert(&other[0..vector.len()]);
            let complex = convert(&data[0..len]);
            let dest = convert_mut(&mut temp[0..len]);
            let other_iter = &other[0..other_points];

            let shifts = create_shifted_copies(vector);

            // The next lines uses + $reg::LEN due to rounding of odd numbers
            let scalar_len = conv_len + Reg::LEN;
            let conv_len = conv_len as isize;

            let partition = Reg::calc_data_alignment_reqs(&data[0..len]);
            let step = if self.is_complex() { 2 } else { 1 };
            let scalar_left_points = partition.left / step;
            let simd = Reg::array_to_regs(partition.center(data));
            Chunk::execute_with_range(
                Complexity::Large,
                &self.multicore_settings,
                &mut dest[scalar_len..points - scalar_len],
                1,
                simd,
                move |dest_range, range, simd| {
                    let mut i = (scalar_len + range.start) as isize;
                    for num in dest_range {
                        let end = (i + conv_len) as usize - scalar_left_points;
                        let shift = end % shifts.len();
                        let end = (end + shifts.len() - 1) / shifts.len();
                        let mut sum = Reg::splat(T::zero());
                        let shifted = &shifts[shift];
                        let simd_iter = simd[end - shifted.len()..end].iter();
                        let iteration = simd_iter.zip(shifted.iter());
                        for (this, other) in iteration {
                            sum = sum + simd_mul(*this, *other);
                        }
                        (*num) = simd_sum(sum);
                        i += 1;
                    }
                },
            );
            for (i, num) in IndexedEdgeIteratorMut::new(dest, scalar_len, scalar_len) {
                *num = Self::convolve_iteration(
                    complex,
                    other_iter,
                    i as isize,
                    conv_len,
                    full_conv_len,
                );
            }
        }
        temp.trade(&mut self.data);
    }

    fn multiply_function_priv<TT, CMut, FA, F>(
        &mut self,
        is_symmetric: bool,
        is_fft_shifted: bool,
        ratio: T,
        convert_mut: CMut,
        function_arg: FA,
        fun: F,
    ) where
        CMut: Fn(&mut [T]) -> &mut [TT],
        FA: Copy + Sync + Send,
        F: Fn(FA, T) -> TT + 'static + Sync,
        TT: Zero + Mul<Output = TT> + Copy + Send + Sync + From<T>,
    {
        let two = T::one() + T::one();
        if !is_symmetric {
            let len = self.len();
            let points = self.points();
            let data = self.data.to_slice_mut();
            let converted = convert_mut(&mut data[0..len]);
            Chunk::execute_with_range(
                Complexity::Medium,
                &self.multicore_settings,
                converted,
                1,
                (ratio, function_arg),
                move |array, range, (ratio, arg)| {
                    let scale = TT::from(ratio);
                    let offset = if points % 2 != 0 { 1 } else { 0 };
                    let max = T::from(points - offset).unwrap() / two;
                    let mut j =
                        -(T::from(points - offset).unwrap()) / two + T::from(range.start).unwrap();
                    for num in array {
                        *num =
                            (*num) * scale * fun(arg, fft_swap_x(is_fft_shifted, j, max) * ratio);
                        j = j + T::one();
                    }
                },
            );
        } else {
            let len = self.len();
            let data = self.data.to_slice_mut();
            let converted = convert_mut(&mut data[0..len]);
            let points = converted.len();
            Chunk::execute_sym_pairs_with_range(
                Complexity::Medium,
                &self.multicore_settings,
                converted,
                1,
                (ratio, function_arg),
                move |array1, range1, array2, range2, (ratio, arg)| {
                    let two = T::from(2.0).unwrap();
                    assert!(array1.len() >= array2.len());
                    assert!(range1.end <= range2.start);
                    let scale = TT::from(ratio);
                    let len1 = array1.len();
                    let len2 = array2.len();
                    let offset = points % 2;
                    let max = T::from(points - offset).unwrap() / two;
                    let center = T::from(points - offset).unwrap() / two;
                    let mut j1 = -center + T::from(range1.start).unwrap();
                    let mut j2 = center - T::from(range2.end - 1).unwrap();
                    let mut i1 = 0;
                    let mut i2 = 0;
                    {
                        let mut iter1 = array1.iter_mut();
                        let mut iter2 = array2.iter_mut().rev();
                        while j1 < j2 {
                            let num = iter1.next().unwrap();
                            (*num) = (*num)
                                * scale
                                * fun(arg, fft_swap_x(is_fft_shifted, j1, max) * ratio);
                            j1 = j1 + T::one();
                            i1 += 1;
                        }
                        while j2 < j1 {
                            let num = iter2.next().unwrap();
                            (*num) = (*num)
                                * scale
                                * fun(arg, fft_swap_x(is_fft_shifted, j2, max) * ratio);
                            j2 = j2 + T::one();
                            i2 += 1;
                        }
                        // At this point we can be sure that `j1 == j2`
                        for (num1, num2) in iter1.zip(iter2) {
                            let arg = scale * fun(arg, fft_swap_x(is_fft_shifted, j1, max) * ratio);
                            *num1 = (*num1) * arg;
                            *num2 = (*num2) * arg;
                            j1 = j1 + T::one();
                        }
                        j2 = j1;
                    }
                    // Now we have to deal with differences in length
                    // `common_length` is the number of iterations we spent
                    // in the previous loop.
                    let pos1 = len1 - i1;
                    let pos2 = len2 - i2;
                    let common_length = pos1.min(pos2);
                    for num in &mut array1[i1 + common_length..len1] {
                        (*num) =
                            (*num) * scale * fun(arg, fft_swap_x(is_fft_shifted, j1, max) * ratio);
                        j1 = j1 + T::one();
                    }
                    for num in &mut array2[0..len2 - common_length - i2] {
                        (*num) =
                            (*num) * scale * fun(arg, fft_swap_x(is_fft_shifted, j2, max) * ratio);
                        j2 = j2 + T::one();
                    }
                },
            );
        }
    }
}

struct WrappingIterator<T>
where
    T: Clone,
{
    start: *const T,
    end: *const T,
    pos: *const T,
    count: usize,
}

impl<T> Iterator for WrappingIterator<T>
where
    T: Clone,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        unsafe {
            if self.count == 0 {
                return None;
            }

            let mut n = self.pos;
            if n < self.end {
                n = n.offset(1);
            } else {
                n = self.start;
            }

            self.pos = n;
            self.count -= 1;
            Some((*n).clone())
        }
    }
}

impl<T> WrappingIterator<T>
where
    T: Clone,
{
    pub fn new(slice: &[T], pos: isize, iter_len: usize) -> Self {
        use std::isize;

        assert!(slice.len() <= isize::MAX as usize);
        let len = slice.len() as isize;
        let mut pos = pos % len;
        while pos < 0 {
            pos += len;
        }

        let start = slice.as_ptr();
        unsafe {
            WrappingIterator {
                start,
                end: start.offset(len - 1),
                pos: start.offset(pos),
                count: iter_len,
            }
        }
    }
}

struct ReverseWrappingIterator<T>
where
    T: Clone,
{
    start: *const T,
    end: *const T,
    pos: *const T,
    count: usize,
}

impl<T> Iterator for ReverseWrappingIterator<T>
where
    T: Clone,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        unsafe {
            if self.count == 0 {
                return None;
            }

            let mut n = self.pos;
            if n > self.start {
                n = n.offset(-1);
            } else {
                n = self.end;
            }

            self.pos = n;
            self.count -= 1;
            Some((*n).clone())
        }
    }
}

impl<T> ReverseWrappingIterator<T>
where
    T: Clone,
{
    pub fn new(slice: &[T], pos: isize, iter_len: usize) -> Self {
        use std::isize;

        assert!(slice.len() <= isize::MAX as usize);
        let len = slice.len() as isize;
        let mut pos = pos % len;
        while pos < 0 {
            pos += len;
        }

        let start = slice.as_ptr();
        unsafe {
            ReverseWrappingIterator {
                start,
                end: start.offset(len - 1),
                pos: start.offset(pos),
                count: iter_len,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fft_swap_x;

    #[test]
    fn fft_swap_x_test() {
        let input = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let expected = [0.0, 0.25, 0.5, 0.75, 1.0, -1.0, -0.75, -0.5, -0.25];
        let mut actual = [0.0; 9];
        for i in 0..actual.len() {
            actual[i] = fft_swap_x(true, input[i], 4.0);
        }
        assert_eq!(&actual, &expected);
    }
}
