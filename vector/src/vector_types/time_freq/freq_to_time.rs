use super::super::{
    Buffer, ComplexNumberSpace, DataDomain, DspVec, ErrorReason, FloatIndex, FrequencyDomain,
    FrequencyDomainOperations, InsertZerosOpsBuffered, MetaData, RededicateForceOps, ScaleOps,
    TimeDomainOperations, ToRealTimeResult, ToSliceMut, ToTimeResult, TransRes, Vector,
};
use super::fft;
use crate::multicore_support::*;
use crate::numbers::*;
use crate::window_functions::*;

/// Defines all operations which are valid on `DataVecs` containing frequency domain data.
/// # Failures
/// All operations in this trait set `self.len()` to `0`
/// if the vector isn't in frequency domain and complex number space.
pub trait FrequencyToTimeDomainOperations<S, T>: ToTimeResult
where
    S: ToSliceMut<T>,
    T: RealNumber,
{
    /// Performs an Inverse Fast Fourier Transformation transforming a frequency domain vector
    /// into a time domain vector.
    ///
    /// This version of the IFFT neither applies a window nor does it scale the
    /// vector.
    /// # Example
    ///
    /// ```
    /// use std::f32;
    /// use basic_dsp_vector::*;
    /// # use num_complex::Complex;
    /// let vector = vec!(Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)).to_complex_freq_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.plain_ifft(&mut buffer);
    /// let actual = &result[..];
    /// let expected = &[Complex::new(1.0, 0.0), Complex::new(-0.5, 0.8660254), Complex::new(-0.5, -0.8660254)];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!((actual[i] - expected[i]).norm() < 1e-4);
    /// }
    /// ```
    fn plain_ifft<B>(self, buffer: &mut B) -> Self::TimeResult
    where
        B: for<'a> Buffer<'a, S, T>;

    /// Performs an Inverse Fast Fourier Transformation transforming a frequency domain vector
    /// into a time domain vector.
    /// # Example
    ///
    /// ```
    /// use std::f32;
    /// use basic_dsp_vector::*;
    /// # use num_complex::Complex;
    /// let vector = vec!(Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(3.0, 0.0)).to_complex_freq_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.ifft(&mut buffer);
    /// let actual = &result[..];
    /// let expected = &[Complex::new(1.0, 0.0), Complex::new(-0.5, 0.8660254), Complex::new(-0.5, -0.8660254)];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!((actual[i] - expected[i]).norm() < 1e-4);
    /// }
    /// ```
    fn ifft<B>(self, buffer: &mut B) -> Self::TimeResult
    where
        B: for<'a> Buffer<'a, S, T>;

    /// Performs an Inverse Fast Fourier Transformation transforming a frequency domain vector
    /// into a time domain vector and removes the FFT window.
    fn windowed_ifft<B>(self, buffer: &mut B, window: &WindowFunction<T>) -> Self::TimeResult
    where
        B: for<'a> Buffer<'a, S, T>;
}

/// Defines all operations which are valid on `DataVecs` containing frequency domain data and
/// the data is assumed to half of complex conjugate symmetric spectrum round 0 Hz where
/// the 0 Hz element itself is real.
/// # Failures
/// All operations in this trait set `self.len()` to `0` if the first element (0Hz)
/// isn't real.
pub trait SymmetricFrequencyToTimeDomainOperations<S, T>: ToRealTimeResult
where
    S: ToSliceMut<T>,
    T: RealNumber,
{
    /// Performs a Symmetric Inverse Fast Fourier Transformation under the assumption that `self`
    /// contains half of a symmetric spectrum starting from 0 Hz. This assumption
    /// isn't verified and no error is raised if the spectrum isn't symmetric. The reason
    /// for this is that there is no robust verification possible.
    ///
    /// The argument indicates whether the resulting real vector should have `2*N`
    /// or `2*N-1` points.
    ///
    /// This version of the IFFT neither applies a window nor does it scale the
    /// vector.
    fn plain_sifft<B>(self, buffer: &mut B) -> TransRes<Self::RealTimeResult>
    where
        B: for<'a> Buffer<'a, S, T>;

    /// Performs a Symmetric Inverse Fast Fourier Transformation under the assumption that `self`
    /// contains half of a symmetric spectrum starting from 0 Hz. This assumption
    /// isn't verified and no error is raised if the spectrum isn't symmetric. The reason
    /// for this is that there is no robust verification possible.
    ///
    /// The argument indicates whether the resulting real vector should have `2*N` or
    /// `2*N-1` points.
    fn sifft<B>(self, buffer: &mut B) -> TransRes<Self::RealTimeResult>
    where
        B: for<'a> Buffer<'a, S, T>;

    /// Performs a Symmetric Inverse Fast Fourier Transformation (SIFFT) and removes the FFT
    /// window. The SIFFT is performed under the assumption that `self`
    /// contains half of a symmetric spectrum starting from 0 Hz. This assumption
    /// isn't verified and no error is raised if the spectrum isn't symmetric. The reason
    /// for this is that there is no robust verification possible.
    ///
    /// The argument indicates whether the resulting real vector should have `2*N` or `2*N-1`
    /// points.
    fn windowed_sifft<B>(
        self,
        buffer: &mut B,
        window: &WindowFunction<T>,
    ) -> TransRes<Self::RealTimeResult>
    where
        B: for<'a> Buffer<'a, S, T>;
}

impl<S, T, N, D> FrequencyToTimeDomainOperations<S, T> for DspVec<S, T, N, D>
where
    DspVec<S, T, N, D>: ToTimeResult,
    <DspVec<S, T, N, D> as ToTimeResult>::TimeResult:
        RededicateForceOps<DspVec<S, T, N, D>> + TimeDomainOperations<S, T>,
    S: ToSliceMut<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: FrequencyDomain,
{
    fn plain_ifft<B>(mut self, buffer: &mut B) -> Self::TimeResult
    where
        B: for<'a> Buffer<'a, S, T>,
    {
        if self.domain() != DataDomain::Frequency {
            self.mark_vector_as_invalid();
            self.number_space.to_complex();
            self.domain.to_freq();
            return Self::TimeResult::rededicate_from_force(self);
        }

        if !self.is_complex() {
            self.zero_interleave_b(buffer, 2);
            self.number_space.to_complex();
        }

        fft(&mut self, buffer, true);

        self.domain.to_freq();
        Self::TimeResult::rededicate_from_force(self)
    }

    fn ifft<B>(mut self, buffer: &mut B) -> Self::TimeResult
    where
        B: for<'a> Buffer<'a, S, T>,
    {
        let points = self.points();
        self.scale(T::one() / T::from(points).unwrap());
        self.ifft_shift();
        self.plain_ifft(buffer)
    }

    fn windowed_ifft<B>(self, buffer: &mut B, window: &WindowFunction<T>) -> Self::TimeResult
    where
        B: for<'a> Buffer<'a, S, T>,
    {
        let mut result = self.ifft(buffer);
        result.unapply_window(window);
        result
    }
}

impl<S, T, N, D> SymmetricFrequencyToTimeDomainOperations<S, T> for DspVec<S, T, N, D>
where
    DspVec<S, T, N, D>: ToRealTimeResult + ToTimeResult + FrequencyDomainOperations<S, T>,
    <DspVec<S, T, N, D> as ToRealTimeResult>::RealTimeResult:
        RededicateForceOps<DspVec<S, T, N, D>> + TimeDomainOperations<S, T>,
    S: ToSliceMut<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: FrequencyDomain,
{
    fn plain_sifft<B>(mut self, buffer: &mut B) -> TransRes<Self::RealTimeResult>
    where
        B: for<'a> Buffer<'a, S, T>,
    {
        if self.domain() != DataDomain::Frequency || !self.is_complex() {
            self.mark_vector_as_invalid();
            self.number_space.to_complex();
            self.domain.to_freq();
            return Err((
                ErrorReason::InputMustBeInFrequencyDomain,
                Self::RealTimeResult::rededicate_from_force(self),
            ));
        }

        if self.points() > 0 && self.data(1).abs() > T::from(1e-10).unwrap() {
            self.mark_vector_as_invalid();
            self.number_space.to_complex();
            self.domain.to_freq();
            return Err((
                ErrorReason::InputMustBeConjSymmetric,
                Self::RealTimeResult::rededicate_from_force(self),
            ));
        }

        self.mirror(buffer);

        fft(&mut self, buffer, true);

        self.domain.to_freq();
        self.pure_complex_to_real_operation(buffer, |x, _arg| x.re, (), Complexity::Small);
        Ok(Self::RealTimeResult::rededicate_from_force(self))
    }

    fn sifft<B>(mut self, buffer: &mut B) -> TransRes<Self::RealTimeResult>
    where
        B: for<'a> Buffer<'a, S, T>,
    {
        let points = self.points();
        self.scale(Complex::<T>::new(
            T::one() / T::from(points).unwrap(),
            T::zero(),
        ));
        self.ifft_shift();
        self.plain_sifft(buffer)
    }

    fn windowed_sifft<B>(
        self,
        buffer: &mut B,
        window: &WindowFunction<T>,
    ) -> TransRes<Self::RealTimeResult>
    where
        B: for<'a> Buffer<'a, S, T>,
    {
        let mut result = self.sifft(buffer)?;
        result.unapply_window(window);
        Ok(result)
    }
}
