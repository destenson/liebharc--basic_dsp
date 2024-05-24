use super::{Simd, SimdFrom};
use crate::numbers::*;
use std::arch::x86_64::*;
use std::mem;
use std::simd::num::{SimdFloat, SimdInt};
use std::simd::StdFloat;
pub use std::simd::{f32x4, f64x2};
use std::simd::{i32x4, i64x2};

/// This value must be read in groups of 2 bits:
/// 10 means that the third position (since it's the third bit pair)
/// will be replaced with the value of the second position (10b = 2d)
const SWAP_IQ_PS: i32 = 0b1011_0001;

impl Simd<f32> for f32x4 {
    type Array = [f32; 4];

    #[inline]
    fn to_array(self) -> Self::Array {
        let mut target = [0.0; 4];
        self.copy_to_slice(&mut target);
        target
    }

    type ComplexArray = [Complex<f32>; 2];

    const LEN: usize = 4;

    #[inline]
    fn from_complex(value: Complex<f32>) -> f32x4 {
        f32x4::from_array([value.re, value.im, value.re, value.im])
    }

    #[inline]
    fn add_real(self, value: f32) -> f32x4 {
        let increment = f32x4::splat(value);
        self + increment
    }

    #[inline]
    fn add_complex(self, value: Complex<f32>) -> f32x4 {
        let increment = f32x4::from_array([value.re, value.im, value.re, value.im]);
        self + increment
    }

    #[inline]
    fn scale_real(self, value: f32) -> f32x4 {
        let scale_vector = f32x4::splat(value);
        self * scale_vector
    }

    #[inline]
    fn scale_complex(self, value: Complex<f32>) -> f32x4 {
        let scaling_real = f32x4::splat(value.re);
        let scaling_imag = f32x4::splat(value.im);
        let parallel = scaling_real * self;
        let shuffled = self.swap_iq();
        let cross = scaling_imag * shuffled;
        unsafe {
            mem::transmute(_mm_addsub_ps(
                mem::transmute(parallel),
                mem::transmute(cross),
            ))
        }
    }

    #[inline]
    fn mul_complex(self, value: f32x4) -> f32x4 {
        let values = value.as_array();
        let scaling_real = f32x4::from_array([values[0], values[0], values[2], values[2]]);
        let scaling_imag = f32x4::from_array([values[1], values[1], values[3], values[3]]);

        let parallel = scaling_real * self;
        let shuffled = self.swap_iq();
        let cross = scaling_imag * shuffled;
        unsafe {
            mem::transmute(_mm_addsub_ps(
                mem::transmute(parallel),
                mem::transmute(cross),
            ))
        }
    }

    #[inline]
    fn div_complex(self, value: f32x4) -> f32x4 {
        let values = self.as_array();
        let scaling_imag = f32x4::from_array([values[0], values[0], values[2], values[2]]);
        let scaling_real = f32x4::from_array([values[1], values[1], values[3], values[3]]);

        let parallel = scaling_real * value;
        let shuffled = value.swap_iq();
        let cross = scaling_imag * shuffled;
        let mul: f32x4 = unsafe {
            mem::transmute(_mm_addsub_ps(
                mem::transmute(parallel),
                mem::transmute(cross),
            ))
        };
        let square = shuffled * shuffled;
        let square_shuffled = square.swap_iq();
        let sum = square + square_shuffled;
        let div = mul / sum;
        div.swap_iq()
    }

    #[inline]
    fn complex_abs_squared(self) -> f32x4 {
        let squared = self * self;
        unsafe {
            mem::transmute(_mm_hadd_ps(
                mem::transmute(squared),
                mem::transmute(squared),
            ))
        }
    }

    #[inline]
    fn complex_abs(self) -> f32x4 {
        let squared_sum = self.complex_abs_squared();
        StdFloat::sqrt(squared_sum)
    }

    #[inline]
    fn sqrt(self) -> f32x4 {
        StdFloat::sqrt(self)
    }
    #[inline]
    fn store_half(self, target: &mut [f32], index: usize) {
        let values = self.as_array();
        target[index] = values[0];
        target[index + 1] = values[1];
    }

    #[inline]
    fn sum_real(&self) -> f32 {
        let values = self.as_array();
        values[0] + values[1] + values[2] + values[3]
    }

    #[inline]
    fn sum_complex(&self) -> Complex<f32> {
        let values = self.as_array();
        Complex::<f32>::new(values[0] + values[1], values[2] + values[3])
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        self.simd_max(other)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        self.simd_min(other)
    }

    #[inline]
    fn swap_iq(self) -> Self {
        unsafe { mem::transmute(_mm_permute_ps(mem::transmute(self), SWAP_IQ_PS)) }
    }
}

impl Simd<f64> for f64x2 {
    type Array = [f64; 2];

    #[inline]
    fn to_array(self) -> Self::Array {
        let mut target = [0.0; 2];
        self.copy_to_slice(&mut target);
        target
    }

    type ComplexArray = [Complex<f64>; 1];

    const LEN: usize = 2;

    #[inline]
    fn from_complex(value: Complex<f64>) -> f64x2 {
        f64x2::from_array([value.re, value.im])
    }

    #[inline]
    fn add_real(self, value: f64) -> f64x2 {
        let increment = f64x2::splat(value);
        self + increment
    }

    #[inline]
    fn add_complex(self, value: Complex<f64>) -> f64x2 {
        let increment = f64x2::from_array([value.re, value.im]);
        self + increment
    }

    #[inline]
    fn scale_real(self, value: f64) -> f64x2 {
        let scale_vector = f64x2::splat(value);
        self * scale_vector
    }
    #[inline]
    fn scale_complex(self, value: Complex<f64>) -> f64x2 {
        let values = self.as_array();
        let complex = Complex::new(values[0], values[1]);
        let result = complex * value;
        f64x2::from_array([result.re, result.im])
    }

    #[inline]
    fn mul_complex(self, value: f64x2) -> f64x2 {
        let self_values = self.as_array();
        let complex = Complex::new(self_values[0], self_values[1]);
        let value_values = value.as_array();
        let value = Complex::new(value_values[0], value_values[1]);
        let result = complex * value;
        f64x2::from_array([result.re, result.im])
    }

    #[inline]
    fn div_complex(self, value: f64x2) -> f64x2 {
        let self_values = self.as_array();
        let value_values = value.as_array();
        let complex = Complex::new(self_values[0], self_values[1]);
        let value = Complex::new(value_values[0], value_values[1]);
        let result = complex / value;
        f64x2::from_array([result.re, result.im])
    }

    #[inline]
    fn complex_abs_squared(self) -> f64x2 {
        let values = self.as_array();
        let result = values[0] * values[0] + values[1] * values[1];
        f64x2::from_array([result, 0.0])
    }

    #[inline]
    fn complex_abs(self) -> f64x2 {
        let values = self.as_array();
        let result = (values[0] * values[0] + values[1] * values[1]).sqrt();
        f64x2::from_array([result, 0.0])
    }

    #[inline]
    fn sqrt(self) -> f64x2 {
        StdFloat::sqrt(self)
    }

    #[inline]
    fn store_half(self, target: &mut [f64], index: usize) {
        let values = self.as_array();
        target[index] = values[0];
    }

    #[inline]
    fn sum_real(&self) -> f64 {
        let values = self.as_array();
        values[0] + values[1]
    }

    #[inline]
    fn sum_complex(&self) -> Complex<f64> {
        let values = self.as_array();
        Complex::<f64>::new(values[0], values[1])
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        self.simd_max(other)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        self.simd_min(other)
    }

    #[inline]
    fn swap_iq(self) -> Self {
        let values = self.as_array();
        f64x2::from_array([values[1], values[0]])
    }
}

impl SimdFrom<f32x4> for i32x4 {
    fn regfrom(value: f32x4) -> Self {
        value.cast::<i32>()
    }
}

impl SimdFrom<i32x4> for f32x4 {
    fn regfrom(value: i32x4) -> Self {
        value.cast::<f32>()
    }
}

impl SimdFrom<f64x2> for i64x2 {
    fn regfrom(value: f64x2) -> Self {
        value.cast::<i64>()
    }
}

impl SimdFrom<i64x2> for f64x2 {
    fn regfrom(value: i64x2) -> Self {
        value.cast::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_test() {
        let vec = f32x4::from_array([1.0, 2.0, 3.0, 4.0]);
        let result = vec.swap_iq();
        let vec_values = vec.as_array();
        let result_values = result.as_array();

        assert_eq!(result_values[0], vec_values[1]);
        assert_eq!(result_values[1], vec_values[0]);
        assert_eq!(result_values[2], vec_values[3]);
        assert_eq!(result_values[3], vec_values[2]);
    }
}
