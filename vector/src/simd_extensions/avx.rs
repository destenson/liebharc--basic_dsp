use super::{Simd, SimdFrom};
use crate::numbers::*;
use std::arch::x86_64::*;
use std::mem;
use std::simd::num::{SimdFloat, SimdInt};
use std::simd::StdFloat;
pub use std::simd::{f32x8, f64x4};
use std::simd::{i32x8, i64x4};

/// This value must be read in groups of 2 bits.
const SWAP_IQ_PS: i32 = 0b1011_0001;

const SWAP_IQ_PD: i32 = 0b0101;

impl Simd<f32> for f32x8 {
    type Array = [f32; 8];

    #[inline]
    fn to_array(self) -> Self::Array {
        let mut target = [0.0; 8];
        self.copy_to_slice(&mut target);
        target
    }

    type ComplexArray = [Complex<f32>; 4];

    const LEN: usize = 8;

    #[inline]
    fn from_complex(value: Complex<f32>) -> f32x8 {
        f32x8::from_array([
            value.re, value.im, value.re, value.im, value.re, value.im, value.re, value.im,
        ])
    }

    #[inline]
    fn add_real(self, value: f32) -> f32x8 {
        let increment = f32x8::splat(value);
        self + increment
    }

    #[inline]
    fn scale_real(self, value: f32) -> f32x8 {
        let scale_vector = f32x8::splat(value);
        self * scale_vector
    }

    #[inline]
    fn scale_complex(self, value: Complex<f32>) -> f32x8 {
        let scaling_real = f32x8::splat(value.re);
        let scaling_imag = f32x8::splat(value.im);
        let parallel = scaling_real * self;
        let shuffled = self.swap_iq();
        let cross = scaling_imag * shuffled;
        unsafe {
            mem::transmute(_mm256_addsub_ps(
                mem::transmute(parallel),
                mem::transmute(cross),
            ))
        }
    }

    #[inline]
    fn mul_complex(self, value: f32x8) -> f32x8 {
        let value_arr = value.as_array();
        let scaling_real = f32x8::from_array([
            value_arr[0],
            value_arr[0],
            value_arr[2],
            value_arr[2],
            value_arr[4],
            value_arr[4],
            value_arr[6],
            value_arr[6],
        ]);
        let scaling_imag = f32x8::from_array([
            value_arr[1],
            value_arr[1],
            value_arr[3],
            value_arr[3],
            value_arr[5],
            value_arr[5],
            value_arr[7],
            value_arr[7],
        ]);
        let parallel = scaling_real * self;
        let shuffled = self.swap_iq();
        let cross = scaling_imag * shuffled;
        unsafe {
            mem::transmute(_mm256_addsub_ps(
                mem::transmute(parallel),
                mem::transmute(cross),
            ))
        }
    }

    #[inline]
    fn div_complex(self, value: f32x8) -> f32x8 {
        let values = value.as_array();
        let scaling_real = f32x8::from_array([
            values[0], values[0], values[2], values[2], values[4], values[4], values[6], values[6],
        ]);
        let scaling_imag = f32x8::from_array([
            values[1], values[1], values[3], values[3], values[5], values[5], values[7], values[7],
        ]);
        let parallel = scaling_real * value;
        let shuffled = value.swap_iq();
        let cross = scaling_imag * shuffled;
        let mul: f32x8 = unsafe {
            mem::transmute(_mm256_addsub_ps(
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
    fn complex_abs_squared(self) -> f32x8 {
        let squared = self * self;
        unsafe {
            mem::transmute(_mm256_hadd_ps(
                mem::transmute(squared),
                mem::transmute(squared),
            ))
        }
    }

    #[inline]
    fn complex_abs(self) -> f32x8 {
        let squared_sum = self.complex_abs_squared();
        StdFloat::sqrt(squared_sum)
    }

    #[inline]
    fn sqrt(self) -> f32x8 {
        StdFloat::sqrt(self)
    }
    #[inline]
    fn store_half(self, target: &mut [f32], index: usize) {
        let values = self.as_array();
        target[index] = values[0];
        target[index + 1] = values[1];
        target[index + 2] = values[2];
        target[index + 3] = values[3];
    }

    #[inline]
    fn sum_real(&self) -> f32 {
        let values = self.as_array();
        values.iter().sum()
    }

    #[inline]
    fn sum_complex(&self) -> Complex<f32> {
        let values = self.as_array();
        Complex::<f32>::new(
            values[0] + values[2] + values[4] + values[6],
            values[1] + values[3] + values[5] + values[7],
        )
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
        unsafe { mem::transmute(_mm256_permute_ps(mem::transmute(self), SWAP_IQ_PS)) }
    }
}

impl Simd<f64> for f64x4 {
    type Array = [f64; 4];

    #[inline]
    fn to_array(self) -> Self::Array {
        let mut target = [0.0; 4];
        self.copy_to_slice(&mut target);
        target
    }

    type ComplexArray = [Complex<f64>; 2];

    const LEN: usize = 4;

    #[inline]
    fn from_complex(value: Complex<f64>) -> f64x4 {
        f64x4::from_array([value.re, value.im, value.re, value.im])
    }

    #[inline]
    fn add_real(self, value: f64) -> f64x4 {
        let increment = f64x4::splat(value);
        self + increment
    }

    #[inline]
    fn scale_real(self, value: f64) -> f64x4 {
        let scale_vector = f64x4::splat(value);
        self * scale_vector
    }

    #[inline]
    fn scale_complex(self, value: Complex<f64>) -> f64x4 {
        let scaling_real = f64x4::splat(value.re);
        let scaling_imag = f64x4::splat(value.im);
        let parallel = scaling_real * self;
        let shuffled = self.swap_iq();
        let cross = scaling_imag * shuffled;
        unsafe {
            mem::transmute(_mm256_addsub_pd(
                mem::transmute(parallel),
                mem::transmute(cross),
            ))
        }
    }

    #[inline]
    fn mul_complex(self, value: f64x4) -> f64x4 {
        let value_arr = value.as_array();
        let scaling_real =
            f64x4::from_array([value_arr[0], value_arr[0], value_arr[2], value_arr[2]]);
        let scaling_imag =
            f64x4::from_array([value_arr[1], value_arr[1], value_arr[3], value_arr[3]]);
        let parallel = scaling_real * self;
        let shuffled = self.swap_iq();
        let cross = scaling_imag * shuffled;
        unsafe {
            mem::transmute(_mm256_addsub_pd(
                mem::transmute(parallel),
                mem::transmute(cross),
            ))
        }
    }

    #[inline]
    fn div_complex(self, value: f64x4) -> f64x4 {
        let values = self.as_array();
        let scaling_imag = f64x4::from_array([values[0], values[0], values[2], values[2]]);
        let scaling_real = f64x4::from_array([values[1], values[1], values[3], values[3]]);

        let parallel = scaling_real * value;
        let shuffled = value.swap_iq();
        let cross = scaling_imag * shuffled;
        let mul: f64x4 = unsafe {
            mem::transmute(_mm256_addsub_pd(
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
    fn complex_abs_squared(self) -> f64x4 {
        let squared = self * self;
        unsafe {
            mem::transmute(_mm256_hadd_pd(
                mem::transmute(squared),
                mem::transmute(squared),
            ))
        }
    }

    #[inline]
    fn complex_abs(self) -> f64x4 {
        let squared_sum = self.complex_abs_squared();
        StdFloat::sqrt(squared_sum)
    }

    #[inline]
    fn sqrt(self) -> f64x4 {
        StdFloat::sqrt(self)
    }
    #[inline]
    fn store_half(self, target: &mut [f64], index: usize) {
        let values = self.as_array();
        target[index] = values[0];
        target[index + 1] = values[1];
    }

    #[inline]
    fn sum_real(&self) -> f64 {
        let values = self.as_array();
        values.iter().sum()
    }

    #[inline]
    fn sum_complex(&self) -> Complex<f64> {
        let values = self.as_array();
        Complex::<f64>::new(values[0] + values[2], values[1] + values[3])
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        self.max(other)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        self.min(other)
    }

    #[inline]
    fn swap_iq(self) -> Self {
        unsafe { mem::transmute(_mm256_permute_pd(mem::transmute(self), SWAP_IQ_PD)) }
    }
}

impl SimdFrom<f32x8> for i32x8 {
    fn regfrom(value: f32x8) -> Self {
        value.cast::<i32>()
    }
}

impl SimdFrom<i32x8> for f32x8 {
    fn regfrom(value: i32x8) -> Self {
        value.cast::<f32>()
    }
}

impl SimdFrom<f64x4> for i64x4 {
    fn regfrom(value: f64x4) -> Self {
        value.cast::<i64>()
    }
}

impl SimdFrom<i64x4> for f64x4 {
    fn regfrom(value: i64x4) -> Self {
        value.cast::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_test_f32() {
        let vec = f32x8::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = vec.swap_iq();
        let result_values = result.as_array();
        let vec_values = vec.as_array();

        assert_eq!(result_values[0], vec_values[1]);
        assert_eq!(result_values[1], vec_values[0]);
        assert_eq!(result_values[2], vec_values[3]);
        assert_eq!(result_values[3], vec_values[2]);
        assert_eq!(result_values[4], vec_values[5]);
        assert_eq!(result_values[5], vec_values[4]);
        assert_eq!(result_values[6], vec_values[7]);
        assert_eq!(result_values[7], vec_values[6]);
    }

    #[test]
    fn shuffle_test_f64() {
        let vec = f64x4::from_array([1.0, 2.0, 3.0, 4.0]);
        let result = vec.swap_iq();
        let vec_values = vec.as_array();
        let result_values = result.as_array();

        assert_eq!(result_values[0], vec_values[1]);
        assert_eq!(result_values[1], vec_values[0]);
        assert_eq!(result_values[2], vec_values[3]);
        assert_eq!(result_values[3], vec_values[2]);
    }
}
