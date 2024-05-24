use super::{Simd, SimdFrom};
use crate::numbers::*;
use std::arch::x86_64::*;
use std::mem;
use std::simd::num::{SimdFloat, SimdInt};
use std::simd::StdFloat;
pub use std::simd::{f32x16, f64x8};
use std::simd::{i32x16, i64x8};

/// This value must be read in groups of 2 bits.
const SWAP_IQ_PS: i32 = 0b1011_0001;

/// This value must be read in groups of 2 bits:
/// 10 means that the third position (since it's the third bit pair)
/// will be replaced with the value of the second position (10b = 2d)
const SWAP_IQ_PD: i32 = 0b1011_0001;

impl Simd<f32> for f32x16 {
    type Array = [f32; 16];

    #[inline]
    fn to_array(self) -> Self::Array {
        let mut target = [0.0; 16];
        self.copy_to_slice(&mut target);
        target
    }

    type ComplexArray = [Complex<f32>; 8];

    const LEN: usize = 16;

    #[inline]
    fn from_complex(value: Complex<f32>) -> f32x16 {
        f32x16::from_array([
            value.re, value.im, value.re, value.im, value.re, value.im, value.re, value.im,
            value.re, value.im, value.re, value.im, value.re, value.im, value.re, value.im,
        ])
    }

    #[inline]
    fn add_real(self, value: f32) -> f32x16 {
        let increment = f32x16::splat(value);
        self + increment
    }

    #[inline]
    fn scale_real(self, value: f32) -> f32x16 {
        let scale_vector = f32x16::splat(value);
        self * scale_vector
    }

    #[inline]
    fn scale_complex(self, value: Complex<f32>) -> f32x16 {
        let scaling_real = f32x16::splat(value.re);
        let scaling_imag = f32x16::splat(value.im);
        let parallel = scaling_real * self;
        let shuffled = self.swap_iq();
        let cross = scaling_imag * shuffled;
        let ones = f32x16::splat(1.0);
        unsafe {
            mem::transmute(_mm512_fmaddsub_ps(
                mem::transmute(parallel),
                mem::transmute(ones),
                mem::transmute(cross),
            ))
        }
    }
    #[inline]
    fn mul_complex(self, value: f32x16) -> f32x16 {
        let value_arr = value.as_array();
        let scaling_real = f32x16::from_array([
            value_arr[0],
            value_arr[0],
            value_arr[2],
            value_arr[2],
            value_arr[4],
            value_arr[4],
            value_arr[6],
            value_arr[6],
            value_arr[8],
            value_arr[8],
            value_arr[10],
            value_arr[10],
            value_arr[12],
            value_arr[12],
            value_arr[14],
            value_arr[14],
        ]);
        let scaling_imag = f32x16::from_array([
            value_arr[1],
            value_arr[1],
            value_arr[3],
            value_arr[3],
            value_arr[5],
            value_arr[5],
            value_arr[7],
            value_arr[7],
            value_arr[9],
            value_arr[9],
            value_arr[11],
            value_arr[11],
            value_arr[13],
            value_arr[13],
            value_arr[15],
            value_arr[15],
        ]);
        let parallel = scaling_real * self;
        let shuffled = self.swap_iq();
        let cross = scaling_imag * shuffled;
        let ones = f32x16::splat(1.0);
        unsafe {
            mem::transmute(_mm512_fmaddsub_ps(
                mem::transmute(parallel),
                mem::transmute(ones),
                mem::transmute(cross),
            ))
        }
    }

    #[inline]
    fn div_complex(self, value: f32x16) -> f32x16 {
        let value_arr = self.as_array();
        let scaling_imag = f32x16::from_array([
            value_arr[0],
            value_arr[0],
            value_arr[2],
            value_arr[2],
            value_arr[4],
            value_arr[4],
            value_arr[6],
            value_arr[6],
            value_arr[8],
            value_arr[8],
            value_arr[10],
            value_arr[10],
            value_arr[12],
            value_arr[12],
            value_arr[14],
            value_arr[14],
        ]);
        let scaling_real = f32x16::from_array([
            value_arr[1],
            value_arr[1],
            value_arr[3],
            value_arr[3],
            value_arr[5],
            value_arr[5],
            value_arr[7],
            value_arr[7],
            value_arr[9],
            value_arr[9],
            value_arr[11],
            value_arr[11],
            value_arr[13],
            value_arr[13],
            value_arr[15],
            value_arr[15],
        ]);
        let parallel = scaling_real * value;
        let shuffled = value.swap_iq();
        let cross = scaling_imag * shuffled;
        let ones = f32x16::splat(1.0);
        let mul: f32x16 = unsafe {
            mem::transmute(_mm512_fmaddsub_ps(
                mem::transmute(parallel),
                mem::transmute(ones),
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
    fn complex_abs_squared(self) -> f32x16 {
        let squared = self * self;
        let ones = f32x16::splat(1.0);
        unsafe {
            mem::transmute(_mm512_fmaddsub_ps(
                mem::transmute(squared),
                mem::transmute(ones),
                mem::transmute(squared),
            ))
        }
    }

    #[inline]
    fn complex_abs(self) -> f32x16 {
        let squared_sum = self.complex_abs_squared();
        StdFloat::sqrt(squared_sum)
    }

    #[inline]
    fn sqrt(self) -> f32x16 {
        StdFloat::sqrt(self)
    }
    #[inline]
    fn store_real(self, target: &mut [f32], index: usize) {
        let values = self.as_array();
        target[index] = values[0];
        target[index + 1] = values[1];
        target[index + 2] = values[4];
        target[index + 3] = values[5];
        target[index + 4] = values[8];
        target[index + 5] = values[9];
        target[index + 6] = values[12];
        target[index + 7] = values[13];
    }
    #[inline]
    fn sum_real(&self) -> f32 {
        let values = self.as_array();
        values[0]
            + values[1]
            + values[2]
            + values[3]
            + values[4]
            + values[5]
            + values[6]
            + values[7]
            + values[8]
            + values[9]
            + values[10]
            + values[11]
            + values[12]
            + values[13]
            + values[14]
            + values[15]
    }
    #[inline]
    fn sum_complex(&self) -> Complex<f32> {
        let values = self.as_array();
        Complex::<f32>::new(
            values[0]
                + values[2]
                + values[4]
                + values[6]
                + values[8]
                + values[10]
                + values[12]
                + values[14],
            values[1]
                + values[3]
                + values[5]
                + values[7]
                + values[9]
                + values[11]
                + values[13]
                + values[15],
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
        unsafe { mem::transmute(_mm512_permute_ps(mem::transmute(self), SWAP_IQ_PS)) }
    }
}

impl Simd<f64> for f64x8 {
    type Array = [f64; 8];

    #[inline]
    fn to_array(self) -> Self::Array {
        let mut target = [0.0; 8];
        self.copy_to_slice(&mut target);
        target
    }

    type ComplexArray = [Complex<f64>; 4];

    const LEN: usize = 8;

    #[inline]
    fn from_complex(value: Complex<f64>) -> f64x8 {
        f64x8::from_array([
            value.re, value.im, value.re, value.im, value.re, value.im, value.re, value.im,
        ])
    }

    #[inline]
    fn add_real(self, value: f64) -> f64x8 {
        let increment = f64x8::splat(value);
        self + increment
    }

    #[inline]
    fn scale_real(self, value: f64) -> f64x8 {
        let scale_vector = f64x8::splat(value);
        self * scale_vector
    }

    #[inline]
    fn scale_complex(self, value: Complex<f64>) -> f64x8 {
        let scaling_real = f64x8::splat(value.re);
        let scaling_imag = f64x8::splat(value.im);
        let parallel = scaling_real * self;
        let shuffled = self.swap_iq();
        let cross = scaling_imag * shuffled;
        let ones = f32x16::splat(1.0);
        unsafe {
            mem::transmute(_mm512_fmaddsub_pd(
                mem::transmute(parallel),
                mem::transmute(ones),
                mem::transmute(cross),
            ))
        }
    }

    #[inline]
    fn mul_complex(self, value: f64x8) -> f64x8 {
        let value_arr = value.as_array();
        let scaling_real = f64x8::from_array([
            value_arr[0],
            value_arr[0],
            value_arr[2],
            value_arr[2],
            value_arr[4],
            value_arr[4],
            value_arr[6],
            value_arr[6],
        ]);
        let scaling_imag = f64x8::from_array([
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
        let ones = f32x16::splat(1.0);
        let cross = scaling_imag * shuffled;
        unsafe {
            mem::transmute(_mm512_fmaddsub_pd(
                mem::transmute(parallel),
                mem::transmute(ones),
                mem::transmute(cross),
            ))
        }
    }

    #[inline]
    fn div_complex(self, value: f64x8) -> f64x8 {
        let values = self.as_array();
        let scaling_imag = f64x8::from_array([
            values[0], values[0], values[2], values[2], values[4], values[4], values[6], values[6],
        ]);
        let scaling_real = f64x8::from_array([
            values[1], values[1], values[3], values[3], values[5], values[5], values[7], values[7],
        ]);

        let parallel = scaling_real * value;
        let shuffled = self.swap_iq();
        let cross = scaling_imag * shuffled;
        let ones = f64x8::splat(1.0);
        let mul: f64x8 = unsafe {
            mem::transmute(_mm512_fmaddsub_pd(
                mem::transmute(parallel),
                mem::transmute(ones),
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
    fn complex_abs_squared(self) -> f64x8 {
        let squared = self * self;
        let ones = f64x8::splat(1.0);
        unsafe {
            mem::transmute(_mm512_fmaddsub_ps(
                mem::transmute(squared),
                mem::transmute(ones),
                mem::transmute(squared),
            ))
        }
    }

    #[inline]
    fn complex_abs(self) -> f64x8 {
        let squared_sum = self.complex_abs_squared();
        StdFloat::sqrt(squared_sum)
    }

    #[inline]
    fn sqrt(self) -> f64x8 {
        StdFloat::sqrt(self)
    }
    #[inline]
    fn store_real(self, target: &mut [f64], index: usize) {
        let values = self.as_array();
        target[index] = values[0];
        target[index + 1] = values[1];
        target[index + 2] = values[4];
        target[index + 3] = values[5];
    }

    #[inline]
    fn sum_real(&self) -> f64 {
        let values = self.as_array();
        values[0]
            + values[1]
            + values[2]
            + values[3]
            + values[4]
            + values[5]
            + values[6]
            + values[7]
    }

    #[inline]
    fn sum_complex(&self) -> Complex<f64> {
        let values = self.as_array();
        Complex::<f64>::new(
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
        unsafe { mem::transmute(_mm512_permute_pd(mem::transmute(self), SWAP_IQ_PD)) }
    }
}

impl SimdFrom<f32x16> for i32x16 {
    fn regfrom(value: f32x16) -> Self {
        value.cast::<i32>()
    }
}

impl SimdFrom<i32x16> for f32x16 {
    fn regfrom(value: i32x16) -> Self {
        value.cast::<f32>()
    }
}

impl SimdFrom<f64x8> for i64x8 {
    fn regfrom(value: f64x8) -> Self {
        value.cast::<i64>()
    }
}

impl SimdFrom<i64x8> for f64x8 {
    fn regfrom(value: i64x8) -> Self {
        value.cast::<f64>()
    }
}
