use super::Simd;
use crate::numbers::*;
use std::ops::*;

#[allow(non_camel_case_types)]
// To stay consistent with the `simd` crate
#[derive(Debug, Clone, Copy)]
pub struct f32x4(f32, f32, f32, f32);

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy)]
pub struct f64x2(f64, f64);

impl f32x4 {
    #[inline]
    pub fn new(v1: f32, v2: f32, v3: f32, v4: f32) -> Self {
        f32x4(v1, v2, v3, v4)
    }

    #[inline]
    pub fn splat(value: f32) -> Self {
        f32x4(value, value, value, value)
    }

    #[inline]
    pub fn from_slice(array: &[f32]) -> Self {
        f32x4(array[0], array[1], array[2], array[3])
    }

    #[inline]
    pub fn copy_to_slice(self, array: &mut [f32]) {
        array[0] = self.extract(0);
        array[1] = self.extract(1);
        array[2] = self.extract(2);
        array[3] = self.extract(3);
    }

    #[inline]
    pub fn extract(self, index: usize) -> f32 {
        match index {
            0 => self.0,
            1 => self.1,
            2 => self.2,
            3 => self.3,
            _ => panic!("{} out of bounds for type f32x4", index),
        }
    }
}

impl f64x2 {
    #[inline]
    pub fn new(v1: f64, v2: f64) -> Self {
        f64x2(v1, v2)
    }

    #[inline]
    pub fn splat(value: f64) -> Self {
        f64x2(value, value)
    }

    #[inline]
    pub fn from_slice(array: &[f64]) -> Self {
        f64x2(array[0], array[1])
    }

    #[inline]
    pub fn copy_to_slice(self, array: &mut [f64]) {
        array[0] = self.extract(0);
        array[1] = self.extract(1);
    }

    #[inline]
    pub fn extract(self, index: usize) -> f64 {
        match index {
            0 => self.0,
            1 => self.1,
            _ => panic!("{} out of bounds for type f32x4", index),
        }
    }
}

impl Add for f32x4 {
    type Output = Self;
    #[inline]
    fn add(self, x: Self) -> Self {
        f32x4(self.0 + x.0, self.1 + x.1, self.2 + x.2, self.3 + x.3)
    }
}

impl Sub for f32x4 {
    type Output = Self;
    #[inline]
    fn sub(self, x: Self) -> Self {
        f32x4(self.0 - x.0, self.1 - x.1, self.2 - x.2, self.3 - x.3)
    }
}

impl Mul for f32x4 {
    type Output = Self;
    #[inline]
    fn mul(self, x: Self) -> Self {
        f32x4(self.0 * x.0, self.1 * x.1, self.2 * x.2, self.3 * x.3)
    }
}

impl Div for f32x4 {
    type Output = Self;
    #[inline]
    fn div(self, x: Self) -> Self {
        f32x4(self.0 / x.0, self.1 / x.1, self.2 / x.2, self.3 / x.3)
    }
}

impl Add for f64x2 {
    type Output = Self;
    #[inline]
    fn add(self, x: Self) -> Self {
        f64x2(self.0 + x.0, self.1 + x.1)
    }
}

impl Sub for f64x2 {
    type Output = Self;
    #[inline]
    fn sub(self, x: Self) -> Self {
        f64x2(self.0 - x.0, self.1 - x.1)
    }
}

impl Mul for f64x2 {
    type Output = Self;
    #[inline]
    fn mul(self, x: Self) -> Self {
        f64x2(self.0 * x.0, self.1 * x.1)
    }
}

impl Div for f64x2 {
    type Output = Self;
    #[inline]
    fn div(self, x: Self) -> Self {
        f64x2(self.0 / x.0, self.1 / x.1)
    }
}

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
        f32x4::new(value.re, value.im, value.re, value.im)
    }

    #[inline]
    fn add_real(self, value: f32) -> f32x4 {
        let increment = f32x4::splat(value);
        self + increment
    }

    #[inline]
    fn scale_real(self, value: f32) -> f32x4 {
        let scale_vector = f32x4::splat(value);
        self * scale_vector
    }

    #[inline]
    fn scale_complex(self, value: Complex<f32>) -> f32x4 {
        let a = Complex::<f32>::new(self.0, self.1);
        let b = Complex::<f32>::new(self.2, self.3);
        let a = a * value;
        let b = b * value;
        f32x4::new(a.re, a.im, b.re, b.im)
    }

    #[inline]
    fn mul_complex(self, value: f32x4) -> f32x4 {
        let a = Complex::<f32>::new(self.0, self.1);
        let b = Complex::<f32>::new(self.2, self.3);
        let c = Complex::<f32>::new(value.0, value.1);
        let a = a * c;
        let d = Complex::<f32>::new(value.2, value.3);
        let b = b * d;
        f32x4::new(a.re, a.im, b.re, b.im)
    }

    #[inline]
    fn div_complex(self, value: f32x4) -> f32x4 {
        let a = Complex::<f32>::new(self.0, self.1);
        let b = Complex::<f32>::new(self.2, self.3);
        let c = Complex::<f32>::new(value.0, value.1);
        let a = a / c;
        let d = Complex::<f32>::new(value.2, value.3);
        let b = b / d;
        f32x4::new(a.re, a.im, b.re, b.im)
    }

    #[inline]
    fn complex_abs_squared(self) -> f32x4 {
        let squared = self * self;
        f32x4::new(squared.0 + squared.1, 0.0, squared.2 + squared.3, 0.0)
    }

    #[inline]
    fn complex_abs(self) -> f32x4 {
        let squared = self * self;
        f32x4::new(
            (squared.0 + squared.1).sqrt(),
            0.0,
            (squared.2 + squared.3).sqrt(),
            0.0,
        )
    }

    #[inline]
    fn sqrt(self) -> f32x4 {
        f32x4::new(self.0.sqrt(), self.1.sqrt(), self.2.sqrt(), self.3.sqrt())
    }

    #[inline]
    fn store_real(self, target: &mut [f32], index: usize) {
        target[index] = self.extract(0);
        target[index + 1] = self.extract(2);
    }

    #[inline]
    fn sum_real(&self) -> f32 {
        self.0 + self.1 + self.2 + self.3
    }

    #[inline]
    fn sum_complex(&self) -> Complex<f32> {
        Complex::<f32>::new(self.0 + self.2, self.1 + self.3)
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        f32x4::new(
            f32::max(self.0, other.0),
            f32::max(self.1, other.1),
            f32::max(self.2, other.2),
            f32::max(self.3, other.3),
        )
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        f32x4::new(
            f32::min(self.0, other.0),
            f32::min(self.1, other.1),
            f32::min(self.2, other.2),
            f32::min(self.3, other.3),
        )
    }

    #[inline]
    fn swap_iq(self) -> Self {
        f32x4::new(
            self.extract(1),
            self.extract(0),
            self.extract(3),
            self.extract(2),
        )
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
        f64x2::new(value.re, value.im)
    }

    #[inline]
    fn add_real(self, value: f64) -> f64x2 {
        let increment = f64x2::splat(value);
        self + increment
    }

    #[inline]
    fn scale_real(self, value: f64) -> f64x2 {
        let scale_vector = f64x2::splat(value);
        self * scale_vector
    }

    #[inline]
    fn scale_complex(self, value: Complex<f64>) -> f64x2 {
        let complex = Complex::new(self.0, self.1);
        let result = complex * value;
        f64x2::new(result.re, result.im)
    }

    #[inline]
    fn mul_complex(self, value: f64x2) -> f64x2 {
        let complex = Complex::new(self.0, self.1);
        let value = Complex::new(value.0, value.1);
        let result = complex * value;
        f64x2::new(result.re, result.im)
    }

    #[inline]
    fn div_complex(self, value: f64x2) -> f64x2 {
        let complex = Complex::new(self.0, self.1);
        let value = Complex::new(value.0, value.1);
        let result = complex / value;
        f64x2::new(result.re, result.im)
    }

    #[inline]
    fn complex_abs_squared(self) -> f64x2 {
        let a = self.0;
        let b = self.1;
        let result = a * a + b * b;
        f64x2::new(result, 0.0)
    }

    #[inline]
    fn complex_abs(self) -> f64x2 {
        let a = self.0;
        let b = self.1;
        let result = (a * a + b * b).sqrt();
        f64x2::new(result, 0.0)
    }

    #[inline]
    fn sqrt(self) -> f64x2 {
        f64x2::new(self.0.sqrt(), self.1.sqrt())
    }

    #[inline]
    fn store_real(self, target: &mut [f64], index: usize) {
        target[index] = self.extract(0);
    }

    #[inline]
    fn sum_real(&self) -> f64 {
        self.0 + self.1
    }

    #[inline]
    fn sum_complex(&self) -> Complex<f64> {
        Complex::<f64>::new(self.0, self.1)
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        f64x2::new(f64::max(self.0, other.0), f64::max(self.1, other.1))
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        f64x2::new(f64::min(self.0, other.0), f64::min(self.1, other.1))
    }

    #[inline]
    fn swap_iq(self) -> Self {
        f64x2::new(self.extract(1), self.extract(0))
    }
}
