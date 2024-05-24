//! Defines the buffers. Finding a good general purpose buffering scheme is hard.
//! So the best option seems to be to create an abstraction so that the buffering can be adjusted
//! to what an application needs.
use super::ToSliceMut;
use crate::numbers::*;

/// A "slice-like" type which also allows to
pub trait BufferBorrow<S: ToSliceMut<T>, T: RealNumber>: ToSliceMut<T> {
    /// Moves the content of this slice into `storage`.
    /// This operation might just copy all contents into `storage` or
    fn trade(self, storage: &mut S);
}

/// A buffer which can be used by other types. Types will call buffers to create new arrays.
/// A buffer may can implement any buffering strategy.
pub trait Buffer<'a, S, T>
where
    S: ToSliceMut<T>,
    T: RealNumber,
{
    /// The type of the burrow which is returned.
    type Borrow: BufferBorrow<S, T>;

    /// Asks the buffer for new storage of exactly size `len`.
    /// The returned array doesn't need to have be initialized with any default value.
    fn borrow(&'a mut self, len: usize) -> Self::Borrow;

    /// Returns the allocated length of all storage within this buffer.
    fn alloc_len(&self) -> usize;
}
