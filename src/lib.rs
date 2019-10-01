// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! # Type Level Sized Vectors
//!
//! This crate provides a `Vec<N, A>` type, which wraps the standard `Vec<A>`
//! and tracks its size `N` at the type level using the [`typenum`][typenum]
//! crate.
//!
//! Because the size is embedded in the type, we can do things like verifying at
//! compile time that index lookups are within bounds.
//!
//! ```compile_fail
//! # #[macro_use] extern crate sized_vec;
//! # extern crate typenum;
//! # use typenum::U8;
//! # fn main() {
//! let vec = svec![1, 2, 3];
//! // This index lookup won't compile, because index `U8` is outside
//! // the vector's length of `U3`:
//! assert_eq!(5, vec[U8::new()]);
//! # }
//! ```
//!
//! ```
//! # #[macro_use] extern crate sized_vec;
//! # extern crate typenum;
//! # use typenum::U2;
//! # fn main() {
//! let vec = svec![1, 2, 3];
//! // On the other hand, this lookup can be verified to be correct
//! // by the type system:
//! assert_eq!(3, vec[U2::new()]);
//! # }
//! ```
//!
//! ## Limitations
//!
//! If this looks too good to be true, it's because it comes with a number of
//! limitations: you won't be able to perform operations on the vector which
//! could leave it with a length that can't be known at compile time. This
//! includes `Extend::extend()` and filtering operations like `Vec::retain()`.
//!
//! `FromIterator::from_iter` is, notably, also not available, but you can use
//! `Vec::try_from_iter` as a replacement. Note that `try_from_iter` needs to be
//! able to infer the size of the resulting vector at compile time; there's no
//! way to construct a vector of arbitrary length.
//!
//! ```
//! # #[macro_use] extern crate sized_vec;
//! # extern crate typenum;
//! # use sized_vec::Vec;
//! # use typenum::U2;
//! # fn main() {
//! let vec = svec![1, 2, 3, 4, 5];
//! let new_vec = Vec::try_from_iter(vec.into_iter().map(|i| i + 10));
//! assert_eq!(Some(svec![11, 12, 13, 14, 15]), new_vec);
//! # }
//! ```
//!
//! [typenum]: https://crates.io/crates/typenum

#![allow(clippy::type_repetition_in_bounds)]

use std::convert::TryFrom;
use typenum::consts::*;
use typenum::{
    Add1, Bit, Diff, Eq, IsEqual, IsLess, IsLessOrEqual, Le, LeEq, Sub1, Sum, True, Unsigned,
};

use std::borrow::{Borrow, BorrowMut};
use std::convert::{AsMut, AsRef};
use std::fmt::{Debug, Error, Formatter};
use std::marker::PhantomData;
use std::ops::{Add, Index, IndexMut, Sub};
use std::slice::{Iter, IterMut};
use std::vec::IntoIter;

pub trait IsTrue {}
impl IsTrue for True {}

/// A type level range.
pub struct Range<Left, Right>
where
    Left: Unsigned + IsLessOrEqual<Right>,
    Right: Unsigned,
    LeEq<Left, Right>: IsTrue,
{
    left: PhantomData<Left>,
    right: PhantomData<Right>,
}

impl<Left, Right> Range<Left, Right>
where
    Left: Unsigned + IsLessOrEqual<Right>,
    Right: Unsigned,
    LeEq<Left, Right>: IsTrue,
{
    /// Create an instance of this Range.
    pub fn new() -> Self {
        Range {
            left: PhantomData,
            right: PhantomData,
        }
    }

    /// Reify the range type into a `std::ops::Range<usize>` value.
    pub fn to_usize() -> ::std::ops::Range<usize> {
        Left::USIZE..Right::USIZE
    }
}

impl<Left, Right> Default for Range<Left, Right>
where
    Left: Unsigned + IsLessOrEqual<Right>,
    Right: Unsigned,
    LeEq<Left, Right>: IsTrue,
{
    fn default() -> Self {
        Self::new()
    }
}

#[macro_export]
macro_rules! svec {
    () => { $crate::vec::Vec::new() };

    ( $($x:expr),* ) => {{
        $crate::Vec::new()$(.push($x))*
    }};

    ( $($x:expr ,)* ) => {{
        $crate::Vec::new()$(.push($x))*
    }};
}

/// A vector of length `N` containing elements of type `A`.
#[derive(PartialEq, Eq, Clone, Hash, PartialOrd, Ord)]
pub struct Vec<N, A>
where
    N: Unsigned,
{
    len: PhantomData<N>,
    vec: ::std::vec::Vec<A>,
}

impl<A> Vec<U0, A> {
    /// Construct an empty vector.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Vec {
            len: PhantomData,
            vec: vec![],
        }
    }
}

impl<N, A> Vec<N, A>
where
    N: Unsigned,
{
    #[inline]
    #[must_use]
    fn trust_me<M: Unsigned>(self) -> Vec<M, A> {
        use std::mem::transmute;
        unsafe { transmute(self) }
    }

    #[must_use]
    fn from_vec(vec: ::std::vec::Vec<A>) -> Vec<N, A> {
        Vec {
            len: PhantomData,
            vec,
        }
    }

    /// Construct a vector of size `N` using a function.
    ///
    /// The function is called with an index to generate a value of `A` for each
    /// index in the vector.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # use sized_vec::Vec;
    /// # use typenum::U3;
    /// # fn main() {
    /// let vec: Vec<U3, _> = Vec::fill(|i| i + 10);
    /// assert_eq!(svec![10, 11, 12], vec);
    /// # }
    /// ```
    #[must_use]
    pub fn fill<F>(f: F) -> Vec<N, A>
    where
        F: FnMut(usize) -> A,
    {
        Vec::from_vec((0..N::USIZE).map(f).collect())
    }

    /// Construct a vector of size `N` containing the same element repeated.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # use sized_vec::Vec;
    /// # use typenum::U3;
    /// # fn main() {
    /// let vec: Vec<U3, _> = Vec::repeat(5);
    /// assert_eq!(svec![5, 5, 5], vec);
    /// # }
    /// ```
    #[must_use]
    pub fn repeat(a: A) -> Vec<N, A>
    where
        A: Clone,
    {
        Vec::from_vec(::std::iter::repeat(a).take(N::USIZE).collect())
    }

    /// Construct a vector of size `N` from an iterator.
    ///
    /// Returns `None` if the iterator didn't contain exactly `N` elements.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # use sized_vec::Vec;
    /// # use typenum::U3;
    /// # fn main() {
    /// let good_vec: Option<Vec<U3, _>> = Vec::try_from_iter(1..=3);
    /// assert_eq!(Some(svec![1, 2, 3]), good_vec);
    ///
    /// let bad_vec: Option<Vec<U3, _>> = Vec::try_from_iter(1..=500);
    /// assert_eq!(None, bad_vec);
    /// # }
    /// ```
    #[must_use]
    pub fn try_from_iter<I>(iter: I) -> Option<Self>
    where
        I: IntoIterator<Item = A>,
    {
        let mut vec = ::std::vec::Vec::with_capacity(N::USIZE);
        vec.extend(iter);
        if vec.len() == N::USIZE {
            Some(Vec::from_vec(vec))
        } else {
            None
        }
    }

    /// Construct a vector of size `N` using the default value.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # use sized_vec::Vec;
    /// # use typenum::U3;
    /// # fn main() {
    /// let vec: Vec<U3, _> = Vec::from_default();
    /// assert_eq!(svec![0, 0, 0], vec);
    /// # }
    /// ```
    #[must_use]
    pub fn from_default() -> Vec<N, A>
    where
        A: Default,
    {
        Vec::from_vec((0..N::USIZE).map(|_| Default::default()).collect())
    }

    /// Push an element onto the end of the vector.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # use sized_vec::Vec;
    /// # use typenum::{U2, U3};
    /// # fn main() {
    /// let vec: Vec<U2, _> = svec![1, 2];
    /// let new_vec: Vec<U3, _> = vec.push(3);
    /// assert_eq!(svec![1, 2, 3], new_vec);
    /// # }
    /// ```
    ///
    /// ```compile_fail
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # use sized_vec::Vec;
    /// # use typenum::{U2, U3};
    /// # fn main() {
    /// let vec: Vec<U2, _> = svec![1, 2];
    /// // Type error, because the new length will be U3, not U2:
    /// let new_vec: Vec<U2, _> = vec.push(3);
    /// # }
    /// ```
    #[must_use]
    pub fn push(mut self, a: A) -> Vec<Add1<N>, A>
    where
        N: Add<B1>,
        Add1<N>: Unsigned,
    {
        self.vec.push(a);
        self.trust_me()
    }

    /// Pop an element off the end of the vector.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # use sized_vec::Vec;
    /// # fn main() {
    /// let vec = svec![1, 2, 3];
    /// let (new_vec, value) = vec.pop();
    /// assert_eq!(svec![1, 2], new_vec);
    /// assert_eq!(3, value);
    /// # }
    /// ```
    ///
    /// ```compile_fail
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # use sized_vec::Vec;
    /// # use typenum::{U2, U3};
    /// # fn main() {
    /// let vec: Vec<U3, _> = svec![1, 2, 3];
    /// // Type error, because the new length will be U2, not U3:
    /// let (new_vec: Vec<U3, _>, value) = vec.pop();
    /// # }
    /// ```
    #[must_use]
    pub fn pop(mut self) -> (Vec<Diff<N, U1>, A>, A)
    where
        N: Sub<U1>,
        Diff<N, U1>: Unsigned,
    {
        let o = self.vec.pop().unwrap();
        (self.trust_me(), o)
    }

    /// Insert an element into the vector at index `Index`.
    #[must_use]
    pub fn insert<Index>(mut self, _: Index, a: A) -> Vec<Add1<N>, A>
    where
        Index: Unsigned + IsLess<N>,
        Le<Index, N>: IsTrue,
        N: Add<B1>,
        Add1<N>: Unsigned,
    {
        self.vec.insert(Index::USIZE, a);
        self.trust_me()
    }

    /// Remove the element at index `Index` from the vector.
    #[must_use]
    pub fn remove<Index>(mut self, _: Index) -> (Vec<Sub1<N>, A>, A)
    where
        Index: Unsigned + IsLess<N>,
        Le<Index, N>: IsTrue,
        N: Sub<B1>,
        Sub1<N>: Unsigned,
    {
        let o = self.vec.remove(Index::USIZE);
        (self.trust_me(), o)
    }

    /// Remove the element at index `Index` from the vector,
    /// replacing it with the element at the end of the vector.
    #[must_use]
    pub fn swap_remove<Index>(mut self, _: Index) -> (Vec<Sub1<N>, A>, A)
    where
        Index: Unsigned + IsLess<N>,
        Le<Index, N>: IsTrue,
        N: Sub<B1>,
        Sub1<N>: Unsigned,
    {
        let o = self.vec.swap_remove(Index::USIZE);
        (self.trust_me(), o)
    }

    /// Append two vectors together.
    #[must_use]
    pub fn append<M>(mut self, mut other: Vec<M, A>) -> Vec<Sum<N, M>, A>
    where
        N: Add<M>,
        M: Unsigned,
        Sum<N, M>: Unsigned,
    {
        self.vec.append(&mut other.vec);
        self.trust_me()
    }

    /// Get a reference to the element at index `Index`.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # use sized_vec::Vec;
    /// # use typenum::U1;
    /// # fn main() {
    /// let vec = svec![1, 2, 3];
    /// assert_eq!(&2, vec.get(U1::new()));
    /// # }
    /// ```
    ///
    /// ```compile_fail
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # use sized_vec::Vec;
    /// # use typenum::U5;
    /// # fn main() {
    /// let vec = svec![1, 2, 3];
    /// // This index is out of bounds, so this won't compile:
    /// assert_eq!(&2, vec.get(U5::new()));
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn get<Index>(&self, _: Index) -> &A
    where
        Index: Unsigned + IsLess<N>,
        Le<Index, N>: IsTrue,
    {
        unsafe { self.vec.get_unchecked(Index::USIZE) }
    }

    /// Get a mutable reference to the element at index `Index`.
    #[inline]
    #[must_use]
    pub fn get_mut<Index>(&mut self, _: Index) -> &mut A
    where
        Index: Unsigned + IsLess<N>,
        Le<Index, N>: IsTrue,
    {
        unsafe { self.vec.get_unchecked_mut(Index::USIZE) }
    }

    /// Get the length of the vector.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        N::USIZE
    }

    /// Test if the vector is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool
    where
        N: IsEqual<U0>,
    {
        Eq::<N, U0>::BOOL
    }

    /// Get an iterator over the elements of the vector.
    #[inline]
    #[must_use]
    pub fn iter(&self) -> Iter<A> {
        self.vec.iter()
    }

    /// Get a mutable iterator over the elements of the vector.
    #[inline]
    #[must_use]
    pub fn iter_mut(&mut self) -> IterMut<A> {
        self.vec.iter_mut()
    }

    /// Get a reference to the slice of elements contained in the vector.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[A] {
        self.vec.as_slice()
    }

    /// Get a mutable reference to the slice of elements contained in the
    /// vector.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [A] {
        self.vec.as_mut_slice()
    }

    /// Truncate the vector to fit the size given by the target type.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # use sized_vec::Vec;
    /// # use typenum::U3;
    /// # fn main() {
    /// let vec_6 = svec![1, 2, 3, 4, 5, 6];
    /// let vec_3: Vec<U3, _> = vec_6.truncate();
    /// assert_eq!(svec![1, 2, 3], vec_3);
    /// # }
    /// ```
    #[must_use]
    pub fn truncate<M>(mut self) -> Vec<M, A>
    where
        M: Unsigned + IsLessOrEqual<N>,
        LeEq<M, N>: IsTrue,
    {
        self.vec.truncate(M::USIZE);
        self.trust_me()
    }

    /// Slice a subset out of the vector.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # use sized_vec::Range;
    /// # use typenum::{U2, U4};
    /// # fn main() {
    /// let vec = svec![1, 2, 3, 4, 5, 6];
    /// let (vec, sub_vec) = vec.slice(Range::<U2, U4>::new());
    /// assert_eq!(svec![1, 2, 5, 6], vec);
    /// assert_eq!(svec![3, 4], sub_vec);
    /// # }
    /// ```
    #[must_use]
    pub fn slice<Left, Right>(
        mut self,
        _: Range<Left, Right>,
    ) -> (
        Vec<Diff<N, Diff<Right, Left>>, A>,
        Vec<Diff<Right, Left>, A>,
    )
    where
        Diff<N, Diff<Right, Left>>: Unsigned,
        Diff<Right, Left>: Unsigned,
        Left: Unsigned + IsLessOrEqual<Right>,
        Right: Unsigned + Sub<Left> + IsLessOrEqual<N>,
        N: Sub<Diff<Right, Left>>,
        LeEq<Left, Right>: IsTrue,
        LeEq<Right, N>: IsTrue,
    {
        let removed = Vec::from_vec(self.vec.drain(Range::<Left, Right>::to_usize()).collect());
        (self.trust_me(), removed)
    }

    /// Remove a subset from the vector and return an iterator over the removed
    /// elements.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # use sized_vec::Range;
    /// # use typenum::{U2, U4};
    /// # fn main() {
    /// let vec = svec![1, 2, 3, 4, 5, 6];
    /// let (vec, iter) = vec.drain(Range::<U2, U4>::new());
    /// assert_eq!(svec![1, 2, 5, 6], vec);
    /// assert_eq!(vec![3, 4], iter.collect::<::std::vec::Vec<_>>());
    /// # }
    /// ```
    #[must_use]
    pub fn drain<Left, Right>(
        self,
        range: Range<Left, Right>,
    ) -> (Vec<Diff<N, Diff<Right, Left>>, A>, impl Iterator<Item = A>)
    where
        Diff<N, Diff<Right, Left>>: Unsigned,
        Diff<Right, Left>: Unsigned,
        Left: Unsigned + IsLessOrEqual<Right>,
        Right: Unsigned + Sub<Left> + IsLessOrEqual<N>,
        N: Sub<Diff<Right, Left>>,
        LeEq<Left, Right>: IsTrue,
        LeEq<Right, N>: IsTrue,
    {
        let (remainder, slice) = self.slice(range);
        (remainder.trust_me(), slice.into_iter())
    }

    /// Drop the contents of the vector, leaving an empty vector.
    #[must_use]
    pub fn clear(mut self) -> Vec<U0, A> {
        self.vec.clear();
        self.trust_me()
    }

    /// Split the vector at `Index`, returning the two sides of the split.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # use typenum::U3;
    /// # fn main() {
    /// let vec = svec![1, 2, 3, 4, 5, 6];
    /// let (left, right) = vec.split_off(U3::new());
    /// assert_eq!(svec![1, 2, 3], left);
    /// assert_eq!(svec![4, 5, 6], right);
    /// # }
    /// ```
    #[must_use]
    pub fn split_off<Index>(mut self, _: Index) -> (Vec<Index, A>, Vec<Diff<N, Index>, A>)
    where
        Index: Unsigned + IsLessOrEqual<N>,
        N: Sub<Index>,
        Diff<N, Index>: Unsigned,
        LeEq<Index, N>: IsTrue,
    {
        let split = self.vec.split_off(Index::USIZE);
        (self.trust_me(), Vec::from_vec(split))
    }

    /// Resize the vector, dropping elements if the new size is smaller,
    /// and padding with copies of `value` if it is larger.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # use sized_vec::Vec;
    /// # use typenum::U5;
    /// # fn main() {
    /// let vec = svec![1, 2, 3];
    /// let vec: Vec<U5, _> = vec.resize(100);
    /// assert_eq!(svec![1, 2, 3, 100, 100], vec);
    /// # }
    /// ```
    #[must_use]
    pub fn resize<M>(mut self, value: A) -> Vec<M, A>
    where
        M: Unsigned,
        A: Clone,
    {
        self.vec.resize(M::USIZE, value);
        self.trust_me()
    }

    /// Map the vector into a vector of elements of `B` using a function.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # fn main() {
    /// let vec = svec![1, 2, 3];
    /// let vec = vec.map(|num| format!("{}", num));
    /// assert_eq!(svec![
    ///     "1".to_string(), "2".to_string(), "3".to_string()
    /// ], vec);
    /// # }
    /// ```
    #[must_use]
    pub fn map<F, B>(self, f: F) -> Vec<N, B>
    where
        F: FnMut(A) -> B,
    {
        Vec::from_vec(self.into_iter().map(f).collect())
    }

    /// Apply a list of functions from `A` to `B` to a vector of `A` in order,
    /// returning a vector of `B`.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # fn main() {
    /// let vec = svec![1, 2, 3];
    /// let fn_vec = vec.clone().map(|i| move |a| a + i);
    /// let vec = vec.apply(fn_vec);
    /// assert_eq!(svec![2, 4, 6], vec);
    /// # }
    /// ```
    #[must_use]
    pub fn apply<F, B>(self, fs: Vec<N, F>) -> Vec<N, B>
    where
        F: FnMut(A) -> B,
    {
        Vec::from_vec(
            self.into_iter()
                .zip(fs.into_iter())
                .map(|(a, mut f)| f(a))
                .collect(),
        )
    }

    /// Merge two vectors together into a new vector using a function.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # fn main() {
    /// let left = svec!["foo", "bar"];
    /// let right = svec!["lol", "omg"];
    /// let vec = left.zip(right, |a, b| format!("{} {}", a, b));
    /// assert_eq!(svec![
    ///     "foo lol".to_string(), "bar omg".to_string()
    /// ], vec);
    /// # }
    /// ```
    #[must_use]
    pub fn zip<B, C, F>(self, other: Vec<N, B>, mut f: F) -> Vec<N, C>
    where
        F: FnMut(A, B) -> C,
    {
        Vec::from_vec(
            self.into_iter()
                .zip(other.into_iter())
                .map(|(a, b)| f(a, b))
                .collect(),
        )
    }

    /// Split a vector into two vectors of the same size using a function.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # fn main() {
    /// let vec = svec![1, 2, 3];
    /// let vec = vec.unzip(|a| (a, a * 2));
    /// assert_eq!((svec![1, 2, 3], svec![2, 4, 6]), vec);
    /// # }
    /// ```
    #[must_use]
    pub fn unzip<B, C, F>(self, f: F) -> (Vec<N, B>, Vec<N, C>)
    where
        F: FnMut(A) -> (B, C),
    {
        let (left, right) = self.into_iter().map(f).unzip();
        (Vec::from_vec(left), Vec::from_vec(right))
    }
}

impl<A> Default for Vec<U0, A> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<N, A> Debug for Vec<N, A>
where
    N: Unsigned,
    A: Debug,
{
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        self.vec.fmt(f)
    }
}

impl<N, M, A> Add<Vec<M, A>> for Vec<N, A>
where
    N: Unsigned + Add<M>,
    M: Unsigned,
    Sum<N, M>: Unsigned,
{
    type Output = Vec<Sum<N, M>, A>;

    fn add(self, other: Vec<M, A>) -> Self::Output {
        self.append(other)
    }
}

impl<N, A> Into<::std::vec::Vec<A>> for Vec<N, A>
where
    N: Unsigned,
{
    fn into(self) -> ::std::vec::Vec<A> {
        self.vec
    }
}

impl<'a, N, A> Into<&'a [A]> for &'a Vec<N, A>
where
    N: Unsigned,
{
    fn into(self) -> &'a [A] {
        &self.vec
    }
}

impl<'a, N, A> Into<&'a mut [A]> for &'a mut Vec<N, A>
where
    N: Unsigned,
{
    fn into(self) -> &'a mut [A] {
        &mut self.vec
    }
}

impl<N, A> IntoIterator for Vec<N, A>
where
    N: Unsigned,
{
    type Item = A;
    type IntoIter = IntoIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.into_iter()
    }
}

impl<'a, N, A> IntoIterator for &'a Vec<N, A>
where
    N: Unsigned,
{
    type Item = &'a A;
    type IntoIter = Iter<'a, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.iter()
    }
}

impl<'a, N, A> IntoIterator for &'a mut Vec<N, A>
where
    N: Unsigned,
{
    type Item = &'a mut A;
    type IntoIter = IterMut<'a, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.iter_mut()
    }
}

impl<N, A> Borrow<[A]> for Vec<N, A>
where
    N: Unsigned,
{
    fn borrow(&self) -> &[A] {
        self.as_slice()
    }
}

impl<N, A> Borrow<::std::vec::Vec<A>> for Vec<N, A>
where
    N: Unsigned,
{
    fn borrow(&self) -> &::std::vec::Vec<A> {
        &self.vec
    }
}

impl<N, A> BorrowMut<[A]> for Vec<N, A>
where
    N: Unsigned,
{
    fn borrow_mut(&mut self) -> &mut [A] {
        self.as_mut_slice()
    }
}

impl<N, A, I> Index<I> for Vec<N, A>
where
    N: Unsigned,
    I: Unsigned + IsLess<N>,
    Le<I, N>: IsTrue,
{
    type Output = A;
    fn index(&self, index: I) -> &Self::Output {
        self.get(index)
    }
}

impl<N, A, I> IndexMut<I> for Vec<N, A>
where
    N: Unsigned,
    I: Unsigned + IsLess<N>,
    Le<I, N>: IsTrue,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.get_mut(index)
    }
}

impl<N, A> AsRef<[A]> for Vec<N, A>
where
    N: Unsigned,
{
    fn as_ref(&self) -> &[A] {
        self.as_slice()
    }
}

impl<N, A> AsRef<Vec<N, A>> for Vec<N, A>
where
    N: Unsigned,
{
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<N, A> AsRef<::std::vec::Vec<A>> for Vec<N, A>
where
    N: Unsigned,
{
    fn as_ref(&self) -> &::std::vec::Vec<A> {
        &self.vec
    }
}

impl<N, A> AsMut<[A]> for Vec<N, A>
where
    N: Unsigned,
{
    fn as_mut(&mut self) -> &mut [A] {
        self.as_mut_slice()
    }
}

impl<N, A> AsMut<Vec<N, A>> for Vec<N, A>
where
    N: Unsigned,
{
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

impl<N, A> TryFrom<std::vec::Vec<A>> for Vec<N, A>
where
    N: Unsigned,
{
    type Error = ();

    /// Construct a vector of size `N` from a `std::vec::Vec`.
    ///
    /// Returns `Err(())` if the source vector didn't contain exactly `N` elements.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # use std::convert::TryFrom;
    /// # use sized_vec::Vec;
    /// # use typenum::U3;
    /// # fn main() {
    /// let good_vec: Result<Vec<U3, _>, _> = Vec::try_from(vec![1, 2, 3]);
    /// assert_eq!(Ok(svec![1, 2, 3]), good_vec);
    ///
    /// let bad_vec: Result<Vec<U3, _>, _> = Vec::try_from(vec![1, 2]);
    /// assert_eq!(Err(()), bad_vec);
    /// # }
    /// ```
    #[must_use]
    fn try_from(vec: ::std::vec::Vec<A>) -> Result<Self, Self::Error> {
        if vec.len() == N::USIZE {
            Ok(Vec::from_vec(vec))
        } else {
            Err(())
        }
    }
}

impl<N, A> TryFrom<Box<[A]>> for Vec<N, A>
where
    N: Unsigned,
{
    type Error = ();

    /// Construct a vector of size `N` from a boxed array.
    ///
    /// Returns `Err(())` if the source vector didn't contain exactly `N` elements.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # use std::convert::TryFrom;
    /// # use sized_vec::Vec;
    /// # use typenum::U3;
    /// # fn main() {
    /// let boxed: Box<[_]> = Box::new([1, 2, 3]);
    /// let good_vec: Result<Vec<U3, _>, _> = Vec::try_from(boxed);
    /// assert_eq!(Ok(svec![1, 2, 3]), good_vec);
    ///
    /// let boxed: Box<[_]> = Box::new([1, 2]);
    /// let bad_vec: Result<Vec<U3, _>, _> = Vec::try_from(boxed);
    /// assert_eq!(Err(()), bad_vec);
    /// # }
    /// ```
    #[must_use]
    fn try_from(array: Box<[A]>) -> Result<Self, Self::Error> {
        if array.len() == N::USIZE {
            Ok(Vec::from_vec(array.into_vec()))
        } else {
            Err(())
        }
    }
}

impl<'a, N, A> TryFrom<&'a [A]> for Vec<N, A>
where
    A: Clone,
    N: Unsigned,
{
    type Error = ();

    /// Construct a vector of size `N` from a slice of clonable values.
    ///
    /// Returns `Err(())` if the source slice didn't contain exactly `N` elements.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # extern crate typenum;
    /// # use std::convert::TryFrom;
    /// # use sized_vec::Vec;
    /// # use typenum::U3;
    /// # fn main() {
    /// let good_vec: Result<Vec<U3, _>, _> = Vec::try_from([1, 2, 3].as_ref());
    /// assert_eq!(Ok(svec![1, 2, 3]), good_vec);
    ///
    /// let bad_vec: Result<Vec<U3, _>, _> = Vec::try_from([1, 2].as_ref());
    /// assert_eq!(Err(()), bad_vec);
    /// # }
    /// ```
    #[must_use]
    fn try_from(slice: &'a [A]) -> Result<Self, Self::Error> {
        Vec::try_from_iter(slice.iter().cloned()).ok_or(())
    }
}

macro_rules! declare_from_array {
    ($s1:expr, $s2:ty) => {
        impl<A> From<[A; $s1]> for Vec<$s2, A> {
            #[must_use]
            fn from(array: [A; $s1]) -> Self {
                let boxed_array: Box<[A]> = Box::new(array);
                let vec = boxed_array.into_vec();
                debug_assert_eq!($s1, vec.len());
                Vec::from_vec(vec)
            }
        }
    };
}

declare_from_array!(0, U0);
declare_from_array!(1, U1);
declare_from_array!(2, U2);
declare_from_array!(3, U3);
declare_from_array!(4, U4);
declare_from_array!(5, U5);
declare_from_array!(6, U6);
declare_from_array!(7, U7);
declare_from_array!(8, U8);
declare_from_array!(9, U9);
declare_from_array!(10, U10);
declare_from_array!(11, U11);
declare_from_array!(12, U12);
declare_from_array!(13, U13);
declare_from_array!(14, U14);
declare_from_array!(15, U15);
declare_from_array!(16, U16);
declare_from_array!(17, U17);
declare_from_array!(18, U18);
declare_from_array!(19, U19);
declare_from_array!(20, U20);
declare_from_array!(21, U21);
declare_from_array!(22, U22);
declare_from_array!(23, U23);
declare_from_array!(24, U24);
declare_from_array!(25, U25);
declare_from_array!(26, U26);
declare_from_array!(27, U27);
declare_from_array!(28, U28);
declare_from_array!(29, U29);
declare_from_array!(30, U30);
declare_from_array!(31, U31);

#[cfg(feature = "generic-array")]
impl<N, A> From<generic_array::GenericArray<A, N>> for Vec<N, A>
where
    N: Unsigned + generic_array::ArrayLength<A>,
{
    /// Construct a vector of size `N` from a `GenericArray`.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate sized_vec;
    /// # use std::convert::TryFrom;
    /// # use sized_vec::Vec;
    /// # use typenum::U3;
    /// # use generic_array::GenericArray;
    /// # fn main() {
    /// let array = GenericArray::from([1, 2, 3]);
    /// let good_vec: Vec<U3, _> = Vec::from(array);
    /// assert_eq!(svec![1, 2, 3], good_vec);
    /// # }
    /// ```
    ///
    /// ```compile_fail
    /// # #[macro_use] extern crate sized_vec;
    /// # use std::convert::TryFrom;
    /// # use sized_vec::Vec;
    /// # use typenum::U3;
    /// # use generic_array::GenericArray;
    /// # fn main() {
    /// let array = GenericArray::from([1, 2]);
    /// let bad_vec: Vec<U3, _> = Vec::from(array);
    /// # }
    /// ```
    #[must_use]
    fn from(array: generic_array::GenericArray<A, N>) -> Self {
        Vec::from_vec(array.into_iter().collect())
    }
}

#[cfg(feature = "serde")]
mod ser {
    use super::*;
    use serde::de::{Deserialize, Deserializer, Error};
    use serde::ser::{Serialize, Serializer};

    impl<N, A> Serialize for Vec<N, A>
    where
        N: Unsigned,
        A: Serialize,
    {
        fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            self.vec.serialize(ser)
        }
    }

    impl<'de, N, A> Deserialize<'de> for Vec<N, A>
    where
        N: Unsigned,
        A: Deserialize<'de>,
    {
        fn deserialize<D>(des: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let vec: std::vec::Vec<_> = Deserialize::deserialize(des)?;
            Self::try_from(vec).map_err(|()| {
                <D as Deserializer<'de>>::Error::custom(format!(
                    "length of sized_vec::Vec was not {}",
                    N::USIZE
                ))
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::*;
        use serde_json::{from_str, to_string};

        #[test]
        fn serialise() {
            let v = svec![1, 2, 3, 4, 5];
            assert_eq!(v, from_str(&to_string(&v).unwrap()).unwrap());
        }
    }
}

#[cfg(feature = "proptest")]
pub mod proptest {
    use super::*;
    use ::proptest::collection::vec as stdvec;
    use ::proptest::strategy::{BoxedStrategy, Strategy, ValueTree};

    /// A strategy for generating a sized vector.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// proptest! {
    ///     #[test]
    ///     fn proptest_a_vector(ref vec in sized_vec::<U16, _>(".*")) {
    ///         assert_eq!(16, vec.len());
    ///     }
    /// }
    /// ```
    pub fn sized_vec<N, A>(element: A) -> BoxedStrategy<Vec<N, <A::Tree as ValueTree>::Value>>
    where
        N: Unsigned + 'static,
        A: Strategy + 'static,
        <A::Tree as ValueTree>::Value: Clone,
    {
        stdvec(element, N::USIZE).prop_map(Vec::from_vec).boxed()
    }

    #[cfg(test)]
    mod tests {
        use super::proptest::sized_vec;
        use crate::*;
        use ::proptest::proptest;

        proptest! {
            #[test]
            fn test_the_proptest(ref vec in sized_vec::<U16, _>(".*")) {
                assert_eq!(16, vec.len())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn basics() {
        let v = svec![1, 2, 3];
        assert_eq!(U3::USIZE, v.len());
        assert_eq!(&2, v.get(U1::new()));
        assert_eq!((svec![1, 2], 3), v.pop());
        let v1 = svec![1, 2, 3];
        let v2 = svec![4, 5, 6];
        let v3 = v1.append(v2);
        assert_eq!(6, v3.len());
        assert_eq!(svec![1, 2, 3, 4, 5, 6], v3);
        let v4: Vec<U4, _> = v3.truncate();
        assert_eq!(4, v4.len());
        assert_eq!(svec![1, 2, 3, 4,], v4);
        assert_eq!(2, v4[U1::new()]);
    }

    #[test]
    fn static_array_conversion() {
        let v = Vec::from([1, 2, 3, 4, 5]);
        assert_eq!(svec![1, 2, 3, 4, 5], v);
    }
}
