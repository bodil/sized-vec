# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic
Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

The minimum compatible `rustc` version for this crate is now 1.34.0.

### Changed

- `Vec::try_from` has been renamed to `Vec::try_from_iter` in honour of
  `std::convert::TryFrom` stabilising.

### Removed

- `Vec::try_from_vec` has been removed, in favour of the equivalent `TryFrom`
  implementation.

### Added

- `Vec::from_default()` constructor.
- `Into` implementation for converting into mutable slices.
- `AsRef` and `Borrow` implementations for getting a reference to the wrapped
  `std::vec::Vec`.
- `Vec` now implements `std::convert::TryFrom` for `std::vec::Vec<A>`,
  `Box<[A]>` and `&[A] where A: Clone`.
- There are now `From` implementations for fixed arrays `[A; n]` for any `n`
  from 0 to 32 exclusive.
- There is a `From` implementation for `GenericArray<A, N>` from the
  [`generic-array`](https://crates.io/crates/generic-array) behind the
  `generic-array` feature flag.

## [0.2.2] - 2019-02-11

### Added

- Added docs for `apply`.

## [0.2.1] - 2019-01-05

### Added

- `repeat` and `fill` constructors.
- An `apply` implementation for the category theorists.

### Fixed

- `zip` and `unzip` will now correctly give warnings if you don't use their
  results.

## [0.2.0] - 2018-08-15

### Added
- `try_from` and `try_from_vec` constructors to safely convert from an iterator
  or `std::vec::Vec`.
- `Into::into` implementations for converting to slices and `std::vec::Vec`.
- Serde `Serialize` and `Deserialize` implementations behind the `serde` feature
  flag.
- A `sized_vec` Proptest strategy behind the `proptest` feature flag.

## [0.1.0] - 2018-08-04

- Initial release.
