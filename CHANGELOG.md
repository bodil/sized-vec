# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic
Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `repeat` and `fill` constructors.
- An `apply` implementation for the category theorists.

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
