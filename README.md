# sized-vec

Rust vectors with type level size.

## Documentation

* [API docs](https://docs.rs/sized-vec/)

## Examples

```rust
#[macro_use]
use sized_vec::Vec;
use typenum::{U2, U8};

fn main() {
    let vec = svec![1, 2, 3, 4, 5];
    // Use typenums for type safe index access:
    assert_eq!(3, vec[U2::new()]);
    // Out of bounds access won't even compile:
    assert_eq!(0, vec[U8::new()]); // <- type error!
}
```

## Licence

Copyright 2018 Bodil Stokke

This software is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

## Code of Conduct

Please note that this project is released with a [Contributor Code of
Conduct][coc]. By participating in this project you agree to abide by its
terms.

[coc]: ./CODE_OF_CONDUCT.md
