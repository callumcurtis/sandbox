# Notes

Based on progress through the [Rust Book](https://doc.rust-lang.org/book/).

## Style

- filenames: snakecase
- indents: four spaces

## Cargo

Rust's build system and package manager. Adds package management
and better support for multi-file projects compared to `rustc`.

```shell
# build
cargo build

# build with performance optimizations
cargo build --release

# build and run
cargo run

# check compilation without producing artifact
cargo check
```

