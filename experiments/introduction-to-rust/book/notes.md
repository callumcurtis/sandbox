# Notes

Based on progress through the [Rust Book](https://doc.rust-lang.org/book/).

Rust is strongly typed with type inference.

## Primitives

- Unless otherwise specified, Rust defaults to an `i32`

## Scope

- Shadowing is supported - can redeclare local variables, e.g., with different mutability and/or types

## String Formatting

- `println!("x = {x} and y + 2 = {}", y + 2);`
- Variables in `{}`, expressions after string

## Range Expressions

- `1..=10` is inclusive
- `1..10` is exclusive

## Enums

- Enum states are called variants
- `Ok`/`Err` as `Return` states for error handling

## Match

- `match` expression has arms
- An arm is of the form `<pattern> => <logic>`
- Each arm is checked in order
- An underscore in a variant acts as a wildcard

## Style

- filenames: snake_case
- indents: four spaces
- variables
    - local: snake_case

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

# open documentation (including crates) in browser
cargo doc --open
```

- Allows installing crates - external source code files

