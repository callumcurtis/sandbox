# Notes

Based on progress through the [Rust Book](https://doc.rust-lang.org/book/).

Rust is strongly typed with type inference.

## Constants

- `const` keyword instead of `let`
- Always immutable (unlike variables)
- Must be declared with a type annotation (unlike variables)
- Expression must be determinable at compile-time
- Can be declared in any scope

## Mutability

- By default, all variables are immutable
- `mut` keyword makes mutable

## Primitives

- Unless otherwise specified, Rust defaults to an `i32`

## Shadowing

- Shadowing is supported - can redeclare variables in the same scope
- Can be used to change the type of a variable
- Can be used to change the mutability of a variable
- Can be used to "modify" an immutable variable by redefining it

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
- local variables: snake_case
- constants: UPPER_SNAKE_CASE

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

