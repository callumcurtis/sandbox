# Notes

Based on progress through the [Rust Book](https://doc.rust-lang.org/book/).

## Data Types

- Rust is statically typed with type inference
- Type must be specified when ambiguous (e.g., `String::parse`)
- Scalar: single value; int, float, bool, char
- Compound: tuple, array

## Integers

- Two dimensions, size and sign: `<i/u><size>` (e.g., `i16`, `u64`)
- Signed ints are stored using 2's complement: `-2^(n-1)..=2^(n-1)-1`
- Int literals can have type suffix (e.g., `57u8`)
- Int literals can be provided in decimal, hex, octal, binary, and byte formats
- Truncation towards 0

## Integer Overflow

- If compiled in debug mode, panics at runtime if integer overflow
- When compiled in release mode, uses 2's complement wrapping
- If integer overflow is intended, use `wrapping_*` methods
- Check if overflow occurs with `checked_*` methods
- Return new value and overflow flag with `overflowing_*` methods
- Saturate min/max values with `saturating_*` methods

## Floats

- `f32` or `f64`
- `f64` by default, since, on most platforms, same performance
- IEEE-754 standard

## Booleans

- One byte

## Chars

- Char literals surrounded in single quotes (string literals in double quotes)
- Four bytes
- Unicode

## Tuples

- Fixed length
- Heterogeneous types
- Definition: `let tup = (500, 6.4, 1);`
- Destructuring assignment: `let (x, y, z) = tup;`
- Element access: `<tuple>.<index>`; e.g., `tup.0`
- Unit: tuple without any values; `()`; represents an empty value or empty return type

## Arrays

- Fixed length
- Homogeneous types
- Allocated on the stack
- Definition: `let a = [1, 2, 3, 4, 5];`
- Typing: `[<type>; <size>]`; e.g., `let a: [i32; 5] = [1, 2, 3, 4, 5];`
- Repeated value definition: `[<value>; <size>]`; e.g., `let a = [3; 5];` <-> `let a = [3, 3, 3, 3, 3];`
- Indexing: `<array>[<index>]`
- OOB errors at runtime (panic)

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

- Filenames: snake_case
- Indents: four spaces
- Local variables: snake_case
- Constants: UPPER_SNAKE_CASE

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

