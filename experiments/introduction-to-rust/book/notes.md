# Notes

Based on progress through the [Rust Book](https://doc.rust-lang.org/book/).

## Data Types

- Rust is statically typed with type inference
- Type must be specified when ambiguous (e.g., `String::parse`)
- Primitives: placed on the stack
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

- Char literals surrounded in single quotes
- Four bytes
- Unicode

## Tuples

- Fixed length
- Heterogeneous types
- Definition: `let tup = (500, 6.4, 1);`
- Destructuring assignment: `let (x, y, z) = tup;`
- Element access: `<tuple>.<index>`; e.g., `tup.0`
- Unit type: tuple without any values; `()`; represents an empty value or empty return type

## Arrays

- Fixed length
- Homogeneous types
- Definition: `let a = [1, 2, 3, 4, 5];`
- Typing: `[<type>; <size>]`; e.g., `let a: [i32; 5] = [1, 2, 3, 4, 5];`
- Repeated value definition: `[<value>; <size>]`; e.g., `let a = [3; 5];` <-> `let a = [3, 3, 3, 3, 3];`
- Indexing: `<array>[<index>]`
- OOB errors at runtime (panic)

## String Literals

- Surrounded in double quotes
- Slice in the binary

## Scope Block

- Example:

  ```rust
  let y = {
      let x = 3;
      x + 1 // missing semicolon is required (expression, not statement)
  };
  ```

## Statements

- Instructions that perform some action and **do not** return a value
- Examples: assignment (`x = y = 6` is invalid syntax), function definition

## Expressions

- Evaluate to a resultant value
- Do not end with semicolon; if a semicolon is added, becomes a statement
- Examples: function calling, arithmetic, scope block

## Conditionals

- `if <condition> { ... } else if <condition> { ... } else { ... }`
- Prefer `match` if many `else if` cases
- Is an expression; can use with statements: `let number = if condition { 5 } else { 6 };`

## Looping

- Three constructs: `loop`, `while`, `for`
- `break` and `continue`
- `loop`: infinite until `break`
- Can pass expression to `break` to return value from `loop`
- Loop label: allows `break` and `continue` to apply to any loop in the hierarchy (instead of innermost)
  - Example: `'counting_up: loop { ... }`
  - Begins with single quote
- `while <condition> { ... }`
- `for <identifier> in <iterable> { ... }`

## Functions

- Definition: `fn <name>([<parameter_name>: <parameter_type>[, <parameter_name>: <parameter_type>[...]]]) [-> <return_type>] { ... }`
- Insensitive to function declaration order; can reference earler/later functions
- Implicitly return the final expression in the body; can `return` early/explicitly
- Return the unit type by default

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
- Reversing: `(<range_expression>).rev()`
- Can omit start/end to use full range: `..5`, `3..`, `..`

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
- Functions: snake_case

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

## Ownership

- Rules that govern how memory in the heap is managed
- Rules are checked at compile time; non-compliance leads to compilation error
- No runtime overhead
- Rules:
  - R1: each value has an owner
  - R2: there can only be one owner at a time
  - R3: when the owner goes out of scope, memory for the value is deallocated (RAII)
- Corollaries:
  - C1: if a value is orphaned (e.g., after reassignment), it is immediately deallocated
- `drop` is responsible for deallocating memory; called when value leaves scope
- Move: copies data on the stack and claims data on the heap; implicit
  - Other references become invalid (R2)
- Clone: if implemented, clones rather than moves data on the heap; explicit `clone` required
- Copy: if implemented; all data must be on the stack; mutually exclusive with `drop`; implicit
  - All scalar primitives are copyable
  - All compound types (including primitives) with copyable elements are also copyable
- Values are deallocated in reverse order of allocation
- Functions: arguments and return values are copied or moved
  - If moved, the argument moves into the function's scope and is no longer valid from the caller
- References (borrowing): does not transfer ownership; immutable by default; `&`, `&mut`, and `*`
  - Alternative to taking and returning ownership in functions
  - Either one mutable reference (prevents data races) or any number of immutable references
  - Reference's scope ends at last use; references' scopes may not overlap, even in the same block
- Slice: **reference** to a contiguous sequence of elements in a collection (e.g., `&s[0..5]`)
  - Internally is a reference to the first element and a length

## Structs

- Definition: `struct <name> { [<field_name>: <field_type>[, <field_name>: <field_type>[...]]] }`
- Instantiation: `let user1 = User { active: true, ... };`
- Field shorthand: `let user1 = User { email, ... };` (where `email` is an in-scope variable)
- Struct update: `let user2 = User { active: false, ..user1 };`
  - Moves/copies remaining fields from source instance; beware invalidation of moved fields
- Access: `user1.email`
- Individual fields can be mutated iff the entire instance is mutable
- Tuple struct: distinct type name; otherwise equivalent to tuple; e.g., `struct AlwaysEqual;`
  - Useful when implementing a trait on a struct without fields
- Fields must be owned by the instance or be references with lifetimes

## Methods

- `impl`emented on structs, enums, and traits

  ```rust
  #[derive(Debug)]
  struct Rectangle {
      width: u32,
      height: u32,
  }

  impl Rectangle {
      fn area(&self) -> u32 {
          self.width * self.height
      }
  }
  ```

- Invocation: `rectangle.area()`
- First parameter `&self`; shorthand for `self: &Self`; `Self` aliases the type applied to `impl`
- Method can take ownership of `self`, borrow it mutably, or borrow it immutably
- Method names can duplicate field names; compiler differentiates based on parentheses in usage
- Compiler uses signature of `self` for "automatic referencing and dereferencing" (i.e., no `->`)
- Associated functions: defined together in `impl` blocks for a given type
- Associated functions that are not methods do not accept `self` as first parameter
  - Often used for constructors; `new` by **convention**
  - Invocation: `Rectangle::new()`
- Multiple `impl` blocks may exist for the same type

## Attributes

- `#[derive(Debug)]` enables `{:?}` and `dbg!` formatting for structs

## Debugging

- `dbg!` macro wraps an expression, prints the result to stderr, and returns ownership of the result

