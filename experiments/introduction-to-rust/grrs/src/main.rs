use std::env::args;

fn main() {
    let pattern = args().nth(1).expect("no pattern given");
    let path = args().nth(2).expect("no path given");

    println!("pattern: {:?}, path: {:?}", pattern, path)
}
