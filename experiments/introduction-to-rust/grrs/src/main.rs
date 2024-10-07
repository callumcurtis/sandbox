use clap::Parser;

// Search for a pattern in a file and display the lines that contain it.
#[derive(Parser)]
struct Cli {
    // The pattern to search for
    pattern: String,
    // The path to the file to read
    path: std::path::PathBuf,
}

fn main() {
    let args = Cli::parse();

    let result = std::fs::read_to_string(&args.path);
    let content = match result {
        Ok(c) => { c },
        Err(_e) => { panic!("failed to read {}", &args.path.display()); }
    };
    println!("content={}", content);
}
