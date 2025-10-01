use clap::Parser;
use clap_derive::{Args, Parser, Subcommand};
use memmap2::Mmap;
use rand::Rng;
use regex::Regex;
use std::{
    fs::File,
    io::{self, BufWriter, Read, Write},
    path::PathBuf,
};

/*
TODO:
Read from stdin and write to stdout.

We want to:
- read utf8 from stdin (skip past any invalid utf8).
- replace all whitespace with space.
- convert all uppercase to lowercase.
- remove all characters not in the alphabet " abcdefghijklmnopqrstuvwxyz0123456789.?,-:;'()"
- convert multiple spaces to a single space.
- replace each character with its corresponding index in the alphabet as u8.
- write the result to stdout as bytes.

Note: our intended input is about around 17GiB.  But we only need to do this conversion once per input.

We also want to add a reverse operation, where we convert from indices (as bytes) into che characters in the alphabet.
(We can't undo the multiple spaces conversion nor the lower casing etc, but that's fine.  I just want it to be readable.)

Let's use clap for the command line switch to choose between the two modes.
*/

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert text to indices
    ToIndices,
    /// Convert indices to text
    ToText,
    /// Generate snippets of specific size.
    GenerateSnippets(SnippetOptions),
}

#[derive(Args)]
struct SnippetOptions {
    /// Size of the snippets to generate
    #[arg(value_parser)]
    size: usize,
    #[arg(value_parser)]
    path: PathBuf,
}

// const ALPHABET: &str = " abcdefghijklmnopqrstuvwxyz0123456789.?,-:;'()";
const ALPHABET: &[char; 46] = &[
    ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
    's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.',
    '?', ',', '-', ':', ';', '\'', '(', ')',
];

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::ToIndices => convert_to_indices(),
        Commands::ToText => convert_to_text(),
        Commands::GenerateSnippets(SnippetOptions { path, size }) => generate_snippets(size, path),
    }
}

/// Map the file at path into a buffer, then keep picking random snippets of size `size` from it, and write them to stdout.
fn generate_snippets(size: usize, path: PathBuf) {
    let file = File::open(path).expect("Failed to open file");

    let mmap = unsafe { Mmap::map(&file).expect("Failed to map file") };

    let mut rng = rand::rng();
    let mut stdout = io::stdout();
    let mut buf_writer = BufWriter::with_capacity(1024 * 1024, &mut stdout);
    let file_size = mmap.len();
    loop {
        let start = rng.random_range(0..file_size - size);
        let snippet = &mmap[start..][..size];

        // Handle broken pipe errors gracefully
        if let Err(e) = buf_writer.write_all(snippet) {
            if e.kind() == io::ErrorKind::BrokenPipe {
                // Pipe closed by the reader - exit gracefully
                return;
            }
            // For other errors, still panic
            panic!("Failed to write snippet to stdout: {}", e);
        }
    }
}

fn convert_to_indices() {
    let stdin = io::stdin();
    let stdout = io::stdout();

    let mut content = String::new();

    stdin
        .lock()
        .read_to_string(&mut content)
        .expect("Failed to read from stdin");
    eprintln!("Read {} bytes", content.len());

    // Stage 1: Clean up text using regex
    // let whitespace_regex = Regex::new(r"\s+").unwrap();
    let whitespace_regex = Regex::new(r"\s").unwrap();
    // I'm a bit suspicous about using - both for character ranges and as a literal character.
    // But it seems to work?
    let invalid_char_regex = Regex::new(r"[^a-z0-9.?,-:;'() ]").unwrap();

    let content = content.to_lowercase();
    eprintln!("to_lowercase {} bytes", content.len());
    let content = whitespace_regex.replace_all(&content, " ");
    eprintln!("replace space {} bytes", content.len());
    let content = invalid_char_regex.replace_all(&content, "");
    eprintln!("remove invalid {} bytes", content.len());

    // Stage 2: Convert to indices
    // let mut writer = BufWriter::new(stdout.lock());
    stdout
        .lock()
        .write_all(
            &content
                .chars()
                .filter_map(|c| {
                    ALPHABET
                        .iter()
                        .position(|&alphabet_char| alphabet_char == c)
                })
                .map(|idx| idx as u8)
                .collect::<Vec<_>>(),
        )
        .expect("Failed to write to stdout");
}

fn convert_to_text() {
    let stdin = io::stdin();
    let stdout = io::stdout();

    let mut bytes = Vec::new();
    stdin
        .lock()
        .read_to_end(&mut bytes)
        .expect("Failed to read from stdin");

    stdout
        .lock()
        .write_all(
            bytes
                .into_iter()
                .filter_map(|idx| ALPHABET.get(idx as usize))
                .collect::<String>()
                .as_bytes(),
        )
        .expect("Failed to write to stdout");
}
