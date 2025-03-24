use clap::Parser;
use clap_derive::{Parser, Subcommand};
use regex::Regex;
use std::io::{self, BufWriter, Read, Write};

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
}

const ALPHABET: &str = " abcdefghijklmnopqrstuvwxyz0123456789.?,-:;'()";

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::ToIndices => convert_to_indices(),
        Commands::ToText => convert_to_text(),
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

    // Stage 1: Clean up text using regex
    let whitespace_regex = Regex::new(r"\s+").unwrap();
    let invalid_char_regex = Regex::new(r"[^a-z0-9.?,-:;'() ]").unwrap();

    let content = content.to_lowercase();
    let content = whitespace_regex.replace_all(&content, " ");
    let content = invalid_char_regex.replace_all(&content, "");

    // Stage 2: Convert to indices
    let mut writer = BufWriter::new(stdout.lock());
    for c in content.chars() {
        if let Some(idx) = ALPHABET.find(c) {
            writer
                .write_all(&[idx as u8])
                .expect("Failed to write to stdout");
        }
    }
}

fn convert_to_text() {
    let stdin = io::stdin();
    let stdout = io::stdout();

    let mut bytes = Vec::new();
    stdin
        .lock()
        .read_to_end(&mut bytes)
        .expect("Failed to read from stdin");

    let mut writer = BufWriter::new(stdout.lock());
    for &idx in &bytes {
        if (idx as usize) < ALPHABET.len() {
            let c = ALPHABET.chars().nth(idx as usize).unwrap();
            write!(writer, "{}", c).expect("Failed to write to stdout");
        }
    }
    writer.flush().expect("Failed to flush stdout");
}
