/*

$ head --bytes=1G ../../corpus.txt | time cargo run --profile=release
    Finished `release` profile [optimized] target(s) in 0.00s
     Running `target/release/two-time-pad`

At 1 char:      4.287564892914652
At 2 chars:     3.9594148396746167
At 3 chars:     3.6839407592906888
At 4 chars:     3.4401139264518545
At 5 chars:     3.2246336739770243
At 10 chars:    2.453036215709043
At 12 chars:    2.2121792326747234

Compare the amount of information we get from 46 sigils, per character.  We need to go lower than half that.

>>> from math import *
>>> log(46, 2)
5.523561956057013
>>> log(46, 2) / 2
2.7617809780285065

Hmm, I think we are overestimating the entropy: if we haven't seen part of the item, we pretend to be completely
agnostic about the whole thing.  Even though we can guess that the rest of the item probably still follows basic
probalitiies for shorter items.  (And our neural net would probably figure that out, too.)

This over-estimate gets worse with longer windows.

In any case, 2.2121792326747234 (at 12 chars) is smaller than 2.7617809780285065

--- Hmm:

$ head --bytes=1G /dev/urandom | time cargo run --profile=release
    Finished `release` profile [optimized] target(s) in 0.00s
     Running `target/release/two-time-pad`
10 2.752359639782636

---

OK, the calculation is all wrong, at least for the longer sequences.

*/

use std::collections::HashMap;
use std::io;
use std::io::prelude::*;

use itertools::Itertools;

// Consider binary search?
pub static ALPHA: &str = " abcdefghijklmnopqrstuvwxyz0123456789.?,-:;'()";
type Key = (u8, u8, u8, u8, u8,
            u8, u8, u8, u8, u8);
const LEN: f64 = 10.0;

pub fn entropy<K>(h: &HashMap<K, usize>) -> f64 {
    let total: f64 = h.values().sum::<usize>() as f64;
    h.iter()
        .map(|(_, count)| {
            let p = *count as f64 / total;
            -p * p.log2()
        })
        .sum()
}

fn main() {
    ctrlc::set_handler(move || {
        println!("received Ctrl+C!");
    })
    .expect("Error setting Ctrl-C handler");

    let stdin = io::stdin();
    let h: HashMap<Key, _> = stdin
        .bytes()
        .filter_map(Result::ok)
        .filter_map(|byte| ALPHA.find(byte as char).map(|u| u as u8))
        .tuple_windows()
        .counts();
    let h = entropy(&h) / LEN;
    println!("{LEN} {h:?}");
}
