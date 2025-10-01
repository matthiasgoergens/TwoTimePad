We have an alphabet of 46 characters.

Each character thus takes about 6 bits to store
```
>>> m.log(46, 2)
5.523561956057013
```

In a 64 bit word we can store 11 characters:
```
>>> 64 / m.log(46, 2)
11.586726193922573
>>> 11 * m.log(46, 2)
60.759181516627145
>>> 12 * m.log(46, 2)
66.28274347268416
```

But that requires arithmetic coding.

If we use a straight-forward encoding, we can store 10 characters at 6 bits each.
```
>>> 64 // m.ceil(m.log(46, 2))
10
```

Or even more straightforward: 8 characters at 8 bits each.  Start with that.

As a naive calculation, when we hack up our corpus, we are replacing each single position in it with a context window that's 8 times as long.

If we just stuck them in a Python `Counter` but in Rust, we'd have to make a trade-off in picking what datatype we use for the value to our key.  Need enough bits, but not waste them.

Alternatively, we could also just sort our context windows.  (Can a prefix-array do something for us here?)

Hmm, how to do the smoothing?

xxx tex_t <- is definitely better than
xxx xxx_x <- even though neither might appear in the corpus.

Using a neural network is an 'obvious' idea for how to do the smoothing.

We could also do bootstrapping to figure out the smoothing parameters?  Or something like Huffmann coding but for fragments?

We want to learn from a supposed real underlying model, but we only have samples to estimate its distribution.

How to recognise that sometimes the past isn't giving us information?  We need a robust bias!

last letter is most important, than second to last etc.  We assume that conditional probabilities change less and less (in the real, underlying model) as we broaden context.

We need to have a model for that decay, either fixed or learned.  And we need more training data to overcome our initial assumption.  Also, as we see more training data, we can be more confident moving away from our bias.

Let's do some math.
