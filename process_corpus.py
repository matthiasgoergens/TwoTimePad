# We need to be able to:
# Load the corpus and make it available.
# We need to move from utf8/ascii view to 'alpha' view [0,46)
# And in the other direction.

# For training, we only need one direction at first.

# OK, let's convert and store the conversion.

alpha = " abcdefghijklmnopqrstuvwxyz0123456789.?,-:;'()"
alphaRE = alpha.replace("-", "\\-") 
assert len(alpha) == 46

def load(validation_split=1.0):
    # About 128 MB
    big_size = 128 * (1 << 20)
    size = round(validation_split * big_size)
    while True:
        f = open("corpus.txt", "r")
        if validation_split < 1.0:
            # Forward to make sure, validation stays ahead of training.
            f.read(big_size)
        while True:
            # text = ' '.join(f.open('r').read() for f in pathlib.Path('data').glob('*.txt')).lower()
            text = f.read(size).lower()
            print(f"Text: {repr(text[:100])} {validation_split}")
            if len(text) < size:
                break
            text = re.sub("\s+", " ", text)
            # text = re.sub(f'[^{alphaRE}]', '', text)
            text = re.sub("[^%s]" % alphaRE, "", text)
            yield text
            if validation_split < 1.0:
                # Forward to make sure, validation stays ahead of training.
                f.read(big_size)
