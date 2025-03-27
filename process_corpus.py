# We need to be able to:
# Load the corpus and make it available.
# We need to move from utf8/ascii view to 'alpha' view [0,46)
# And in the other direction.

# For training, we only need one direction at first.

# OK, let's convert and store the conversion.

alpha = b" abcdefghijklmnopqrstuvwxyz0123456789.?,-:;'()"
alphaRE = alpha.replace(b"-", b"\\-")
assert len(alpha) == 46
