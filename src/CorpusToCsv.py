import sys
import random
from itertools import islice, izip, chain

# Should be: output/corpus
fn = sys.argv[1]
print fn

corpus = file (fn,'r').read()
print len(corpus)

alpha = " abcdefghijklmnopqrstuvwxyz0123456789.?,-:;'()"

# b - a

def chr_ (i):
    return alpha[i]
def ord_ (a):
    return alpha.index(a)
def onehot (i):
  m = len(alpha) * [0]
  m[i] = 1
  return m
def subtract(a, b):
  ia = alpha.index(a)
  ib = alpha.index(b)
  return (ib - ia) % len(alpha)


def samples(w, corpus):
  while True:
    start = random.randrange(len(corpus)-w)
    yield corpus[start : start + w]

def gen_sample(w, a, b, corpus):
    aw = corpus[a:a+w]
    corpus[b:b+w]

width = 31
sl = samples(width, corpus)
sr = samples(width, corpus)

# Just a test:
x = '7'
y = 'x'
assert (alpha[subtract(alpha[subtract(x, y)], y)] == x)

for a, b in izip (sl, sr):
  # print repr(a), repr(b),
  print ','.join([a[width // 2]] + map (str, list(chain(*[onehot(subtract(ax, bx)) for ax, bx in izip(a, b)]))))
