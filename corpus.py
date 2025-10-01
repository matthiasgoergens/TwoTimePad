from collections import Counter
import string
import re
from pprint import pprint

text = open('corpus.txt','r').read().lower()
keep=string.ascii_lowercase + r"'\-"

clean = re.sub(f"[^{keep}]", " ", 
    text.replace('--', ' '))

words = clean.split()
c = Counter(words)
print(len(words))
print(len(c))
pprint(c)
