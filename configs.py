import itertools
import json
import numpy as np

indices = np.arange(28)
r = 2
for r in range(2, 28):
    combs = list(itertools.combinations(indices, r))

    print(len(combs))