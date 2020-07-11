import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("changed_bits.csv")
print(df.nbits.max())

df = df[df.nbits>0]

plt.figure()
plt.hist(df.nbits,bins=100)
plt.yscale('log')
plt.show()