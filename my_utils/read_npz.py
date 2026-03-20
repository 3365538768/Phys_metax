import sys
import numpy as np

file = sys.argv[1]
data = np.load(file)

print("keys:", data.files)

for k in data.files:
    arr = data[k]
    print(f"\n[{k}]")
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)
    if arr.size > 0:
        print("min:", arr.min())
        print("max:", arr.max())
        print("sample:", arr.reshape(-1)[:100])