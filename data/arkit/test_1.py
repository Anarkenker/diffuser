import numpy as np
arr = np.load("arkit_demo.npy")  # 路径按你的来
labels = arr[:, 3]
uniq, cnt = np.unique(labels, return_counts=True)
print(dict(zip(uniq, cnt)))
print("label==1 点数:", (labels == 1).sum())
