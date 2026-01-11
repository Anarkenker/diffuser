import numpy as np
from PIL import Image
m = np.array(Image.open("labels.png"))
print("labels.png shape:", m.shape, "unique:", np.unique(m), "sum:", m.sum())
