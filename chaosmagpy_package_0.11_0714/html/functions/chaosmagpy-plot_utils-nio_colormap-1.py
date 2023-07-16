import numpy as np
import chaosmagpy as cp
import matplotlib.pyplot as plt

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

figh = 0.35 + 0.15 + 0.22
fig, ax = plt.subplots(1, 1, figsize=(6.4, figh))
fig.subplots_adjust(top=0.792, bottom=0.208, left=0.023, right=0.977)
ax.imshow(gradient, aspect='auto', cmap='nio')
# ax.set_axis_off()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()