#gaussian function("is a mathematical function with a characteristic bell-shaped curve.")

import torch
import numpy as np
import matplotlib.pyplot as plt

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create meshgrid
X, Y = np.mgrid[-4.0:4.0:0.01, -4.0:4.0:0.01]

# Convert to tensors and send to device
x = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(Y, dtype=torch.float32).to(device)

# Compute 2D Gaussian
z = torch.exp(-(x**2 + y**2) / 2.0)

# Plot the result
plt.imshow(z.cpu().numpy(), extent=[-4, 4, -4, 4], origin='lower')
plt.title("2D Gaussian Distribution")
plt.colorbar(label='z')
plt.tight_layout()
plt.show()
