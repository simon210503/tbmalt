import numpy as np
import torch
torch.set_default_dtype(torch.float64)

# Fractional positions of 8 atoms in the diamond cubic unit cell
unit_cell_positions = np.array([
    [0.000, 0.000, 0.000],
    [0.250, 0.250, 0.250],
    [0.500, 0.500, 0.000],
    [0.750, 0.750, 0.250],
    [0.500, 0.000, 0.500],
    [0.750, 0.250, 0.750],
    [0.000, 0.500, 0.500],
    [0.250, 0.750, 0.750]
])

# Generate 2x2x2 supercell fractional positions
supercell_size = 2
offsets = np.array([[i, j, k] for i in range(supercell_size)
                              for j in range(supercell_size)
                              for k in range(supercell_size)])

# Compute all fractional positions
fractional_positions = []
for offset in offsets:
    for pos in unit_cell_positions:
        frac_pos = (pos + offset) / supercell_size
        fractional_positions.append(frac_pos)

# Convert to numpy array
fractional_positions = torch.from_numpy(np.array(fractional_positions))

#print(fractional_positions)