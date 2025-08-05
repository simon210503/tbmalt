import h5py
import torch
import os
from tbmalt.data.units import length_units

# Lade HDF5-Daten
sample_path = 'dft_test.hdf5'
f = h5py.File(sample_path, 'r')

# Erstelle Zielordner, falls nicht vorhanden
output_dir = 'genfiles512'
os.makedirs(output_dir, exist_ok=True)
for ii in range(1, 10):
    datapoint = ii
    geodata = f[f'fnetdata/dataset/datapoint{datapoint}/geometry']

    # Extrahiere relevante Daten
    Z = torch.tensor(geodata['localattoatnum'][()])  # Ordnungszahlen
    coords = torch.tensor(geodata['coordinates'][()])  # (N, 3)
    basis = torch.tensor(geodata['basis'][()]) / length_units['a']

    # Erstelle Zielordner, falls nicht vorhanden
    output_path = os.path.join(output_dir, f'datapoint{datapoint}.gen')

    # Schreibe die .gen Datei
    with open(output_path, "w") as fout:
        fout.write(f" {len(Z)}  F\n")  # Anzahl der Atome + Flag (z. B. S für scaled)
        fout.write(" Si\n")
        for i, (z, pos) in enumerate(zip(Z, coords), start=1):
            x, y, z = pos.tolist()
            fout.write(f"  {i:3d}  1 {x:20.10E} {y:20.10E} {z:20.10E}\n")
        fout.write(f"{0:20.10E} {0:20.10E} {0:20.10E}\n")
        for vec in basis:
            fout.write("".join(f"{x.item():20.10E}" for x in vec) + "\n")

    print(f".gen Datei gespeichert unter: {output_path}")