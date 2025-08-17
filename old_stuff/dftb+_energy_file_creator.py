import os
import re

# Verzeichnis mit den Dateien
input_dir = "output"
output_file = "electronic_energies_512.txt"
pattern = re.compile(r"Total energy:\s+(-?\d+\.\d+)\s+H")

with open(output_file, "w") as out:
    for i in range(1, 7):
        filename = os.path.join(input_dir, f"detailed_{i}.out")
        if not os.path.isfile(filename):
            print('file doesnt exist')
            continue  # Datei existiert nicht
        with open(filename, "r") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    energy = float(match.group(1))
                    out.write(f"{i} {energy:.10f}\n")
                    break  # Nur den ersten passenden Treffer pro Datei verwenden
