import torch
import h5py
from tbmalt import OrbitalInfo, Geometry
from tbmalt.physics.dftb.feeds import HubbardFeed, SkFeed, SkfOccupationFeed, RepulsiveSplineFeed, DftbpRepulsiveSpline, PairwiseRepulsiveEnergyFeed
from tbmalt.physics.dftb import Dftb2
from tbmalt.tools.downloaders import download_dftb_parameter_set
from ase.build import molecule
torch.set_default_dtype(torch.float64)

from torch.utils.data import DataLoader, Dataset, random_split
import re

import random

# VALUES THAT MIGHT BE CHANGED

N = 10  # Amount of datapoints used
seed = random.randint(0, 999999)  # random seed
random.seed(seed)  

print(f"Seed: {seed}")

datapoints = random.sample(range(2, 6307), N)
print("Datapoints:", datapoints)

path = "pbc.h5"
f = h5py.File('dft.hdf5', 'r')
data = [f[f'fnetdata/dataset/datapoint{num}'] for num in datapoints]
dataSi64 = f['fnetdata/dataset/datapoint1']



si64energy = torch.tensor(dataSi64['targets']).flatten()
dsetenergy = torch.stack([torch.tensor(d['targets']) for d in data]).flatten()

print(si64energy)
print(dsetenergy)

chem_pot = si64energy / GeoSi64.n_atoms
formation_energy = dsetenergy - chem_pot * Geodset.n_atoms

print(formation_energy)