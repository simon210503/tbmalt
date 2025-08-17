import torch
import h5py
from tbmalt import OrbitalInfo, Geometry
from tbmalt.physics.dftb.feeds import HubbardFeed, SkFeed, SkfOccupationFeed, RepulsiveSplineFeed, DftbpRepulsiveSpline, PairwiseRepulsiveEnergyFeed
from tbmalt.physics.dftb import Dftb2
from tbmalt.tools.downloaders import download_dftb_parameter_set
from ase.build import molecule
from tbmalt.data.units import length_units
from tbmalt.common.maths.interpolation import CubicSpline
torch.set_default_dtype(torch.float64)
#torch.set_num_threads(1)
#torch.set_num_interop_threads(1)

url = 'https://github.com/dftbparams/pbc/releases/download/v0.3.0/pbc-0-3.tar.xz'
path = "pbc.h5"
#download_dftb_parameter_set(url, path)

cutoff = torch.tensor([5.0])
lattice_constant = 5.43 * length_units['a']

#print(lattice_constant)

GeoSi2 = Geometry(
    torch.tensor([14, 14]),
    torch.tensor([      
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25]]),
    torch.tensor([
        [lattice_constant / 2, lattice_constant / 2, 0.0],
        [0.0, lattice_constant / 2, lattice_constant / 2],
        [lattice_constant / 2, 0.0, lattice_constant / 2]]), 
    frac = True,
    cutoff = cutoff)

orb = OrbitalInfo(GeoSi2.atomic_numbers, shell_dict={14: [0, 1, 2]})

species = [14]

h_feed = SkFeed.from_database(path, species, 'hamiltonian')
s_feed = SkFeed.from_database(path, species, 'overlap')
o_feed = SkfOccupationFeed.from_database(path, species)
u_feed = HubbardFeed.from_database(path, species)
r_feed = PairwiseRepulsiveEnergyFeed.from_database(path, species)

dftb2 = Dftb2(h_feed, s_feed, o_feed, u_feed, r_feed, filling_temp=297)
dftb2(GeoSi2, orb)

print(dftb2.total_energy)
print(dftb2.repulsive_energy)
energy_per_atom = dftb2.total_energy / GeoSi2.n_atoms
print(energy_per_atom)