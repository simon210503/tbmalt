import torch
from tbmalt import OrbitalInfo, Geometry
from tbmalt.physics.dftb.feeds import HubbardFeed, SkFeed, SkfOccupationFeed, RepulsiveSplineFeed
from tbmalt.physics.dftb import Dftb2
from tbmalt.tools.downloaders import download_dftb_parameter_set
from ase.build import molecule
torch.set_default_dtype(torch.float64)

url = 'https://github.com/dftbparams/auorg/releases/download/v1.1.0/auorg-1-1.tar.xz'
path = "auorg.h5"
#download_dftb_parameter_set(url, path)

H2O = Geometry(torch.tensor([8, 1, 1]), 
               torch.tensor([[0.0, -1.0, 0.0],
                             [0.0, 0.0, 0.78306400000],
                             [0.0, 0.0, -0.78306400000]], requires_grad=True),
               units='angstrom'
               )

orb = OrbitalInfo(H2O.atomic_numbers, shell_dict={1: [0], 8: [0, 1]})

species = [1, 8]

h_feed = SkFeed.from_database(path, species, 'hamiltonian')
s_feed = SkFeed.from_database(path, species, 'overlap')
o_feed = SkfOccupationFeed.from_database(path, species)
u_feed = HubbardFeed.from_database(path, species)
r_feed = RepulsiveSplineFeed.from_database(path, species)

mix_params = {'mix_param': 0.2, 'init_mix_param': 0.2,
                'generations': 3, 'tolerance': 1e-10}

dftb2 = Dftb2(h_feed, s_feed, o_feed, u_feed, r_feed)
dftb2(H2O, orb)

print(dftb2.q_final_atomic)
print(dftb2.total_energy)