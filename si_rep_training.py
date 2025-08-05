import torch
import h5py
from tbmalt import Geometry, OrbitalInfo
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, PairwiseRepulsiveEnergyFeed
from tbmalt.common.maths.interpolation import CubicSpline
#from tbmalt.tools.downloaders import download_dftb_parameter_set
from tbmalt.ml.loss_function import Loss, mse_loss
from tbmalt.structures.geometry import atomic_pair_distances
from tbmalt.data.units import length_units
from new_feeds import pairwise_repulsive, PTBPRepulsive, DFTBGammaRepulsive, xTBRepulsive
from torch.nn import ModuleDict, Parameter
from ase.build import molecule
from si64pos import fractional_positions
from formation_calc import select_random_datapoints, calc_reference_formation_energies, load_Geo_dset, get_energies_from_file
Tensor = torch.Tensor
# This must be set until typecasting from HDF5 databases has been implemented.
torch.set_default_dtype(torch.float64)

import random

# VALUES THAT MIGHT BE CHANGED

N = 10  # Amount of datapoints used
seed = random.randint(0, 999999)  # random seed
random.seed(seed)  

print(f"Seed: {seed}")

datapoints = random.sample(range(1, 1001), N)
print("Datapoints:", datapoints)


# Files and paths to be loaded
parameter_path = "pbc.h5"
sample_path = 'dft.hdf5'

# Constants for initial calculation
cutoff = torch.Tensor([5.0])
lattice_constant = 10.91929849 * length_units['a']
filling_temp = 297
shell_dict = {14: [0, 1, 2]}
species = [14]
Repulsive = DFTBGammaRepulsive

# Parameters for machine learning
model = 'spline'
fit_model = True
number_of_epochs = 500
lr = 0.1
loss_func = mse_loss
device = torch.device('cpu')

# Initial Parameters for Repulsive
alpha = {
        14: Parameter(Tensor([2.5]),requires_grad = True)
        }

# Effective charge of cores
Z = {
        14: Parameter(Tensor([14.0]),requires_grad = True)
    }


Geodset = load_Geo_dset(sample_path, datapoints)
GeoSi64 = Geometry(torch.full((64,), 14),
                fractional_positions,
                torch.diag(torch.tensor([lattice_constant, lattice_constant, lattice_constant])),
                frac=True)
                
formation_energy = calc_reference_formation_energies(sample_path, datapoints)

print("Reference formation energy:", formation_energy)

# Reference of target properties
targets = {'formation_energy': formation_energy}

elec_energy_defects = get_energies_from_file(datapoints)
elec_energy_Si64 = -88.2635544138

# Cutoff for repulsive
cutoff_rep = {
     '(14, 14)': cutoff
     }

# Prepare input for r_feed
Si_pair_repulsive = pairwise_repulsive(GeoSi64, alpha, Z, Repulsive, cutoff_rep)

# Define repulsive
new_r_feed = PairwiseRepulsiveEnergyFeed(Si_pair_repulsive)

def calc_formation_energy():
    rep_energy_Si64 = new_r_feed.forward(GeoSi64)
    total_energy_Si64 = elec_energy_Si64 + rep_energy_Si64
    chem_pot = total_energy_Si64 / GeoSi64.n_atoms
    rep_energy_defects = new_r_feed.forward(Geodset)
    total_energy_defects = elec_energy_defects + rep_energy_defects
    formation_energy = total_energy_defects - Geodset.n_atoms * chem_pot
    return formation_energy

print(calc_formation_energy())

# Define a delegate to obtain predictions from the trained model
def prediction_delegate(calculator, targets, **kwargs):
     predictions = dict()
     predictions["formation_energy"] = calc_formation_energy()
     return predictions

# Define a delegate to obtain reference results
def reference_delegate(calculator, targets, **kwargs):
     references = dict()
     references["formation_energy"] = targets['formation_energy']
     return references

# Define parameters to optimize

variable = list(alpha.values()) + list(Z.values())

print(variable)

# Define the loss entity
loss_entity = Loss(prediction_delegate, reference_delegate,
                   loss_functions=loss_func, reduction='mean')

# Define optimizer
#optimizer = torch.optim.Adam([variable], lr=lr)
optimizer = getattr(torch.optim, 'Adam')(params=variable, lr=lr)

# Execution
loss_list = []
loss_list.append(0)
for epoch in range(number_of_epochs):
    _loss = 0
    print('epoch', epoch)
    calc_formation_energy()
    total_loss, raw_losses = loss_entity(calc_formation_energy, targets)
    _loss = _loss + total_loss
    optimizer.zero_grad()
    _loss.retain_grad()

    # Invoke the autograd engine
    _loss.backward(retain_graph=True)

    # Update the model
    optimizer.step()
    print("loss:", _loss)
    loss_list.append(_loss.detach())

print(variable)

print("Reference formation energy:", formation_energy)
print("Final formation energy:", calc_formation_energy())

    # Plot the loss
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.linewidth"] = 1.5
plt.tick_params(direction='in', labelsize='26', width=1.5, length=5, top='on',
                    right='on', zorder=10)
plt.plot(torch.linspace(1, number_of_epochs, number_of_epochs),
         loss_list[1:])
plt.xlabel("Iteration", fontsize=28)
plt.ylabel("Loss",  fontsize=28)
#plt.savefig('loss_hubbard.pdf', dpi=500, bbox_inches='tight')
plt.show()