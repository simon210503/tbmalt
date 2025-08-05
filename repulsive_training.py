import torch
from tbmalt import Geometry, OrbitalInfo
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, PairwiseRepulsiveEnergyFeed
from tbmalt.common.maths.interpolation import CubicSpline
#from tbmalt.tools.downloaders import download_dftb_parameter_set
from tbmalt.ml.loss_function import Loss, mse_loss
from tbmalt.structures.geometry import atomic_pair_distances
from new_feeds import xTBRepulsive, pairwise_repulsive, PTBPRepulsive, DFTBGammaRepulsive
import torch.nn as nn
from torch.nn import ModuleDict, Parameter


from ase.build import molecule

Tensor = torch.Tensor

# This must be set until typecasting from HDF5 databases has been implemented.
torch.set_default_dtype(torch.float64)

# Provide a list of moecules for training
molecule_names = ['H2O', 'CH4', 'CO2']

# Reference of target properties
targets = {'total_repulsive_energy': Tensor([0.0718, 0.0142, 0.3641])} 

# Provide information about the orbitals on each atom; this is keyed by atomic
# numbers and valued by azimuthal quantum numbers like so:
#   {Z₁: [ℓᵢ, ℓⱼ, ..., ℓₙ], Z₂: [ℓᵢ, ℓⱼ, ..., ℓₙ], ...}
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}

# Before running this example, please use setup.ipynb to download the parameter set needed
# Location at which the DFTB parameter set database is located
parameter_db_path = 'mio.h5'

# Type of ML model
model = 'spline'

# Whether performing model fitting
fit_model = True

# Number of training cycles
number_of_epochs = 500

# Learning rate
lr = 0.01

# Loss function
loss_func = mse_loss

device = torch.device('cpu')
# Construct the `Geometry` and `OrbitalInfo` objects. The former is analogous
# to the ase.Atoms object while the latter provides information about what
# orbitals are present and which atoms they belong to.

geometry = Geometry.from_ase_atoms(list(map(molecule, molecule_names)))

orbs = OrbitalInfo(geometry.atomic_numbers, shell_dict)

print('Geometry:', geometry)
print('OrbitalInfo:', orbs)

# Identify which species are present
species = torch.unique(geometry.atomic_numbers)
# Strip out padding species and convert to a standard list.
species = species[species != 0].tolist()

# Load the Hamiltonian feed model
h_feed = SkFeed.from_database(parameter_db_path, species, 'hamiltonian',
                              interpolation=CubicSpline, requires_grad_onsite=False, requires_grad_offsite=False)

# Load the overlap feed model
s_feed = SkFeed.from_database(parameter_db_path, species, 'overlap', interpolation=CubicSpline)

# Load the occupation feed object
o_feed = SkfOccupationFeed.from_database(parameter_db_path, species)

# Load the Hubbard-U feed object
u_feed = HubbardFeed.from_database(parameter_db_path, species)

# Initial Parameters for Repulsive
alpha = {
        1: Parameter(Tensor([1.0]),requires_grad = True),
        6: Parameter(Tensor([1.01]),requires_grad = True),
        8: Parameter(Tensor([1.02]),requires_grad = True)
        }

# Effective charge of cores
Z = {
        1: Parameter(Tensor([1.0]),requires_grad = True),
        6: Parameter(Tensor([6.0]),requires_grad = True),
        8: Parameter(Tensor([8.0]),requires_grad = True)
    }

cutoff = {}
for species_pair, _, _ in atomic_pair_distances(
    geometry, True, True):
    cutoff[str((species_pair[0].item(), species_pair[1].item()))
            ]= Tensor([5.0])
 
# Prepare input for r_feed
H2O_pair_repulsive = pairwise_repulsive(geometry, alpha, Z, PTBPRepulsive, cutoff)

# Define repulsive
r_feed = PairwiseRepulsiveEnergyFeed(H2O_pair_repulsive)

dftb_calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, r_feed, filling_scheme=None)

# Define a delegate to obtain predictions from the trained model
def prediction_delegate(calculator, targets, **kwargs):
     predictions = dict()
     predictions["total_repulsive_energy"] = calculator.repulsive_energy
     return predictions

# Define a delegate to obtain reference results
def reference_delegate(calculator, targets, **kwargs):
     references = dict()
     references["total_repulsive_energy"] = targets['total_repulsive_energy']
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
    dftb_calculator(geometry, orbs, grad_mode="direct")
    total_loss, raw_losses = loss_entity(dftb_calculator, targets)
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