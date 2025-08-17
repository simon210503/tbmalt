import torch
from tbmalt import Geometry, OrbitalInfo
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed
from tbmalt.common.maths.interpolation import CubicSpline
#from tbmalt.tools.downloaders import download_dftb_parameter_set
from tbmalt.ml.loss_function import Loss, mse_loss
import torch.nn as nn

from ase.build import molecule

Tensor = torch.Tensor

# This must be set until typecasting from HDF5 databases has been implemented.
torch.set_default_dtype(torch.float64)

# Provide a list of moecules for training
molecule_names = ['H2O']

# Reference of target properties
targets = {'total_energy': torch.tensor([-4.0779379326])}

# Provide information about the orbitals on each atom; this is keyed by atomic
# numbers and valued by azimuthal quantum numbers like so:
#   {Z₁: [ℓᵢ, ℓⱼ, ..., ℓₙ], Z₂: [ℓᵢ, ℓⱼ, ..., ℓₙ], ...}
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}

# Before running this example, please use setup.ipynb to download the parameter set needed
# Location at which the DFTB parameter set database is located
parameter_db_path = 'auorg.h5'

# Type of ML model
model = 'spline'

# Whether performing model fitting
fit_model = True

# Number of training cycles
number_of_epochs = 250

# Learning rate
lr = 0.002

# Loss function
loss_func = mse_loss

device = torch.device('cpu')
# Construct the `Geometry` and `OrbitalInfo` objects. The former is analogous
# to the ase.Atoms object while the latter provides information about what
# orbitals are present and which atoms they belong to.
geometry = Geometry(
        torch.tensor([8,1,1], device=device),
        torch.tensor([
            [0.00000000, -0.71603315, -0.00000000],
            [0.00000000, -0.14200298, 0.77844804 ],
            [-0.00000000, -0.14200298, -0.77844804]],
            device=device),units='a')

orbs = OrbitalInfo(geometry.atomic_numbers, shell_dict)

print('Geometry:', geometry)
print('OrbitalInfo:', orbs)

# Identify which species are present
species = torch.unique(geometry.atomic_numbers)
# Strip out padding species and convert to a standard list.
species = species[species != 0].tolist()

# Load the Hamiltonian feed model
h_feed = SkFeed.from_database(parameter_db_path, species, 'hamiltonian',
                              interpolation=CubicSpline, requires_grad_onsite=True, requires_grad_offsite=True)

# Load the overlap feed model
s_feed = SkFeed.from_database(parameter_db_path, species, 'overlap',
                              interpolation=CubicSpline)

# Load the occupation feed object
o_feed = SkfOccupationFeed.from_database(parameter_db_path, species)

# Load the Hubbard-U feed object
u_feed = HubbardFeed.from_database(parameter_db_path, species)

#When add r_feed, loss: tensor(1.4389e-09) on epoch 0
r_feed = RepulsiveSplineFeed.from_database(parameter_db_path, species)

dftb_calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, r_feed, filling_scheme=None)

torch.manual_seed(42) # Set random seed for reproducibility
# Create a random tensor for the Hamiltonian matrix
zero_tensor = torch.rand((6, 6), dtype=torch.double, requires_grad=True)

def return_random_tensor(geometry, orbs):
    return zero_tensor

# Assign the random tensor to the Hamiltonian feed
h_feed.matrix = return_random_tensor

# Test
H = h_feed.matrix(geometry, orbs)
print('H:', H)

# Define a delegate to obtain predictions from the trained model
def prediction_delegate(calculator, targets, **kwargs):
     predictions = dict()
     predictions["energy"] = calculator.total_energy
     
     return predictions

# Define a delegate to obtain reference results
def reference_delegate(calculator, targets, **kwargs):
     references = dict()
     references["energy"] = targets['total_energy']

     return references

# Define parameters to optimize
variable = h_feed.matrix(geometry, orbs)
print(variable)

# Define the loss entity
loss_entity = Loss(prediction_delegate, reference_delegate,
                   loss_functions=loss_func, reduction='mean')

# Define optimizer
optimizer = torch.optim.Adam([variable], lr=lr)

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