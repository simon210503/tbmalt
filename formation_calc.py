import torch
import h5py
import numpy as np

from torch import Tensor
from torch.nn import Parameter
from tbmalt import OrbitalInfo, Geometry
from tbmalt.physics.dftb.feeds import HubbardFeed, SkFeed, SkfOccupationFeed, PairwiseRepulsiveEnergyFeed
from tbmalt.physics.dftb import Dftb2
from tbmalt.data.units import length_units, energy_units

from new_feeds import pairwise_repulsive
from si64pos import fractional_positions
from saveload import load_Geo_dset

torch.set_default_dtype(torch.float64)


def calc_reference_formation_energies(sample_path: str, datapoints: list[int]) -> Tensor:
    """
    Calculate the reference formation energies for a set of datapoints.

    The formation energy is computed relative to the chemical potential 
    derived from a 64-atom Si reference structure.

    Parameters
    ----------
    sample_path : str
        Path to the HDF5 file containing the dataset.
    datapoints : list[int]
        List of indices of datapoints to evaluate.

    Returns
    -------
    torch.Tensor
        Tensor containing the formation energies for the given datapoints.
    """
    lattice_constant = 10.91929849 * length_units['a']
    GeoSi64 = Geometry(
        torch.full((64,), 14),
        fractional_positions,
        torch.diag(torch.tensor([lattice_constant, lattice_constant, lattice_constant])),
        frac=True
    )
    f = h5py.File(sample_path, 'r')
    data = [f[f'fnetdata/dataset/datapoint{num}'] for num in datapoints]
    si64energy = -347.23490576 * energy_units['ev']
    all_targets_np = np.array([d['targets'] for d in data])
    dsetenergy = torch.from_numpy(all_targets_np).flatten() * energy_units['ev']
    Geodset = load_Geo_dset(sample_path, datapoints)
    chem_pot = si64energy / GeoSi64.n_atoms
    formation_energy = dsetenergy - chem_pot * Geodset.n_atoms
    return formation_energy


def calc_elec_energies_dset(sample_path1: str, sample_path2: str, datapoints: list[int]) -> Tensor:
    """
    Calculate defect electronic energies by subtracting DFTB corrections from DFT values.

    Parameters
    ----------
    sample_path1 : str
        Path to the HDF5 file with DFT total energies.
    sample_path2 : str
        Path to the HDF5 file containing DFT–DFTB delta energies.
    datapoints : list[int]
        List of indices of datapoints to evaluate.

    Returns
    -------
    torch.Tensor
        Electronic defect energies after correction.
    """
    f = h5py.File(sample_path1, 'r')
    data = [f[f'fnetdata/dataset/datapoint{num}'] for num in datapoints]
    all_targets_np = np.array([d['targets'] for d in data])
    dft_energy = torch.from_numpy(all_targets_np).flatten() * energy_units['ev']

    f = h5py.File(sample_path2, 'r')
    data = [f[f'fnetdata/dataset/datapoint{num}'] for num in datapoints]
    all_targets_np = np.array([d['targets'] for d in data])
    dsetdelta = torch.from_numpy(all_targets_np).flatten() * energy_units['ev']
    dftb_elec_en = dft_energy - dsetdelta
    return dftb_elec_en


def prepare_system(
    sample_path: str,
    datapoints: list[int],
    repulsive_model: str,
    alpha: dict[int, Parameter] | None = None,
    Z: dict[int, Parameter] | None = None,
) -> dict:
    """
    Prepare dataset geometries, parameters, and repulsive energy feed.

    This function loads defect geometries, calculates reference energies,
    sets up parameters (alpha, Z), and constructs the repulsive feed.

    Parameters
    ----------
    sample_path : str
        Path to the dataset HDF5 file.
    datapoints : list[int]
        List of datapoint indices.
    repulsive_model : str
        Which repulsive model to use ('pbc', 'siband', or custom).
    alpha : dict[int, Parameter], optional
        Dictionary of alpha parameters per atomic species.
    Z : dict[int, Parameter], optional
        Dictionary of effective nuclear charges per atomic species.

    Returns
    -------
    dict
        Dictionary of prepared system parameters including geometries,
        energies, and repulsive feed.
    """
    lattice_constant = 10.91929849 * length_units['a']

    Geodset = load_Geo_dset(sample_path, datapoints)
    formation_energy = calc_reference_formation_energies(sample_path, datapoints)
    if sample_path == 'dft.hdf5':
        elec_energy_defects = calc_elec_energies_dset('dft.hdf5', 'dft_dftb_elecen_siband.hdf5', datapoints)
    elif sample_path == 'dft_test.hdf5':
        elec_energy_defects = calc_elec_energies_dset('dft_test.hdf5', 'dft_dftb_elecen_siband_test.hdf5', datapoints)
    elec_energy_Si64 = -88.2635544138  # reference value

    GeoSi64 = Geometry(
        torch.full((64,), 14),
        fractional_positions,
        torch.diag(torch.tensor([lattice_constant] * 3)),
        frac=True
    )

    if alpha is None:
        alpha = {14: Parameter(Tensor([2.5]), requires_grad=True)}
    if Z is None:
        Z = {14: Parameter(Tensor([14.0]), requires_grad=True)}

    cutoff = Tensor([6.0])
    cutoff_rep = {'(14, 14)': cutoff}

    if repulsive_model == 'pbc':
        repulsive_feed = PairwiseRepulsiveEnergyFeed.from_database('pbc.h5', [14])
    elif repulsive_model == 'siband':
        repulsive_feed = PairwiseRepulsiveEnergyFeed.from_database('siband.h5', [14])
    else:
        Si_pair_repulsive = pairwise_repulsive(GeoSi64, alpha, Z, repulsive_model, cutoff_rep)
        repulsive_feed = PairwiseRepulsiveEnergyFeed(Si_pair_repulsive)

    params = {
        "Geodset": Geodset,
        "GeoSi64": GeoSi64,
        "formation_energy": formation_energy,
        "elec_energy_defects": elec_energy_defects,
        "elec_energy_Si64": elec_energy_Si64,
        "alpha": alpha,
        "Z": Z,
        "repulsive_feed": repulsive_feed
    }

    return params


def calc_formation_energy(params: dict) -> Tensor:
    """
    Compute formation energy using prepared system parameters.

    The formation energy is calculated by combining electronic and repulsive energies
    for defect structures and the reference Si64 crystal.

    Parameters
    ----------
    params : dict
        Dictionary containing system parameters prepared by `prepare_system`.

    Returns
    -------
    torch.Tensor
        Formation energies of the defects.
    """
    rep_feed = params['repulsive_feed']
    Geodset = params['Geodset']
    GeoSi64 = params['GeoSi64']
    elec_energy_Si64 = params['elec_energy_Si64']
    elec_energy_defects = params['elec_energy_defects']

    rep_energy_Si64 = rep_feed.forward(GeoSi64)
    total_energy_Si64 = elec_energy_Si64 + rep_energy_Si64
    chem_pot = total_energy_Si64 / GeoSi64.n_atoms

    rep_energy_defects = rep_feed.forward(Geodset)
    total_energy_defects = elec_energy_defects + rep_energy_defects

    formation_energy = total_energy_defects - Geodset.n_atoms * chem_pot
    return formation_energy


def calc_electronic_energies_dset(parameterset_path: str, sample_path: str, datapoints: list[int]) -> Tensor:
    """
    (OBSOLETE) Compute electronic energies of dataset geometries using DFTB2.

    Parameters
    ----------
    parameterset_path : str
        Path to DFTB parameter set.
    sample_path : str
        Path to dataset HDF5 file.
    datapoints : list[int]
        List of datapoint indices.

    Returns
    -------
    torch.Tensor
        Electronic energies of the dataset geometries.
    """
    Geodset = load_Geo_dset(sample_path, datapoints)
    shell_dict={14: [0, 1, 2]}
    orbdset = OrbitalInfo(Geodset.atomic_numbers, shell_dict=shell_dict)

    species = [14]

    h_feed = SkFeed.from_database(parameterset_path, species, 'hamiltonian')
    s_feed = SkFeed.from_database(parameterset_path, species, 'overlap')
    o_feed = SkfOccupationFeed.from_database(parameterset_path, species)
    u_feed = HubbardFeed.from_database(parameterset_path, species)

    dftb2 = Dftb2(h_feed, s_feed, o_feed, u_feed, filling_temp=297)

    dftb2(Geodset, orbdset)
    elec_energy = dftb2.total_energy
    return elec_energy


def calc_electronic_energies_Si64(parameterset_path: str) -> Tensor:
    """
    (OBSOLETE) Compute electronic energy of the reference 64-atom Si system using DFTB2.

    Parameters
    ----------
    parameterset_path : str
        Path to DFTB parameter set.

    Returns
    -------
    torch.Tensor
        Electronic energy of the Si64 reference structure.
    """
    lattice_constant = 10.91929849 * length_units['a']
    GeoSi64 = Geometry(
        torch.full((64,), 14),
        fractional_positions,
        torch.diag(torch.tensor([lattice_constant, lattice_constant, lattice_constant])),
        frac=True
    )
    shell_dict={14: [0, 1, 2]}
    orbSi64 = OrbitalInfo(GeoSi64.atomic_numbers, shell_dict=shell_dict)

    species = [14]

    h_feed = SkFeed.from_database(parameterset_path, species, 'hamiltonian')
    s_feed = SkFeed.from_database(parameterset_path, species, 'overlap')
    o_feed = SkfOccupationFeed.from_database(parameterset_path, species)
    u_feed = HubbardFeed.from_database(parameterset_path, species)

    dftb2 = Dftb2(h_feed, s_feed, o_feed, u_feed, filling_temp=297)

    dftb2(GeoSi64, orbSi64)
    elec_energy = dftb2.total_energy
    return elec_energy


def calc_dftb_formation_energies(parameterset_path: str, sample_path: str, datapoints: list[int]) -> Tensor:
    """
    (OBSOLETE) Calculate formation energies using a full DFTB2 model.

    Parameters
    ----------
    parameterset_path : str
        Path to DFTB parameter set.
    sample_path : str
        Path to dataset HDF5 file.
    datapoints : list[int]
        List of datapoint indices.

    Returns
    -------
    torch.Tensor
        Formation energies for the given dataset.
    """
    GeoSi64 = Geometry(
        torch.full((64,), 14),
        fractional_positions,
        torch.diag(torch.tensor([lattice_constant] * 3)),
        frac=True
    )
    Geodset = load_Geo_dset(sample_path, datapoints)
    shell_dict={14: [0, 1, 2]}
    orbdset = OrbitalInfo(Geodset.atomic_numbers, shell_dict=shell_dict)
    orbSi64 = OrbitalInfo(GeoSi64.atomic_numbers, shell_dict=shell_dict)

    species = [14]

    h_feed = SkFeed.from_database(parameterset_path, species, 'hamiltonian')
    s_feed = SkFeed.from_database(parameterset_path, species, 'overlap')
    o_feed = SkfOccupationFeed.from_database(parameterset_path, species)
    u_feed = HubbardFeed.from_database(parameterset_path, species)
    r_feed = PairwiseRepulsiveEnergyFeed.from_database(parameterset_path, species)

    dftb2 = Dftb2(h_feed, s_feed, o_feed, u_feed, r_feed, filling_temp=297)

    dftb2(GeoSi64, orbSi64)
    chem_pot = dftb2.total_energy / GeoSi64.n_atoms

    dftb2(Geodset, orbdset)
    formation_energy = dftb2.total_energy - Geodset.n_atoms * chem_pot
    return formation_energy


def get_energies_from_file(datapoints: list[int], filename: str = "electronic_energies_64.txt") -> Tensor:
    """
    (OBSOLETE)
    Load energies for specified datapoints from a text file.

    The file must contain lines of the form "<index> <energy>".
    Missing datapoints will be filled with NaN.

    Parameters
    ----------
    datapoints : list[int]
        List of datapoint indices.
    filename : str, optional
        Path to text file containing energies. Default is 'electronic_energies_64.txt'.

    Returns
    -------
    torch.Tensor
        1D tensor of energies, with NaN values for missing datapoints.
    """
    energies = {}

    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    idx = int(parts[0])
                    energy = float(parts[1])
                    energies[idx] = energy
                except ValueError:
                    continue

    tensor_values = [
        energies.get(dp, float('nan')) for dp in datapoints
    ]
    return torch.tensor(tensor_values, dtype=torch.float64)
