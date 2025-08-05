import torch
import numpy as np
import h5py
from tbmalt import OrbitalInfo, Geometry
from tbmalt.physics.dftb.feeds import HubbardFeed, SkFeed, SkfOccupationFeed, PairwiseRepulsiveEnergyFeed
from tbmalt.physics.dftb import Dftb2
from tbmalt.data.units import length_units, energy_units
from si64pos import fractional_positions
torch.set_default_dtype(torch.float64)

import random

def select_random_datapoints(N, seed, max_dpoint: int=1000):
    random.seed(seed) 
    datapoints = random.sample(range(1, max_dpoint+1), N)
    return datapoints

def load_Geo_dset(sample_path, datapoints):
    f = h5py.File(sample_path, 'r')
    geodata = [f[f'fnetdata/dataset/datapoint{num}/geometry'] for num in datapoints]
    Geodset = Geometry([torch.tensor(d['localattoatnum'][()]) for d in geodata],
               [torch.tensor(d['coordinates'][()]) for d in geodata],
               [torch.tensor(d['basis'][()]) for d in geodata],
               frac = True)
    return Geodset

def calc_reference_formation_energies(sample_path, datapoints):
    lattice_constant = 10.91929849 * length_units['a']
    GeoSi64 = Geometry(torch.full((64,), 14),
                    fractional_positions,
                    torch.diag(torch.tensor([lattice_constant, lattice_constant, lattice_constant])),
                    frac=True)
    f = h5py.File(sample_path, 'r')
    data = [f[f'fnetdata/dataset/datapoint{num}'] for num in datapoints]
    si64energy = -347.23490576 * energy_units['ev']
    all_targets_np = np.array([d['targets'] for d in data])
    dsetenergy = torch.from_numpy(all_targets_np).flatten() * energy_units['ev']
    Geodset = load_Geo_dset(sample_path, datapoints)
    chem_pot = si64energy / GeoSi64.n_atoms
    formation_energy = dsetenergy - chem_pot * Geodset.n_atoms
    return formation_energy

def calc_elec_energies_dset(sample_path1, sample_path2, datapoints):
    dft_energy = calc_reference_formation_energies(sample_path1, datapoints)

    f = h5py.File(sample_path2, 'r')
    data = [f[f'fnetdata/dataset/datapoint{num}'] for num in datapoints]
    all_targets_np = np.array([d['targets'] for d in data])
    dsetdelta = torch.from_numpy(all_targets_np).flatten() * energy_units['ev']
    dftb_elec_en = dft_energy - dsetdelta
    return dftb_elec_en

parameterset_path = "pbc.h5"

def calc_electronic_energies_dset(parameterset_path, sample_path, datapoints):
    """OBSOLETE"""
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


def calc_electronic_energies_Si64(parameterset_path):
    """OBSOLETE"""
    lattice_constant = 10.91929849 * length_units['a']
    GeoSi64 = Geometry(torch.full((64,), 14),
                    fractional_positions,
                    torch.diag(torch.tensor([lattice_constant, lattice_constant, lattice_constant])),
                    frac=True)
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


def calc_dftb_formation_energies(parameterset_path, sample_path, datapoints):
    """OBSOLETE"""
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

def get_energies_from_file(datapoints, filename="electronic_energies_64.txt"):
    """
    Gibt einen 1D Tensor mit Energien für die angegebenen Datenpunkte zurück.
    Falls ein Datenpunkt fehlt, wird NaN eingetragen.

    :param datapoints: Liste von Integer-Datenpunkten
    :param filename: Datei mit Energien im Format "<index> <energie>"
    :return: torch.Tensor mit den Energien (float), NaN bei fehlenden Einträgen
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



if __name__ == '__main__':
    N = 10
    datapoints = [1, 2, 3, 4, 5, 6]
    print(calc_elec_energies_dset('dft.hdf5', 'dft_dftb_elecen_siband.hdf5', datapoints))


    """
    seed = random.randint(0, 999999)  
    print(f"Seed: {seed}")
    lattice_constant = 10.91929849 * length_units['a']
    #datapoints = select_random_datapoints(N, seed)
    datapoints = [1101, 4663, 6257, 517, 2090, 966, 4059, 6234, 3683, 3869]
    print(f"Datapoints: {datapoints}")

    # Files and paths to be loaded
    GeoSi64 = Geometry(torch.full((64,), 14),
                    fractional_positions,
                    torch.diag(torch.tensor([lattice_constant, lattice_constant, lattice_constant])),
                    frac=True)

    sample_path = 'dft.hdf5'

    #print(calc_reference_formation_energies(sample_path, datapoints))
    print(calc_dftb_formation_energies(parameterset_path, sample_path, datapoints))
    #print(calc_electronic_energies_dset(parameterset_path, sample_path, datapoints))
    #print(calc_electronic_energies_Si64(parameterset_path))
    """
