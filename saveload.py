import os
import ast
import re
import h5py
import torch
from typing import Any, List, Union, Tuple

from torch import Tensor
from torch.nn import Parameter

from tbmalt import OrbitalInfo, Geometry
from new_feeds import xTBRepulsive, PTBPRepulsive, DFTBGammaRepulsive


def save_results_to_file(result: list[Any], file_path: str) -> None:
    """
    Save the output of train64test64() to a text file in a human-readable format.

    Parameters
    ----------
    result : list
        The result returned by train64test64.
    file_path : str
        Path to the file where the result will be saved.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(repr(result))  # Save the full list in a repr-safe way


def save_results_to_directory(results_list: list[Any], directory: str) -> None:
    """
    Save a list of results to individual text files inside a directory.

    Parameters
    ----------
    results_list : list
        List of results (each can be any serializable Python object).
    directory : str
        Directory path where the result files will be stored.
    """
    os.makedirs(directory, exist_ok=True)
    for idx, item in enumerate(results_list):
        file_path = os.path.join(directory, f"result_{idx}.txt")
        with open(file_path, "w") as f:
            f.write(repr(item))


def load_results_from_file(filepath: str) -> list[Union[str, Any]]:
    """
    Load results line by line from a text file.

    Uses `ast.literal_eval` for safe evaluation of simple Python literals.
    For unsupported or complex types (e.g., Tensor, Geometry, Parameter),
    the raw line is returned as a string.

    Parameters
    ----------
    filepath : str
        Path to the text file containing saved results.

    Returns
    -------
    results : list of Union[str, Any]
        Parsed results. Each entry is either a Python object (int, float, list, dict, etc.)
        or a raw string if parsing failed.
    """
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            try:
                results.append(ast.literal_eval(stripped))
            except Exception:
                results.append(stripped)
    return results


def load_Geo_dset(sample_path: str, datapoints: list[int]) -> Geometry:
    """
    Load a dataset of geometries from an HDF5 file.

    Parameters
    ----------
    sample_path : str
        Path to the HDF5 sample file.
    datapoints : list of int
        Indices of datapoints to extract from the dataset.

    Returns
    -------
    Geodset : Geometry
        Geometry object constructed from the selected dataset entries.
    """
    f = h5py.File(sample_path, 'r')
    geodata = [f[f'fnetdata/dataset/datapoint{num}/geometry'] for num in datapoints]
    Geodset = Geometry(
        [torch.tensor(d['localattoatnum'][()]) for d in geodata],
        [torch.tensor(d['coordinates'][()]) for d in geodata],
        [torch.tensor(d['basis'][()]) for d in geodata],
        frac=True
    )
    return Geodset


def load_parameters(file_path: str) -> list[float]:
    """
    Load numeric parameters from a text file.

    Parameters
    ----------
    file_path : str
        Path to the text file containing results.

    Returns
    -------
    parameters : list of float
        Extracted floating-point numbers from the result file.
    """
    raw_alpha = load_results_from_file(file_path)[1]
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", raw_alpha)
    return [float(n) for n in numbers]


def load_repulsives(base_path: str, cutoff: float = 6.0) -> Tuple[xTBRepulsive, PTBPRepulsive, DFTBGammaRepulsive]:
    """
    Load repulsive potentials (xTB, PTBP, DFTB-Gamma) from parameter files.

    Parameters
    ----------
    base_path : str
        Path to the base directory containing result subfolders for Gamma, xTB, and PTBP.
    cutoff : float, default=8.0
        Cutoff radius for the repulsive potentials.

    Returns
    -------
    xTB : xTBRepulsive
        xTB repulsive potential module.
    PTBP : PTBPRepulsive
        PTBP repulsive potential module.
    Gamma : DFTBGammaRepulsive
        DFTB-Gamma repulsive potential module.
    """
    Gamma_alpha = load_parameters(f'{base_path}/Gamma_result/result_8.txt')
    Gamma_Z = load_parameters(f'{base_path}/Gamma_result/result_9.txt')
    xTB_alpha = load_parameters(f'{base_path}/xTB_result/result_8.txt')
    xTB_Z = load_parameters(f'{base_path}/xTB_result/result_9.txt')
    PTBP_alpha = load_parameters(f'{base_path}/PTBP_result/result_8.txt')
    PTBP_Z = load_parameters(f'{base_path}/PTBP_result/result_9.txt')

    Gamma = DFTBGammaRepulsive(
        [Parameter(Tensor(Gamma_Z)), Parameter(Tensor(Gamma_Z)),
         Parameter(Tensor(Gamma_alpha)), Parameter(Tensor(Gamma_alpha))],
        cutoff
    )
    xTB = xTBRepulsive(
        [Parameter(Tensor(xTB_Z)), Parameter(Tensor(xTB_Z)),
         Parameter(Tensor(xTB_alpha)), Parameter(Tensor(xTB_alpha)), 1.5],
        cutoff
    )
    PTBP = PTBPRepulsive(
        [Parameter(Tensor(PTBP_Z)), Parameter(Tensor(PTBP_Z)),
         Parameter(Tensor(PTBP_alpha)), Parameter(Tensor(PTBP_alpha))],
        cutoff
    )

    return xTB, PTBP, Gamma

def load_MSE_from_file(base_path):

    MSE = []
    for ii in range(5):
        MSE.append(load_results_from_file(os.path.join(base_path, f'split{ii}/xTB_result/result_10.txt'))[0])
        MSE.append(load_results_from_file(os.path.join(base_path, f'split{ii}/PTBP_result/result_10.txt'))[0])
        MSE.append(load_results_from_file(os.path.join(base_path, f'split{ii}/Gamma_result/result_10.txt'))[0])
        MSE.append(load_results_from_file(os.path.join(base_path, f'split{ii}/pbc_result/result_0.txt'))[0])

    return MSE

def load_MSE512_from_file(base_path):

    MSE = []
    MSE.append(load_results_from_file(os.path.join(base_path, f'xTB_result/result_10.txt'))[0])
    MSE.append(load_results_from_file(os.path.join(base_path, f'PTBP_result/result_10.txt'))[0])
    MSE.append(load_results_from_file(os.path.join(base_path, f'Gamma_result/result_10.txt'))[0])
    MSE.append(load_results_from_file(os.path.join(base_path, f'pbc_result/result_0.txt'))[0])

    return MSE


def load_testdpoints_from_file(base_path):
    dpoints = load_results_from_file(os.path.join(base_path, 'Gamma_result/result_1.txt'))[0]
    return dpoints
        

if __name__ == '__main__':

    from pathlib import Path
    from utils import stringlist_list_converter
    base_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/logs/64routine/split3')
    #zahlen = load_testdpoints_from_file(base_path)
    #kleiner_gleich_6 = [x for x in zahlen if x <= 6]
    #print(kleiner_gleich_6)

    print(load_parameters(os.path.join(base_path, 'Gamma_result/result_8.txt')))