import torch
import random
import numpy as np
from collections import Counter
from saveload import load_Geo_dset
from tbmalt.structures.geometry import atomic_pair_distances
from typing import List, Union, Dict, Any

def get_atom_count_from_hdf5(path: str, datapoints: Union[List[int], range]) -> torch.Tensor:
    """
    Returns the number of atoms for each structure in an HDF5 dataset.

    Args:
        path (str): Path to the HDF5 file.
        datapoints (list or range): List or range of datapoint indices to retrieve.

    Returns:
        torch.Tensor: 1D tensor containing the atom counts for the specified datapoints.
    """
    Geo = load_Geo_dset(path, datapoints)
    return Geo.n_atoms


def vacancy_datapoints(path: str, datapoints: Union[List[int], range]) -> torch.Tensor:
    """
    Returns the indices of structures with exactly 63 atoms.

    Args:
        path (str): Path to the HDF5 file.
        datapoints (list or range): List or range of datapoint indices to check.

    Returns:
        torch.Tensor: 1D tensor with 1-based indices of structures with 63 atoms.
    """
    t = get_atom_count_from_hdf5(path, datapoints)
    return torch.where(t == 63)[0] + 1


def find_structures_with_atom_count(path: str, datapoints: Union[List[int], range], atom_count: int) -> torch.Tensor:
    """
    Finds the indices of structures in an HDF5 file
    that have a specific number of atoms.

    Args:
        path (str): Path to the HDF5 file.
        datapoints (list | range): List or range of datapoints to check.
        atom_count (int): Desired number of atoms.

    Returns:
        torch.Tensor: 1D tensor containing the indices (1-based) 
                      of structures matching the atom count.
    """
    t = get_atom_count_from_hdf5(path, datapoints)
    return torch.where(t == atom_count)[0] + 1


def get_atom_count_from_gen(filepath: str) -> int:
    """
    OBSOLETE: Reads the first integer from a file, assumed to represent atom count.

    Args:
        filepath (str): Path to the file.

    Returns:
        int: Number of atoms read from the first line.
    """
    with open(filepath, "r") as f:
        first_line = f.readline().strip()
    return int(first_line.split()[0])


def split_data(dpoints: List[int], portions: int = 5) -> List[List[int]]:
    """
    Splits a list of datapoints into roughly equal-sized portions.

    Args:
        dpoints (list): List of datapoints to split.
        portions (int, optional): Number of portions to split into. Defaults to 5.

    Returns:
        list of list: List containing sublists of datapoints.
    """
    splits = []
    total = len(dpoints)
    base_size = total // portions
    remainder = total % portions

    start = 0
    for i in range(portions):
        end = start + base_size + (1 if i < remainder else 0)
        splits.append(dpoints[start:end])
        start = end

    return splits


def all_distances(Geometry: Any, cutoff: Union[float, torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Computes all pairwise atomic distances in the geometry up to a cutoff.

    Args:
        Geometry: Geometry object (from TBMalt) containing atomic positions.
        cutoff (float or torch.Tensor): Distance cutoff(s) for filtering.

    Returns:
        torch.Tensor or list: Tensor of distances if scalar cutoff, list of tensors if cutoff is 1D tensor.
    """
    distances_list = []
    for _, _, distances in atomic_pair_distances(
        Geometry, ignore_self=True, force_batch_index=True
    ):
        distances_list.append(distances)

    all_distances_tensor = torch.cat(distances_list, dim=0)

    if torch.numel(torch.as_tensor(cutoff)) == 1:
        return all_distances_tensor[all_distances_tensor <= cutoff]
    else:
        cutoff_tensor = torch.as_tensor(cutoff, device=all_distances_tensor.device)
        mask = all_distances_tensor[None, :] <= cutoff_tensor[:, None]
        return [all_distances_tensor[m] for m in mask]


def count_distances(Geometry: Any, cutoff: Union[float, torch.Tensor], decimals: int = 3) -> Dict[float, int]:
    """
    Counts the number of occurrences of each distance in the geometry,
    rounded to a specified number of decimals.

    Args:
        Geometry: Geometry object containing atomic positions.
        cutoff (float or torch.Tensor): Maximum distance(s) to consider.
        decimals (int, optional): Number of decimal places to round distances. Defaults to 3.

    Returns:
        dict: Mapping from rounded distance to count.
    """
    alldistances = all_distances(Geometry, cutoff)

    if isinstance(alldistances, torch.Tensor):
        values = alldistances.cpu().numpy()
    else:
        values = np.array(alldistances)

    rounded = np.round(values, decimals=decimals)
    return dict(Counter(rounded.tolist()))



def shortest_distance(distances: Union[torch.Tensor, List[torch.Tensor]]) -> Union[float, List[float]]:
    """
    Get the shortest non-zero distance(s) from a distance matrix or list of them.

    Args:
        distances (Union[torch.Tensor, List[torch.Tensor]]):
            - If Tensor: shape (N, N) distance matrix.
            - If List[Tensor]: list of distance matrices.

    Returns:
        float or List[float]: The shortest distance (single tensor case) 
                              or list of shortest distances (list case).
    """

    def _min_nonzero(d: torch.Tensor) -> float:
        vals = d.flatten()
        nonzero = vals[vals > 0]
        if nonzero.numel() == 0:
            raise ValueError("No valid non-zero distances found")
        return nonzero.min().item()

    if isinstance(distances, list):
        return [_min_nonzero(d) for d in distances]
    else:
        return _min_nonzero(distances)

def sum_dict_values(d: Dict[Any, float]) -> float:
    """
    Sums all values in a dictionary.

    Args:
        d (dict): Dictionary with numerical values.

    Returns:
        float: Sum of all values.
    """
    return sum(d.values())


def stringlist_list_converter(string_list: List[str]) -> List[float]:
    """
    Converts a list of strings containing numbers in brackets into a list of floats.

    Args:
        string_list (list of str): List of strings to convert.

    Returns:
        list of float: Extracted numbers as floats.
    """
    full_str = " ".join(string_list)
    start = full_str.find('[') + 1
    end = full_str.find(']')
    numbers_str = full_str[start:end]
    numbers = [float(x) for x in numbers_str.replace(',', ' ').split()]
    return numbers


def random_sample_from_tensor(tensor: torch.Tensor, n: int) -> List[Any]:
    """
    Randomly selects n elements from a 1D tensor.

    Args:
        tensor (torch.Tensor): 1D tensor to sample from.
        n (int): Number of elements to sample.

    Returns:
        list: Randomly sampled elements as a list.
    """
    if tensor.ndim != 1:
        raise ValueError("The tensor must be 1-dimensional.")
    if n > tensor.size(0):
        raise ValueError("n cannot be larger than the length of the tensor.")

    indices = torch.randperm(tensor.size(0))[:n]
    return tensor[indices].tolist()


def select_random_datapoints(N: int, seed: int, max_dpoint: int = 6306) -> List[int]:
    """
    Randomly selects N unique datapoints from 1 to max_dpoint using a fixed seed.

    Args:
        N (int): Number of datapoints to select.
        seed (int): Random seed.
        max_dpoint (int, optional): Maximum datapoint index. Defaults to 6306.

    Returns:
        list of int: List of randomly selected datapoints.
    """
    random.seed(seed)
    return random.sample(range(1, max_dpoint + 1), N)


if __name__ == "__main__":
    zahlen =[]
    for ii in range(1, 6307):
        Geo = load_Geo_dset("dft.hdf5", [ii])
        alldist = all_distances(Geo, 8.0)
        shortest = shortest_distance(alldist)
        zahlen.append(shortest)
        print(zahlen)

    indizes_sortiert = sorted(range(len(zahlen)), key=lambda i: zahlen[i])
    sort = [zahlen[i] for i in indizes_sortiert]

    print("Zahlen:", zahlen)
    print("Sortierte Indizes:", indizes_sortiert)
    print("Sortierte Werte:", sort)
    with open("ergebnisse.txt", "w") as f:
        f.write(f"Zahlen: {zahlen}\n")
        f.write(f"Indizes sortiert: {indizes_sortiert}\n")
            


