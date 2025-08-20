import os
import torch

import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from tbmalt.physics.dftb.feeds import DftbpRepulsiveSpline
from tbmalt.io.skf import Skf

from utils import get_atom_count_from_hdf5, all_distances, select_random_datapoints
from saveload import load_repulsives, load_Geo_dset
from new_feeds import DFTBGammaRepulsive, xTBRepulsive, PTBPRepulsive


def plot_loss(loss_history: list[float]) -> None:
    """
    Plot the training loss curve over epochs.

    Args:
        loss_history (list[float]): A list of loss values for each epoch.
    """
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.tick_params(
        direction='in', labelsize=26, width=1.5, length=5,
        top=True, right=True, zorder=10
    )
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Epoch", fontsize=28)
    plt.ylabel("Loss", fontsize=28)
    plt.show()


def save_loss_plot(loss_history: list[float], path: str, filename: str) -> None:
    """
    Save the training loss curve as an image file.

    Args:
        loss_history (list[float]): A list of loss values for each epoch.
        path (str): Directory where the plot should be saved.
        filename (str): File name of the saved plot.
    """
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.tick_params(
        direction='in', labelsize=26, width=1.5, length=5,
        top=True, right=True, zorder=10
    )

    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Epoch", fontsize=28)
    plt.ylabel("Loss", fontsize=28)

    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_formation_energies_new(
    datapoints: list[int],
    sample_path: str,
    target_test: Tensor | np.ndarray | list[float],
    model_test: Tensor | np.ndarray | list[float],
    target_train: Tensor | np.ndarray | list[float] | None = None,
    model_train: Tensor | np.ndarray | list[float] | None = None,
    filepath: str = "plots/formation_energy_plot.png"
) -> None:
    """
    Plot target vs. model formation energies with markers grouped by atom counts.

    Args:
        datapoints (list[int]): List of datapoint indices.
        sample_path (str): Path to the dataset.
        target_test (Tensor | np.ndarray | list[float]): Target values for test set.
        model_test (Tensor | np.ndarray | list[float]): Model predictions for test set.
        target_train (Tensor | np.ndarray | list[float] | None): Target values for training set.
        model_train (Tensor | np.ndarray | list[float] | None): Model predictions for training set.
        filepath (str): Path where the plot should be saved.
    """
    def to_numpy(x: Tensor | np.ndarray | list[float] | None) -> np.ndarray:
        if x is None:
            return np.array([])
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, list):
            return np.array(x)
        elif isinstance(x, np.ndarray):
            return x
        else:
            raise TypeError(f"Unsupported data type: {type(x)}")

    target_train_np = to_numpy(target_train)
    model_train_np = to_numpy(model_train)
    target_test_np = to_numpy(target_test)
    model_test_np = to_numpy(model_test)

    if target_test_np.size == 0:
        print("Warnung: Keine Testdaten zum Plotten vorhanden. Der Plot wird nicht gespeichert.")
        return

    atom_counts_test = get_atom_count_from_hdf5(sample_path, datapoints).tolist()
    atom_counts_train: list[int] = []
    if target_train_np.size > 0:
        atom_counts_train = get_atom_count_from_hdf5(sample_path, datapoints).tolist()

    plt.figure(figsize=(6, 6))
    unique_atom_counts = sorted(set(atom_counts_train + atom_counts_test))

    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']

    color_map = {atom_num: colors[i] for i, atom_num in enumerate(unique_atom_counts)}
    marker_map = {atom_num: markers[i] for i, atom_num in enumerate(unique_atom_counts)}

    for atom_num in unique_atom_counts:
        plt.scatter([], [], color=color_map[atom_num], marker=marker_map[atom_num], label=str(atom_num), edgecolor='k')

    if target_train_np.size > 0:
        for atom_num in unique_atom_counts:
            idx_train = [i for i, a in enumerate(atom_counts_train) if a == atom_num]
            if idx_train:
                plt.scatter(
                    target_train_np[idx_train],
                    model_train_np[idx_train],
                    color=color_map[atom_num],
                    marker=marker_map[atom_num],
                    alpha=0.7,
                    edgecolor="k",
                )

    for atom_num in unique_atom_counts:
        idx_test = [i for i, a in enumerate(atom_counts_test) if a == atom_num]
        if idx_test:
            plt.scatter(
                target_test_np[idx_test],
                model_test_np[idx_test],
                color=color_map[atom_num],
                marker=marker_map[atom_num],
                alpha=0.7,
                edgecolor="k",
            )

    all_targets = np.concatenate([target_train_np, target_test_np])
    if all_targets.size > 0:
        min_val, max_val = all_targets.min(), all_targets.max()
        plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1)

    plt.xlabel("Target Formation Energy [Ha]")
    plt.ylabel("Model Formation Energy [Ha]")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(fontsize=9)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_formation_energies(
    target_test: Tensor | np.ndarray | list[float],
    model_test: Tensor | np.ndarray | list[float],
    target_train: Tensor | np.ndarray | list[float] | None = None,
    model_train: Tensor | np.ndarray | list[float] | None = None,
    filepath: str = "plots/formation_energy_plot.png"
) -> None:
    """
    Deprecated version of the formation energies plot.

    Args:
        target_test (Tensor | np.ndarray | list[float]): Target values for test set.
        model_test (Tensor | np.ndarray | list[float]): Model predictions for test set.
        target_train (Tensor | np.ndarray | list[float] | None): Target values for training set.
        model_train (Tensor | np.ndarray | list[float] | None): Model predictions for training set.
        filepath (str): Path where the plot should be saved.
    """
    def to_numpy(x: Tensor | np.ndarray | list[float] | None) -> np.ndarray:
        if x is None:
            return np.array([])
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, list):
            return np.array(x)
        elif isinstance(x, np.ndarray):
            return x
        else:
            raise TypeError(f"Unsupported data type: {type(x)}")

    target_train_np = to_numpy(target_train)
    model_train_np = to_numpy(model_train)
    target_test_np = to_numpy(target_test)
    model_test_np = to_numpy(model_test)

    if target_test_np.size == 0:
        print("Warnung: Keine Testdaten zum Plotten vorhanden. Der Plot wird nicht gespeichert.")
        return

    plt.figure(figsize=(6, 6))
    if target_train_np.size > 0:
        plt.scatter(target_train_np, model_train_np, label="Training", color="blue", alpha=0.6)
    plt.scatter(target_test_np, model_test_np, label="Testing", color="orange", alpha=0.6)

    all_targets = np.concatenate([target_train_np, target_test_np])
    if all_targets.size > 0:
        min_val, max_val = all_targets.min(), all_targets.max()
        plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1)

    plt.xlabel("Target Formation Energy [Ha]")
    plt.ylabel("Model Formation Energy [Ha]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_errors_bar64(errors: list[float], metric_name: str = "MSE", filepath: str = "plots/errors_bar.png") -> None:
    """
    Plot error values as grouped bar plots across multiple splits.

    Args:
        errors (list[float]): List of error values, reshaped into (num_splits, num_models).
        metric_name (str): Name of the error metric to display.
        filepath (str): Path where the plot should be saved.
    """
    model_names = ["xTB", "PTBP", "Gamma", "pbc"]
    num_splits = 5
    num_models = len(model_names)

    errors = np.array(errors).reshape((num_splits, num_models))
    x = np.arange(num_splits)
    bar_width = 0.18
    offsets = np.linspace(-bar_width * 1.5, bar_width * 1.5, num_models)

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(model_names):
        plt.bar(x + offsets[i], errors[:, i], width=bar_width, label=model)

    plt.xlabel("Splits")
    plt.ylabel(metric_name)
    plt.xticks(x, [f"Split {i}" for i in range(num_splits)])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_errors_512(errors: list[float], metric_name: str = "MSE", filepath: str = "plots/errors/512_bar.png") -> None:
    """
    Plot error values for four models as a simple bar chart.

    Args:
        errors (list[float]): List of exactly 4 error values.
        metric_name (str): Name of the error metric to display.
        filepath (str): Path where the plot should be saved.
    """
    model_names = ["xTB", "PTBP", "Gamma", "pbc"]

    if len(errors) != 4:
        raise ValueError(f"Fehlerliste muss genau 4 Werte enthalten, aber hat {len(errors)}.")

    plt.figure(figsize=(8, 5))
    plt.bar(model_names, errors, color="skyblue")

    plt.ylabel(metric_name)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_repulsive_curves(base_path: str, all_distances: list[Tensor] | None = None, save_path: str | None = None) -> None:
    """
    Plot repulsive energy curves for different models (xTB, PTBP, Gamma, PBC).

    Args:
        base_path (str): Path to load trained repulsives.
        all_distances (list[Tensor] | None): Optional list of distances to mark on the plot.
        save_path (str | None): File path to save the plot. If None, the plot is shown.
    """
    xTB, PTBP, Gamma = load_repulsives(base_path)

    file_path = "pbc.h5"
    atom_pair = [14, 14]
    skf_data = Skf.read(path=file_path, atom_pair=atom_pair, device="cpu", dtype=torch.float64)
    PBC = DftbpRepulsiveSpline.from_skf(skf_data)

    r = torch.arange(4, 9, 0.1)

    a = Gamma.forward(Tensor(r))
    b = xTB.forward(Tensor(r))
    c = PTBP.forward(Tensor(r))
    d = PBC.forward(Tensor(r))

    fig, ax = plt.subplots()
    ax.plot(r.numpy(), a.detach().numpy(), 'r', label='Gamma')
    ax.plot(r.numpy(), b.detach().numpy(), 'b', label='xTB')
    ax.plot(r.numpy(), c.detach().numpy(), 'g', label='PTBP')
    ax.plot(r.numpy(), d.detach().numpy(), 'y', label='pbc')

    if all_distances is not None:
        for i, d in enumerate(all_distances):
            ax.axvline(
                x=d.item(),
                color='k',
                linestyle='--',
                alpha=0.3,
                linewidth=0.5,
                label='distances' if i == 0 else None
            )

    ax.set_xlabel('Distance [bohr]')
    ax.set_ylabel('Repulsive energy [Ha]')
    ax.legend(loc="upper right")
    ax.grid(True)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_repulsives_relaxed_w_distances(log_path: str, save_dir: str) -> None:
    """
    Plot repulsive curves for multiple geometries with distance markers.

    Args:
        log_path (str): Path to load repulsives.
        save_dir (str): Directory to save the generated plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    for ii in range(1, 7):
        Geometry = load_Geo_dset('dft.hdf5', [ii])
        distances = all_distances(Geometry, 8.0)
        file_name = f"{ii}.png"
        save_path = os.path.join(save_dir, file_name)
        plot_repulsive_curves(log_path, distances, save_path)


def plot_distance_counts(datapoints: list[int], save_path: str = None) -> None:
    cutoffs = torch.arange(0, 11, 0.1)
    plt.figure(figsize=(7,5))

    for dp in datapoints:
        Geometry = load_Geo_dset('dft.hdf5', [dp])
        all_dists_list = all_distances(Geometry, cutoffs)
        counts = [d.numel() for d in all_dists_list]
        plt.plot(cutoffs.numpy(), counts, marker='o', label=f'Datapoint {dp}')

    plt.xlabel("Cutoff [bohr]")
    plt.ylabel("Absolute amount of repulsive interactions")
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_normalized_distance_counts(datapoints: list[int], save_path: str = None) -> None:
    cutoffs = torch.arange(0, 11, 0.1)
    plt.figure(figsize=(7,5))

    for dp in datapoints:
        Geometry = load_Geo_dset('dft.hdf5', [dp])
        n_atoms = Geometry.n_atoms
        all_dists_list = all_distances(Geometry, cutoffs)
        counts = torch.tensor([d.numel() for d in all_dists_list], dtype=torch.float32)
        normalized_counts = counts / n_atoms
        plt.plot(cutoffs.numpy(), normalized_counts.numpy(), marker='o', label=f'Datapoint {dp}')

    plt.xlabel("Cutoff [bohr]")
    plt.ylabel("Repulsive Interactions per number of atoms")
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_distance_distribution(datapoints: list[int], save_path: str = None) -> None:
    all_distances_flat = []

    for dp in datapoints:
        Geometry = load_Geo_dset('dft.hdf5', [dp])
        dists_list = all_distances(Geometry, 10.0)
        for d in dists_list:
            all_distances_flat.extend(d.flatten().tolist())

    all_distances_tensor = torch.tensor(all_distances_flat)

    plt.figure(figsize=(7,5))
    plt.hist(all_distances_tensor.numpy(), bins=50, color='skyblue', edgecolor='black', density=True)
    plt.xlabel("Distance [bohr]")
    plt.ylabel("Distribution of distances")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_distance_distributions_aligned(list1: list[int], list2: list[int], list3: list[int],
                                        max_cutoff: float = 10.0, bin_width: float = 0.1,
                                        save_path: str = None) -> None:

    def gather_distances(datapoints: list[int]) -> torch.Tensor:
        all_dists = []
        for dp in datapoints:
            Geometry = load_Geo_dset('dft.hdf5', [dp])
            dists_list = all_distances(Geometry, max_cutoff)
            for d in dists_list:
                all_dists.extend(d.flatten().tolist())
        return torch.tensor(all_dists)

    dist1 = gather_distances(list1)
    dist2 = gather_distances(list2)
    dist3 = gather_distances(list3)

    bins = np.arange(3.5, max_cutoff + bin_width, bin_width)

    plt.figure(figsize=(8,5))
    plt.hist(dist1.numpy(), bins=bins, alpha=0.5, density=True, label="Set 1", color='blue', edgecolor='black')
    plt.hist(dist2.numpy(), bins=bins, alpha=0.5, density=True, label="Set 2", color='green', edgecolor='black')
    plt.hist(dist3.numpy(), bins=bins, alpha=0.5, density=True, label="Set 3", color='red', edgecolor='black')

    plt.xlabel("Distance [bohr]")
    plt.ylabel("Distribution of distances")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    from utils import find_structures_with_atom_count
    path = 'dft.hdf5'
    alld = list(range(1,6307))
    datapoints1 = find_structures_with_atom_count(path, alld, 63).tolist()
    datapoints2 = find_structures_with_atom_count(path, alld, 64).tolist()
    datapoints3 = find_structures_with_atom_count(path, alld, 65).tolist()
    #plot_normalized_distance_counts(datapoints)
    #plot_distance_distribution(datapoints)
    plot_distance_distributions_aligned(datapoints1, datapoints2, datapoints3, save_path = 'plots/distance_distribution.png')