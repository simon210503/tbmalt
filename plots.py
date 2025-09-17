import os
import torch

import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from pathlib import Path
from tbmalt.physics.dftb.feeds import DftbpRepulsiveSpline
from tbmalt.io.skf import Skf

from utils import get_atom_count_from_hdf5, all_distances, select_random_datapoints
from saveload import load_repulsives, load_Geo_dset, load_testdpoints_from_file, load_parameters
from formation_calc import calc_reference_formation_energies, calc_formation_energy, prepare_system
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
    filepath: str = "plots/formation_energy_plot.png",
    axis_limits: tuple[float, float] | None = None  # optionales symmetrisches Limit
) -> None:
    """
    Plot target vs. model formation energies with markers grouped by dataset IDs and custom legend labels.
    Optionally, set symmetric axis limits.
    """
    def to_numpy(x):
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

    # IDs aus HDF5
    atom_counts_test = get_atom_count_from_hdf5(sample_path, datapoints).tolist()
    atom_counts_train: list[int] = []
    if target_train_np.size > 0:
        atom_counts_train = get_atom_count_from_hdf5(sample_path, datapoints).tolist()

    # Mapping IDs → Labels
    label_map = {
        63: "Vacancies",
        64: "Perfect/MD",
        65: "Interstitials",
        511: "Vacancy",
        512: "Perfect",
        513: "Interstitials"
    }

    plt.figure(figsize=(6, 6))
    unique_ids = sorted(set(atom_counts_train + atom_counts_test))

    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']

    color_map = {uid: colors[i % len(colors)] for i, uid in enumerate(unique_ids)}
    marker_map = {uid: markers[i % len(markers)] for i, uid in enumerate(unique_ids)}

    # Dummy scatter für Legende
    for uid in unique_ids:
        plt.scatter([], [], color=color_map[uid], marker=marker_map[uid], 
                    label=label_map.get(uid, str(uid)), edgecolor='k')

    # Training
    if target_train_np.size > 0:
        for uid in unique_ids:
            idx_train = [i for i, a in enumerate(atom_counts_train) if a == uid]
            if idx_train:
                plt.scatter(
                    target_train_np[idx_train],
                    model_train_np[idx_train],
                    color=color_map[uid],
                    marker=marker_map[uid],
                    alpha=0.7,
                    edgecolor='k'
                )

    # Testing
    for uid in unique_ids:
        idx_test = [i for i, a in enumerate(atom_counts_test) if a == uid]
        if idx_test:
            plt.scatter(
                target_test_np[idx_test],
                model_test_np[idx_test],
                color=color_map[uid],
                marker=marker_map[uid],
                alpha=0.7,
                edgecolor='k'
            )

    # Diagonale
    if axis_limits is not None:
        min_val, max_val = axis_limits
    else:
        all_targets = np.concatenate([target_train_np, target_test_np])
        if all_targets.size > 0:
            min_val, max_val = all_targets.min(), all_targets.max()
        else:
            min_val, max_val = 0, 1  # Fallback, falls keine Daten vorhanden

    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1)

    # Achsenlimits
    if axis_limits is not None:
        plt.xlim(axis_limits)
        plt.ylim(axis_limits)


    # Labels und Ticks
    plt.xlabel("Target Formation Energy [Ha]", fontsize=18)
    plt.ylabel("Model Formation Energy [Ha]", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.grid(True)
    plt.tight_layout()
    plt.legend(fontsize=15)

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

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(model_names):
        ax.bar(x + offsets[i], errors[:, i], width=bar_width, label=model)

    ax.set_ylabel(f"{metric_name} [Ha$^2$]", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Split {i}" for i in range(num_splits)], fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    fig.tight_layout()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_errors_512(errors: list[float], metric_name: str = "MSE", filepath: str = "plots/errors/512_bar.png") -> None:
    """
    Plot error values for four models as a simple bar chart.

    Args:
        errors (list[float]): List of exactly 4 error values.
        metric_name (str): Name of the error metric to display.
        filepath (str): Path where the plot should be saved.
    """
    model_names = ["xTB", "PTBP", "Gamma", "pbc"]
    colors = ["b", "g", "r", "y"]  # gleiche Reihenfolge wie in plot_repulsive_curves

    if len(errors) != 4:
        raise ValueError(f"Fehlerliste muss genau 4 Werte enthalten, aber hat {len(errors)}.")

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(model_names, errors, color=colors)

    ax.set_ylabel(f"{metric_name} [Ha$^2$]", fontsize=18)
    ax.set_xticklabels(model_names, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)

    # Legende mit den Model-Namen
    ax.legend(bars, model_names, fontsize=12, loc="upper right")

    ax.grid(axis='y', linestyle='--', alpha=0.6)
    fig.tight_layout()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)



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

    r = torch.arange(3.5, 6.5, 0.01)

    a = Gamma.forward(Tensor(r))
    b = xTB.forward(Tensor(r))
    c = PTBP.forward(Tensor(r))
    d = PBC.forward(Tensor(r))

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(r.numpy(), a.detach().numpy(), 'r', label='Gamma')
    ax.plot(r.numpy(), b.detach().numpy(), 'b', label='xTB')
    ax.plot(r.numpy(), c.detach().numpy(), 'g', label='PTBP')
    ax.plot(r.numpy(), d.detach().numpy(), 'y', label='pbc')


    if all_distances is not None:
        if isinstance(all_distances, torch.Tensor):
            values = all_distances.cpu().numpy()
        else:
            values = np.array(all_distances)

        rounded = np.round(values, decimals=3)
        for i, dist in enumerate(rounded):
            ax.axvline(
                x=dist.item(),
                color='k',
                linestyle='--',
                alpha=0.3,
                linewidth=0.5
            )

    # Sichtbereich so setzen, dass die Kurven bis zum Rand gehen
    ax.set_xlim(r.min().item(), r.max().item())

    ax.set_xlabel('Distance [Bohr]', fontsize=12)
    ax.set_ylabel('Repulsive Potential [Ha]', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        fig.tight_layout()
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
        distances = all_distances(Geometry, 6.0)
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
    plt.hist(dist1.numpy(), bins=bins, alpha=0.5, density=True, label="Vacancies", color='blue', edgecolor='black')
    plt.hist(dist2.numpy(), bins=bins, alpha=0.5, density=True, label="Perfect/MD", color='green', edgecolor='black')
    plt.hist(dist3.numpy(), bins=bins, alpha=0.5, density=True, label="Interstitials", color='red', edgecolor='black')

    plt.xlabel("Distance [bohr]")
    plt.ylabel("Distribution of distances")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_formation_energies_from_saved_params(base_path, base_save_path, sample_path, axis_limit):
    Gamma_alpha = load_parameters(f'{base_path}/Gamma_result/result_8.txt')
    Gamma_Z = load_parameters(f'{base_path}/Gamma_result/result_9.txt')
    xTB_alpha = load_parameters(f'{base_path}/xTB_result/result_8.txt')
    xTB_Z = load_parameters(f'{base_path}/xTB_result/result_9.txt')
    PTBP_alpha = load_parameters(f'{base_path}/PTBP_result/result_8.txt')
    PTBP_Z = load_parameters(f'{base_path}/PTBP_result/result_9.txt')

    from torch.nn import Parameter

    # Beispielhafte Funktion, um die geladenen Werte zu verpacken
    def wrap_as_dict(value: float | list[float], key: int = 14) -> dict[int, Parameter]:
        """
        Verpackt einen Wert oder eine Liste in das gewünschte Format:
        {14: Parameter(Tensor([...]), requires_grad=True)}

        Args:
            value (float | list[float]): Eingabewert(e).
            key (int): Schlüssel des Dictionaries.

        Returns:
            dict[int, Parameter]: Dictionary mit dem Schlüssel und Parameter.
        """
        if isinstance(value, list):
            tensor = torch.tensor(value, dtype=torch.float32)
        else:
            tensor = torch.tensor([value], dtype=torch.float32)

        return {key: Parameter(tensor, requires_grad=True)}


    # Verwendung für deine Variablen
    Gamma_alpha = wrap_as_dict(Gamma_alpha, key=14)
    Gamma_Z = wrap_as_dict(Gamma_Z, key=14)
    xTB_alpha = wrap_as_dict(xTB_alpha, key=14)
    xTB_Z = wrap_as_dict(xTB_Z, key=14)
    PTBP_alpha = wrap_as_dict(PTBP_alpha, key=14)
    PTBP_Z = wrap_as_dict(PTBP_Z, key=14)

    dpoints = load_testdpoints_from_file(base_path)
    target_test = calc_reference_formation_energies(sample_path, dpoints)

    params = prepare_system(sample_path, dpoints, DFTBGammaRepulsive, Gamma_alpha, Gamma_Z)
    model_test = calc_formation_energy(params)
    plot_formation_energies_new(dpoints, sample_path, target_test, model_test, 
                                filepath=os.path.join(base_save_path, 'Gamma.png'), axis_limits=axis_limit)
    params = prepare_system(sample_path, dpoints, xTBRepulsive, xTB_alpha, xTB_Z)
    model_test = calc_formation_energy(params)
    plot_formation_energies_new(dpoints, sample_path, target_test, model_test, 
                                filepath=os.path.join(base_save_path, 'xTB.png'), axis_limits=axis_limit)
    params = prepare_system(sample_path, dpoints, PTBPRepulsive, PTBP_alpha, PTBP_Z)
    model_test = calc_formation_energy(params)
    plot_formation_energies_new(dpoints, sample_path, target_test, model_test, 
                                filepath=os.path.join(base_save_path, 'PTBP.png'), axis_limits=axis_limit)
    params = prepare_system(sample_path, dpoints, 'pbc', PTBP_alpha, PTBP_Z)
    model_test = calc_formation_energy(params)
    plot_formation_energies_new(dpoints, sample_path, target_test, model_test, 
                                filepath=os.path.join(base_save_path, 'pbc.png'), axis_limits=axis_limit)
    params = prepare_system(sample_path, dpoints, 'siband', PTBP_alpha, PTBP_Z)
    model_test = calc_formation_energy(params)
    plot_formation_energies_new(dpoints, sample_path, target_test, model_test, 
                                filepath=os.path.join(base_save_path, 'siband.png'), axis_limits=axis_limit)


def plot_formation_energies_from_saved_params_random_dpoints(base_path, base_save_path, sample_path, seed, axis_limit):
    Gamma_alpha = load_parameters(f'{base_path}/Gamma_result/result_8.txt')
    Gamma_Z = load_parameters(f'{base_path}/Gamma_result/result_9.txt')
    xTB_alpha = load_parameters(f'{base_path}/xTB_result/result_8.txt')
    xTB_Z = load_parameters(f'{base_path}/xTB_result/result_9.txt')
    PTBP_alpha = load_parameters(f'{base_path}/PTBP_result/result_8.txt')
    PTBP_Z = load_parameters(f'{base_path}/PTBP_result/result_9.txt')

    from torch.nn import Parameter
    from utils import select_random_datapoints

    # Beispielhafte Funktion, um die geladenen Werte zu verpacken
    def wrap_as_dict(value: float | list[float], key: int = 14) -> dict[int, Parameter]:
        """
        Verpackt einen Wert oder eine Liste in das gewünschte Format:
        {14: Parameter(Tensor([...]), requires_grad=True)}

        Args:
            value (float | list[float]): Eingabewert(e).
            key (int): Schlüssel des Dictionaries.

        Returns:
            dict[int, Parameter]: Dictionary mit dem Schlüssel und Parameter.
        """
        if isinstance(value, list):
            tensor = torch.tensor(value, dtype=torch.float32)
        else:
            tensor = torch.tensor([value], dtype=torch.float32)

        return {key: Parameter(tensor, requires_grad=True)}


    # Verwendung für deine Variablen
    Gamma_alpha = wrap_as_dict(Gamma_alpha, key=14)
    Gamma_Z = wrap_as_dict(Gamma_Z, key=14)
    xTB_alpha = wrap_as_dict(xTB_alpha, key=14)
    xTB_Z = wrap_as_dict(xTB_Z, key=14)
    PTBP_alpha = wrap_as_dict(PTBP_alpha, key=14)
    PTBP_Z = wrap_as_dict(PTBP_Z, key=14)

    dpoints = select_random_datapoints(1000, seed)
    target_test = calc_reference_formation_energies(sample_path, dpoints)

    params = prepare_system(sample_path, dpoints, DFTBGammaRepulsive, Gamma_alpha, Gamma_Z)
    model_test = calc_formation_energy(params)
    plot_formation_energies_new(dpoints, sample_path, target_test, model_test, 
                                filepath=os.path.join(base_save_path, 'Gamma.png'), axis_limits=axis_limit)
    params = prepare_system(sample_path, dpoints, xTBRepulsive, xTB_alpha, xTB_Z)
    model_test = calc_formation_energy(params)
    plot_formation_energies_new(dpoints, sample_path, target_test, model_test, 
                                filepath=os.path.join(base_save_path, 'xTB.png'), axis_limits=axis_limit)
    params = prepare_system(sample_path, dpoints, PTBPRepulsive, PTBP_alpha, PTBP_Z)
    model_test = calc_formation_energy(params)
    plot_formation_energies_new(dpoints, sample_path, target_test, model_test, 
                                filepath=os.path.join(base_save_path, 'PTBP.png'), axis_limits=axis_limit)
    params = prepare_system(sample_path, dpoints, 'pbc', PTBP_alpha, PTBP_Z)
    model_test = calc_formation_energy(params)
    plot_formation_energies_new(dpoints, sample_path, target_test, model_test, 
                                filepath=os.path.join(base_save_path, 'pbc.png'), axis_limits=axis_limit)
    params = prepare_system(sample_path, dpoints, 'siband', PTBP_alpha, PTBP_Z)
    model_test = calc_formation_energy(params)
    plot_formation_energies_new(dpoints, sample_path, target_test, model_test, 
                                filepath=os.path.join(base_save_path, 'siband.png'), axis_limits=axis_limit)


if __name__ == "__main__":
    """
    from utils import find_structures_with_atom_count
    path = 'dft.hdf5'
    alld = list(range(1,6307))
    datapoints1 = find_structures_with_atom_count(path, alld, 63).tolist()
    datapoints2 = find_structures_with_atom_count(path, alld, 64).tolist()
    datapoints3 = find_structures_with_atom_count(path, alld, 65).tolist()
    #plot_normalized_distance_counts(datapoints)
    #plot_distance_distribution(datapoints)
    plot_distance_distributions_aligned(datapoints1, datapoints2, datapoints3, save_path = 'plots/distance_distribution.png')

    #from saveload import load_MSE_from_file
    base_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/logs/64routine/split0')
    save_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/plots/64routine/formation_energies/split0')
    sample_path = 'dft.hdf5'
    #MSE = load_MSE_from_file(base_path)
    #plot_errors_bar64(MSE, metric_name='MSE', filepath=os.path.join(save_path, 'MSE.png'))
    axis_limit = (-0.1, 0.65)

    plot_formation_energies_from_saved_params(base_path, save_path, sample_path, axis_limit)

    print(1)

    axis_limit = (-0.02, 0.41)

    sample_path = 'dft_test.hdf5'

    base_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/logs/512routine/512test')
    save_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/plots/512routine/formation_energies/512test')
    plot_formation_energies_from_saved_params(base_path, save_path, sample_path, axis_limit)

    print(2)

    axis_limit = (0.12, 0.48)

    sample_path = 'dft.hdf5'

    base_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/logs/63_only_routine/split0')
    save_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/plots/63_only_routine/formation_energies/split0')
    plot_formation_energies_from_saved_params(base_path, save_path, sample_path, axis_limit)

    print(3)

    axis_limit = (-0.02, 0.43)

    sample_path = 'dft_test.hdf5'

    base_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/logs/63_train_512routine/512test')
    save_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/plots/63_train_512routine/formation_energies/512test')
    plot_formation_energies_from_saved_params(base_path, save_path, sample_path, axis_limit)

    print(4)

    axis_limit = (-0.02, 0.48)

    sample_path = 'dft.hdf5'

    base_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/logs/6364_only_routine/split0')
    save_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/plots/6364_only_routine/formation_energies/split0')
    plot_formation_energies_from_saved_params(base_path, save_path, sample_path, axis_limit)

    print(5)

    axis_limit = (-0.02, 0.44)

    sample_path = 'dft_test.hdf5'

    base_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/logs/6364_train_512routine/512test')
    save_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/plots/6364_train_512routine/formation_energies/512test')
    plot_formation_energies_from_saved_params(base_path, save_path, sample_path, axis_limit)

    print(6)

    axis_limit = (-0.02, 0.78)

    sample_path = 'dft.hdf5'

    base_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/logs/63_train_512routine/512test')
    save_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/plots/extra63')
    plot_formation_energies_from_saved_params_random_dpoints(base_path, save_path, sample_path, 1234, axis_limit)

    print(7)

    axis_limit = (-0.02, 0.6)

    base_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/logs/6364_train_512routine/512test')
    save_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/plots/extra6364')
    plot_formation_energies_from_saved_params_random_dpoints(base_path, save_path, sample_path, 12345, axis_limit)

    print(8)
    Geo = load_Geo_dset('dft.hdf5', [1])
    distances = all_distances(Geo, 6.0)

    base_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/logs/6364_train_512routine/512test')
    save_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/plots/6364_train_512routine/formation_energies/512test/1.png')
    plot_repulsive_curves(base_path, distances, save_path)

    base_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/logs/512routine/512test')
    save_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/plots/512routine/MSE.png')

    from saveload import load_MSE_from_file, load_MSE512_from_file

    #MSE = load_MSE_from_file(base_path)
    MSE = load_MSE512_from_file(base_path)

    plot_errors_512(MSE, filepath=save_path)
    """

    base_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/logs/512routine/512test')
    save_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/plots/512routine/repulsives')
    plot_repulsives_relaxed_w_distances(base_path, save_path)

    base_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/logs/63_train_512routine/512test')
    save_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/plots/63_train_512routine/formation_energies/512test')
    plot_repulsives_relaxed_w_distances(base_path, save_path)

    base_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/logs/6364_train_512routine/512test')
    save_path = Path(r'C:/Users/simon/Desktop/runs_for_thesis/plots/6364_train_512routine/formation_energies/512test')
    plot_repulsives_relaxed_w_distances(base_path, save_path)