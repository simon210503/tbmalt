import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import ast
from formation_calc import select_random_datapoints
from training import train_model
from testing import test_model
from new_feeds import PTBPRepulsive, DFTBGammaRepulsive, xTBRepulsive
from torch.nn import Parameter



def split_data(dpoints, portions: int = 5):
    """
    Teilt die Liste `dpoints` in `portions` möglichst gleich große Teilmengen auf.
    Funktioniert auch, wenn len(dpoints) nicht durch portions teilbar ist.
    """
    splits = []
    total = len(dpoints)
    base_size = total // portions
    remainder = total % portions

    start = 0
    for i in range(portions):
        # Füge 1 zum Split hinzu, solange noch "Rest" übrig ist
        end = start + base_size + (1 if i < remainder else 0)
        splits.append(dpoints[start:end])
        start = end

    return splits


def train64test64(traindpoints, testdpoints, repulsive_model, initial_alpha, initial_Z):
    """
    Train on (portions - 1) parts of randomly split data and test on the remaining part.
    
    Args:
        total_dpoints (int): Total number of datapoints available.
        seed (int): Random seed for reproducibility.
        max_dpoint (int): Maximum index of datapoints to be used.
        portions (int): Number of equally-sized parts to split the data into.
        repulsive_model: The repulsive model class to be used.
        initial_alpha (dict): Initial alpha parameters.
        initial_Z (dict): Initial Z parameters.
    
    Returns:
        list: [
            traindpoints, testdpoints,
            loss_history, params,
            target_formation_energy, final_formation_energy,
            final_alpha, final_Z,
            MSE, MAE
        ]
    """
    sample_path = 'dft.hdf5'

    loss_history, params, target_formation_energy_training, model_formation_energy_training, final_alpha, final_Z = train_model(
        sample_path,
        traindpoints,
        repulsive_model=repulsive_model,
        alpha=initial_alpha,
        Z=initial_Z
    )

    MSE, MAE, target_formation_energy_testing, model_formation_energy_testing = test_model(
        sample_path,
        testdpoints,
        repulsive_model=repulsive_model,
        alpha=final_alpha,
        Z=final_Z
    )

    return [
        traindpoints,
        testdpoints,
        loss_history,
        params,
        target_formation_energy_testing,
        model_formation_energy_testing,
        target_formation_energy_training,
        model_formation_energy_training,
        final_alpha,
        final_Z,
        MSE,
        MAE
    ]



def train64test512(training_dpoints, repulsive_model, initial_alpha, initial_Z):
    """
    Train on a random subset of datapoints from one file, test on a fixed set from another file.
    
    Args:
        training_dpoints (int): Number of datapoints to draw from training file.
        seed (int): Random seed for reproducibility.
        max_dpoint (int): Maximum datapoint index for training selection.
        repulsive_model: The repulsive model class to be used.
        initial_alpha (dict): Initial alpha parameters.
        initial_Z (dict): Initial Z parameters.
    
    Returns:
        list: [
            traindpoints, testdpoints,
            loss_history, params,
            target_formation_energy, final_formation_energy,
            final_alpha, final_Z,
            MSE, MAE
        ]
    """
    traindpoints = training_dpoints
    testdpoints = [1, 2, 3, 4, 5, 6]

    training_sample_path = 'dft.hdf5'
    testing_sample_path = 'dft_test.hdf5'

    loss_history, params, target_formation_energy_training, model_formation_energy_training, final_alpha, final_Z = train_model(
        training_sample_path,
        traindpoints,
        repulsive_model=repulsive_model,
        alpha=initial_alpha,
        Z=initial_Z
    )

    MSE, MAE, target_formation_energy_testing, model_formation_energy_testing = test_model(
        testing_sample_path,
        testdpoints,
        repulsive_model=repulsive_model,
        alpha=final_alpha,
        Z=final_Z
    )

    return [
        traindpoints,
        testdpoints,
        loss_history,
        params,
        target_formation_energy_testing,
        model_formation_energy_testing,
        target_formation_energy_training,
        model_formation_energy_training,
        final_alpha,
        final_Z,
        MSE,
        MAE
    ]

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def get_atom_count_from_gen(filepath):
    with open(filepath, "r") as f:
        first_line = f.readline().strip()
    return int(first_line.split()[0])

def plot_formation_energies_new(
    datapoints,
    genfiles_path,
    target_test,
    model_test,
    target_train=None,
    model_train=None,
    filepath="plots/formation_energy_plot.png"
):
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

    atom_counts_test = [get_atom_count_from_gen(f"{genfiles_path}/datapoint{dp}.gen") for dp in datapoints]
    atom_counts_train = []
    if target_train_np.size > 0:
        atom_counts_train = [get_atom_count_from_gen(f"{genfiles_path}/datapoint{dp}.gen") for dp in datapoints]

    plt.figure(figsize=(6, 6))

    unique_atom_counts = sorted(set(atom_counts_train + atom_counts_test))

    colors = ['red', 'blue', 'green']  # deutliche Farben
    markers = ['o', 's', '^']  # Kreis, Quadrat, Dreieck

    color_map = {atom_num: colors[i] for i, atom_num in enumerate(unique_atom_counts)}
    marker_map = {atom_num: markers[i] for i, atom_num in enumerate(unique_atom_counts)}

    # Dummy-Punkte nur für die Legende (unsichtbar außerhalb des Bereichs)
    for atom_num in unique_atom_counts:
        plt.scatter([], [], color=color_map[atom_num], marker=marker_map[atom_num], label=str(atom_num), edgecolor='k')

    # Trainingspunkte plotten (ohne Label)
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
    # Testpunkte plotten (ohne Label)
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

    plt.xlabel("Target Formation Energy")
    plt.ylabel("Model Formation Energy")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(fontsize=9)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_formation_energies(
    target_test,
    model_test,
    target_train=None,
    model_train=None,
    filepath="plots/formation_energy_plot.png"
):
    # Hilfsfunktion: konvertiert beliebige Eingabetypen in NumPy-Arrays
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

    # Konvertierung
    target_train_np = to_numpy(target_train)
    model_train_np = to_numpy(model_train)
    target_test_np = to_numpy(target_test)
    model_test_np = to_numpy(model_test)

    # Prüfen, ob wenigstens Testdaten vorhanden sind
    if target_test_np.size == 0:
        print("Warnung: Keine Testdaten zum Plotten vorhanden. Der Plot wird nicht gespeichert.")
        return

    # Plot erstellen
    plt.figure(figsize=(6, 6))
    if target_train_np.size > 0:
        plt.scatter(target_train_np, model_train_np, label="Training", color="blue", alpha=0.6)
    plt.scatter(target_test_np, model_test_np, label="Testing", color="orange", alpha=0.6)

    # Diagonale Linie
    all_targets = np.concatenate([target_train_np, target_test_np])
    if all_targets.size > 0:
        min_val, max_val = all_targets.min(), all_targets.max()
        plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1)

    plt.xlabel("Target Formation Energy")
    plt.ylabel("Model Formation Energy")
    plt.title("Formation Energy Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Verzeichnis anlegen und speichern
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()


import os
import torch
from torch.nn import Parameter

def save_results_to_file(result, file_path):
    """
    Save the output of train64test64() to a text file in a human-readable format.
    
    Args:
        result (list): The result returned by train64test64.
        file_path (str): Path to the file where the result will be saved.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(repr(result))  # Save the full list in a repr-safe way


import os

def save_results_to_directory(results_list, directory):
    os.makedirs(directory, exist_ok=True)
    for idx, item in enumerate(results_list):
        file_path = os.path.join(directory, f"result_{idx}.txt")
        with open(file_path, "w") as f:
            f.write(repr(item))




def load_results_from_file(filepath):
    """
    Lädt die Ergebnisse zeilenweise aus der Datei.
    Für sichere Datentypen wird ast.literal_eval verwendet.
    Für komplexe Typen (z. B. Tensor, Geometry, Parameter) wird der Text als String zurückgegeben.
    """
    import ast

    results = []
    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            try:
                results.append(ast.literal_eval(stripped))
            except Exception:
                # fallback: als roher String speichern
                results.append(stripped)
    return results



def full_routine64(total_dpoints, seed, max_dpoint, portions):
    from torch.nn import Parameter
    MSE = []
    MAE = []
    datapoints = select_random_datapoints(total_dpoints, seed, max_dpoint)
    dpoints = split_data(datapoints, portions)
    genfilepath = 'geometries'
    print(dpoints)

    for ii in range(5):
        print(f"\n🔁 Zyklus {ii+1}/5")

        xTB_alpha = {14: Parameter(torch.tensor([0.9549]), requires_grad=True)}
        PTBP_alpha = {14: Parameter(torch.tensor([1.12357]), requires_grad=True)}
        Gamma_alpha = {14: Parameter(torch.tensor([2.7285]), requires_grad=True)}
        xTB_Z = {14: Parameter(torch.tensor([13.2505]), requires_grad=True)}
        PTBP_Z = {14: Parameter(torch.tensor([8.4406]), requires_grad=True)}
        Gamma_Z = {14: Parameter(torch.tensor([9.2531]), requires_grad=True)}

        testdpoints = dpoints[ii]
        traindpoints = [dp for i, part in enumerate(dpoints) if i != ii for dp in part]

        split_dir = f"plots/formation_energies/split{ii}"
        log_dir = f"logs/split{ii}"
        os.makedirs(split_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # xTB
        print("→ Training mit xTBRepulsive")
        results = train64test64(traindpoints, testdpoints, xTBRepulsive, xTB_alpha, xTB_Z)
        plot_formation_energies_new(
            datapoints=testdpoints,
            genfiles_path=genfilepath,
            target_test=results[4],
            model_test=results[5],
            filepath=f"{split_dir}/xTB.png"
        )
        save_results_to_directory(results, f"{log_dir}/xTB_result")
        MSE.append(results[10])
        MAE.append(results[11])

        # PTBP
        print("→ Training mit PTBPRepulsive")
        results = train64test64(traindpoints, testdpoints, PTBPRepulsive, PTBP_alpha, PTBP_Z)
        plot_formation_energies_new(
            datapoints=testdpoints,
            genfiles_path=genfilepath,
            target_test=results[4],
            model_test=results[5],
            filepath=f"{split_dir}/PTBP.png"
        )
        save_results_to_directory(results, f"{log_dir}/PTBP_result")
        MSE.append(results[10])
        MAE.append(results[11])

        # Gamma
        print("→ Training mit DFTBGammaRepulsive")
        results = train64test64(traindpoints, testdpoints, DFTBGammaRepulsive, Gamma_alpha, Gamma_Z)
        plot_formation_energies_new(
            datapoints=testdpoints,
            genfiles_path=genfilepath,
            target_test=results[4],
            model_test=results[5],
            filepath=f"{split_dir}/Gamma.png"
        )
        save_results_to_directory(results, f"{log_dir}/Gamma_result")
        MSE.append(results[10])
        MAE.append(results[11])

        # Testmodel: pbc
        print("→ Test mit Modell: pbc")
        mse, mae, target, model = test_model('dft.hdf5', testdpoints, 'pbc', xTB_alpha, xTB_Z)
        plot_formation_energies_new(
            datapoints=testdpoints,
            genfiles_path=genfilepath,
            target_test=target,
            model_test=model,
            filepath=f"{split_dir}/pbc.png"
        )
        save_results_to_directory([mse, mae, target, model], f"{log_dir}/pbc_result")
        MSE.append(mse)
        MAE.append(mae)

        # Testmodel: siband
        print("→ Test mit Modell: siband")
        mse, mae, target, model = test_model('dft.hdf5', testdpoints, 'siband', xTB_alpha, xTB_Z)
        plot_formation_energies_new(
            datapoints=testdpoints,
            genfiles_path=genfilepath,
            target_test=target,
            model_test=model,
            filepath=f"{split_dir}/siband.png"
        )
        save_results_to_directory([mse, mae, target, model], f"{log_dir}/siband_result")

    return MSE, MAE


def full_routine512(total_dpoints, seed, max_dpoint, portions):
    from torch.nn import Parameter
    xTB_alpha = {14: Parameter(torch.tensor([0.9549]), requires_grad=True)}
    PTBP_alpha = {14: Parameter(torch.tensor([1.12357]), requires_grad=True)}
    Gamma_alpha = {14: Parameter(torch.tensor([2.7285]), requires_grad=True)}
    xTB_Z = {14: Parameter(torch.tensor([13.2505]), requires_grad=True)}
    PTBP_Z = {14: Parameter(torch.tensor([8.4406]), requires_grad=True)}
    Gamma_Z = {14: Parameter(torch.tensor([9.2531]), requires_grad=True)}

    MSE = []
    MAE = []
    genfilepath = 'genfiles512'
    datapoints = select_random_datapoints(total_dpoints, seed, max_dpoint)
    print(datapoints)

    split_dir = f"plots/formation_energies/512test"
    log_dir = f"logs/512test"
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("→ Training mit xTBRepulsive")
    results = train64test512(datapoints, xTBRepulsive, xTB_alpha, xTB_Z)
    plot_formation_energies_new(
        datapoints=[1, 2, 3, 4, 5, 6],
        genfiles_path=genfilepath,
        target_test=results[4],
        model_test=results[5],
        filepath=f"{split_dir}/xTB.png"
    )
    save_results_to_directory(results, f"{log_dir}/xTB_result")
    MSE.append(results[10])
    MAE.append(results[11])

    print("→ Training mit PTBPRepulsive")
    results = train64test512(datapoints, PTBPRepulsive, PTBP_alpha, PTBP_Z)
    plot_formation_energies_new(
        datapoints=[1, 2, 3, 4, 5, 6],
        genfiles_path=genfilepath,
        target_test=results[4],
        model_test=results[5],
        filepath=f"{split_dir}/PTBP.png"
    )
    save_results_to_directory(results, f"{log_dir}/PTBP_result")
    MSE.append(results[10])
    MAE.append(results[11])

    print("→ Training mit DFTBGammaRepulsive")
    results = train64test512(datapoints, DFTBGammaRepulsive, Gamma_alpha, Gamma_Z)
    plot_formation_energies_new(
        datapoints=[1, 2, 3, 4, 5, 6],
        genfiles_path=genfilepath,
        target_test=results[4],
        model_test=results[5],
        filepath=f"{split_dir}/Gamma.png"
    )
    save_results_to_directory(results, f"{log_dir}/Gamma_result")
    MSE.append(results[10])
    MAE.append(results[11])

    testdpoints = [1, 2, 3, 4, 5, 6]

    print("→ Test mit Modell: pbc")
    mse, mae, target, model = test_model('dft_test.hdf5', testdpoints, 'pbc', xTB_alpha, xTB_Z)
    plot_formation_energies_new(
        datapoints=testdpoints,
        genfiles_path=genfilepath,
        target_test=target,
        model_test=model,
        filepath=f"{split_dir}/pbc.png"
    )
    save_results_to_directory([mse, mae, target, model], f"{log_dir}/pbc_result")
    MSE.append(mse)
    MAE.append(mae)

    print("→ Test mit Modell: siband")
    mse, mae, target, model = test_model('dft_test.hdf5', testdpoints, 'siband', xTB_alpha, xTB_Z)
    plot_formation_energies_new(
        datapoints=testdpoints,
        genfiles_path=genfilepath,
        target_test=target,
        model_test=model,
        filepath=f"{split_dir}/siband.png"
    )
    save_results_to_directory([mse, mae, target, model], f"{log_dir}/siband_result")

    return MSE, MAE



def plot_errors_bar64(errors, metric_name="MSE", filepath="plots/errors_bar.png"):
    """
    Plottet Balkendiagramm für MSE oder MAE. Erwartet eine Liste der Länge 5×4 = 20.
    Die Reihenfolge ist: [xTB0, PTBP0, Gamma0, pbc0, xTB1, ..., pbc4]
    """

    model_names = ["xTB", "PTBP", "Gamma", "pbc"]
    num_splits = 5
    num_models = len(model_names)

    # Konvertieren in NumPy-Array für einfache Handhabung
    errors = np.array(errors).reshape((num_splits, num_models))

    # Balkenbreite und Positionen
    x = np.arange(num_splits)
    bar_width = 0.18
    offsets = np.linspace(-bar_width * 1.5, bar_width * 1.5, num_models)

    plt.figure(figsize=(10, 6))

    # Für jedes Modell Balken einfügen
    for i, model in enumerate(model_names):
        plt.bar(x + offsets[i], errors[:, i], width=bar_width, label=model)

    # Beschriftung
    plt.xlabel("Datensplit (ii)")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} für verschiedene Modelle je Split")
    plt.xticks(x, [f"Split {i}" for i in range(num_splits)])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Ordner erstellen und Plot speichern
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_errors_512(errors, metric_name="MSE", filepath="plots/errors/512_bar.png"):
    """
    Plottet Balkendiagramm für MSE oder MAE aus full_routine512.
    Erwartet eine Liste mit 4 Werten: [xTB, PTBP, Gamma, pbc]
    """
    model_names = ["xTB", "PTBP", "Gamma", "pbc"]

    if len(errors) != 4:
        raise ValueError(f"Fehlerliste muss genau 4 Werte enthalten, aber hat {len(errors)}.")

    plt.figure(figsize=(8, 5))
    plt.bar(model_names, errors, color="skyblue")

    plt.ylabel(metric_name)
    plt.title(f"{metric_name} für Modelle (512 Testpunkte)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Verzeichnis anlegen und speichern
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()



if __name__ == "__main__":
    total_dpoints = 1000
    seed = 832478
    max_dpoint = 6306
    portions = 5
    repulsive_model = xTBRepulsive


    datapoints = select_random_datapoints(total_dpoints, seed, max_dpoint)
    dpoints = split_data(datapoints, portions)
    testdpoints = dpoints[0]
    traindpoints = [dp for i, part in enumerate(dpoints) if i != 0 for dp in part]

    #results = train64test64(traindpoints, testdpoints, repulsive_model, xTB_alpha, custom_Z)
    #save_results_to_file(results, 'res.txt')
    #load_results_from_file(results, 'res.txt')
    #train64test512(traindpoints, repulsive_model, xTB_alpha, custom_Z)
    MSE, MAE = full_routine64(total_dpoints, seed, max_dpoint, portions)
    plot_errors_bar64(MSE, metric_name='MSE', filepath = "plots/MSE64.png")
    plot_errors_bar64(MAE, metric_name='MAE', filepath = "plots/MAE64.png")
    MSE, MAE = full_routine512(total_dpoints, seed, max_dpoint, portions)
    plot_errors_512(MSE, metric_name='MSE', filepath = "plots/MSE512.png")
    plot_errors_512(MAE, metric_name='MAE', filepath = "plots/MAE512.png")