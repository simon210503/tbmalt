import torch
import os
from torch.nn import Parameter
from typing import List, Dict, Any, Type

from training import train_model
from testing import test_model
from new_feeds import PTBPRepulsive, DFTBGammaRepulsive, xTBRepulsive
from utils import (
    split_data, 
    select_random_datapoints,
    find_structures_with_atom_count,
    random_sample_from_tensor
)
from saveload import save_results_to_directory
from plots import (
    plot_formation_energies_new, 
    plot_errors_bar64, 
    plot_repulsives_relaxed_w_distances,
    plot_errors_512
)


def train64test64(
    traindpoints: List[int], 
    testdpoints: List[int], 
    repulsive_model: Type, 
    initial_alpha: Dict[int, Parameter], 
    initial_Z: Dict[int, Parameter], 
    weight: bool = False
) -> List[Any]:
    """
    Train on a subset of datapoints and test on the remaining subset.

    Args:
        traindpoints (List[int]): Indices of datapoints used for training.
        testdpoints (List[int]): Indices of datapoints used for testing.
        repulsive_model (Type): Repulsive model class to train/test.
        initial_alpha (Dict[int, Parameter]): Initial alpha parameters for each element.
        initial_Z (Dict[int, Parameter]): Initial Z parameters for each element.
        weight (bool, optional): Whether to use weighting in training/testing. Defaults to False.

    Returns:
        List[Any]: Contains training/test datapoints, loss history, parameters, 
                   formation energies, final model parameters, and error metrics.
    """
    sample_path = 'dft.hdf5'

    loss_history, params, target_formation_energy_training, model_formation_energy_training, final_alpha, final_Z = train_model(
        sample_path,
        traindpoints,
        repulsive_model=repulsive_model,
        alpha=initial_alpha,
        Z=initial_Z,
        use_weights=weight
    )

    MSE, MAE, target_formation_energy_testing, model_formation_energy_testing = test_model(
        sample_path,
        testdpoints,
        repulsive_model=repulsive_model,
        alpha=final_alpha,
        Z=final_Z,
        use_weights=weight
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


def train64test512(
    training_dpoints: List[int], 
    repulsive_model: Type, 
    initial_alpha: Dict[int, Parameter], 
    initial_Z: Dict[int, Parameter], 
    weight: bool = False
) -> List[Any]:
    """
    Train on a random subset of datapoints from one file, test on a fixed set from another file.

    Args:
        training_dpoints (List[int]): Indices of training datapoints.
        repulsive_model (Type): Repulsive model class.
        initial_alpha (Dict[int, Parameter]): Initial alpha parameters for training.
        initial_Z (Dict[int, Parameter]): Initial Z parameters for training.
        weight (bool, optional): Whether to use weighting in training/testing. Defaults to False.

    Returns:
        List[Any]: Contains training/test datapoints, loss history, parameters, 
                   formation energies, final model parameters, and error metrics.
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
        Z=initial_Z,
        use_weights=weight
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


def full_routine64(
    total_dpoints: int, 
    seed: int, 
    max_dpoint: int, 
    portions: int, 
    weight: bool = False
) -> None:
    """
    Perform full 64-routine training and testing cycles on different models with k-fold-like splitting.

    Args:
        total_dpoints (int): Total number of available datapoints.
        seed (int): Random seed for reproducibility.
        max_dpoint (int): Maximum index of datapoints to select from.
        portions (int): Number of splits for k-fold-like training/testing.
        weight (bool, optional): Whether to use weighted training/testing. Defaults to False.

    Returns:
        None
    """
    MSE = []
    MAE = []
    datapoints = select_random_datapoints(total_dpoints, seed, max_dpoint)
    dpoints = split_data(datapoints, portions)
    sample_path = 'dft.hdf5'
    print(dpoints)

    for ii in range(5):
        print(f"\n🔁 Zyklus {ii+1}/5")

        xTB_alpha = {14: Parameter(torch.tensor([0.4709]), requires_grad=True)}
        PTBP_alpha = {14: Parameter(torch.tensor([1.7996]), requires_grad=True)}
        Gamma_alpha = {14: Parameter(torch.tensor([1.9513]), requires_grad=True)}
        xTB_Z = {14: Parameter(torch.tensor([3.2796]), requires_grad=True)}
        PTBP_Z = {14: Parameter(torch.tensor([3.0571]), requires_grad=True)}
        Gamma_Z = {14: Parameter(torch.tensor([4.7265]), requires_grad=True)}

        testdpoints = dpoints[ii]
        traindpoints = [dp for i, part in enumerate(dpoints) if i != ii for dp in part]

        split_dir = f"plots/64routine/formation_energies/split{ii}"
        log_dir = f"logs/64routine/split{ii}"
        os.makedirs(split_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # xTB
        print("→ Training mit xTBRepulsive")
        results = train64test64(traindpoints, testdpoints, xTBRepulsive, xTB_alpha, xTB_Z, weight=weight)
        plot_formation_energies_new(
            datapoints=testdpoints,
            sample_path=sample_path,
            target_test=results[4],
            model_test=results[5],
            filepath=f"{split_dir}/xTB.png"
        )
        save_results_to_directory(results, f"{log_dir}/xTB_result")
        MSE.append(results[10])
        MAE.append(results[11])

        # PTBP
        print("→ Training mit PTBPRepulsive")
        results = train64test64(traindpoints, testdpoints, PTBPRepulsive, PTBP_alpha, PTBP_Z, weight=weight)
        plot_formation_energies_new(
            datapoints=testdpoints,
            sample_path=sample_path,
            target_test=results[4],
            model_test=results[5],
            filepath=f"{split_dir}/PTBP.png"
        )
        save_results_to_directory(results, f"{log_dir}/PTBP_result")
        MSE.append(results[10])
        MAE.append(results[11])

        # Gamma
        print("→ Training mit DFTBGammaRepulsive")
        results = train64test64(traindpoints, testdpoints, DFTBGammaRepulsive, Gamma_alpha, Gamma_Z, weight=weight)
        plot_formation_energies_new(
            datapoints=testdpoints,
            sample_path=sample_path,
            target_test=results[4],
            model_test=results[5],
            filepath=f"{split_dir}/Gamma.png"
        )
        save_results_to_directory(results, f"{log_dir}/Gamma_result")
        MSE.append(results[10])
        MAE.append(results[11])

        # Testmodel: pbc
        print("→ Test mit Modell: pbc")
        mse, mae, target, model = test_model(sample_path, testdpoints, 'pbc', xTB_alpha, xTB_Z, use_weights=weight)
        plot_formation_energies_new(
            datapoints=testdpoints,
            sample_path=sample_path,
            target_test=target,
            model_test=model,
            filepath=f"{split_dir}/pbc.png"
        )
        save_results_to_directory([mse, mae, target, model], f"{log_dir}/pbc_result")
        MSE.append(mse)
        MAE.append(mae)

        # Testmodel: siband
        print("→ Test mit Modell: siband")
        mse, mae, target, model = test_model(sample_path, testdpoints, 'siband', xTB_alpha, xTB_Z, use_weights=weight)
        plot_formation_energies_new(
            datapoints=testdpoints,
            sample_path=sample_path,
            target_test=target,
            model_test=model,
            filepath=f"{split_dir}/siband.png"
        )
        save_results_to_directory([mse, mae, target, model], f"{log_dir}/siband_result")

        save_dir = f'plots/64routine/repulsives/{ii}'
        plot_repulsives_relaxed_w_distances(log_dir, save_dir)

    # Fehlerplots
    plot_errors_bar64(MSE, metric_name='MSE', filepath="plots/64routine/MSE64.png")
    plot_errors_bar64(MAE, metric_name='MAE', filepath="plots/64routine/MAE64.png")



def full_routine512(
    total_dpoints: int,
    seed: int,
    max_dpoint: int,
    weight: bool = False
) -> None:
    """
    Execute a full training and testing routine for 512-atom systems using
    xTB, PTBP, and DFTBGamma repulsive models, including plotting and saving results.

    Args:
        total_dpoints (int): Total number of random datapoints to select for training/testing.
        seed (int): Seed for random selection of datapoints.
        max_dpoint (int): Maximum index of datapoints in the dataset.
        weight (bool, optional): Whether to use weighting in the training routines. Defaults to False.

    Returns:
        None
    """
    xTB_alpha = {14: Parameter(torch.tensor([0.4709]), requires_grad=True)}
    PTBP_alpha = {14: Parameter(torch.tensor([1.7996]), requires_grad=True)}
    Gamma_alpha = {14: Parameter(torch.tensor([1.9513]), requires_grad=True)}
    xTB_Z = {14: Parameter(torch.tensor([3.2796]), requires_grad=True)}
    PTBP_Z = {14: Parameter(torch.tensor([3.0571]), requires_grad=True)}
    Gamma_Z = {14: Parameter(torch.tensor([4.7265]), requires_grad=True)}

    MSE: List[float] = []
    MAE: List[float] = []
    sample_path = 'dft_test.hdf5'
    datapoints = select_random_datapoints(total_dpoints, seed, max_dpoint)
    print(datapoints)

    split_dir = f"plots/512routine/formation_energies/512test"
    log_dir = f"logs/512routine/512test"
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Training and plotting for each repulsive model
    print("→ Training mit xTBRepulsive")
    results = train64test512(datapoints, xTBRepulsive, xTB_alpha, xTB_Z, weight=weight)
    plot_formation_energies_new(
        datapoints=[1, 2, 3, 4, 5, 6],
        sample_path=sample_path,
        target_test=results[4],
        model_test=results[5],
        filepath=f"{split_dir}/xTB.png"
    )
    save_results_to_directory(results, f"{log_dir}/xTB_result")
    MSE.append(results[10])
    MAE.append(results[11])

    print("→ Training mit PTBPRepulsive")
    results = train64test512(datapoints, PTBPRepulsive, PTBP_alpha, PTBP_Z, weight=weight)
    plot_formation_energies_new(
        datapoints=[1, 2, 3, 4, 5, 6],
        sample_path=sample_path,
        target_test=results[4],
        model_test=results[5],
        filepath=f"{split_dir}/PTBP.png"
    )
    save_results_to_directory(results, f"{log_dir}/PTBP_result")
    MSE.append(results[10])
    MAE.append(results[11])

    print("→ Training mit DFTBGammaRepulsive")
    results = train64test512(datapoints, DFTBGammaRepulsive, Gamma_alpha, Gamma_Z, weight=weight)
    plot_formation_energies_new(
        datapoints=[1, 2, 3, 4, 5, 6],
        sample_path=sample_path,
        target_test=results[4],
        model_test=results[5],
        filepath=f"{split_dir}/Gamma.png"
    )
    save_results_to_directory(results, f"{log_dir}/Gamma_result")
    MSE.append(results[10])
    MAE.append(results[11])

    testdpoints = [1, 2, 3, 4, 5, 6]

    print("→ Test mit Modell: pbc")
    mse, mae, target, model = test_model(sample_path, testdpoints, 'pbc', xTB_alpha, xTB_Z)
    plot_formation_energies_new(
        datapoints=testdpoints,
        sample_path=sample_path,
        target_test=target,
        model_test=model,
        filepath=f"{split_dir}/pbc.png"
    )
    save_results_to_directory([mse, mae, target, model], f"{log_dir}/pbc_result")
    MSE.append(mse)
    MAE.append(mae)

    print("→ Test mit Modell: siband")
    mse, mae, target, model = test_model(sample_path, testdpoints, 'siband', xTB_alpha, xTB_Z)
    plot_formation_energies_new(
        datapoints=testdpoints,
        sample_path=sample_path,
        target_test=target,
        model_test=model,
        filepath=f"{split_dir}/siband.png"
    )
    save_results_to_directory([mse, mae, target, model], f"{log_dir}/siband_result")

    plot_errors_512(MSE, metric_name='MSE', filepath=f'plots/512routine/MSE.png')
    plot_errors_512(MAE, metric_name='MAE', filepath=f'plots/512routine/MAE.png')
    save_dir = f'plots/512routine/repulsives'
    plot_repulsives_relaxed_w_distances(log_dir, save_dir)


def full_routine_atomcount_train(
    N_train: int,
    N_plot: int,
    seed: int,
    weight: bool,
    atom_count: int = 63
) -> None:
    """
    Train and evaluate models for structures with a specific number of atoms.

    Args:
        N_train (int): Number of training datapoints to sample.
        N_plot (int): Number of datapoints to be plotted.
        seed (int): Seed for random selection of datapoints.
        weight (bool): Whether to use weighted training.
        atom_count (int, optional): Number of atoms in structures to select. Defaults to 63.

    Returns:
        None
    """
    sample_path = 'dft.hdf5'

    all_dpoints = list(range(1, 6307))
    selected_indices = find_structures_with_atom_count(sample_path, all_dpoints, atom_count)
    traindpoints = random_sample_from_tensor(selected_indices, N_train)
    print(traindpoints)

    testdpoints = traindpoints


    plotdpoints = select_random_datapoints(N_plot, seed, 6306)
    print(plotdpoints)

    xTB_alpha = {14: Parameter(torch.tensor([0.4709]), requires_grad=True)}
    PTBP_alpha = {14: Parameter(torch.tensor([1.7996]), requires_grad=True)}
    Gamma_alpha = {14: Parameter(torch.tensor([1.9513]), requires_grad=True)}
    xTB_Z = {14: Parameter(torch.tensor([3.2796]), requires_grad=True)}
    PTBP_Z = {14: Parameter(torch.tensor([3.0571]), requires_grad=True)}
    Gamma_Z = {14: Parameter(torch.tensor([4.7265]), requires_grad=True)}

    split_dir = f"plots/{atom_count}_train/formation_energies"
    log_dir = f"logs/{atom_count}_train"
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    MSE: List[float] = []
    MAE: List[float] = []

    # Training and testing for each repulsive model
    print("→ Training mit xTBRepulsive")
    results = train64test64(traindpoints, testdpoints, xTBRepulsive, xTB_alpha, xTB_Z, weight=weight)
    plot_formation_energies_new(
        datapoints=plotdpoints,
        sample_path=sample_path,
        target_test=results[4],
        model_test=results[5],
        filepath=f"{split_dir}/xTB.png"
    )
    save_results_to_directory(results, f"{log_dir}/xTB_result")
    MSE.append(results[10])
    MAE.append(results[11])

    print("→ Training mit PTBPRepulsive")
    results = train64test64(traindpoints, testdpoints, PTBPRepulsive, PTBP_alpha, PTBP_Z, weight=weight)
    plot_formation_energies_new(
        datapoints=plotdpoints,
        sample_path=sample_path,
        target_test=results[4],
        model_test=results[5],
        filepath=f"{split_dir}/PTBP.png"
    )
    save_results_to_directory(results, f"{log_dir}/PTBP_result")
    MSE.append(results[10])
    MAE.append(results[11])

    print("→ Training mit DFTBGammaRepulsive")
    results = train64test64(traindpoints, testdpoints, DFTBGammaRepulsive, Gamma_alpha, Gamma_Z, weight=weight)
    plot_formation_energies_new(
        datapoints=plotdpoints,
        sample_path=sample_path,
        target_test=results[4],
        model_test=results[5],
        filepath=f"{split_dir}/Gamma.png"
    )
    save_results_to_directory(results, f"{log_dir}/Gamma_result")
    MSE.append(results[10])
    MAE.append(results[11])

    print("→ Test mit Modell: pbc")
    mse, mae, target, model = test_model(sample_path, testdpoints, 'pbc', xTB_alpha, xTB_Z, use_weights=weight)
    plot_formation_energies_new(
        datapoints=plotdpoints,
        sample_path=sample_path,
        target_test=target,
        model_test=model,
        filepath=f"{split_dir}/pbc.png"
    )
    save_results_to_directory([mse, mae, target, model], f"{log_dir}/pbc_result")
    MSE.append(mse)
    MAE.append(mae)

    print("→ Test mit Modell: siband")
    mse, mae, target, model = test_model(sample_path, testdpoints, 'siband', xTB_alpha, xTB_Z, use_weights=weight)
    plot_formation_energies_new(
        datapoints=plotdpoints,
        sample_path=sample_path,
        target_test=target,
        model_test=model,
        filepath=f"{split_dir}/siband.png"
    )
    save_results_to_directory([mse, mae, target, model], f"{log_dir}/siband_result")

    plot_errors_512(MSE, metric_name='MSE', filepath=f'plots/{atom_count}_train/MSE.png')
    plot_errors_512(MAE, metric_name='MAE', filepath=f'plots/{atom_count}_train/MAE.png')
    save_dir = f'plots/{atom_count}_train/repulsives'
    plot_repulsives_relaxed_w_distances(log_dir, save_dir)

def full_routine_63_64_train(
    N_test: int,
    seed: int,
    weight: bool
) -> None:
    """
    Train and evaluate models for structures with a specific number of atoms.

    Args:
        N_train (int): Number of training datapoints to sample.
        N_test (int): Number of test datapoints to sample.
        seed (int): Seed for random selection of datapoints.
        weight (bool): Whether to use weighted training.
        atom_count (int, optional): Number of atoms in structures to select. Defaults to 63.

    Returns:
        None
    """
    sample_path = 'dft.hdf5'

    all_dpoints = list(range(1, 6307))
    selected_indices63 = find_structures_with_atom_count(sample_path, all_dpoints, 63).tolist()
    selected_indices64 = find_structures_with_atom_count(sample_path, all_dpoints, 64).tolist()
    traindpoints = selected_indices63 + selected_indices64
    print(traindpoints)

    testdpoints = traindpoints

    plotdpoints = select_random_datapoints(N_test, seed, 6306)
    print(plotdpoints)

    xTB_alpha = {14: Parameter(torch.tensor([0.4709]), requires_grad=True)}
    PTBP_alpha = {14: Parameter(torch.tensor([1.7996]), requires_grad=True)}
    Gamma_alpha = {14: Parameter(torch.tensor([1.9513]), requires_grad=True)}
    xTB_Z = {14: Parameter(torch.tensor([3.2796]), requires_grad=True)}
    PTBP_Z = {14: Parameter(torch.tensor([3.0571]), requires_grad=True)}
    Gamma_Z = {14: Parameter(torch.tensor([4.7265]), requires_grad=True)}

    split_dir = f"plots/6364_train/formation_energies"
    log_dir = f"logs/6364_train"
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    MSE: List[float] = []
    MAE: List[float] = []

    # Training and testing for each repulsive model
    print("→ Training mit xTBRepulsive")
    results = train64test64(traindpoints, testdpoints, xTBRepulsive, xTB_alpha, xTB_Z, weight=weight)
    plot_formation_energies_new(
        datapoints=plotdpoints,
        sample_path=sample_path,
        target_test=results[4],
        model_test=results[5],
        filepath=f"{split_dir}/xTB.png"
    )
    save_results_to_directory(results, f"{log_dir}/xTB_result")
    MSE.append(results[10])
    MAE.append(results[11])

    print("→ Training mit PTBPRepulsive")
    results = train64test64(traindpoints, testdpoints, PTBPRepulsive, PTBP_alpha, PTBP_Z, weight=weight)
    plot_formation_energies_new(
        datapoints=plotdpoints,
        sample_path=sample_path,
        target_test=results[4],
        model_test=results[5],
        filepath=f"{split_dir}/PTBP.png"
    )
    save_results_to_directory(results, f"{log_dir}/PTBP_result")
    MSE.append(results[10])
    MAE.append(results[11])

    print("→ Training mit DFTBGammaRepulsive")
    results = train64test64(traindpoints, testdpoints, DFTBGammaRepulsive, Gamma_alpha, Gamma_Z, weight=weight)
    plot_formation_energies_new(
        datapoints=plotdpoints,
        sample_path=sample_path,
        target_test=results[4],
        model_test=results[5],
        filepath=f"{split_dir}/Gamma.png"
    )
    save_results_to_directory(results, f"{log_dir}/Gamma_result")
    MSE.append(results[10])
    MAE.append(results[11])

    print("→ Test mit Modell: pbc")
    mse, mae, target, model = test_model(sample_path, testdpoints, 'pbc', xTB_alpha, xTB_Z, use_weights=weight)
    plot_formation_energies_new(
        datapoints=plotdpoints,
        sample_path=sample_path,
        target_test=target,
        model_test=model,
        filepath=f"{split_dir}/pbc.png"
    )
    save_results_to_directory([mse, mae, target, model], f"{log_dir}/pbc_result")
    MSE.append(mse)
    MAE.append(mae)

    print("→ Test mit Modell: siband")
    mse, mae, target, model = test_model(sample_path, testdpoints, 'siband', xTB_alpha, xTB_Z, use_weights=weight)
    plot_formation_energies_new(
        datapoints=plotdpoints,
        sample_path=sample_path,
        target_test=target,
        model_test=model,
        filepath=f"{split_dir}/siband.png"
    )
    save_results_to_directory([mse, mae, target, model], f"{log_dir}/siband_result")

    plot_errors_512(MSE, metric_name='MSE', filepath=f'plots/6364_train/MSE.png')
    plot_errors_512(MAE, metric_name='MAE', filepath=f'plots/6364_train/MAE.png')
    save_dir = f'plots/6364_train/repulsives'
    plot_repulsives_relaxed_w_distances(log_dir, save_dir)

if __name__ == "__main__":
    total_dpoints = 6306
    seed = 3648295765
    max_dpoint = 6306
    portions = 5
    repulsive_model = xTBRepulsive


    #datapoints = select_random_datapoints(total_dpoints, seed, max_dpoint)
    #dpoints = split_data(datapoints, portions)
    #testdpoints = dpoints[0]
    #traindpoints = [dp for i, part in enumerate(dpoints) if i != 0 for dp in part]
    full_routine64(total_dpoints, 432589237, max_dpoint, portions, weight = True)
    full_routine512(total_dpoints, 549345823, max_dpoint, weight = True)
    full_routine_atomcount_train(1301, 1000, 38478291, True, 63)
    full_routine_63_64_train(1000, 7438290578932, True)