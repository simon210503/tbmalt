import torch
import random
from typing import Tuple, Union

from torch.nn import Parameter
from torch import Tensor

from tbmalt.ml.loss_function import l1_loss, mse_loss
from formation_calc import calc_formation_energy, prepare_system, calc_reference_formation_energies
from new_feeds import PTBPRepulsive, DFTBGammaRepulsive, xTBRepulsive
from utils import select_random_datapoints

torch.set_default_dtype(torch.float64)


def test_model(
    sample_path: str,
    datapoints: list[int],
    repulsive_model: Union[type[PTBPRepulsive], type[DFTBGammaRepulsive], type[xTBRepulsive]],
    alpha: dict[int, Parameter],
    Z: dict[int, Parameter],
    use_weights: bool = False,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, Tensor, Tensor]:
    """
    Test a repulsive potential model by computing formation energies and evaluating errors.

    Parameters
    ----------
    sample_path : str
        Path to the dataset file (HDF5).
    datapoints : list of int
        Indices of the datapoints to be tested.
    repulsive_model : {PTBPRepulsive, DFTBGammaRepulsive, xTBRepulsive}
        The repulsive potential model class to evaluate.
    alpha : dict[int, torch.nn.Parameter]
        Dictionary mapping atomic numbers to learnable alpha parameters.
    Z : dict[int, torch.nn.Parameter]
        Dictionary mapping atomic numbers to learnable Z parameters.
    use_weights : bool, optional
        If True, increases weights for small datapoints (<= 6). Default is False.
    device : torch.device, optional
        Device on which to run the calculations. Default is CPU.

    Returns
    -------
    mse : float
        Mean squared error of the model predictions against reference values.
    mae : float
        Mean absolute error of the model predictions against reference values.
    target_formation_energy : torch.Tensor
        Reference formation energies for the given datapoints.
    model_formation_energy : torch.Tensor
        Predicted formation energies from the model.
    """
    weights = torch.ones(len(datapoints), device=device)
    if use_weights:
        weights[torch.tensor(datapoints, device=device) <= 6] = 200.0

    # Prepare system
    params = prepare_system(sample_path, datapoints, repulsive_model, alpha, Z)

    # Compute predictions and targets
    model_formation_energy = calc_formation_energy(params)
    target_formation_energy = calc_reference_formation_energies(sample_path, datapoints)

    # Compute loss metrics
    mse = mse_loss(model_formation_energy, target_formation_energy, weights=weights)
    mae = l1_loss(model_formation_energy, target_formation_energy, weights=weights)

    print(f"\nTest-Ergebnisse:")
    print(f"  MSE (mean squared error):     {mse.item():.6f}")
    print(f"  MAE (mean absolute error):    {mae.item():.6f}")

    return mse.item(), mae.item(), target_formation_energy, model_formation_energy


if __name__ == "__main__":
    # Example usage with adjustable parameters
    SAMPLE_PATH = "dft_test.hdf5"
    seed = random.randint(0, 999999)
    DATAPOINTS = select_random_datapoints(6, seed, 6)  # Example, can be adjusted
    print(DATAPOINTS)

    REPULSIVE = PTBPRepulsive
    xTB_alpha = {14: Parameter(torch.tensor([1.0]), requires_grad=True)}
    PTBP_alpha = {14: Parameter(torch.tensor([1.0]), requires_grad=True)}
    Gamma_alpha = {14: Parameter(torch.tensor([2.7]), requires_grad=True)}
    custom_Z = {14: Parameter(torch.tensor([14.0]), requires_grad=True)}

    print(test_model(SAMPLE_PATH, DATAPOINTS, PTBPRepulsive, PTBP_alpha, custom_Z, 512))
