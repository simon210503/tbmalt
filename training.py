import torch
import random
from typing import Tuple, List, Dict, Optional, Union

from torch.nn import Parameter
from torch import Tensor

from tbmalt.ml.loss_function import Loss, mse_loss
from new_feeds import pairwise_repulsive, PTBPRepulsive, DFTBGammaRepulsive, xTBRepulsive
from formation_calc import prepare_system, calc_formation_energy
from plots import plot_loss, save_loss_plot
from utils import select_random_datapoints

torch.set_default_dtype(torch.float64)
Tensor = torch.Tensor


class EarlyStopping:
    """
    Implements early stopping to halt training if the validation loss does not improve.

    Attributes
    ----------
    patience : int
        Number of epochs to wait for improvement before stopping.
    min_delta : float
        Minimum change in the monitored loss to qualify as an improvement.
    counter : int
        Counter for epochs without improvement.
    best_loss : Optional[float]
        Best observed loss.
    early_stop : bool
        Whether training should be stopped.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-5):
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.counter: int = 0
        self.best_loss: Optional[float] = None
        self.early_stop: bool = False

    def __call__(self, val_loss: float) -> None:
        """
        Update early stopping status based on current validation loss.

        Parameters
        ----------
        val_loss : float
            Current validation loss to evaluate improvement.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        else:
            if abs(val_loss - self.best_loss) < self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.counter = 0
            if val_loss < self.best_loss:
                self.best_loss = val_loss


def train_model(
    sample_path: str,
    datapoints: List[int],
    lr: float = 0.05,
    epochs: int = 10000,
    device: torch.device = torch.device('cpu'),
    repulsive_model=DFTBGammaRepulsive,
    alpha: Optional[Dict[int, Parameter]] = None,
    Z: Optional[Dict[int, Parameter]] = None,
    early_stopping: bool = True,
    patience: int = 5,
    min_delta: float = 1e-6,
    use_weights: bool = False
) -> Tuple[List[float], Dict[str, Dict[int, Parameter]], List[float], List[float], Dict[int, Parameter], Dict[int, Parameter]]:
    """
    Train a repulsive potential model using Adam optimizer and optional early stopping.

    Parameters
    ----------
    sample_path : str
        Path to the dataset file (HDF5).
    datapoints : list[int]
        Indices of datapoints to train on.
    lr : float, optional
        Learning rate for the optimizer.
    epochs : int, optional
        Maximum number of training epochs.
    device : torch.device, optional
        Device for computation (CPU or GPU).
    repulsive_model : class, optional
        Repulsive potential model class to train.
    alpha : dict[int, Parameter], optional
        Initial alpha parameters per atomic number.
    Z : dict[int, Parameter], optional
        Initial Z parameters per atomic number.
    early_stopping : bool, optional
        Whether to stop early if validation loss stagnates.
    patience : int, optional
        Number of epochs without improvement before stopping.
    min_delta : float, optional
        Minimum improvement in loss to reset early stopping counter.
    use_weights : bool, optional
        Whether to apply higher weights to small datapoints (<= 6).

    Returns
    -------
    loss_history : list[float]
        History of total loss per epoch.
    params : dict
        Optimized parameters and related data structures.
    target_formation_energy : list[float]
        Reference formation energies for the training datapoints.
    final_formation_energy : list[float]
        Predicted formation energies from the trained model.
    alpha : dict[int, Parameter]
        Optimized alpha parameters.
    Z : dict[int, Parameter]
        Optimized Z parameters.
    """
    # Prepare system with externally provided alpha/Z
    params = prepare_system(sample_path, datapoints, repulsive_model, alpha=alpha, Z=Z)
    targets = {'formation_energy': params['formation_energy']}
    alpha = params['alpha']
    Z = params['Z']

    variables = list(alpha.values()) + list(Z.values())

    # Optional: weighting
    if use_weights:
        weights = torch.ones(len(datapoints), device=device)
        weights[torch.tensor(datapoints, device=device) <= 6] = 200.0
        loss_entity = Loss(
            lambda calc, tgt: {'formation_energy': calc_formation_energy(params)},
            lambda calc, tgt: {'formation_energy': tgt['formation_energy']},
            loss_functions={
                'formation_energy': lambda pred, ref, **kwargs: mse_loss(pred, ref, weights=weights)
            },
            reduction='mean'
        )
    else:
        loss_entity = Loss(
            lambda calc, tgt: {'formation_energy': calc_formation_energy(params)},
            lambda calc, tgt: {'formation_energy': tgt['formation_energy']},
            loss_functions=mse_loss,
            reduction='mean'
        )

    optimizer = torch.optim.Adam(variables, lr=lr)
    loss_history: List[float] = []

    if early_stopping:
        stopper = EarlyStopping(patience=patience, min_delta=min_delta)

    for epoch in range(epochs):
        optimizer.zero_grad()
        formation_energy = calc_formation_energy(params)
        total_loss, _ = loss_entity(lambda: formation_energy, targets)
        total_loss.backward()
        optimizer.step()

        current_loss = total_loss.item()
        loss_history.append(current_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {current_loss:.6f}")

        if early_stopping:
            stopper(current_loss)
            if stopper.early_stop:
                print(f"\nFrühes Stoppen bei Epoch {epoch+1} (Loss: {current_loss:.6f})")
                break

    print("\nTraining abgeschlossen.")

    final_formation_energy = calc_formation_energy(params).detach().tolist()
    target_formation_energy = targets['formation_energy']
    if isinstance(target_formation_energy, torch.Tensor):
        target_formation_energy = target_formation_energy.detach().tolist()

    print("\nOptimized parameters:")
    print(f"  alpha (14): {alpha[14].item():.6f}")
    print(f"  Z (14):     {Z[14].item():.6f}")

    return loss_history, params, target_formation_energy, final_formation_energy, alpha, Z


if __name__ == "__main__":
    """
    Example training run using PTBPRepulsive potentials with early stopping.
    Adjust datapoints, learning rate, epochs, and model type as needed.
    """
    SAMPLE_PATH = 'dft.hdf5'
    seed = random.randint(0, 999999)
    DATAPOINTS = select_random_datapoints(100, seed)  # Example selection
    print(DATAPOINTS)

    LR = 0.05
    EPOCHS = 200
    REPULSIVE = PTBPRepulsive

    xTB_alpha = {14: Parameter(torch.tensor([1.0]), requires_grad=True)}
    PTBP_alpha = {14: Parameter(torch.tensor([1.0]), requires_grad=True)}
    Gamma_alpha = {14: Parameter(torch.tensor([2.7]), requires_grad=True)}
    custom_Z = {14: Parameter(torch.tensor([14.0]), requires_grad=True)}

    loss_hist, _, _, _, _, _ = train_model(
        SAMPLE_PATH,
        DATAPOINTS,
        lr=LR,
        epochs=EPOCHS,
        repulsive_model=REPULSIVE,
        alpha=PTBP_alpha,
        Z=custom_Z,
        early_stopping=True,
        patience=5,
        min_delta=1e-5
    )

    plot_loss(loss_hist)
