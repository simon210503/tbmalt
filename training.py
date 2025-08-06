import torch
import random
import os
from torch.nn import Parameter
import matplotlib.pyplot as plt
from tbmalt import Geometry
from tbmalt.ml.loss_function import Loss, mse_loss
from tbmalt.physics.dftb.feeds import PairwiseRepulsiveEnergyFeed
from tbmalt.data.units import length_units
from new_feeds import pairwise_repulsive, PTBPRepulsive, DFTBGammaRepulsive, xTBRepulsive
from formation_calc import (
    select_random_datapoints,
    load_Geo_dset,
    calc_reference_formation_energies,
    get_energies_from_file
)
from si64pos import fractional_positions


torch.set_default_dtype(torch.float64)
Tensor = torch.Tensor

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
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

def prepare_system(
    sample_path: str,
    datapoints: list[int],
    repulsive_model,
    alpha: dict = None,
    Z: dict = None,
):
    """Lade Geometrien, bereite Parameter, Repulsive Energie-Feed vor."""
    lattice_constant = 10.91929849 * length_units['a']

    Geodset = load_Geo_dset(sample_path, datapoints)
    formation_energy = calc_reference_formation_energies(sample_path, datapoints)
    if sample_path == 'dft.hdf5':
        elec_energy_defects = get_energies_from_file(datapoints)
    elif sample_path == 'dft_test.hdf5':
        elec_energy_defects = get_energies_from_file(datapoints, 'electronic_energies_512.txt')
    elec_energy_Si64 = -88.2635544138  # Referenzwert

    GeoSi64 = Geometry(
        torch.full((64,), 14),
        fractional_positions,
        torch.diag(torch.tensor([lattice_constant] * 3)),
        frac=True
    )

    # Fallback: Wenn keine Parameter übergeben wurden
    if alpha is None:
        alpha = {14: Parameter(Tensor([2.5]), requires_grad=True)}
    if Z is None:
        Z = {14: Parameter(Tensor([14.0]), requires_grad=True)}

    cutoff = Tensor([8.0]) #5 first; 8 second; 11 third neighbour included
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



def calc_formation_energy(params):
    """Berechne Formation Energy anhand der Parameter."""

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


def train_model(
    sample_path: str,
    datapoints: list[int],
    lr: float = 0.05,
    epochs: int = 500,
    device: torch.device = torch.device('cpu'),
    repulsive_model=DFTBGammaRepulsive,
    alpha: dict = None,
    Z: dict = None,
    early_stopping: bool = True,
    patience: int = 10,
    min_delta: float = 1e-6
):
    # Vorbereitung mit extern übergebenen alpha/Z
    params = prepare_system(sample_path, datapoints, repulsive_model, alpha=alpha, Z=Z)
    targets = {'formation_energy': params['formation_energy']}
    alpha = params['alpha']
    Z = params['Z']

    variables = list(alpha.values()) + list(Z.values())

    loss_entity = Loss(
        lambda calc, tgt: {'formation_energy': calc_formation_energy(params)},
        lambda calc, tgt: {'formation_energy': tgt['formation_energy']},
        loss_functions=mse_loss,
        reduction='mean'
    )

    optimizer = torch.optim.Adam(variables, lr=lr)
    loss_history = []

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

    """
    print("\nFormation Energy Comparison (in Hartree):")
    print(f"{'Datapoint':<12} {'Target':>12} {'Final':>12}")
    print("-" * 36)
    for dp, tgt, fin in zip(datapoints, target_formation_energy, final_formation_energy):
        print(f"{dp:<12} {tgt:>12.6f} {fin:>12.6f}")
    """

    print("\noptimized parameter:")
    print(f"  alpha (14): {alpha[14].item():.6f}")
    print(f"  Z (14):     {Z[14].item():.6f}")

    return loss_history, params, target_formation_energy, final_formation_energy, alpha, Z



def plot_loss(loss_history):
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


def save_loss_plot(loss_history, path: str, filename: str):
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

    # Stelle sicher, dass der Pfad existiert
    os.makedirs(path, exist_ok=True)
    
    # Speichern
    save_path = os.path.join(path, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # Plot-Fenster schließen, um Speicher zu sparen



if __name__ == "__main__":
    # Beispielaufruf mit veränderbaren Parametern:
    SAMPLE_PATH = 'dft.hdf5'
    seed = random.randint(0, 999999)
    DATAPOINTS = select_random_datapoints(100, seed)  # Beispiel, kann angepasst werden
    print(DATAPOINTS)
    LR = 0.05
    EPOCHS = 200
    REPULSIVE = PTBPRepulsive

    xTB_alpha = {14: Parameter(torch.tensor([1.0]), requires_grad=True)}
    PTBP_alpha = {14: Parameter(torch.tensor([1.0]), requires_grad=True)}
    Gamma_alpha = {14: Parameter(torch.tensor([2.7]), requires_grad=True)}
    custom_Z = {14: Parameter(torch.tensor([14.0]), requires_grad=True)}

    loss_hist, _, _, _, _, _ = train_model(
        "dft.hdf5",
        DATAPOINTS,
        lr=LR,
        epochs=EPOCHS,
        repulsive_model=REPULSIVE,
        alpha=PTBP_alpha,
        Z=custom_Z,
        early_stopping=True,       # aktiviert Early Stopping
        patience=5,               # z.B. wenn sich der Loss 15 Epochen lang nicht verbessert
        min_delta=1e-5             # minimale Verbesserung, die als relevant gilt
    )

    plot_loss(loss_hist)