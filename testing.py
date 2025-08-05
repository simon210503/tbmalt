import torch
import random
from training import calc_formation_energy, prepare_system
from formation_calc import calc_reference_formation_energies, select_random_datapoints, get_energies_from_file
from new_feeds import PTBPRepulsive, DFTBGammaRepulsive, xTBRepulsive
from torch.nn import Parameter

torch.set_default_dtype(torch.float64)

import torch.nn.functional as F

def test_model(sample_path, datapoints, repulsive_model, alpha, Z):
    
    # System vorbereiten
    params = prepare_system(sample_path, datapoints, repulsive_model, alpha, Z)

    # Vorhersage und Zielwerte berechnen
    model_formation_energy = calc_formation_energy(params)
    target_formation_energy = calc_reference_formation_energies(sample_path, datapoints)

    # Loss berechnen
    mse = F.mse_loss(model_formation_energy, target_formation_energy)
    mae = F.l1_loss(model_formation_energy, target_formation_energy)

    print(f"\nTest-Ergebnisse:")
    print(f"  MSE (mean squared error):     {mse.item():.6f}")
    print(f"  MAE (mean absolute error):    {mae.item():.6f}")

    return mse.item(), mae.item(), target_formation_energy, model_formation_energy


if __name__ == "__main__":
    # Beispielaufruf mit veränderbaren Parametern:
    SAMPLE_PATH = 'dft_test.hdf5'
    seed = random.randint(0, 999999)
    DATAPOINTS = select_random_datapoints(6, seed, 6)  # Beispiel, kann angepasst werden
    print(DATAPOINTS)
    REPULSIVE = PTBPRepulsive
    xTB_alpha = {14: Parameter(torch.tensor([1.0]), requires_grad=True)}
    PTBP_alpha = {14: Parameter(torch.tensor([1.0]), requires_grad=True)}
    Gamma_alpha = {14: Parameter(torch.tensor([2.7]), requires_grad=True)}
    custom_Z = {14: Parameter(torch.tensor([14.0]), requires_grad=True)}

    print(test_model(SAMPLE_PATH, DATAPOINTS, PTBPRepulsive, PTBP_alpha, custom_Z, 512))