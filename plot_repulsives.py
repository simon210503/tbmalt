import torch
import re
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Parameter
from torch import Tensor
from traintest import load_results_from_file
from new_feeds import DFTBGammaRepulsive, xTBRepulsive, PTBPRepulsive
from tbmalt.structures.geometry import atomic_pair_distances
from formation_calc import load_Geo_dset
from collections import Counter
from tbmalt.physics.dftb.feeds import DftbpRepulsiveSpline
from tbmalt.io.skf import Skf


def load_parameters(file_path):
    raw_alpha = load_results_from_file(file_path)[1]
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", raw_alpha)
    return [float(n) for n in numbers]


def plot_repulsive_curves(base_path, all_distances=None, save_path=None):
    Gamma_alpha = load_parameters(f'{base_path}/Gamma_result/result_8.txt')
    Gamma_Z = load_parameters(f'{base_path}/Gamma_result/result_9.txt')
    xTB_alpha = load_parameters(f'{base_path}/xTB_result/result_8.txt')
    xTB_Z = load_parameters(f'{base_path}/xTB_result/result_9.txt')
    PTBP_alpha = load_parameters(f'{base_path}/PTBP_result/result_8.txt')
    PTBP_Z = load_parameters(f'{base_path}/PTBP_result/result_9.txt')

    Gamma = DFTBGammaRepulsive(
        [Parameter(Tensor(Gamma_Z)), Parameter(Tensor(Gamma_Z)),
         Parameter(Tensor(Gamma_alpha)), Parameter(Tensor(Gamma_alpha))],
        8.0
    )
    xTB = xTBRepulsive(
        [Parameter(Tensor(xTB_Z)), Parameter(Tensor(xTB_Z)),
         Parameter(Tensor(xTB_alpha)), Parameter(Tensor(xTB_alpha)), 1.5],
        8.0
    )
    PTBP = PTBPRepulsive(
        [Parameter(Tensor(PTBP_Z)), Parameter(Tensor(PTBP_Z)),
         Parameter(Tensor(PTBP_alpha)), Parameter(Tensor(PTBP_alpha))],
        8.0
    )

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
    ax.plot(r.numpy(), c.detach().numpy(), 'y', label='pbc')

    # Falls all_distances angegeben ist, Punkte hinzufügen
    if all_distances is not None:
        for i, d in enumerate(all_distances):
            ax.axvline(
                x=d.item(),
                color='k',
                linestyle='--',
                alpha=0.3,          # noch durchsichtiger
                linewidth=0.5,      # dünnere Linie
                label='distances' if i == 0 else None
            )



    ax.set_xlabel('distance [bohr]')
    ax.set_ylabel('repulsive energy [Ha]')
    ax.set_title('Si-Si repulsive optimized for a batch of defects')
    ax.legend(loc="upper right")
    ax.grid(True)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)  # schließt die Figur, damit sie nicht im Speicher bleibt
    else:
        plt.show()


def count_distances(all_distances, decimals=3):
    """
    Zählt, wie oft jede Distanz in all_distances vorkommt,
    nachdem auf `decimals` Nachkommastellen gerundet wurde.
    
    Args:
        all_distances (torch.Tensor | list | numpy.ndarray): Liste oder Tensor mit Distanzen.
        decimals (int): Anzahl der Nachkommastellen zum Runden (default: 3).
    
    Returns:
        dict: Schlüssel = gerundete Distanz (float), Wert = Anzahl der Vorkommen.
    """
    # In Python-Liste mit Floats umwandeln
    if isinstance(all_distances, torch.Tensor):
        values = all_distances.cpu().numpy()
    else:
        values = np.array(all_distances)
    
    # Runden
    rounded = np.round(values, decimals=decimals)
    
    # In Liste und zählen
    counts = Counter(rounded.tolist())
    return dict(counts)


if __name__ == '__main__':
    #plot_repulsive_curves('logs/512test')

    for ii in range(1, 7):
        base_path = 'logs/512test'

        Geo = load_Geo_dset('dft.hdf5', [ii])

        distances_list = []

        for _, _, distances in atomic_pair_distances(Geo, ignore_self=True, force_batch_index=True):
            distances_list.append(distances)

        all_distances = torch.cat(distances_list, dim=0)
        all_distances = all_distances[all_distances <= 8]
        plot_repulsive_curves(base_path, all_distances, save_path=f'repulsive_plots/{ii}.png')

    #print(count_distances(all_distances))
