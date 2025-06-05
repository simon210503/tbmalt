from __future__ import annotations
import warnings
import re
import numpy as np
from numpy import ndarray as Array
from itertools import combinations_with_replacement
from typing import List, Literal, Optional, Dict, Tuple, Union, Type
from scipy.interpolate import CubicSpline as ScipyCubicSpline
import torch
from torch import Tensor
from torch.nn import Parameter, ParameterDict, ModuleDict, Module

from tbmalt import Geometry, OrbitalInfo, Periodicity
from tbmalt.structures.geometry import atomic_pair_distances
from tbmalt.ml.integralfeeds import IntegralFeed
from tbmalt.io.skf import Skf, VCRSkf
from tbmalt.physics.dftb.slaterkoster import sub_block_rot
from tbmalt.data.elements import chemical_symbols
from tbmalt.ml import Feed
from tbmalt.common.batch import pack, prepeat_interleave, bT, bT2
from tbmalt.common.maths.interpolation import PolyInterpU, BicubInterpSpl
from tbmalt.common.maths.interpolation import CubicSpline
from tbmalt.common import unique
from tbmalt.physics.dftb.feeds import PairwiseRepulsiveEnergyFeed



class xTBRepulsive(Feed):
     
    """Repulsive in form of the xTB-Repulsive.

    Computes the repulsive energy term (E_rep) between atoms A and B.

    This expression is commonly used in semiempirical quantum chemical methods to model 
    the short-range repulsive interaction between atoms. The energy is calculated as:

        E_rep = (Z_A^eff * Z_B^eff / R_AB) * exp(-sqrt(α_A * α_B) * (R_AB)^k_f)

    Where:
    - Z_A^eff, Z_B^eff: Effective nuclear charges of atoms A and B
    - R_AB: Distance between atoms A and B
    - α_A, α_B: Element-specific repulsion parameters for atoms A and B
    - k_f: Empirical exponent controlling the distance dependence of the repulsion

    Arguments:
        coefficients: List containing import parameter
            c[0] := Z_A^eff
            c[1] := Z_B^eff
            c[2] := α_A
            c[3] := α_B
    """

    def __init__(
            self, coefficients: Parameter):

        super().__init__()
        self.coefficients = coefficients

    def forward(self, distances: Tensor) -> Tensor:
        """Evaluate the repulsive interaction at the specified distance(s).

        Arguments:
            distances: Distance(s) at which the repulsive term is to be
                evaluated.

        Returns:
            repulsive: Repulsive interaction energy as evaluated at the
                specified distances.
        """
        results = torch.zeros_like(distances)
        c = self.coefficients
        z1 = c[0]
        z2 = c[1]
        a1 = c[2]
        a2 = c[3]
        kf = 1.5

        results = z1 * z2 / distances * torch.exp(-torch.sqrt(a1 * a2) * distances**kf)

        return results
    
class PTBPRepulsive(Feed):
     
    """Repulsive in form of the PTBP-Repulsive.

    The repulsive is calculated as the follwing form:

    E_rep = (Z_A^eff * Z_B^eff / R_AB) * (1 - erf(R_AB / sqrt(α_A^2 + α_B^2)))

    Arguments:
        coefficients: List containing import parameter
            c[0] := Z_A^eff
            c[1] := Z_B^eff
            c[2] := α_A
            c[3] := α_B
    """

    def __init__(
            self, coefficients: Parameter):

        super().__init__()
        self.coefficients = coefficients

    def forward(self, distances: Tensor) -> Tensor:
        """Evaluate the repulsive interaction at the specified distance(s).

        Arguments:
            distances: Distance(s) at which the repulsive term is to be
                evaluated.

        Returns:
            repulsive: Repulsive interaction energy as evaluated at the
                specified distances.
        """
        results = torch.zeros_like(distances)
        c = self.coefficients
        z1 = c[0]
        z2 = c[1]
        a1 = c[2]
        a2 = c[3]
        gamma = 1 / torch.sqrt(a1**2 + a2**2)

        results = z1 * z2 / distances * (1 - torch.erf(gamma * distances))

        return results

def pairwise_repulsive(Geometry, alpha, Z, Repulsive):
    """
    Delivers input for PairwiseRepulsiveEnergyFeed

    Arguments:
        Geometry: Geometry of a system in the tbmalt notation
        alpha: Dictionary contaning element specific repulsion parameters
                (with atomic number as key)
        Z: Dictionary contaning element specific effective charge
                (with atomic number as key)
        Repulsive: Type of Repulsive to be used. The following options exist:
            - xTBRepulsive
            - PTBPRepulsive

    Returns:
        A torch `ModuleDict` of pair-wise distance dependent
        repulsive feeds, keyed by strings representing tuples 
        of the form `"(z₁, z₂)"`, where `z₁` & `z₂` are the 
        atomic numbers of the associated element pair (with `z₁ ≤ z₂`).
        This can be used as input for the PairwiseRepulsiveEnergyFeed class.
    """
    Dict = ModuleDict({})
    for species_pair, _, _ in atomic_pair_distances(
        Geometry, True, True):
        Dict[str((species_pair[0].item(), species_pair[1].item()))
             ] = Repulsive([Z[species_pair[0].item()],
                               Z[species_pair[1].item()],
                               alpha[species_pair[0].item()],
                               alpha[species_pair[1].item()]])
    return Dict


if __name__ == '__main__':

    alpha = {
        1: Parameter(Tensor([2.0]),requires_grad = True),
        8: Parameter(Tensor([2.0]),requires_grad = True)
    }

    Z = {
        1: Parameter(Tensor([1.0]),requires_grad = True),
        8: Parameter(Tensor([8.0]),requires_grad = True)
    }
    
    H2O_geo = Geometry(torch.tensor([8, 1, 1]), 
               torch.tensor([[0.0, -1.0, 0.0],
                             [0.0, 0.0, 0.78306400000],
                             [0.0, 0.0, -0.78306400000]], requires_grad=True),
               units='angstrom'
               )

    xTB_pair_repulsive = pairwise_repulsive(H2O_geo, alpha, Z, xTBRepulsive)

    xTB_total_repulsive = PairwiseRepulsiveEnergyFeed(xTB_pair_repulsive)

    print(xTB_total_repulsive.forward(H2O_geo))

    PTBP_pair_repulsive = pairwise_repulsive(H2O_geo, alpha, Z, PTBPRepulsive)

    PTBP_total_repulsive = PairwiseRepulsiveEnergyFeed(PTBP_pair_repulsive)
    
    print(PTBP_total_repulsive.forward(H2O_geo))
    