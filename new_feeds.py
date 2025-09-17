import torch
from torch import Tensor
from torch.nn import Parameter, ModuleDict

from tbmalt.structures.geometry import atomic_pair_distances
from tbmalt.ml import Feed


class xTBRepulsive(Feed):
     
    """Repulsive potential in form of the xTB-Repulsive.

    Computes the repulsive potential term (V_rep) between atoms A and B.

    This expression is commonly used in semiempirical quantum chemical methods to model 
    the short-range repulsive interaction between atoms. The energy is calculated as:

        E_rep = (Z_A^eff * Z_B^eff / R_AB) * exp(-sqrt(α_A * α_B) * (R_AB)^k_f)

    Where:
    - Z_A^eff, Z_B^eff: Effective nuclear charges of atoms A and B
    - R_AB: Distance between atoms A and B
    - α_A, α_B: Element-specific repulsion parameters for atoms A and B
    - k_f: Exponent controlling the distance dependence of the repulsion

    Arguments:
        coefficients: List containing important parameters
            c[0] := Z_A^eff
            c[1] := Z_B^eff
            c[2] := α_A
            c[3] := α_B
            c[4] := k_f
        cutoff: Cutoff radius for the repulsive beyond which interactions
            are assumed to be zero.
    """

    def __init__(
            self, coefficients: Parameter, cutoff: Tensor):
        """Initialize the xTB repulsive potential module."""
        super().__init__()
        self.coefficients = coefficients
        self.cutoff = cutoff

    def forward(self, distances: Tensor) -> Tensor:
        """Evaluate the repulsive interaction energy.

        Arguments:
            distances: Distance(s) at which the repulsive term is evaluated.

        Returns:
            Repulsive interaction energy tensor at the specified distances.
        """
        results = torch.zeros_like(distances)
        c = self.coefficients
        z1 = c[0]
        z2 = c[1]
        a1 = c[2]
        a2 = c[3]
        kf = c[4]
        mask = distances < self.cutoff

        results[mask] = z1 * z2 / distances[mask] * torch.exp(-torch.sqrt(a1 * a2) * distances[mask]**kf)

        return results
    
    def derivative(self, distances: Tensor) -> Tensor:
        """Evaluate the analytic derivative of the repulsive interaction.

        Arguments:
            distances: Distance(s) at which the derivative is evaluated.

        Returns:
            Derivative of the repulsive interaction energy.
        """
        results = torch.zeros_like(distances)
        c = self.coefficients
        z1 = c[0]
        z2 = c[1]
        a1 = c[2]
        a2 = c[3]
        kf = c[4]

        V = self.forward(distances)
        results = V * (-1 / distances - kf * distances**(kf-1) * torch.sqrt(a1 * a2))

        return results

    
class PTBPRepulsive(Feed):
     
    """Repulsive potential in form of the PTBP-Repulsive.

    The repulsive potential is calculated as:

        E_rep = (Z_A^eff * Z_B^eff / R_AB) * (1 - erf(R_AB / sqrt(α_A^2 + α_B^2)))

    Arguments:
        coefficients: List containing important parameters
            c[0] := Z_A^eff
            c[1] := Z_B^eff
            c[2] := α_A
            c[3] := α_B
        cutoff: Cutoff radius for the repulsive beyond which interactions
            are assumed to be zero.
    """

    def __init__(
            self, coefficients: Parameter, cutoff: Tensor):
        """Initialize the PTBP repulsive potential module."""
        super().__init__()
        self.coefficients = coefficients
        self.cutoff = cutoff

    def forward(self, distances: Tensor) -> Tensor:
        """Evaluate the repulsive interaction energy.

        Arguments:
            distances: Distance(s) at which the repulsive term is evaluated.

        Returns:
            Repulsive interaction energy tensor at the specified distances.
        """
        results = torch.zeros_like(distances)
        c = self.coefficients
        z1 = c[0]
        z2 = c[1]
        a1 = c[2]
        a2 = c[3]
        gamma = 1 / torch.sqrt(a1**2 + a2**2)
        mask = distances < self.cutoff

        results[mask] = z1 * z2 / distances[mask] * (1 - torch.erf(gamma * distances[mask]))

        return results
    
    def derivative(self, distances: Tensor) -> Tensor:
        """Evaluate the analytic derivative of the PTBP repulsive interaction.

        Arguments:
            distances: Distance(s) at which the derivative is evaluated.

        Returns:
            Derivative of the PTBP repulsive interaction energy.
        """
        results = torch.zeros_like(distances)
        c = self.coefficients
        z1 = c[0]
        z2 = c[1]
        a1 = c[2]
        a2 = c[3]
        gamma = 1 / torch.sqrt(a1**2 + a2**2)

        V = self.forward(distances)
        mask = distances < self.cutoff

        if mask.any():
            r = distances[mask]
            V_masked = V[mask]

            term1 = - V_masked / r
            exp_term = torch.exp(-(gamma * r)**2)
            term2 = - z1 * z2 * 2 * gamma * exp_term / (r * torch.sqrt(torch.tensor(torch.pi, device=distances.device)))

            results[mask] = term1 + term2

        return results
    
class DFTBGammaRepulsive(Feed):
     
    """Repulsive in form of the DFTB-Gamma.

    The repulsion is modeled by the analytical Gamma function form used in 
    density-functional tight binding (DFTB) theory.

    Arguments:
        coefficients: List containing important parameters
            c[0] := Z_A^eff
            c[1] := Z_B^eff
            c[2] := α_A
            c[3] := α_B
        cutoff: Cutoff radius for the repulsive beyond which interactions
            are assumed to be zero.
    """

    def __init__(
            self, coefficients: Parameter, cutoff: Tensor):
        """Initialize the DFTB-Gamma repulsive potential module."""
        super().__init__()
        self.coefficients = coefficients
        self.cutoff = cutoff

    def forward(self, distances: Tensor) -> Tensor:
        """Evaluate the repulsive interaction energy.

        Arguments:
            distances: Distance(s) at which the repulsive term is evaluated.

        Returns:
            Repulsive interaction energy tensor at the specified distances.
        """
        results = torch.zeros_like(distances)
        c = self.coefficients
        z1 = c[0]
        z2 = c[1]
        a1 = c[2]
        a2 = c[3]
        mask = distances < self.cutoff

        if a1 == a2:
            results[mask] = self._equal_gamma(distances[mask], a1)
        else:
            results[mask] = self._unequal_gamma(distances[mask], a1, a2)

        return results * z1 * z2
    
    def derivative(self, distances: Tensor) -> Tensor:
        """Evaluate the analytic derivative of the DFTB-Gamma repulsive interaction.

        Note:
            Currently implemented only for the special case `a1 == a2`.

        Arguments:
            distances: Distance(s) at which the derivative is evaluated.

        Returns:
            Derivative of the DFTB-Gamma repulsive interaction energy.
        """
        results = torch.zeros_like(distances)
        c = self.coefficients
        z1 = c[0]
        z2 = c[1]
        a1 = c[2]
        a2 = c[3]

        if a1 != a2:
            raise NotImplementedError("Derivative for a1 != a2 is not implemented.")
        
        poly = (
            -a1 / distances
            - 11 * a1**2 / 16
            - 3 * a1**3 * distances / 16
            - a1**4 * distances**2 / 48
            - 1 / distances**2
            + 3 * a1**2 / 16
            + a1**3 * distances / 24
        )

        results = z1 * z2 * torch.exp(-a1 * distances) * poly

        return results
    
    def _Gamma(self, a, b, R):
        """Evaluate the auxiliary Gamma function used for unequal α values."""
        zaehler1 = b**4 * a
        nenner1 = 2 * (a**2 - b**2)**2
        zaehler2 = b**6 - 3 * b**4 * a**2
        nenner2 = (a**2 - b**2)**3 * R
        result = zaehler1 / nenner1 - zaehler2 / nenner2
        return result
    
    def _equal_gamma(self, distances, a1):
        """Evaluate the repulsive Gamma function for the case `a1 == a2`."""
        term1 = 1 / distances
        term2 = 11 * a1 / 16
        term3 = 3 * a1**2 * distances / 16
        term4 = a1**3 * distances**2 / 48

        poly = term1 + term2 + term3 + term4

        exp = torch.exp(-a1 * distances)
        results = exp * poly
        return results
    
    def _unequal_gamma(self, distances, a1, a2):
        """Evaluate the repulsive Gamma function for the case `a1 != a2`."""
        exp1 = torch.exp(-a1 * distances)
        exp2 = torch.exp(-a2 * distances)
        term2 = exp1 * self._Gamma(a1, a2, distances)
        term3 = exp2 * self._Gamma(a2, a1, distances)
        results = term2 + term3

        return results

def pairwise_repulsive(Geometry, alpha, Z, Repulsive, cutoff):
    """Construct a dictionary of pairwise repulsive potential modules.

    Arguments:
        Geometry: Geometry of a system in the tbmalt notation.
        alpha: Dictionary containing element-specific repulsion parameters (atomic number as key).
        Z: Dictionary containing element-specific effective charges (atomic number as key).
        Repulsive: Type of repulsive potential to be used. Options:
            - xTBRepulsive
            - PTBPRepulsive
            - DFTBGammaRepulsive
        cutoff: Dictionary of cutoff radii keyed by element pairs.

    Returns:
        A torch `ModuleDict` of pairwise distance-dependent
        repulsive feeds, keyed by strings representing tuples 
        of the form `"(z₁, z₂)"`, where `z₁` & `z₂` are the 
        atomic numbers of the associated element pair (with `z₁ ≤ z₂`).
    """
    Dict = ModuleDict({})
    for species_pair, _, _ in atomic_pair_distances(
        Geometry, True, True):
        cond1 = species_pair[0].item() <= 2
        cond2 = species_pair[1].item() <= 2
        if cond1 and cond2:
            kb = 1
        else:
            kb = 1.5
        Dict[str((species_pair[0].item(), species_pair[1].item()))
             ] = Repulsive([Z[species_pair[0].item()],
                            Z[species_pair[1].item()],
                            alpha[species_pair[0].item()],
                            alpha[species_pair[1].item()],
                            kb], 
                            cutoff[str((species_pair[0].item(),
                                         species_pair[1].item()))
             ])
    return Dict