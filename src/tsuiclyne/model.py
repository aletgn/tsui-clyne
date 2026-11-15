import os
import sys
sys.path.append('../../src/')

import numpy as np

from tsuiclyne.material import Material
from tsuiclyne.mechanics import force, neutral_axis, stiffness, curvature_diff
from tsuiclyne.mechanics import stress_substrate_bottom, stress_substrate_interface, stress_layer, stress_deposit_layer
from tsuiclyne.thermal import th_neutral_axis, th_curvature_diff, th_stiffness, force_cte
from tsuiclyne.thermal import th_stress_substrate_bottom, th_stress_substrate_interface, th_stress_layer
from tsuiclyne.mechanics import force_resultant, moment_resultant, redistribute
from tsuiclyne.util import line, dict_to_h5


class TsuiClaynePlane:
    """Model the stress evolution in a substrate-deposit system.

    Simulates the mechanical and thermal stress distributions during
    deposition, cooling and substrate removal, following the Tsui-Clayne
    analytical approach."""

    def __init__(self, substrate: Material, deposit: Material, sigma_q: float, w: float=None, n: int=None, name: str="Untitled") -> None:
        """Initialise the substrate-deposit model.

            Parameters
            ----------
            substrate : Material
                Substrate material properties.
            deposit : Material
                Deposit material properties.
            sigma_q : float
                Quench stress applied during deposition.
            w : float, optional
                Deposit layer thickness. Must be provided if ``n`` is not specified.
            n : int, optional
                Number of deposition layers. Must be provided if ``w`` is not specified.
            name : str, optional
                Model name, by default ``"Untitled"``.

            Attributes
            ----------
            name : str
                Model name.
            substrate : Material
                Substrate material properties.
            deposit : Material
                Deposit material properties.
            b : float
                Common width of the substrate and deposit.
            sigma_q : float
                Quench stress.
            w : float
                Deposition layer thickness.
            n : int
                Number of deposition steps.
            steps : np.ndarray
                Deposition step indices.
            midpoints : np.ndarray
                Vertical coordinates of deposit layer midpoints.

            F : np.ndarray or None
                Deposition force at each step.
            delta_0 : float or None
                Initial neutral axis position.
            delta : np.ndarray or None
                Neutral axis positions at each deposition step.
            delta_prev : np.ndarray or None
                Neutral axis positions at the preceding deposition steps.
            sigma : np.ndarray or None
                Composite bending stiffness at each deposition step.
            k_diff : np.ndarray or None
                Curvature increments during deposition.

            ssb_dep_hist : np.ndarray
                Substrate bottom stress contributions from each deposition step.
            ssb_dep : float
                Total substrate bottom stress after deposition.
            sst_dep_hist : np.ndarray
                Substrate interface stress contributions from each deposition step.
            sst_dep : float
                Total substrate interface stress after deposition.
            sd_layer : np.ndarray
                Stress distribution within individual deposited layers.
            sd_dep : np.ndarray
                Total deposit stress distribution after deposition.
            done_deposition : bool
                Whether deposition calculations have been completed.

            th_scale : float
                Scaling factor for thermal stresses, used to convert to MPa.
            delta_eps : float or None
                Differential thermal strain.
            th_delta : float or None
                Neutral axis position under thermal loading.
            th_k_diff : float or None
                Thermal curvature increment.
            th_sigma : float or None
                Composite bending stiffness under thermal loading.
            th_F : float or None
                Equivalent thermal force.
            ssb_th : float or np.ndarray
                Thermal stress at the substrate bottom.
            sst_th : float or np.ndarray
                Thermal stress at the substrate-deposit interface.
            sd_th : np.ndarray
                Thermal stress distribution in the deposit.
            done_cooling : bool
                Whether cooling calculations have been completed.

            slope : float or None
                Slope of the linear substrate stress distribution.
            intercept : float or None
                Intercept of the linear substrate stress distribution.
            F_s : float
                Substrate force resultant.
            M_s : float
                Substrate moment resultant.
            slope_red : float or None
                Slope of the redistributed deposit stress distribution.
            intercept_red : float or None
                Intercept of the redistributed deposit stress distribution.
            F_chk : float or None
                Force resultant of the redistributed deposit stress.
            M_chk : float or None
                Moment resultant of the redistributed deposit stress.
            F_res : float or None
                Force equilibrium residual.
            M_res : float or None
                Moment equilibrium residual.
            done_removal : bool
                Whether removal calculations have been completed.

            ssb : float
                Total substrate bottom stress after cooling.
            sst : float
                Total substrate interface stress after cooling.
            sd : np.ndarray
                Total deposit stress distribution after cooling.
            sd_rem : np.ndarray
                Deposit stress distribution after substrate removal.
            """
        self.name = name
        self.substrate = substrate
        self.deposit = deposit

        # b must be unique for both substrate and deposit
        assert abs(self.substrate.b - self.deposit.b) < 0.0001
        self.b = self.substrate.b

        self.sigma_q = sigma_q  # Quench stress

        if (w is None) == (n is None):
            raise ValueError("Exactly one of 'w' or 'n' must be provided.")

        if w is not None:
            print(f"Provided step size (w): {w}")
            self.w = w
            self.n = np.ceil(deposit.h / self.w)
        else:
            print(f"Provided layers(n): {n}")
            self.n = n
            self.w = deposit.h / self.n

        self.steps = np.arange(1, self.n + 1, dtype=int)
        self.midpoints = self.steps * self.w - self.w / 2

        # deposition
        self.F = None
        self.delta_0 = None
        self.delta = None
        self.delta_prev = None
        self.sigma = None
        self.k_diff = None
        self.ssb_dep_hist = np.zeros_like(self.steps, dtype=float)
        self.ssb_dep = 0.
        self.sst_dep_hist = np.zeros_like(self.steps, dtype=float)
        self.sst_dep = 0.
        self.sd_layer = np.zeros_like(self.steps, dtype=float)
        self.sd_dep = np.zeros_like(self.steps, dtype=float)
        self.done_deposition = False

        # cooling
        self.th_scale = 1000 # to MPa
        self.delta_eps = None
        self.th_delta = None
        self.th_k_diff = None
        self.th_sigma = None
        self.th_F = None
        self.ssb_th = np.zeros_like(self.steps, dtype=float)
        self.sst_th = np.zeros_like(self.steps, dtype=float)
        self.sd_th = np.zeros_like(self.steps, dtype=float)
        self.done_cooling = False

        # removal
        self.slope = None
        self.intercept = None
        self.F_s = 0.
        self.M_s = 0.
        self.slope_red = None
        self.intercept_red = None
        self.F_chk = None
        self.M_chk = None
        self.F_res = None
        self.M_res = None
        self.sd_rem = np.zeros_like(self.steps, dtype=float)
        self.done_removal = False

        # total after cooling
        self.ssb = 0.
        self.sst = 0.
        self.sd = np.zeros_like(self.steps, dtype=float)


    def _update_total_stress(self, sb: float, st: float, sd: np.ndarray, removal: bool=False) -> None:
        self.ssb += sb
        self.sst += st
        self.sd += sd
        print(f"\nTotal stress so far:")
        if not removal:
            print(f"- substrate bottom {self.ssb}\n- substrate interface {self.sst}.")
        print(f"- deposit interface {self.sd[0]}\n- deposit top {self.sd[-1]}.\n")


    def deposition(self) -> None:
        """Calculate mechanical stresses generated during deposition.

        Computes the deposition forces, neutral axis positions, composite
        bending stiffness, curvature increments and resulting stress
        distributions in the substrate and deposit.

        Updates the deposition-related attributes and sets
        ``done_deposition`` to True.
        """
        print("Compute force.")
        self.F = force(self.steps, self.sigma_q, self.b,
                       self.substrate.h, self.substrate.E_eff,
                       self.w, self.deposit.E_eff)

        print("Compute positions of neutral axis.")
        self.delta_0 = neutral_axis(0, self.substrate.h, self.substrate.E_eff, self.w, self.deposit.E_eff)
        self.delta = neutral_axis(self.steps, self.substrate.h, self.substrate.E_eff, self.w, self.deposit.E_eff)
        self.delta_prev = np.concatenate([[self.delta_0], self.delta[:-1]])

        print("Compute stiffness.")
        self.sigma = stiffness(self.steps, self.b,
                               self.substrate.h, self.substrate.E_eff,
                               self.w, self.deposit.E_eff, self.delta)

        print("Compute curvature differences.")
        self.k_diff = curvature_diff(self.steps, self.w, self.F, self.delta_prev, self.sigma)

        print("Done deposition.")
        self.done_deposition = True

        print("Computing stress.")
        self.ssb_dep_hist = stress_substrate_bottom(self.steps, self.b,
                                                    self.substrate.h, self.substrate.E_eff,
                                                    self.w, self.deposit.E_eff,
                                                    self.F, self.delta, self.k_diff)
        self.sst_dep_hist = stress_substrate_interface(self.steps, self.b,
                                                       self.substrate.h, self.substrate.E_eff,
                                                       self.w, self.deposit.E_eff,
                                                       self.F, self.delta, self.k_diff)
        self.ssb_dep = self.ssb_dep_hist.sum()
        self.sst_dep = self.sst_dep_hist.sum()
        print(f"Stress in substrate. Bottom: {self.ssb_dep:.5f}, Interface: {self.sst_dep:.5f}.")

        self.sd_layer = stress_layer(self.steps, self.b, self.w, self.deposit.E_eff,
                                     self.F, self.delta, self.k_diff)
        self.sd_dep = stress_deposit_layer(self.steps, self.steps, self.b,
                                           self.substrate.h, self.substrate.E_eff,
                                           self.w, self.deposit.E_eff, self.F, self.delta, self.k_diff)
        print(f"Stress in deposit (edges). Interface: {self.sd_dep[0]:.5f}, Top: {self.sd_dep[-1]:.5f}.")
        self._update_total_stress(self.ssb_dep, self.sst_dep, self.sd_dep)


    def cooling(self, delta_T: float | None = None, delta_eps: float | None = None) -> None:
        """Calculate thermal stresses generated during cooling.

        Parameters
        ----------
        delta_T : float, optional
            Temperature change during cooling. Used to calculate the
            differential thermal strain if ``delta_eps`` is not provided.
        delta_eps : float, optional
            Differential thermal strain between the deposit and substrate.
            If provided, takes precedence over ``delta_T``.

        Raises
        ------
        Exception
            If deposition calculations have not been completed.

        Notes
        -----
        Thermal stresses are scaled by ``th_scale`` to convert their
        numerical values to MPa.

        Updates the thermal-related attributes and sets ``done_cooling``
        to True.
        """
        if not self.done_deposition:
            raise Exception("Must run deposition first.")

        if delta_eps is None:
            self.delta_eps = (self.deposit.alpha - self.substrate.alpha) * delta_T
        else:
            self.delta_eps = delta_eps

        print("Compute position of neutral axis")
        self.th_delta = th_neutral_axis(self.substrate.h, self.substrate.E_eff,
                                        self.deposit.h, self.deposit.E_eff)

        print("Compute curvature difference.")
        self.th_k_diff = th_curvature_diff(self.substrate.h, self.substrate.E_eff,
                                           self.deposit.h, self.deposit.E_eff, delta_eps=self.delta_eps)

        print("Compute (thermal) stiffness.")
        self.th_sigma = th_stiffness(self.b, self.substrate.h, self.substrate.E_eff,
                                     self.deposit.h, self.deposit.E_eff, self.th_delta)

        print("Compute thermal force.")
        self.th_F = force_cte(self.substrate.h, self.deposit.h, self.th_k_diff, self.th_sigma)

        print("Done cooling.")
        self.done_cooling = True

        print("Computing thermal stress.")
        self.ssb_th = th_stress_substrate_bottom(self.b, self.substrate.h, self.substrate.E_eff,
                                                 self.th_F, self.th_delta, self.th_k_diff) * self.th_scale
        self.sst_th = th_stress_substrate_interface(self.b, self.substrate.h, self.substrate.E_eff,
                                                    self.th_F, self.th_delta, self.th_k_diff) * self.th_scale
        print(f"Thermal stress in substrate. Bottom: {self.ssb_th:.5f}, Interface: {self.sst_th:.5f}.")

        self.sd_th = th_stress_layer(self.steps, self.b, self.w,
                                     self.deposit.h, self.deposit.E_eff,
                                     self.th_F, self.th_delta, self.th_k_diff) * self.th_scale
        print(f"Thermal stress in deposit (edges). Interface: {self.sd_th[0]:.5f}, Top: {self.sd_th[-1]:.5f}.")
        self._update_total_stress(self.ssb_th, self.sst_th, self.sd_th)


    def removal(self, ssb: float | None = None, sst: float | None = None) -> None:
        """Calculate stress redistribution following substrate removal.

        Fits a linear stress distribution through the substrate bottom
        and interface stress values, calculates the corresponding force
        and moment resultants, and redistributes these resultants into
        the deposit.

        Parameters
        ----------
        ssb : float, optional
            Substrate bottom stress. If not provided, uses the total
            stress stored in ``self.ssb``.
        sst : float, optional
            Substrate-deposit interface stress. If not provided, uses
            the total stress stored in ``self.sst``.

        Notes
        -----
        Force and moment equilibrium are checked using absolute residuals
        with a tolerance of 1e-4.

        Updates the redistributed stress distribution ``sd_rem`` and
        sets ``done_removal`` to True.
        """
        print(f"Compute line pars for total distribution in substrate (linear).")
        if ssb is None and sst is None:
            print(f"Taking (substrate): bottom {self.ssb}, interface {self.sst}.")
            ssb = self.ssb
            sst = self.sst
        else:
            print(f"Values: bottom {ssb}, interface {sst} already provided.")

        self.slope, self.intercept = line(ssb, -self.substrate.h, sst)
        print(f"Line pars: slope {self.slope}, intercept {self.intercept}.")

        print(f"Compute resultants.")
        self.F_s = force_resultant(self.b, self.substrate.h, self.slope, self.intercept)
        self.M_s = moment_resultant(self.b, self.substrate.h, self.slope, self.intercept)
        print(f"Resultants: force {self.F_s}, moment {self.M_s}.")

        print(f"Redistribution of stress in substrate.")
        self.slope_red, self.intercept_red = redistribute(self.b, self.deposit.h, self.F_s, self.M_s)
        print(f"Line pars of redistribution: slope: {self.slope_red}, intercept: {self.intercept_red}")

        print(f"Checking is redistributed stress is equivalent to original.")
        self.F_chk = self.b * (self.intercept_red * self.deposit.h + self.slope_red * self.deposit.h**2 / 2)
        self.M_chk = self.b * (self.intercept_red * self.deposit.h**2 / 2 + self.slope_red * self.deposit.h**3 / 3)

        self.F_res = abs(self.F_s + self.F_chk)
        self.M_res = abs(self.M_s + self.M_chk)
        print(f"Residuals: force {self.F_res}, moment: {self.M_res}")
        assert self.F_res < 0.0001 and self.M_res < 0.0001
        print("Passed.")
        self.done_removal = True

        print("Compute stress in deposit.")
        # self.sd_rem = self.sd - self.midpoints * self.slope_red - self.intercept_red
        self.sd_rem = - self.midpoints * self.slope_red - self.intercept_red
        print(f"Stress in deposit after removal (edges): interface {self.sd_rem[0]}, top {self.sd_rem[-1]}")
        self._update_total_stress(0., 0., self.sd_rem, True)


    def export(self, path: str, prefix: str = "") -> None:
        """Export model attributes to an HDF5 file.

        Parameters
        ----------
        path : str
            Path to the HDF5 file.
        prefix : str, optional
            Prefix added to each exported dataset name, by default ``""``.

        Notes
        -----
        The substrate and deposit Material objects are excluded from
        the export. Their individual properties can be exported separately.
        """
        _dict = {f"{prefix}{k}": v for k, v in vars(self).items()}
        _dict.pop(f"{prefix}substrate")
        _dict.pop(f"{prefix}deposit")
        dict_to_h5(path, _dict)


    def __repr__(self):
        return (f"{self.__class__.__name__}(\n"
                f"substrate={self.substrate!r},\n"
                f"deposit={self.deposit!r},\n"
                f"sigma_q={self.sigma_q!r}, "
                f"w={self.w!r}, "
                f"n={self.n!r},\n"
                f"name={self.name!r}\n)")