"""Model for the 3D poromechanics problem. Can also be used for the pure mechanics
problem.

NOTE: When running this model on large grids, the main part of the computation time is
spent on solving the resulting linear system. Two approaches are provided:
    1. A direct solver.
    2. An iterative solver, based on GMRES with a block-diagonal preconditioner that is
       in part based on PyAMG (assumed to be available).

To control the linear solver, search of the line 'self.linear_solver =' (there are two
occurrences, one each for the mechanics and poromechanical model) and set this to either
'direct' or 'iterative'.

"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


from collections import namedtuple
import sympy as sym
from porepy.viz.data_saving_model_mixin import VerificationDataSaving
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.applications.md_grids.domains import nd_cube_domain
import porepy as pp
import numpy as np
from dataclasses import make_dataclass
import scipy.sparse as sps
import time
from typing import Callable, Union
import warnings

import pickle
import dataclasses

import matplotlib

# matplotlib.use("Tkagg")
import matplotlib.pyplot as plt


Field = namedtuple("Field", ["name", "is_scalar", "is_cc", "relative"])


class DataSaving(VerificationDataSaving):
    """Mixin class to save relevant data."""

    def collect_data(self):
        """Collect data from the verification setup.

        Returns:
            ManuPoroMechSaveData object containing the results of the verification for
            the current time.

        """

        mdg: pp.MixedDimensionalGrid = self.mdg
        sd: pp.Grid = mdg.subdomains()[0]
        t = self.time_manager.time

        collected_data = {}

        export_data = []

        for field in self.fields:
            exact_value = getattr(self.exact_sol, field.name)(sd=sd, time=t)
            ad_representation = getattr(self, field.name)([sd])
            approx_value = ad_representation.value(self.equation_system)
            if isinstance(exact_value, (int, float)):
                exact_value = exact_value * np.ones_like(approx_value)
            elif isinstance(exact_value, np.ndarray) and (
                exact_value.ndim == 2 and exact_value.shape[1] == 1
            ):
                exact_value = np.ravel(exact_value * np.ones(sd.num_cells), order="F")
            elif isinstance(exact_value, np.ndarray) and (
                exact_value.ndim == 1 and exact_value.shape[0] == 3
            ):
                exact_value = np.ravel(
                    exact_value.reshape((-1, 1)) * np.ones(sd.num_cells), order="F"
                )

            if field.is_cc:
                export_data.append((sd, f"{field.name}_approx", approx_value))
                export_data.append((sd, f"{field.name}_exact", exact_value))
                export_data.append(
                    (sd, f"{field.name}_error", approx_value - exact_value)
                )

            error, ref_norm = self.l2_error(
                grid=sd,
                true_array=exact_value,
                approx_array=approx_value,
                name=field.name,
                is_scalar=field.is_scalar,
                is_cc=field.is_cc,
                relative=False,
            )
            collected_data[field.name] = (error, ref_norm)

        # exporter = pp.Exporter(self.mdg, file_name=f"results_{sd.num_cells}_cells")
        # exporter.write_vtu(export_data)

        collected_data["time"] = t
        collected_data["cell_diameter"] = sd.cell_diameters(
            cell_wise=False, func=np.min
        )

        # Convert to dataclass, big thanks to https://stackoverflow.com/a/77325321
        return make_dataclass(
            "SavedData", ((k, type(v)) for k, v in collected_data.items())
        )(**collected_data)

    def l2_error(
        self,
        grid: pp.GridLike,
        true_array: np.ndarray,
        approx_array: np.ndarray,
        name: str,
        is_scalar: bool,
        is_cc: bool,
        relative: bool = False,
    ) -> pp.number:
        """Compute discrete L2-error as given in [1].

        It is possible to compute the absolute error (default) or the relative error.

        Raises:
            NotImplementedError if a mortar grid is given and ``is_cc=False``.
            ZeroDivisionError if the denominator in the relative error is zero.

        Parameters:
            grid: Either a subdomain grid or a mortar grid.
            true_array: Array containing the true values of a given variable.
            approx_array: Array containing the approximate values of a given variable.
            is_scalar: Whether the variable is a scalar quantity. Use ``False`` for
                vector quantities. For example, ``is_scalar=True`` for pressure, whereas
                ``is_scalar=False`` for displacement.
            is_cc: Whether the variable is associated to cell centers. Use ``False``
                for variables associated to face centers. For example, ``is_cc=True``
                for pressures, whereas ``is_scalar=False`` for subdomain fluxes.
            relative: Compute the relative error (if True) or the absolute error (if False).

        Returns:
            Discrete L2-error between the true and approximated arrays.

        References:

            - [1] Nordbotten, J. M. (2016). Stable cell-centered finite volume
                discretization for Biot equations. SIAM Journal on Numerical Analysis,
                54(2), 942-968.

        """
        # Sanity check
        if isinstance(grid, pp.MortarGrid) and not is_cc:
            raise NotImplementedError("Interface variables can only be cell-centered.")

        mech_data = self.mdg.subdomain_data(grid)[pp.PARAMETERS][self.stress_keyword]

        if hasattr(self, "darcy_keyword"):
            fluid_data = self.mdg.subdomain_data(grid)[pp.PARAMETERS][
                self.darcy_keyword
            ]
            permeability = fluid_data["second_order_tensor"].values[0, 0]
            has_fluid = True
        else:
            has_fluid = False

        stiffness = mech_data["fourth_order_tensor"]

        try:
            cosserat_parameter = mech_data["cosserat_parameter"]
        except KeyError:
            cosserat_parameter = np.zeros(grid.num_cells)

        mu = stiffness.mu
        lmbda = stiffness.lmbda

        # Obtain proper measure, e.g., cell volumes for cell-centered quantities and face
        # areas for face-centered quantities.
        if is_cc:
            vol = grid.cell_volumes

            if name == "rotation":
                meas = vol / mu
            elif name == "total_pressure":
                if not has_fluid:
                    meas = vol / lmbda
                else:
                    meas = 0
            elif name == "displacement":
                meas = vol * mu
            elif name == "pressure":
                meas = vol

        else:
            assert isinstance(grid, pp.Grid)  # to please mypy
            fi, ci, sgn = sps.find(grid.cell_faces)
            fc_cc = grid.face_centers[::, fi] - grid.cell_centers[::, ci]
            n = grid.face_normals[::, fi] / grid.face_areas[fi]
            dist_fc_cc = np.abs(np.sum(fc_cc * n, axis=0))

            # Distance between neighboring cells
            dist_cc_cc = np.bincount(fi, weights=dist_fc_cc, minlength=grid.num_cells)

            meas = dist_cc_cc / grid.dim

            debug = True

        if not is_scalar:
            meas = meas.repeat(grid.dim)

        # Obtain numerator and denominator to determine the error.
        numerator = np.sqrt(np.sum(meas * np.abs(true_array - approx_array) ** 2))

        denominator = (
            np.sqrt(np.sum(meas * np.abs(true_array) ** 2)) if relative else 1.0
        )

        if name == "total_pressure":
            mean_val = np.mean((approx_array - true_array) * vol)
            mean_val_true = np.mean(true_array * vol)
            numerator_2 = np.sqrt(
                np.sum(vol / mu * (approx_array - true_array) - mean_val) ** 2
            )
            denominator_2 = np.sqrt(
                np.sum(vol / mu * (true_array - mean_val_true) ** 2)
            )

            numerator += numerator_2
            denominator += denominator_2

        # Deal with the case when the denominator is zero when computing the relative error.
        if np.isclose(denominator, 0) and not relative:
            raise ZeroDivisionError("Attempted division by zero.")

        return numerator, denominator


class ExactSolution:
    def __init__(self, setup):
        self.nd = setup.nd
        # Heterogeneity factor.
        heterogeneity: float = 1

        # Lamé parameters
        lame_lmbda_base = setup.solid.lame_lambda
        lame_mu_base = setup.solid.shear_modulus
        biot_coefficient = setup.solid.biot_coefficient
        permeability = setup.solid.permeability
        porosity = setup.solid.porosity

        fluid_compressibility = setup.fluid.reference_component.compressibility

        pi = sym.pi

        t, x, y, z = self._symbols()
        all_vars = [x, y, z]
        # Characteristic function: 1 if x > 0.5 and y > 0.5, 0 otherwise
        char_func = sym.Piecewise((1, ((x > 0.5) & (y > 0.5) & (z > 0.5))), (0, True))

        def make_heterogeneous(v, invert: bool):
            # Helper function to include the heterogeneity into a function.
            if invert:
                return v / ((1 - char_func) + char_func * heterogeneity)
            else:
                return v * ((1 - char_func) + char_func * heterogeneity)

        lame_lmbda = make_heterogeneous(lame_lmbda_base, False)
        lame_mu = make_heterogeneous(lame_mu_base, False)

        # The displacement is the curl of pot
        u = [0, 0, 0]

        rot = [0, 0, 0]
        # The solid pressure is the divergence of the displacement, hence 0
        # (div curl = 0)
        g = 1000
        solid_p = (x * x + y * y + z * z) * g / lame_lmbda
        # solid_p = 1000 * z / lame_lmbda
        # Fluid pressure is set to zero, this is a pure mechanics problem.
        fluid_p = 0

        q = [
            -permeability * sym.diff(fluid_p, x),
            -permeability * sym.diff(fluid_p, y),
            -permeability * sym.diff(fluid_p, z),
        ]

        nd = 3

        # Exact gradient of the displacement
        grad_u = [[sym.diff(u[i], var) for var in all_vars] for i in range(self.nd)]
        rot_dim = len(rot)

        displacement_stress = [
            [
                2 * lame_mu * grad_u[0][0],
                2 * lame_mu * grad_u[0][1],
                2 * lame_mu * grad_u[0][2],
            ],
            [
                2 * lame_mu * grad_u[1][0],
                2 * lame_mu * grad_u[1][1],
                2 * lame_mu * grad_u[1][2],
            ],
            [
                2 * lame_mu * grad_u[2][0],
                2 * lame_mu * grad_u[2][1],
                2 * lame_mu * grad_u[2][2],
            ],
        ]

        # Exact elastic stress
        sigma_total = [
            [
                lame_lmbda * solid_p
                + 2 * lame_mu * grad_u[0][0]
                - biot_coefficient * fluid_p,
                2 * lame_mu * (grad_u[0][1] - rot[2]),
                2 * lame_mu * (grad_u[0][2] + rot[1]),
            ],
            [
                2 * lame_mu * (grad_u[1][0] + rot[2]),
                lame_lmbda * solid_p
                + 2 * lame_mu * grad_u[1][1]
                - biot_coefficient * fluid_p,
                2 * lame_mu * (grad_u[1][2] - rot[0]),
            ],
            [
                2 * lame_mu * (grad_u[2][0] - rot[1]),
                2 * lame_mu * (grad_u[2][1] + rot[0]),
                lame_lmbda * solid_p
                + 2 * lame_mu * grad_u[2][2]
                - biot_coefficient * fluid_p,
            ],
        ]
        # Mechanics source term
        source_mech = [
            sum(
                [
                    sym.diff(sigma_total[i][j], v)
                    for (j, v) in zip(range(self.nd), all_vars)
                ]
            )
            for i in range(self.nd)
        ]

        stress_asymmetry = [
            sigma_total[2][1] - sigma_total[1][2],
            sigma_total[0][2] - sigma_total[2][0],
            sigma_total[1][0] - sigma_total[0][1],
        ]
        source_rot = [-stress_asymmetry[i] / (2 * lame_mu) for i in range(self.nd)]

        source_p = sym.diff(u[0], x) + sym.diff(u[1], y) + sym.diff(u[2], z) - solid_p

        ## Public attributes
        # Primary variables
        self.u = u  # displacement
        self.rot = [rot[i] * 2 * lame_mu for i in range(self.nd)]
        # The solid pressure used in the convergence analysis is different from the one
        # used to derive the analytical solutions. Change.
        self.solid_p = (
            solid_p * lame_lmbda - biot_coefficient * fluid_p
        )  # Solid pressure
        self.f_pressure = fluid_p  # Fluid pressure

        # Secondary variables
        self.sigma_total = sigma_total  # poroelastic (total) stress
        self._displacement_stress = displacement_stress  # Displacement stress

        # The 3d expression will be different
        total_rotation = [[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]

        # Exact divergence of the mass flux
        div_mf = sym.diff(q[0], x) + sym.diff(q[1], y) + sym.diff(q[2], z)

        # Exact flow accumulation
        accum_flow = fluid_compressibility * fluid_p + biot_coefficient * solid_p

        # Exact flow source
        source_flow = accum_flow + div_mf
        self.source_flow = source_flow
        self.total_rot = total_rotation  # Cosserat couple stress

        self.fluid_flux = q  # Darcy flux

        # Source terms
        self.source_mech = source_mech  # Source term entering the momentum balance
        self.source_rotation = source_rot  # Source term entering the rotation balance
        self.source_p = source_p  # Source term entering the solid pressure balance

        # Heterogeneous material parameters. Make these available, so that a model can
        # be populated with these parameters.
        self.lame_lmbda = lame_lmbda  # Lamé parameter
        self.lame_mu = lame_mu  # Lamé parameter

    def _symbols(self):
        if self.nd == 2:
            return sym.symbols("t x y")
        else:
            return sym.symbols("t x y z")

    def _cc(
        self, sd: pp.Grid
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.nd == 2:
            return sd.cell_centers[0], sd.cell_centers[1]
        else:
            return sd.cell_centers[0], sd.cell_centers[1], sd.cell_centers[2]

    def _fc(
        self, sd: pp.Grid
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.nd == 2:
            return sd.face_centers[0], sd.face_centers[1]
        else:
            return sd.face_centers[0], sd.face_centers[1], sd.face_centers[2]

    def displacement(self, sd: pp.Grid, time: float, cc=True) -> np.ndarray:
        """Evaluate exact displacement at the cell centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(2 * sd.num_cells, )`` containing the exact displacements
            at the cell centers for the given ``time``.

        Notes:
            The returned displacement is given in PorePy's flattened vector format.

        """
        # Symbolic variables
        t, *x = self._symbols()

        # Lambdify expression
        u_fun: list[Callable] = [
            sym.lambdify((t, *x), self.u[i], "numpy") for i in range(self.nd)
        ]

        # Cell-centered displacements
        if cc:
            u_cc: list[np.ndarray] = [
                u_fun[i](time, *self._cc(sd)) for i in range(self.nd)
            ]
        else:
            u_cc: list[np.ndarray] = [
                u_fun[i](time, *self._fc(sd)) for i in range(self.nd)
            ]

        # Flatten array
        u_flat: np.ndarray = np.asarray(u_cc).ravel("F")

        if u_flat.size == 3:
            if cc:
                u_flat = np.zeros(3 * sd.num_cells)
            else:
                u_flat = np.zeros(3 * sd.num_faces)

        return u_flat

    def rotation(self, sd: pp.Grid, time: float, cc=True) -> np.ndarray:
        """Evaluate exact rotation at the cell centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd.num_cells, )`` containing the exact rotation at the cell
            centers for the given ``time``.

        Notes:
            The returned rotation is given in PorePy's flattened vector format.

        """
        # Symbolic variables
        t, *x = self._symbols()

        # Lambdify expression
        rot_fun = sym.lambdify((t, *x), self.rot, "numpy")

        # Cell-centered rotation
        if cc:
            return np.asarray(rot_fun(time, *self._cc(sd))).ravel("F")
        else:
            return np.asarray(rot_fun(time, *self._fc(sd))).ravel("F")

    def total_pressure(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact solid pressure at the cell centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd.num_cells, )`` containing the exact solid pressure at the
            cell centers for the given ``time``.

        Notes:
            The returned solid pressure is given in PorePy's flattened vector format.

        """
        # Symbolic variables
        t, *x = self._symbols()

        # Lambdify expression
        p_fun = sym.lambdify((t, *x), self.solid_p, "numpy")

        # Cell-centered solid pressure
        p_cc = p_fun(time, *self._cc(sd))

        return p_cc

    def pressure(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact solid pressure at the cell centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd.num_cells, )`` containing the exact solid pressure at the
            cell centers for the given ``time``.

        Notes:
            The returned solid pressure is given in PorePy's flattened vector format.

        """
        # Symbolic variables
        t, *x = self._symbols()

        # Lambdify expression
        p_fun = sym.lambdify((t, *x), self.f_pressure, "numpy")

        # Cell-centered solid pressure
        p_cc = p_fun(time, *self._cc(sd))

        return p_cc

    def stress(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact poroelastic force at the face centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(2 * sd.num_faces, )`` containing the exact poroealstic
            force at the face centers for the given ``time``.

        Notes:
            - The returned poroelastic force is given in PorePy's flattened vector
              format.
            - Recall that force = (stress dot_prod unit_normal) * face_area.

        """
        # Symbolic variables
        t, *x = self._symbols()

        # Get cell centers and face normals
        fc = sd.face_centers
        fn = sd.face_normals

        # Lambdify expression
        # sigma_total_fun: list[list[Callable]] = [
        #    [
        #        sym.lambdify((x, y, t), self.sigma_total[0][0], "numpy"),
        #        sym.lambdify((x, y, t), self.sigma_total[0][1], "numpy"),
        #    ],
        #    [
        #        sym.lambdify((x, y, t), self.sigma_total[1][0], "numpy"),
        #        sym.lambdify((x, y, t), self.sigma_total[1][1], "numpy"),
        #    ],
        # ]
        sigma_total_fun = [
            [
                sym.lambdify((t, *x), self.sigma_total[i][j], "numpy")
                for j in range(sd.dim)
            ]
            for i in range(sd.dim)
        ]

        # Face-centered poroelastic force
        force_total_fc = [
            sum(
                [
                    sigma_total_fun[i][j](time, *self._fc(sd)) * fn[j]
                    for j in range(sd.dim)
                ]
            )
            for i in range(sd.dim)
        ]

        # Flatten array
        force_total_flat: np.ndarray = np.asarray(force_total_fc).ravel("F")

        return force_total_flat

    def displacement_stress(self, sd: pp.Grid, time: float, cc=True) -> np.ndarray:
        t, *x = self._symbols()

        fc = sd.face_centers
        fn = sd.face_normals

        stress_fun = [
            [
                sym.lambdify((t, *x), self._displacement_stress[i][j], "numpy")
                for j in range(sd.dim)
            ]
            for i in range(self.nd)
        ]
        stress_total_fc = [
            sum([stress_fun[i][j](time, *self._fc(sd)) * fn[j] for j in range(sd.dim)])
            for i in range(self.nd)
        ]
        stress_total_flat = np.asarray(stress_total_fc).ravel("F")
        return stress_total_flat

    def total_rotation(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact poroelastic force at the face centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(2 * sd.num_faces, )`` containing the exact poroealstic
            force at the face centers for the given ``time``.

        Notes:
            - The returned poroelastic force is given in PorePy's flattened vector
              format.
            - Recall that force = (stress dot_prod unit_normal) * face_area.

        """
        # Symbolic variables
        t, *x = self._symbols()

        # Get cell centers and face normals
        fc = sd.face_centers
        fn = sd.face_normals

        rot_dim = 1 if self.nd == 2 else 3

        stress_fun = [
            [
                sym.lambdify((t, *x), self.total_rot[i][j], "numpy")
                for j in range(sd.dim)
            ]
            for i in range(rot_dim)
        ]

        force_total_fc = [
            sum([stress_fun[i][j](time, *self._fc(sd)) * fn[j] for j in range(sd.dim)])
            for i in range(rot_dim)
        ]

        # Flatten array
        force_total_flat: np.ndarray = np.asarray(force_total_fc).ravel("F")

        return force_total_flat

    def darcy_flux(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact Darcy flux [m^3 * s^-1] at the face centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd.num_faces, )`` containing the exact Darcy fluxes at
            the face centers for the given ``time``.

        Note:
            The returned fluxes are already scaled with ``sd.face_normals``.

        """
        # Symbolic variables
        t, *x = self._symbols()

        # Get list of face indices
        fc = sd.face_centers
        fn = sd.face_normals

        # Lambdify expression

        q_fun: list[Callable] = [
            sym.lambdify((t, *x), self.fluid_flux[i], "numpy") for i in range(sd.dim)
        ]

        # Face-centered Darcy fluxes
        q_fc: np.ndarray = np.asarray(
            sum([q_fun[i](time, *self._fc(sd)) * fn[i] for i in range(sd.dim)])
        )

        return q_fc

    # -----> Sources
    def mechanics_source(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Compute exact source term for the momentum balance equation.

        TODO: It should be possible to extend this to a general source term by use of a
        keyword argument and getattr.

        Parameters:
            sd: Subdomain grid. time: Time in seconds.

        Returns:
            Exact right hand side of the momentum balance equation with ``shape=( 2 *
            sd.num_cells, )``.

        Notes:
            The returned array is given in PorePy's flattened vector format.

        """
        # Symbolic variables
        t, *x = self._symbols()

        # Get cell centers and cell volumes
        vol = sd.cell_volumes

        # Lambdify expression
        source_mech_fun: list[Callable] = [
            sym.lambdify((t, *x), self.source_mech[i], "numpy") for i in range(self.nd)
        ]

        # Evaluate and integrate source
        source_mech: list[np.ndarray] = [
            source_mech_fun[i](time, *self._cc(sd)) * vol for i in range(self.nd)
        ]

        # Flatten array
        source_mech_flat: np.ndarray = np.asarray(source_mech).ravel("F")

        # Invert sign according to sign convention.
        return -source_mech_flat

    def rotation_source(self, sd: pp.Grid, time: float) -> np.ndarray:
        t, *x = self._symbols()
        vol = sd.cell_volumes

        # Lambdify expression
        source_rot_fun = sym.lambdify((t, *x), self.source_rotation, "numpy")

        # Evaluate and integrate source
        source_rot = (
            np.array(source_rot_fun(time, *self._cc(sd))).reshape((-1, 1)) * vol
        )

        return source_rot.ravel(order="F")

    def total_pressure_source(self, sd: pp.Grid, time: float) -> np.ndarray:
        t, *x = self._symbols()
        vol = sd.cell_volumes

        # Lambdify expression
        source_p_fun = sym.lambdify((t, *x), self.source_p, "numpy")

        # Evaluate and integrate source
        source_p = source_p_fun(time, *self._cc(sd)) * vol

        return source_p

    def fluid_pressure_source(self, sd: pp.Grid, time: float) -> np.ndarray:
        t, *x = self._symbols()
        vol = sd.cell_volumes

        # Lambdify expression
        source_p_fun = sym.lambdify((t, *x), self.source_flow, "numpy")

        # Evaluate and integrate source
        source_p = source_p_fun(time, *self._cc(sd)) * vol

        return source_p


class UnitCubeGrid(pp.ModelGeometry):
    """Class for setting up the geometry of the unit square domain.

    The domain may be assigned different material parameters in the region x > 0.5 and y
    > 0.5. To ensure the region with different material parameters is the same in all
    refinement levels, we want to have the lines x=0.5 and y=0.5 as grid lines. This is
    achieved by different means: For a Cartesian grid, we simply have to make sure the
    number of cells in the x and y direction is even (this is done by the default
    meshing parameters provided in self.meshing_arguments(), but will have to be taken
    care of by the user if the default parameters is overridden). For a simplex grid,
    the lines are defined as fractures in self.set_fractures(), and marked as
    constraints in self.meshing_kwargs().

    Furthermore, if the grid nodes are perturbed, the perturbation should not be applied
    to the nodes on the boundary of the domain, nor to the nodes at x=0.5 and y=0.5. The
    latter is needed to ensure the region with the different material parameters is the
    same in all realizations of the perturbed grid (It would have sufficied to keep keep
    the nodes at {x=0.5, y>0.5} and {x>0.5, y=0.5} fixed, but it is just as easy to keep
    all nodes at x=0.5 and y=0.5 fixed). This is achieved in self.set_geometry().

    """

    params: dict
    """Simulation model parameters."""

    def set_geometry(self) -> None:
        self.set_domain()
        self.set_fractures()
        # Create a fracture network.
        self.fracture_network = pp.create_fracture_network(self.fractures, self.domain)

        solution_type = self.params.get("analytical_solution")

        def _fracture_constraints_2d():
            if solution_type == "heterogeneous_lame":
                fracs = [
                    pp.LineFracture(np.array([[0.5, 0.5], [0.5, 1]])),
                    pp.LineFracture(np.array([[0.5, 0.5], [1, 0.5]])),
                ]
                constraints = np.array([0, 1])
            else:
                fracs = []
                constraints = np.array([], dtype=int)
            return fracs, constraints

        pert_rate = self.params.get("perturbation", 0.0)

        if self.grid_type() == "simplex":
            # self.params.get("prismatic_extrusion"):
            # Create a 2d simplex grid, extrude it

            # Fractures and constraints for the 2d grid
            fracs, constraints = _fracture_constraints_2d()

            network = pp.create_fracture_network(
                fractures=fracs,
                domain=nd_cube_domain(2, 1),
            )
            tmp_mdg = pp.create_mdg(
                "simplex",
                self.meshing_arguments(),
                network,
                constraints=constraints,
                **self.meshing_kwargs(),
            )
            # Use the same grid refinement level in the z-direction
            num_layers = int(np.round(1 / self.meshing_arguments()["cell_size"]))
            z_coord = np.linspace(0, 1, num_layers + 1)
            mdg, _ = pp.grid_extrusion.extrude_mdg(tmp_mdg, z_coord)

            mdg.compute_geometry()

            # Compute cell centers in 2d grids. These will be assigned to the 3d grid
            # later on
            sd = tmp_mdg.subdomains()[0]
            if False:
                exp = pp.Exporter(sd, "simplex_2d")
                exp.write_vtu()

            # Find the circumcenter of the triangles
            cc_2d = self._circumcenter_2d(sd)

            sd_3d = mdg.subdomains()[0]
            n_3d = (
                sd_3d.cell_nodes()
                .tocsc()
                .indices.reshape((6, sd_3d.num_cells), order="F")
            )

            nz_3d = sd_3d.nodes[2]
            z = np.mean(nz_3d[n_3d], axis=0)

            cc_3d = np.tile(cc_2d, (1, num_layers))
            cc_3d[2] = z

            sd_3d.cell_centers = cc_3d

        elif self.grid_type() == "cartesian":
            if False:  # pert_rate == 0:
                # Create a standard 3d Cartesian grid.
                mdg = pp.create_mdg(
                    self.grid_type(),
                    self.meshing_arguments(),
                    self.fracture_network,
                    **self.meshing_kwargs(),
                )
            else:
                # We will create a 2d Cartesian grid, perturb nodes, then extrude.

                # Fractures and constraints for the 2d grid
                fracs, constraints = [], []

                network = pp.create_fracture_network(
                    fractures=fracs,
                    domain=nd_cube_domain(2, 1),
                )
                tmp_mdg = pp.create_mdg(
                    self.grid_type(),
                    self.meshing_arguments(),
                    network,
                    constraints=constraints,
                    **self.meshing_kwargs(),
                )
                sd = tmp_mdg.subdomains()[0]
                x, y = sd.nodes[0], sd.nodes[1]
                h = sd.cell_diameters(cell_wise=False, func=np.min)
                if self.params.get("h2_perturbation", False):
                    pert_rate *= np.sqrt(h)
                match solution_type:
                    case "heterogeneous_lame":
                        pert_nodes = np.logical_not(
                            np.logical_or.reduce(
                                (
                                    # Exterior boundary
                                    np.isin(x, [0, 1]),
                                    np.isin(y, [0, 1]),
                                    # Interior boundary
                                    np.logical_and(x == 0.5, y >= 0.5),
                                    np.logical_and(y == 0.5, x >= 0.5),
                                )
                            )
                        )
                    case _:
                        # Default; No perturbations on the boundary
                        pert_nodes = np.logical_not(
                            np.logical_or.reduce(
                                (np.isin(x, [0, 1]), np.isin(y, [0, 1]))
                            )
                        )
                # Set the random seed
                np.random.seed(42)
                # Perturb the nodes
                x[pert_nodes] += (
                    pert_rate * h * (np.random.rand(pert_nodes.sum()) - 0.5)
                )
                y[pert_nodes] += (
                    pert_rate * h * (np.random.rand(pert_nodes.sum()) - 0.5)
                )
                if False:
                    sd.nodes[0] = x
                    sd.nodes[1] = y
                    exp = pp.Exporter(sd, "unperturbed_2d")
                    exp.write_vtu()
                # Use the same grid refinement level in the z-direction
                num_layers = int(np.round(1 / self.meshing_arguments()["cell_size"]))
                z_coord = np.linspace(0, 1, num_layers + 1)

                mdg, _ = pp.grid_extrusion.extrude_mdg(tmp_mdg, z=z_coord)
                mdg.compute_geometry()

        else:
            raise ValueError("Unknown grid type")

        mdg.set_boundary_grid_projections()
        self.mdg = mdg
        pp.set_local_coordinate_projections(self.mdg)

        self.nd: int = self.mdg.dim_max()

        self.set_well_network()

    def _circumcenter_2d(self, sd):
        cn = sd.cell_nodes().tocsc()
        ni = cn.indices.reshape((3, sd.num_cells), order="F")
        cc = sd.cell_centers.copy()
        x = sd.nodes[0]
        y = sd.nodes[1]

        x0 = x[ni[0]]
        y0 = y[ni[0]]
        x1 = x[ni[1]]
        y1 = y[ni[1]]
        x2 = x[ni[2]]
        y2 = y[ni[2]]

        D = 2 * (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))
        assert np.all(D != 0)
        xc = (
            (x0**2 + y0**2) * (y1 - y2)
            + (x1**2 + y1**2) * (y2 - y0)
            + (x2**2 + y2**2) * (y0 - y1)
        ) / D
        yc = (
            -(
                (x0**2 + y0**2) * (x1 - x2)
                + (x1**2 + y1**2) * (x2 - x0)
                + (x2**2 + y2**2) * (x0 - x1)
            )
            / D
        )

        d_01 = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        d_12 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        d_20 = np.sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2)

        dot_0 = (x1 - x0) * (x2 - x0) + (y1 - y0) * (y2 - y0)
        dot_1 = (x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)
        dot_2 = (x0 - x2) * (x1 - x2) + (y0 - y2) * (y1 - y2)

        angle_0 = np.arccos(dot_0 / (d_01 * d_20)) * 180 / np.pi
        angle_1 = np.arccos(dot_1 / (d_01 * d_12)) * 180 / np.pi
        angle_2 = np.arccos(dot_2 / (d_12 * d_20)) * 180 / np.pi

        assert np.allclose(angle_0 + angle_1 + angle_2, 180)

        all_sharp = np.logical_and.reduce([angle_0 < 85, angle_1 < 85, angle_2 < 85])

        cc[0, all_sharp] = xc[all_sharp]
        cc[1, all_sharp] = yc[all_sharp]

        return cc

    def set_domain(self) -> None:
        """Set domain."""
        self._domain = nd_cube_domain(3, 1.0)

    def meshing_arguments(self) -> dict[str, float]:
        """Set meshing arguments."""
        default_mesh_arguments = {"cell_size": 0.1}
        return self.params.get("meshing_arguments", default_mesh_arguments)


class SourceTerms:
    """Modified source terms to be added to the balance equations."""

    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Body force."""

        external_sources = pp.ad.TimeDependentDenseArray(
            name="source_mechanics",
            domains=self.mdg.subdomains(),
        )

        return external_sources

    def source_rotation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Rotation source."""

        external_sources = pp.ad.TimeDependentDenseArray(
            name="source_rotation",
            domains=self.mdg.subdomains(),
        )

        return external_sources

    def source_solid_mass(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Solid pressure source."""

        external_sources = pp.ad.TimeDependentDenseArray(
            name="source_total_pressure",
            domains=self.mdg.subdomains(),
        )

        return external_sources

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid pressure source."""

        external_sources = pp.ad.TimeDependentDenseArray(
            name="source_fluid_pressure",
            domains=self.mdg.subdomains(),
        )

        return external_sources


class BoundaryConditions(pp.momentum_balance.BoundaryConditionsMomentumBalance):
    def bc_values_displacement(
        self, boundary_grid: pp.BoundaryGrid | pp.Grid
    ) -> np.ndarray:
        """Displacement values for the Dirichlet boundary condition.

        Parameters:
            boundary_grid: Boundary grid to evaluate values on.

        Returns:
            An array with shape (boundary_grid.num_cells,) containing the displacement
            values on the provided boundary grid.

        """
        if isinstance(boundary_grid, pp.BoundaryGrid):
            if hasattr(self, "exact_sol"):
                return self.exact_sol.displacement(boundary_grid, 0)
            else:
                return np.zeros(boundary_grid.num_cells * self.nd)

        val = pp.ad.TimeDependentDenseArray(
            name="bc_values_displacement",
            domains=self.mdg.subdomains(),
        )
        return val

    def update_all_boundary_conditions(self) -> None:
        """Set values for the rotation and the volumetric strain on boundaries."""
        super().update_all_boundary_conditions()


class StiffnessTensor:
    def stiffness_tensor(self, sd: pp.Grid) -> pp.FourthOrderTensor:
        exact_sol = ExactSolution(self)
        if self._is_time_dependent:
            t, *x = exact_sol._symbols()
        else:
            x = exact_sol._symbols()

        def evaluate(funct):
            val = sym.lambdify(x, funct, "numpy")(
                *exact_sol._cc(self.mdg.subdomains()[0])
            )
            if isinstance(val, (float, int)):
                val = val * np.ones(self.mdg.subdomains()[0].num_cells)
            return val

        # Set stiffness matrix
        lame_lmbda = evaluate(exact_sol.lame_lmbda)
        lame_mu = evaluate(exact_sol.lame_mu)
        stiffness = pp.FourthOrderTensor(lmbda=lame_lmbda, mu=lame_mu)
        return stiffness


class MBSolutionStrategy(pp.momentum_balance.SolutionStrategyMomentumBalance):
    """Solution strategy for the verification setup."""

    def __init__(self, params: dict):
        """Constructor for the class."""
        super().__init__(params)

        self.results = []
        """Results object that stores exact and approximated solutions and errors."""

        self.flux_variable: str = "darcy_flux"
        """Keyword to access the Darcy fluxes."""

        self.stress_variable: str = "thermoporoelastic_force"
        """Keyword to access the poroelastic force."""
        # Field = namedtuple("Field", ["name", "is_scalar", "is_cc", "relative"])

        self.fields = [Field("displacement", False, True, True)]
        self.fields.append(Field("stress", False, False, True))
        self.fields.append(Field("displacement_stress", False, False, True))

        if isinstance(self, SetupTpsa):
            self.fields.append(Field("total_pressure", True, True, True))
            self.fields.append(Field("rotation", params["nd"] == 2, True, True))
            self.fields.append(Field("total_rotation", params["nd"] == 2, False, True))

    def solve_linear_system(self) -> np.ndarray:
        """Solve linear system.

        Default method is a direct solver. The linear solver is chosen in the
        initialize_linear_solver of this model. Implemented options are
            - scipy.sparse.spsolve with and without call to umfpack
            - pypardiso.spsolve

        See also:
            :meth:`initialize_linear_solver`

        Returns:
            np.ndarray: Solution vector.

        """
        A, b = self.linear_system
        t_0 = time.time()
        solver = self.linear_solver
        if solver == "direct":
            # This is the default option which is invoked unless explicitly overridden
            # by the user. We need to check if the pypardiso package is available.
            try:
                from pypardiso import spsolve as sparse_solver  # type: ignore
            except ImportError:
                # Fall back on the standard scipy sparse solver.
                sparse_solver = sps.linalg.spsolve
                warnings.warn(
                    """PyPardiso could not be imported,
                    falling back on scipy.sparse.linalg.spsolve"""
                )
            x = sparse_solver(A, b)
        elif solver == "iterative":
            import pyamg
            from scipy.sparse.linalg import LinearOperator

            eq_sys = self.equation_system

            sd = self.mdg.subdomains()

            u_dof = eq_sys.dofs_of([self.displacement(sd)])
            rot_dof = eq_sys.dofs_of([self.rotation(sd)])
            p_solid_dof = eq_sys.dofs_of([self.total_pressure(sd)])

            displacemnt_rows = eq_sys.assembled_equation_indices[
                "momentum_balance_equation"
            ]
            rotation_rows = eq_sys.assembled_equation_indices[
                "angular_momentum_balance_equation"
            ]
            solid_mass_rows = eq_sys.assembled_equation_indices["solid_mass_equation"]

            A_00 = A[displacemnt_rows][:, u_dof]
            A_11 = A[rotation_rows][:, rot_dof]
            A_22 = A[solid_mass_rows][:, p_solid_dof]
            A_10 = A[rotation_rows][:, u_dof]
            A_20 = A[solid_mass_rows][:, u_dof]

            # Define the nullspace for the 3d linear elasticity problem
            nullspace = np.zeros((A_00.shape[0], 3))
            # Translation dofs
            nullspace[::3, 0] = 1
            nullspace[1::3, 1] = 1
            nullspace[2::3, 2] = 1
            # Rotation dofs

            amg_elasticity = pyamg.smoothed_aggregation_solver(A_00, B=nullspace)
            amg_rotation = pyamg.smoothed_aggregation_solver(A_11)
            amg_total_pressure = pyamg.smoothed_aggregation_solver(A_22)

            if True:
                data = self.mdg.subdomain_data(sd[0])
                C = data[pp.PARAMETERS][self.stress_keyword]["fourth_order_tensor"]
                mu = C.mu

                mu_rot = np.repeat(-sd[0].cell_volumes / mu, self._rotation_dimension())
                rotation_solver = sps.dia_matrix((1 / mu_rot, 0), A_11.shape)

                mu = -sd[0].cell_volumes * (1 / C.mu + 1 / C.lmbda)
                total_pressure_solver = sps.dia_matrix((1 / mu, 0), A_22.shape)
                # import scipy.sparse.linalg as spla
                # tps = spla.factorized(A_22)

            def block_preconditioner(r):
                r_0 = r[u_dof]
                r_1 = r[rot_dof]
                r_2 = r[p_solid_dof]
                x = np.zeros_like(r)
                x_0 = amg_elasticity.aspreconditioner().matvec(r_0)
                # x_1 = amg_rotation.aspreconditioner().matvec(r_1 - A_10 @ x_0)
                x_1 = rotation_solver @ (r_1 - A_10 @ x_0)
                x_2 = amg_total_pressure.aspreconditioner().matvec(r_2 - A_20 @ x_0)
                # x_2 = total_pressure_solver @ (r_2 -A_20 @ x_0)
                # x_2 = tps(r_2 -A_20 @ x_0)

                x[u_dof] = x_0
                x[rot_dof] = x_1
                x[p_solid_dof] = x_2

                return x

            precond = LinearOperator(A.shape, matvec=block_preconditioner)

            debug = []

            def print_resid(x):
                # pass
                print(np.linalg.norm(b - A @ x))

            t = time.time()
            x = np.zeros_like(b)
            for _ in range(100):
                x, info = pyamg.krylov.fgmres(
                    A, b, tol=1e-12, M=precond, callback=print_resid, x0=x, maxiter=100
                )
                if info == 0:
                    break
            print("time " + str(time.time() - t))

            if info != 0:
                raise ValueError("GMRES failed")
        else:
            raise ValueError(f"Unknown linear solver: {solver}")

        print(f"Solved linear system in {time.time() - t_0:.2e} seconds.")
        return x

    def _initialize_linear_solver(self) -> None:
        """Initialize linear solver.

        The default linear solver is Pardiso; this can be overridden by user choices.
        If Pardiso is not available, backup solvers will automatically be invoked in
        :meth:`solve_linear_system`.

        To use a custom solver in a model, override this method (and possibly
        :meth:`solve_linear_system`).

        Raises:
            ValueError if the chosen solver is not among the three currently supported,
            see linear_solve.

        """
        self.linear_solver = "direct"

    def tmp_initialize_data_saving(self) -> None:
        # Something is wrong with numba compilation in the exporter. For now, we drop
        # this step.
        pass

    def tmp_save_data_time_step(self) -> None:
        """No saving of data"""
        # pass

    def before_nonlinear_loop(self) -> None:
        """Update values of external sources."""
        super().before_nonlinear_loop()

        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time

        # Mechanics source
        mech_source = self.exact_sol.mechanics_source(sd=sd, time=t)
        pp.set_solution_values(
            name="source_mechanics", values=mech_source, data=data, iterate_index=0
        )
        rotation_source = self.exact_sol.rotation_source(sd=sd, time=t)
        pp.set_solution_values(
            name="source_rotation", values=rotation_source, data=data, iterate_index=0
        )
        total_pressure_source = self.exact_sol.total_pressure_source(sd=sd, time=t)
        pp.set_solution_values(
            name="source_total_pressure",
            values=total_pressure_source,
            data=data,
            iterate_index=0,
        )

        boundary_faces = self.domain_boundary_sides(sd).all_bf
        u = self.exact_sol.displacement(sd, 0, cc=False)
        if u.size != sd.num_faces * sd.dim:
            u = np.ravel(u.reshape((-1, 1)) * np.ones(sd.num_faces), order="F")

        tmp = np.zeros((sd.dim, sd.num_faces))
        tmp[:, boundary_faces] = u.reshape((sd.dim, -1), order="f")[:, boundary_faces]
        pp.set_solution_values(
            name="bc_values_displacement",
            values=tmp.ravel("F"),
            data=data,
            iterate_index=0,
        )

    def set_discretization_parameters(self) -> None:
        """Set parameters for the subproblems and the combined problem.

        The parent class' definitions of permeability, stiffness parameters, and the Biot
        and thermal stress tensors are owerwritten.
        """
        super().set_discretization_parameters()

        # Instantiate exact solution object after materials have been set
        self.exact_sol = ExactSolution(self)
        if self._is_time_dependent:
            t, *x = self.exact_sol._symbols()
        else:
            x = self.exact_sol._symbols()

        def evaluate(funct):
            val = sym.lambdify(x, funct, "numpy")(
                *self.exact_sol._cc(self.mdg.subdomains()[0])
            )
            if isinstance(val, (float, int)):
                val = val * np.ones(self.mdg.subdomains()[0].num_cells)
            return val

        for sd, data in self.mdg.subdomains(dim=self.nd, return_data=True):
            cc = self.exact_sol._cc(self.mdg.subdomains()[0])

        # Instantiate exact solution object after materials have been set
        self.exact_sol = ExactSolution(self)

        data[pp.PARAMETERS][self.stress_keyword]["fourth_order_tensor"] = (
            self.stiffness_tensor(sd)
        )


class SolutionStrategyPoromech(pp.poromechanics.SolutionStrategyPoromechanics):
    """Solution strategy for the verification setup."""

    def __init__(self, params: dict):
        """Constructor for the class."""
        super().__init__(params)

        self.results = []
        """Results object that stores exact and approximated solutions and errors."""

        self.flux_variable: str = "darcy_flux"
        """Keyword to access the Darcy fluxes."""

        self.stress_variable: str = "thermoporoelastic_force"
        """Keyword to access the poroelastic force."""
        # Field = namedtuple("Field", ["name", "is_scalar", "is_cc", "relative"])

        self.rotation_variable: str = "rotation"
        """Name of the rotation variable."""

        self.total_pressure_variable: str = "total_pressure"
        """Name of the volumetric strain variable."""
        self.fields = [
            Field("displacement", False, True, True),
            Field("rotation", params["nd"] == 2, True, True),
            Field("total_pressure", True, True, True),
            Field("pressure", True, True, True),
        ]

        self.fields.append(Field("stress", False, False, True))
        self.fields.append(Field("total_rotation", params["nd"] == 2, False, True))
        self.fields.append(Field("darcy_flux", True, False, True))
        self.fields.append(Field("displacement_stress", False, False, True))

    def initial_condition(self):
        super().initial_condition()

        # Initial guess for rotation and volumetric strain
        num_cells = sum(sd.num_cells for sd in self.mdg.subdomains(dim=self.nd))

        rotation_dims = 1 if self.nd == 2 else 3

        rotation_vals = np.zeros((rotation_dims, num_cells)).ravel("F")
        self.equation_system.set_variable_values(
            rotation_vals,
            [self.rotation_variable],
            time_step_index=0,
            iterate_index=0,
        )

        total_pressure_vals = np.zeros(num_cells)

        self.equation_system.set_variable_values(
            total_pressure_vals,
            [self.total_pressure_variable],
            time_step_index=0,
            iterate_index=0,
        )

    def before_nonlinear_loop(self) -> None:
        """Update values of external sources."""
        super().before_nonlinear_loop()

        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time

        # Mechanics source
        mech_source = self.exact_sol.mechanics_source(sd=sd, time=t)
        pp.set_solution_values(
            name="source_mechanics", values=mech_source, data=data, iterate_index=0
        )
        rotation_source = self.exact_sol.rotation_source(sd=sd, time=t)
        pp.set_solution_values(
            name="source_rotation", values=rotation_source, data=data, iterate_index=0
        )
        total_pressure_source = self.exact_sol.total_pressure_source(sd=sd, time=t)
        pp.set_solution_values(
            name="source_total_pressure",
            values=total_pressure_source,
            data=data,
            iterate_index=0,
        )

        fluid_pressure_source = self.exact_sol.fluid_pressure_source(sd=sd, time=t)
        pp.set_solution_values(
            name="source_fluid_pressure",
            values=fluid_pressure_source,
            data=data,
            iterate_index=0,
        )

        boundary_faces = self.domain_boundary_sides(sd).all_bf
        u = self.exact_sol.displacement(sd, 0, cc=False)

        tmp = np.zeros((sd.dim, sd.num_faces))
        tmp[:, boundary_faces] = u.reshape((sd.dim, -1), order="f")[:, boundary_faces]
        pp.set_solution_values(
            name="bc_values_displacement",
            values=tmp.ravel("F"),
            data=data,
            iterate_index=0,
        )

    def tmp_initialize_data_saving(self) -> None:
        # Something is wrong with numba compilation in the exporter. For now, we drop
        # this step.
        pass

    def tmp_save_data_time_step(self) -> None:
        """No saving of data"""
        # pass

    def solve_linear_system(self) -> np.ndarray:
        """Solve linear system.

        Default method is a direct solver. The linear solver is chosen in the
        initialize_linear_solver of this model. Implemented options are
            - scipy.sparse.spsolve with and without call to umfpack
            - pypardiso.spsolve

        See also:
            :meth:`initialize_linear_solver`

        Returns:
            np.ndarray: Solution vector.

        """
        A, b = self.linear_system
        t_0 = time.time()
        solver = self.linear_solver
        if solver == "direct":
            # This is the default option which is invoked unless explicitly overridden
            # by the user. We need to check if the pypardiso package is available.
            try:
                from pypardiso import spsolve as sparse_solver  # type: ignore
            except ImportError:
                # Fall back on the standard scipy sparse solver.
                sparse_solver = sps.linalg.spsolve
                warnings.warn(
                    """PyPardiso could not be imported,
                    falling back on scipy.sparse.linalg.spsolve"""
                )
            x = sparse_solver(A, b)
        elif solver == "iterative":
            import pyamg
            from scipy.sparse.linalg import LinearOperator

            eq_sys = self.equation_system

            sd = self.mdg.subdomains()

            u_dof = eq_sys.dofs_of([self.displacement(sd)])
            rot_dof = eq_sys.dofs_of([self.rotation(sd)])
            p_solid_dof = eq_sys.dofs_of([self.total_pressure(sd)])
            p_fluid_dof = eq_sys.dofs_of([self.pressure(sd)])

            displacemnt_rows = eq_sys.assembled_equation_indices[
                "momentum_balance_equation"
            ]
            rotation_rows = eq_sys.assembled_equation_indices[
                "angular_momentum_balance_equation"
            ]
            solid_mass_rows = eq_sys.assembled_equation_indices[
                "Solid_mass_equation_poromechanics"
            ]
            fluid_mass_rows = eq_sys.assembled_equation_indices["mass_balance_equation"]

            A_00 = A[displacemnt_rows][:, u_dof]
            A_11 = A[rotation_rows][:, rot_dof]
            A_22 = A[solid_mass_rows][:, p_solid_dof]
            A_33 = A[fluid_mass_rows][:, p_fluid_dof]

            # Define the nullspace for the 3d linear elasticity problem
            nullspace = np.zeros((A_00.shape[0], 6))
            # Translation dofs
            nullspace[::3, 0] = 1
            nullspace[1::3, 1] = 1
            nullspace[2::3, 2] = 1
            # Rotation dofs
            nullspace[::3, 3] = 1
            nullspace[1::3, 3] = -1
            nullspace[1::3, 4] = 1
            nullspace[2::3, 4] = -1
            nullspace[2::3, 5] = 1
            nullspace[::3, 5] = -1

            nullspace_fluid = np.ones(A_33.shape[0]).reshape((-1, 1))

            amg_elasticity = pyamg.smoothed_aggregation_solver(A_00, B=nullspace)
            amg_rotation = pyamg.smoothed_aggregation_solver(A_11)
            amg_total_pressure = pyamg.smoothed_aggregation_solver(A_22)
            amg_fluid_pressure = pyamg.smoothed_aggregation_solver(
                A_33, B=nullspace_fluid
            )

            if True:
                data = self.mdg.subdomain_data(sd[0])
                C = data[pp.PARAMETERS][self.stress_keyword]["fourth_order_tensor"]
                mu = C.mu

                mu_rot = np.repeat(-sd[0].cell_volumes / mu, self._rotation_dimension())
                rotation_solver = sps.dia_matrix((1 / mu_rot, 0), A_11.shape)

                mu = -sd[0].cell_volumes * (1 / C.mu + 1 / C.lmbda)
                total_pressure_solver = sps.dia_matrix((1 / mu, 0), A_22.shape)
                import scipy.sparse.linalg as spla

            def block_preconditioner(r):
                r_0 = r[u_dof]
                r_1 = r[rot_dof]
                r_2 = r[p_solid_dof]
                r_3 = r[p_fluid_dof]
                x = np.zeros_like(r)

                x_0 = amg_elasticity.aspreconditioner().matvec(r_0)
                x_1 = amg_rotation.aspreconditioner().matvec(r_1)
                x_2 = amg_total_pressure.aspreconditioner().matvec(r_2)
                x_3 = amg_fluid_pressure.aspreconditioner().matvec(r_3)

                x[u_dof] = x_0
                x[rot_dof] = x_1
                x[p_solid_dof] = x_2
                x[p_fluid_dof] = x_3

                return x

            precond = LinearOperator(A.shape, matvec=block_preconditioner)

            debug = []

            def print_resid(x):
                print(np.linalg.norm(b - A @ x))
                # pass

            x = np.zeros_like(b)
            for _ in range(100):
                x, info = pyamg.krylov.fgmres(
                    A, b, tol=1e-10, M=precond, callback=print_resid, x0=x, maxiter=40
                )
                if info == 0:
                    break

            if info != 0:
                raise ValueError("GMRES failed")
        else:
            raise ValueError(f"Unknown linear solver: {solver}")

        print(f"Solved linear system in {time.time() - t_0:.2e} seconds.")
        return x

    def _initialize_linear_solver(self) -> None:
        """Initialize linear solver.

        The default linear solver is Pardiso; this can be overridden by user choices.
        If Pardiso is not available, backup solvers will automatically be invoked in
        :meth:`solve_linear_system`.

        To use a custom solver in a model, override this method (and possibly
        :meth:`solve_linear_system`).

        Raises:
            ValueError if the chosen solver is not among the three currently supported,
            see linear_solve.

        """
        self.linear_solver = "direct"

    def set_discretization_parameters(self) -> None:
        """Set parameters for the subproblems and the combined problem.

        The parent class' definitions of permeability, stiffness parameters, and the Biot
        and thermal stress tensors are owerwritten.
        """
        super().set_discretization_parameters()

        # Instantiate exact solution object after materials have been set
        self.exact_sol = ExactSolution(self)
        if self._is_time_dependent:
            t, *x = self.exact_sol._symbols()
        else:
            x = self.exact_sol._symbols()

        def evaluate(funct):
            val = sym.lambdify(x, funct, "numpy")(
                *self.exact_sol._cc(self.mdg.subdomains()[0])
            )
            if isinstance(val, (float, int)):
                val = val * np.ones(self.mdg.subdomains()[0].num_cells)
            return val

        for sd, data in self.mdg.subdomains(dim=self.nd, return_data=True):
            cc = self.exact_sol._cc(self.mdg.subdomains()[0])

            # Set stiffness matrix
            lame_lmbda = evaluate(self.exact_sol.lame_lmbda)
            lame_mu = evaluate(self.exact_sol.lame_mu)
            stiffness = pp.FourthOrderTensor(lmbda=lame_lmbda, mu=lame_mu)
            data[pp.PARAMETERS][self.stress_keyword]["fourth_order_tensor"] = stiffness
            data[pp.PARAMETERS][self.darcy_keyword]["mpfa_inverter"] = "python"

    def _is_time_dependent(self):
        return False

    def _is_nonlinear_problem(self) -> bool:
        """The problem is linear."""
        return False

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary conditions on all external boundaries.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object. Per default Dirichlet-type BC are assigned,
            requiring pressure values on the bonudary.

        """
        # Define boundary faces.
        top = self.domain_boundary_sides(sd).top
        # Define boundary condition on all boundary faces.
        return pp.BoundaryCondition(sd, top, "dir")


class DisplacementStress:
    def displacement_stress(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Linear elastic mechanical stress [Pa].

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Operator for the stress.

        """
        # TODO: This is common to the standard (one-field) mechanical stress. See if we
        # can find a way to unify.
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            return self.create_boundary_operator(
                name=self.stress_keyword,
                domains=domains,  # type: ignore[call-arg]
            )

        # Check that the subdomains are grids.
        if not all([isinstance(g, pp.Grid) for g in domains]):
            raise ValueError(
                """Argument subdomains a mixture of grids and boundary grids."""
            )
        # By now we know that subdomains is a list of grids, so we can cast it as such
        # (in the typing sense).

        for sd in domains:
            # The mechanical stress is only defined on subdomains of co-dimension 0.
            if sd.dim != self.nd:
                raise ValueError("Subdomain must be of co-dimension 0.")

        # No need to facilitate changing of stress discretization, only one is
        # available at the moment.
        discr = self.stress_discretization(domains)
        # Fractures in the domain
        interfaces = self.subdomains_to_interfaces(domains, [1])

        # Boundary conditions on external boundaries
        boundary_operator = self.combine_boundary_operators_mechanical_stress(domains)
        proj = pp.ad.MortarProjections(self.mdg, domains, interfaces, dim=self.nd)
        stress = (
            discr.stress_displacement() @ self.displacement(domains)
            + discr.bound_stress() @ boundary_operator
            + discr.bound_stress()
            @ proj.mortar_to_primary_avg
            @ self.interface_displacement(interfaces)
        )
        return stress


class EquationsPoromechanics:
    """Combines mass and momentum balance equations."""

    def fluid_mass(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """The full measure of cell-wise fluid mass.

        The product of fluid density and porosity is assumed constant cell-wise, and
        integrated over the cell volume.

        Note:
            This implementation assumes constant porosity and must be overridden for
            variable porosity. This has to do with wrapping of scalars as vectors or
            matrices and will hopefully be improved in the future. Extension to variable
            density is straightforward.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the cell-wise fluid mass.

        """
        biot = self.biot_coefficient(subdomains)
        inv_lambda = self.inv_lambda(subdomains)
        eff_compressibility = (
            pp.ad.Scalar(self.fluid.reference_component.compressibility)
            + inv_lambda * biot**2
        )

        fluid_contribution = eff_compressibility * self.pressure(subdomains)

        solid_contribution = biot * inv_lambda * self.total_pressure(subdomains)

        mass = self.volume_integral(
            fluid_contribution + solid_contribution, subdomains, dim=1
        )
        mass.set_name("fluid_mass")
        return mass


class ConstitutiveLawsPoromechanicsRunscript:
    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpfaAd:
        """Discretization object for the Darcy flux term.

        Parameters:
            subdomains: List of subdomains where the Darcy flux is defined.

        Returns:
            Discretization of the Darcy flux.

        """
        return pp.ad.TpfaAd(self.darcy_keyword, subdomains)

    def fluid_density(self, subdomains):
        if len(subdomains) == 0:
            nc = 0
        else:
            nc = subdomains[0].num_cells
        val = 1
        return pp.wrap_as_dense_ad_array(val, nc)


class SetupTpsa(  # type: ignore[misc]
    UnitCubeGrid,
    SourceTerms,
    MBSolutionStrategy,
    DataSaving,
    DisplacementStress,
    StiffnessTensor,
    # EquationsMechanicsRealStokes,
    # pp.momentum_balance.ConstitutiveLawsThreeFieldMomentumBalance,
    # pp.momentum_balance.VariablesThreeFieldMomentumBalance,
    # pp.momentum_balance.SolutionStrategyMomentumBalanceThreeField,
    # pp.momentum_balance.ThreeFieldMomentumBalanceEquations,
    pp.momentum_balance.TpsaMomentumBalanceMixin,
    BoundaryConditions,
    pp.momentum_balance.MomentumBalance,
):
    pass


class SetupTpsaPoromechanics(  # type: ignore[misc]
    UnitCubeGrid,
    SourceTerms,
    BoundaryConditions,
    StiffnessTensor,
    SolutionStrategyPoromech,
    ConstitutiveLawsPoromechanicsRunscript,
    EquationsPoromechanics,
    pp.poromechanics.TpsaPoromechanicsMixin,
    DataSaving,
    DisplacementStress,
    # EquationsPoromechanics,
    # VariablesThreeFieldPoromechanics,
    # pp.momentum_balance.ConstitutiveLawsMomentumBalance,
    # pp.momentum_balance.VariablesThreeFieldMomentumBalance,
    # pp.momentum_balance.SolutionStrategyMomentumBalanceThreeField,
    # pp.poromechanics.BoundaryConditionsPoromechanics,
    pp.poromechanics.Poromechanics,
):
    pass


# For each of the problems considered, the boolean run_{problem} determines if the
# convergence analysis should be run, and plot_{problem} determines if the results
# should be plotted (assumes that the convergence analysis has been run).
run_elasticity = False
plot_elasticity = False

run_poromechanics = True
plot_poromechanics = True

# Number of grid levels for the convergence analysis.
refinement_levels = 3


def run_convergence_analysis(
    grid_type: str,
    refinement_levels: int,
    cosserat_parameters: list[float],
    lame_lambdas: list[float],
    analytical_solution: str,
    heterogeneity: list[float],
    use_cosserat: bool = False,
    perturbation: float = 0.0,
    h2_perturbation: bool = False,
    nd: int = 2,
    use_circumcenter=True,
    prismatic_extrusion=False,
):
    # Function to run the convergence analysis for elasticity (both homogeneous and
    # with heterogeneity).
    all_results = []

    print(" ")
    print(f" {grid_type}")

    for het in heterogeneity:
        cos_results = []
        for lambda_param in lame_lambdas:
            solid = pp.SolidConstants(lame_lambda=lambda_param, biot_coefficient=0)

            conv_analysis = ConvergenceAnalysis(
                model_class=SetupTpsa,
                model_params={
                    "grid_type": grid_type,
                    "meshing_arguments": {"cell_size": 0.25},
                    "perturbation": perturbation,
                    "h2_perturbation": h2_perturbation,
                    "heterogeneity": het,
                    "cosserat_parameter": 0,
                    "material_constants": {"solid": solid},
                    "nd": 3,
                    "analytical_solution": analytical_solution,
                    "use_circumcenter": use_circumcenter,
                    "prismatic_extrusion": prismatic_extrusion,
                },
                levels=refinement_levels,
                spatial_refinement_rate=2,
                temporal_refinement_rate=1,
            )

            res = conv_analysis.run_analysis()
            cos_results.append(res)
        all_results.append(cos_results)

    return all_results


def run_poromech_convergence_analysis(
    grid_type: str,
    refinement_levels: int,
    cosserat_parameters: list[float],
    lame_lambdas: list[float],
    permeabilities: list[float],
    perturbation: float = 0.0,
    h2_perturbation: bool = False,
    nd: int = 2,
    use_circumcenter=True,
    prismatic_extrusion=False,
    analytical_solution: str = "poromechanics",
):
    # Function to run the convergence analysis for poromechanics

    all_results = []
    # We assume that lambda is fixed to a single value
    assert len(lame_lambdas) == 1
    lambda_param = lame_lambdas[0]

    for cos_param in cosserat_parameters:
        cos_results = []

        for perm in permeabilities:
            solid = pp.SolidConstants(
                lame_lambda=lambda_param, permeability=perm, biot_coefficient=1
            )
            fluid = pp.FluidComponent(compressibility=0.0)
            reference_values = pp.ReferenceVariableValues(pressure=0)
            conv_analysis = ConvergenceAnalysis(
                model_class=SetupTpsaPoromechanics,
                model_params={
                    "grid_type": grid_type,
                    "meshing_arguments": {"cell_size": 0.25},
                    "perturbation": perturbation,
                    "h2_perturbation": h2_perturbation,
                    "heterogeneity": 1.0,
                    "cosserat_parameter": cos_param,
                    "material_constants": {
                        "solid": solid,
                        "fluid": fluid,
                        "reference_variable_values": reference_values,
                    },
                    "nd": 3,
                    "analytical_solution": analytical_solution,
                    "use_circumcenter": use_circumcenter,
                    "prismatic_extrusion": prismatic_extrusion,
                },
                levels=refinement_levels,
                spatial_refinement_rate=2,
                temporal_refinement_rate=1,
            )

            res = conv_analysis.run_analysis()
            cos_results.append(res)
        all_results.append(cos_results)

    return all_results


def _add_convergence_lines(ax, fontsize, fontsize_ticks):
    # Helper function to add convergence lines to the plot
    x_vals = ax.get_xlim()
    dx = x_vals[1] - x_vals[0]
    y_vals = ax.get_ylim()
    dy = y_vals[1] - y_vals[0]
    diff = min(dx, dy)

    x_0 = x_vals[0] + 0.12 * dx
    y_0 = y_vals[0] + 0.02 * dy
    x_1 = x_0 + 0.1 * diff
    y_1 = y_0 + 0.1 * diff
    y_2 = y_0 + 0.2 * diff

    ax.plot([x_0, x_1], [y_1, y_0], color="black")
    ax.plot([x_0, x_1], [y_2 + 0.1 * diff, y_0 + 0.1 * diff], color="black")
    ax.text(x_0 - 0.05 * dx, y_1 - 0.07, "1", fontsize=fontsize_ticks)
    ax.text(x_0 - 0.05 * dx, y_2 + 0.10 * diff, "2", fontsize=fontsize_ticks)


def plot_and_save(ax, legend_handles, file_name, y_label):
    # Main function for plotting convergence results

    ax.set_xlabel(r"log$_2$($\delta^{-1}$)", fontsize=fontsize_label)
    ax.set_ylabel(y_label, fontsize=fontsize_label)

    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)

    x_vals = ax.get_xlim()
    y_vals = ax.get_ylim()
    y_min = np.floor(y_vals[0])
    y_max = np.ceil(y_vals[1])

    # By inspection, we know that the x values are between 1 and 4.5
    x_min = 1
    x_max = 4.5
    # Set the x and y limits
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

    # Set integer ticks
    xt = matplotlib.ticker.MaxNLocator(integer=True)
    yt = matplotlib.ticker.MaxNLocator(integer=True)
    ax.xaxis.set_major_locator(xt)
    ax.yaxis.set_major_locator(yt)

    # Set 4 (num_minor_bins) minor ticks between each major tick
    xtick_diff = np.diff(ax.get_xticks())[0]
    ytick_diff = np.diff(ax.get_yticks())[0]
    num_minor_bins = 4

    dx_bins = num_minor_bins / xtick_diff
    dy_bins = num_minor_bins / ytick_diff

    x_vals = ax.get_xlim()
    y_vals = ax.get_ylim()

    x_min = np.floor(x_vals[0] * dx_bins) / dx_bins
    x_max = np.ceil(x_vals[1] * dx_bins) / dx_bins
    ax.set_xlim([x_min, x_max])
    y_min = np.floor(y_vals[0] * dy_bins) / dy_bins
    y_max = np.ceil(y_vals[1] * dy_bins) / dy_bins
    ax.set_ylim([y_min, y_max])

    xt_minor = matplotlib.ticker.AutoMinorLocator(num_minor_bins)
    yt_minor = matplotlib.ticker.AutoMinorLocator(num_minor_bins)

    ax.xaxis.set_minor_locator(xt_minor)
    ax.yaxis.set_minor_locator(yt_minor)

    _add_convergence_lines(ax, fontsize_label, fontsize_ticks)

    ax.grid(which="major", linewidth=1.5)
    ax.grid(which="minor", linewidth=0.75)
    if len(legend_handles) > 0:
        ax.legend(handles=legend_handles, fontsize=fontsize_legend, loc="upper right")

    border = 0.02
    ax.plot(
        [x_min, x_max],
        [y_min + border, y_min + border],
        linestyle="-",
        color="black",
        linewidth=1.75,
    )
    ax.plot(
        [x_min, x_max],
        [y_max - border, y_max - border],
        linestyle="-",
        color="black",
        linewidth=1.75,
    )
    ax.plot(
        [x_min + border / 2, x_min + border / 2],
        [y_min, y_max],
        linestyle="-",
        color="black",
        linewidth=1.75,
    )
    ax.plot(
        [x_max - border, x_max - border],
        [y_min, y_max],
        linestyle="-",
        color="black",
        linewidth=1.75,
    )
    # plt.draw()
    # plt.show()
    # fig.patch.set_edgecolor("black")
    plt.savefig(f"{file_name}.png", bbox_inches="tight", pad_inches=0)


# Control of plotting
fontsize_label = 20
fontsize_ticks = 18
fontsize_legend = 16


elasticity_filename_stem = "elasticity_3d"

# To run the convergence analysis only for a subset of the grids considered, modify the
# following lists.

# Type of the base grid.
grid_types = ["cartesian", "cartesian", "cartesian", "simplex"]

# Perturbation of the grid nodes.
perturbations = [0.0, 0.3, 0.3, 0]
# Whether the perturbation is scaled by h^1.5 (True) or h (False). The name
# 'h2_perturbations' reflects an earlier version of the code (where the perturbation
# were scaled by h^2, thus really small), but has not been updated.
h2_perturbations = [False, False, True, False]
# Whether the circumcenter of the cell should be used as the cell center. This is a good
# choice for simplex grids.
circumcenter = [False, False, False, True]
# Whether the grid should be extruded in the z-direction. With the latest version of the
# setup, this parameter is superfluous, as the grid is always extruded, but it is kept
# for implementational convenience.
extrusion = [False, False, False, True]

if run_elasticity:
    print("Running elasticity convergence")
    for i in range(len(grid_types)):
        params = {
            "grid_type": grid_types[i],
            "refinement_levels": refinement_levels,
            "cosserat_parameters": [0],
            "lame_lambdas": [1e10],
            "perturbation": perturbations[i],
            "h2_perturbation": h2_perturbations[i],
            "nd": 3,
            "analytical_solution": "homogeneous",
            "use_circumcenter": circumcenter[i],
            "prismatic_extrusion": extrusion[i],
            "heterogeneity": [1],
        }
        elasticity_results = run_convergence_analysis(**params)
        filename = (
            f"{elasticity_filename_stem}_{grid_types[i]}_pert_{perturbations[i]}_h2_{h2_perturbations[i]}_circumcenter_{circumcenter[i]}_extrusion_{extrusion[i]}".replace(
                ".", "-"
            )
            + ".pkl"
        )

        for m in range(len(elasticity_results)):
            for j in range(len(elasticity_results[m])):
                for k in range(len(elasticity_results[m][j])):
                    elasticity_results[m][j][k] = dataclasses.asdict(
                        elasticity_results[m][j][k]
                    )

        with open(filename, "wb") as f:
            pickle.dump([elasticity_results, params], f)

if plot_elasticity:
    print("Plotting elasticity convergence")
    for grid_ind in range(len(grid_types)):
        full_stem = f"{elasticity_filename_stem}_{grid_types[grid_ind]}_pert_{perturbations[grid_ind]}_h2_{h2_perturbations[grid_ind]}_circumcenter_{circumcenter[grid_ind]}_extrusion_{extrusion[grid_ind]}".replace(
            ".", "-"
        )
        filename = f"{full_stem}.pkl"
        with open(filename, "rb") as f:
            res, params = pickle.load(f)
        i = 0  # There is only one cosserat parameter

        colors = ["orange", "blue", "green", "red"]
        markers = ["o", "s", "D", "v"]
        fig, ax = plt.subplots()
        legend_handles = []
        for j in range(len(res[i])):  # Loop over lambda
            print(f"lambda: {params['lame_lambdas'][j]}")
            all_errors = []
            cell_volumes = []

            error_all_levels = []

            # For scaling, use the analytical solution on the finest grid.
            ref_val = 0
            for key, val in res[i][j][-1].items():
                if key in [
                    "displacement",
                    "total_pressure",
                    "displacement_stress",
                    "rotation",
                ]:
                    ref_val += val[1]

            for k in range(len(res[i][j])):
                error = 0
                error_str = ""
                key_list = []
                error_this_level = []
                for key, val in res[i][j][k].items():
                    if key in [
                        "displacement",
                        "displacement_stress",
                        "total_pressure",
                        "stress",
                        "rotation",
                    ]:
                        if key != "stress":
                            error += val[0]
                            error_this_level.append(val[0])
                        error_str += f"{val[0] / val[1] ** 0:.5f}, "
                        key_list.append(key)
                # print(error)
                if k == 0:
                    print(key_list)
                print(error_str)
                all_errors.append(error / ref_val)
                cell_volumes.append(res[i][j][k]["cell_diameter"])
                error_all_levels.append(error_this_level)

            if params["lame_lambdas"][j] == 1:
                l_val = "1"
            else:
                if params["lame_lambdas"][j] > 1e18:
                    l_val = f"$\infty$"
                elif params["lame_lambdas"][j] > 1e9:
                    l_val = r"$10^{10}$"
                else:
                    exp = f"{int(np.log10(params['lame_lambdas'][j]))}"
                    l_val = f"$10^{exp}$"

            tmp = ax.plot(
                -np.log2(cell_volumes),
                np.log2(all_errors),
                marker=markers[j],
                color=colors[j],
                label=f"$\lambda$: {l_val}",
            )
            legend_handles.append(tmp[0])

            arr = np.clip(np.array(error_all_levels), a_min=1e-10, a_max=None)
            print("Log error")
            for row in range(arr.shape[0] - 1):
                print(f"{np.log2(arr[row] / arr[row + 1])}")
            print(" ")
            arr = np.asarray(all_errors)
            print(np.log2(arr[:-1] / arr[1:]))
            print("")

        # plot_and_save(ax, legend_handles, full_stem, "log$_2$($e_{*}$)")
        # print(" *************** ")

        # # Also plot primary variables
        # fig, ax = plt.subplots()
        # legend_handles = []
        # for j in range(len(res[i])):  # Loop over lambda
        #     displacement_error = []

        #     for k in range(len(res[i][j])):
        #         # Displacement error, scaled by reference value
        #         displacement_error.append(
        #             res[i][j][k]["displacement"][0] / res[i][j][-1]["displacement"][1]
        #         )

        #     if params["lame_lambdas"][j] == 1:
        #         l_val = "1"
        #     else:
        #         if params["lame_lambdas"][j] > 1e18:
        #             l_val = f"$\infty$"
        #         elif params["lame_lambdas"][j] > 1e9:
        #             l_val = r"$10^{10}$"
        #         else:
        #             l_val = f"1e{int(np.log10(params['lame_lambdas'][j]))}"

        #     tmp = ax.plot(
        #         -np.log2(cell_volumes),
        #         np.log2(displacement_error),
        #         marker=markers[j],
        #         color=colors[j],
        #         label=f"$\lambda$: {l_val}",
        #         linestyle="-",
        #     )
        #     legend_handles.append(tmp[0])

        # plot_and_save(ax, [], "primary_variables_" + full_stem, "log$_2$($e$)")


###### Poromechanics section

poromech_filename_stem = "poromechanics_3d"

grid_types = ["cartesian"]
grid_types = ["cartesian", "cartesian", "cartesian", "simplex"]
perturbations = [0.0, 0.3, 0.3, 0]
h2_perturbations = [False, False, True, False]
circumcenter = [False, False, False, True]
extrusion = [False, False, False, True]
cosserat_parameters = [0]
lame_lambdas = [1e10]
if run_poromechanics:
    #
    for i in range(0, len(grid_types)):
        params = {
            "grid_type": grid_types[i],
            "refinement_levels": refinement_levels,
            "cosserat_parameters": cosserat_parameters,
            "lame_lambdas": lame_lambdas,
            "perturbation": perturbations[i],
            "h2_perturbation": h2_perturbations[i],
            "permeabilities": [1],
            "nd": 3,
            "use_circumcenter": circumcenter[i],
            "prismatic_extrusion": extrusion[i],
            "analytical_solution": "poromechanics",
        }
        cosserat_results = run_poromech_convergence_analysis(**params)
        filename = f"{poromech_filename_stem}_{grid_types[i]}_pert_{perturbations[i]}_h2_{h2_perturbations[i]}_circumcenter_{circumcenter[i]}_extrusion_{extrusion[i]}.pkl"

        for i in range(len(cosserat_results)):
            for j in range(len(cosserat_results[i])):
                for k in range(len(cosserat_results[i][j])):
                    cosserat_results[i][j][k] = dataclasses.asdict(
                        cosserat_results[i][j][k]
                    )

        with open(filename, "wb") as f:
            pickle.dump([cosserat_results, params], f)

if plot_poromechanics:
    for grid_ind in range(len(grid_types)):
        full_stem = f"{poromech_filename_stem}_{grid_types[grid_ind]}_pert_{perturbations[grid_ind]}_h2_{h2_perturbations[grid_ind]}_circumcenter_{circumcenter[grid_ind]}_extrusion_{extrusion[grid_ind]}"
        filename = f"{full_stem}.pkl"
        with open(filename, "rb") as f:
            res, params = pickle.load(f)

        colors = ["orange", "blue", "green", "red"]
        markers = ["o", "s", "D", "v"]
        to_plot = []
        fig, ax = plt.subplots()
        legend_handles = []
        i = 0  # There is only one Cosserat
        for j in range(len(res[i])):  # Loop over permeability
            print(f"Permeability: {params['permeabilities'][j]}")
            all_errors = []
            cell_volumes = []
            error_all_levels = []

            # For scaling, use the analytical solution on the finest grid.
            ref_val = 0
            for key, val in res[i][j][-1].items():
                if key in [
                    "displacement",
                    "total_pressure",
                    "displacement_stress",
                    "rotation",
                    "pressure",
                    "darcy_flux",
                ]:
                    ref_val += val[1]

            for k in range(0, len(res[i][j])):
                error = 0
                error_str = ""
                key_list = []
                errors_this_level = []
                for key, val in res[i][j][k].items():
                    if key in [
                        "displacement",
                        "rotation",
                        "pressure",
                        "total_pressure",
                        "stress",
                        "darcy_flux",
                        "displacement_stress",
                    ]:
                        if key != "stress":
                            error += val[0]
                            errors_this_level.append(val[0])
                            error_str += f"{val[0] / val[1] ** 0:.5f}, "
                            key_list.append(key)
                # print(error)
                if k == 0:
                    print(key_list)
                print(error_str)
                all_errors.append(error / ref_val)
                cell_volumes.append(res[i][j][k]["cell_diameter"])
                error_all_levels.append(errors_this_level)

            if params["permeabilities"][j] == 1:
                l_val = "1"
            elif params["permeabilities"][j] < 1e-8:
                l_val = "0"
            elif params["permeabilities"][j] < 1e-3:
                l_val = r"$10^{-4}$"
            else:
                l_val = r"$10^{-2}$"

            tmp = ax.plot(
                -np.log2(cell_volumes),
                np.log2(all_errors),
                marker=markers[j],
                color=colors[j],
                label=f"$\kappa$: {l_val}",
            )
            legend_handles += tmp

            arr = np.clip(np.array(error_all_levels), a_min=1e-10, a_max=None)
            print("Log error")
            for row in range(arr.shape[0] - 1):
                print(f"{np.log2(arr[row] / arr[row + 1])}")
            print("")
            arr = np.asarray(all_errors)
            print(np.log2(arr[:-1] / arr[1:]))
            print(" ")

        print(" *************** ")
