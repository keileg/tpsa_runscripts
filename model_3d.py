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
        t: number = self.time_manager.time

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
                relative=field.relative,
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
        heterogeneity: float = setup.params.get("heterogeneity")

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

        match setup.params.get("analytical_solution"):

            case "homogeneous":
                pot = (
                    sym.sin(2 * pi * x) ** 2
                    * sym.sin(2 * pi * y) ** 2
                    * sym.sin(2 * pi * z) ** 2
                )
                # The displacement is the curl of pot
                u = [
                    sym.diff(pot, y) - sym.diff(pot, z),
                    sym.diff(pot, z) - sym.diff(pot, x),
                    sym.diff(pot, x) - sym.diff(pot, y),
                ]

                rot = [
                    100 * x * (1 - x) * sym.sin(pi * y) * sym.sin(pi * z),
                    100 * y * (1 - y) * sym.sin(pi * x) * sym.sin(pi * z),
                    100 * z * (1 - z) * sym.sin(pi * x) * sym.sin(pi * y),
                ]
                # The solid pressure is the divergence of the displacement, hence 0
                # (div curl = 0)
                solid_p = sym.diff(u[0], x) + sym.diff(u[1], y) + sym.diff(u[2], z)
                # Fluid pressure is set to zero, this is a pure mechanics problem.
                fluid_p = 0

            case "heterogeneous_lame":

                beta_scaling = 1 + heterogeneity**-1
                scaling = (1 - char_func) + char_func * heterogeneity

                # The displacement is the curl of pot
                u_base = [
                    sym.sin(2 * pi * x)
                    * y
                    * (1 - y)
                    * (1 / 2 - y)
                    * z
                    * (1 - z)
                    * (z - 1 / 2),
                    sym.sin(2 * pi * y)
                    * x
                    * (1 - x)
                    * (1 / 2 - x)
                    * z
                    * (1 - z)
                    * (z - 1 / 2),
                    sym.sin(2 * pi * z)
                    * x
                    * (1 - x)
                    * (1 / 2 - x)
                    * y
                    * (1 - y)
                    * (y - 1 / 2),
                ]
                u = [
                    u_base[0] / (scaling * beta_scaling),
                    u_base[1] / (scaling * beta_scaling),
                    u_base[2] / (scaling * beta_scaling),
                ]

                rot_base = [
                    x
                    * (1 - x)
                    * (x - 0.5)
                    * y
                    * (1 - y)
                    * (y - 0.5)
                    * z
                    * (1 - z)
                    * (z - 0.5),
                    x
                    * (1 - x)
                    * (x - 0.5)
                    * y
                    * (1 - y)
                    * (y - 0.5)
                    * z
                    * (1 - z)
                    * (z - 0.5),
                    x
                    * (1 - x)
                    * (x - 0.5)
                    * y
                    * (1 - y)
                    * (y - 0.5)
                    * z
                    * (1 - z)
                    * (z - 0.5),
                ]
                rot = [
                    rot_base[0] / (scaling * beta_scaling),
                    rot_base[1] / (scaling * beta_scaling),
                    rot_base[2] / (scaling * beta_scaling),
                ]
                # The solid pressure is the first component of the displacement. This
                # has already been scaled.
                solid_p = u[0]
                # Fluid pressure is set to zero, this is a pure mechanics problem.
                fluid_p = 0

            case "poromechanics":
                u = [
                    y * (1 - y) * z * (1 - z) * sym.sin(2 * pi * x),
                    x * (1 - x) * z * (1 - z) * sym.sin(2 * pi * y),
                    x * (1 - x) * y * (1 - y) * sym.sin(2 * pi * z),
                ]

                rot = [
                    x * (1 - x) * sym.sin(pi * y) * sym.sin(pi * z),
                    y * (1 - y) * sym.sin(pi * x) * sym.sin(pi * z),
                    z * (1 - z) * sym.sin(pi * x) * sym.sin(pi * y),
                ]

                solid_p = x * (1 - x) * y * (1 - y) * z * (1 - z)
                fluid_p = u[0]

            case _:
                raise NotImplementedError("Unknown analytical solution")

        lame_lmbda = make_heterogeneous(lame_lmbda_base, False)
        lame_mu = make_heterogeneous(lame_mu_base, False)

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
        source_rot = source_rot_fun(time, *self._cc(sd)) * vol

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

    def set_domain(self) -> None:
        """Set domain."""
        self._domain = nd_cube_domain(3, 1.0)

    def meshing_arguments(self) -> dict[str, float]:
        """Set meshing arguments."""
        default_mesh_arguments = {"cell_size": 0.1}
        return self.params.get("meshing_arguments", default_mesh_arguments)

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

    def _circumcenter_3d(self, sd):
        from scipy.spatial import ConvexHull

        cn = sd.cell_nodes().tocsc()
        ni = cn.indices.reshape((sd.dim + 1, sd.num_cells), order="F")

        x0 = x[ni[0]]
        y0 = y[ni[0]]
        z0 = z[ni[0]]
        x1 = x[ni[1]]
        y1 = y[ni[1]]
        z1 = z[ni[1]]
        x2 = x[ni[2]]
        y2 = y[ni[2]]
        z2 = z[ni[2]]
        x3 = x[ni[3]]
        y3 = y[ni[3]]
        z3 = z[ni[3]]

        # https://en.wikipedia.org/wiki/Tetrahedron
        # https://rodolphe-vaillant.fr/entry/127/find-a-tetrahedron-circumcenter
        A = np.array(
            [
                [x1 - x0, y1 - y0, z1 - z0],
                [x2 - x0, y2 - y0, z2 - z0],
                [x3 - x0, y3 - y0, z3 - z0],
            ]
        )

        iA = [np.linalg.inv(A[:, :, i]) for i in range(A.shape[2])]

        B = 0.5 * np.array(
            [
                (x1**2 + y1**2 + z1**2) - (x0**2 + y0**2 + z0**2),
                (x2**2 + y2**2 + z2**2) - (x0**2 + y0**2 + z0**2),
                (x3**2 + y3**2 + z3**2) - (x0**2 + y0**2 + z0**2),
            ]
        )

        center = np.array([iA[i] @ B[:, i] for i in range(A.shape[2])])

        if True:
            distance_node_center = []
            for ind in ni:
                dist = np.sqrt(np.sum((sd.nodes[:, ind] - center.T) ** 2, axis=0))
                distance_node_center.append(dist)

            max_distance = np.max(np.abs(distance_node_center), axis=0)
            min_distance = np.min(np.abs(distance_node_center), axis=0)
            assert np.max(max_distance - min_distance) < 1e-10

        ind_sets = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

        ind = ind_sets[0]
        x[ni[ind]]
        added_volume = np.zeros(sd.num_cells)

        face_polygons = []
        fn = np.reshape(sd.face_nodes.tocsc().indices, (sd.dim, -1), order="F")
        face_polygons = [sd.nodes[:, fn[:, i]] for i in range(sd.num_faces)]

        cf = np.reshape(sd.cell_faces.tocsc().indices, (sd.dim + 1, -1), order="F")

        cell_polyhedron = [
            [face_polygons[fi] for fi in cf[:, ci]] for ci in range(sd.num_cells)
        ]

        xn = sd.nodes

        v1 = xn[:, fn[1]] - xn[:, fn[0]]
        v2 = xn[:, fn[2]] - xn[:, fn[0]]

        face_cross = np.vstack(
            (
                v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0],
            )
        )

        face_cross_area = np.linalg.norm(face_cross, axis=0)

        distances_from_faces = []
        inside_cell = []

        # Turn this into a loop over the rows in cf
        for ci in range(cf.shape[0]):

            fi = cf[ci]

            # Is the face cross vector pointing into the cell?
            cross_points_into_cell = np.sign(
                np.sum(
                    face_cross[:, fi] * (sd.cell_centers - sd.face_centers[:, fi]),
                    axis=0,
                )
            )
            # The height of the cell, measured as the distance from the current
            # face to the oposite node
            height = 6 * sd.cell_volumes / face_cross_area[fi]
            unit_vec = face_cross[:, fi] / face_cross_area[fi]
            # Distance from the plane of the current face to the computed circumcenter
            distances_from_faces.append(
                np.sum(unit_vec * (center.T - sd.face_centers[:, fi]), axis=0) / height
            )
            #
            inside_cell.append(
                np.logical_and(
                    cross_points_into_cell == np.sign(distances_from_faces[-1]),
                    cross_points_into_cell != 0,
                )
            )

        all_inside = np.all(inside_cell, axis=0)
        print(f"{all_inside.sum()} cells have the circumcenter inside the cell")
        min_distance = np.min(np.abs(distances_from_faces), axis=0)

        replace = np.logical_and(all_inside, min_distance > 0.05)
        if False:

            for ci in range(sd.num_cells):
                inside.append(
                    pp.geometry_property_checks.point_in_polyhedron(
                        np.array(cell_polyhedron[0]), center[0]
                    )[0]
                )

            iv = []

            for i in range(sd.num_cells):
                hull = ConvexHull(sd.nodes[:, ni[:, i]].T, incremental=True)
                init_volume = hull.volume
                iv.append(init_volume)

                hull.add_points([center[i]])

                added_volume[i] = (hull.volume - init_volume) / init_volume

            replace = np.where(added_volume <= 0)[0]

        sd.cell_centers[:, replace] = center.T[:, replace]

        print(f"Replaced {replace.sum()} out of {sd.num_cells} cell centers")

        fc = sd.cell_face_as_dense()
        # Note: fc = -1 (boundary faces) will not be found in replace
        both_replaced = np.all(np.isin(fc, np.where(replace)[0]), axis=0)
        xc = sd.cell_centers
        cc_vec = xc[:, fc[0, both_replaced]] - xc[:, fc[1, both_replaced]]

        normal = sd.face_normals[:, both_replaced]

        cc_vec_cross_normal = np.vstack(
            (
                cc_vec[1] * normal[2] - cc_vec[2] * normal[1],
                cc_vec[2] * normal[0] - cc_vec[0] * normal[2],
                cc_vec[0] * normal[1] - cc_vec[1] * normal[0],
            )
        )
        assert np.linalg.norm(cc_vec_cross_normal) < 1e-10


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

        self.exact_sol: ManuThermoPoroMechExactSolution2d
        """Exact solution object."""

        self.results: list[ManuThermoPoroMechSaveData] = []
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
            if False:
                nullspace[::3, 3] = 1
                nullspace[1::3, 3] = -1
                nullspace[1::3, 4] = 1
                nullspace[2::3, 4] = -1
                nullspace[2::3, 5] = 1
                nullspace[::3, 5] = -1
            elif False:
                nullspace[::3, 3] = -sd[0].cell_centers[1]
                nullspace[1::3, 3] = sd[0].cell_centers[0]
                nullspace[1::3, 4] = sd[0].cell_centers[2]
                nullspace[2::3, 4] = -sd[0].cell_centers[1]
                nullspace[2::3, 5] = sd[0].cell_centers[0]
                nullspace[::3, 5] = -sd[0].cell_centers[2]

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
                if False:
                    x_0 = amg_elasticity.solve(r_0, tol=1e-5, accel="cg")
                    x_1 = amg_rotation.solve(r_1 - A_00 @ x_0, tol=1e-5)
                    x_2 = amg_total_pressure.solve(r_2 - A_20 @ x_0, tol=1e-5)

                elif False:
                    x_0 = amg_elasticity.aspreconditioner().matvec(r_0)
                    x_1 = amg_rotation.aspreconditioner().matvec(r_1)
                    x_2 = amg_total_pressure.aspreconditioner().matvec(r_2)

                else:
                    x_0 = amg_elasticity.aspreconditioner().matvec(r_0)
                    # x_1 = amg_rotation.aspreconditioner().matvec(r_1 - A_10 @ x_0)
                    x_1 = rotation_solver @ (r_1 - A_10 @ x_0)
                    x_2 = amg_total_pressure.aspreconditioner().matvec(r_2 - A_20 @ x_0)
                    # x_2 = total_pressure_solver @ (r_2 -A_20 @ x_0)
                    # x_2 = tps(r_2 -A_20 @ x_0)

                x[u_dof] = x_0
                x[rot_dof] = x_1
                x[p_solid_dof] = x_2

                r_0_new = r_0 - A_00 @ x_0
                r_1_new = r_1 - A_10 @ x_0 - A_11 @ x_1
                r_2_new = r_2 - A_20 @ x_0 - A_22 @ x_2

                return x

            precond = LinearOperator(A.shape, matvec=block_preconditioner)

            debug = []

            def print_resid(x):
                # pass
                print(np.linalg.norm(b - A @ x))

            if False:
                x = np.zeros_like(b[p_solid_dof])

                for _ in range(20):
                    x += amg_total_pressure.solve(b[p_solid_dof] - A_22 @ x)
                    sp_resid(x)

            if False:

                def el_resid(x):
                    print(np.linalg.norm(b[u_dof] - A_00 @ x))

                x, info = pyamg.krylov.fgmres(
                    A_00,
                    b[u_dof],
                    tol=1e-6,
                    M=amg_elasticity.aspreconditioner(),
                    callback=el_resid,
                    maxiter=200,
                )

                def rot_resid(x):
                    print(np.linalg.norm(b[rot_dof] - A_11 @ x))

                x, info = pyamg.krylov.fgmres(
                    A_11,
                    b[rot_dof],
                    tol=1e-6,
                    M=amg_rotation.aspreconditioner(),
                    callback=rot_resid,
                    maxiter=200,
                )

                def sp_resid(x):
                    print(np.linalg.norm(b[p_solid_dof] - A_22 @ x))

                x, info = pyamg.krylov.fgmres(
                    A_22,
                    b[p_solid_dof],
                    tol=1e-6,
                    M=amg_total_pressure.aspreconditioner(),
                    x0=np.random.rand(p_solid_dof.size),
                    callback=sp_resid,
                    maxiter=200,
                )

                import scipy.sparse.linalg as spla

                M = spla.spilu(A)
                Mop = LinearOperator(A.shape, matvec=lambda x: M.solve(x))
                x, info = pyamg.krylov.fgmres(
                    A, b, tol=1e-6, M=Mop, callback=print_resid, maxiter=100
                )

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
        self.linear_solver = "iterative"

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

        self.exact_sol: ExactSolution
        """Exact solution object."""

        self.results: list[ManuThermoPoroMechSaveData] = []
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

                # tps = spla.factorized(A_22)

                # fps = spla.factorized(A_33)

            def block_preconditioner(r):
                r_0 = r[u_dof]
                r_1 = r[rot_dof]
                r_2 = r[p_solid_dof]
                x = np.zeros_like(r)

            def block_preconditioner(r):
                r_0 = r[u_dof]
                r_1 = r[rot_dof]
                r_2 = r[p_solid_dof]
                r_3 = r[p_fluid_dof]
                x = np.zeros_like(r)
                if False:
                    x_0 = amg_elasticity.solve(r_0, tol=1e-5, accel="cg")
                    x_1 = amg_rotation.solve(r_1, tol=1e-5)
                    x_2 = amg_total_pressure.solve(r_2, tol=1e-5)
                elif False:
                    x_0 = amg_elasticity.aspreconditioner().matvec(r_0)
                    x_1 = rotation_solver @ (r_1)
                    x_2 = tps(r_2)

                    x_3 = fps(r_3)

                else:
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
        self.linear_solver = "iterative"

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
                name=self.stress_keyword, domains=domains  # type: ignore[call-arg]
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
