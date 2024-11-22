from collections import namedtuple
import sympy as sym
from porepy.viz.data_saving_model_mixin import VerificationDataSaving
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.applications.md_grids.domains import nd_cube_domain
import porepy as pp
import numpy as np
from dataclasses import make_dataclass
import scipy.sparse as sps

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

        for field in self.fields:
            exact_value = getattr(self.exact_sol, field.name)(sd=sd, time=t)
            ad_representation = getattr(self, field.name)([sd])
            approx_value = ad_representation.value(self.equation_system)
            error, ref_norm = self.l2_error(
                grid=sd,
                true_array=exact_value,
                approx_array=approx_value,
                name=field.name,
                is_scalar=field.is_scalar,
                is_cc=field.is_cc,
                relative=field.relative,
            )
            if field.name == "couple_stress" and sd.num_cells > 20 and False:
                g = pp.CartGrid([sd.cart_dims[0] + 1, sd.cart_dims[1]])
                pp.plot_grid(g, cell_value=approx_value[: g.num_cells])
                pp.plot_grid(g, cell_value=exact_value[: g.num_cells])
                pp.plot_grid(
                    g,
                    cell_value=approx_value[: g.num_cells] - exact_value[: g.num_cells],
                )
                debug = True

            if field.name == "displacement" and False:
                pp.plot_grid(sd, cell_value=approx_value[::2])
                pp.plot_grid(sd, cell_value=exact_value[::2])
                pp.plot_grid(sd, cell_value=approx_value[::2] - exact_value[::2])
                debug = True

            if (
                field.is_cc
                and sd.num_cells > 30
                and (field.name == "rotation" or field.name == "volumetric_strain")
            ):
                if False:
                    if field.name == "displacement":
                        stride = 2
                    else:
                        stride = 1
                    if isinstance(exact_value, list):
                        ev = exact_value[0]
                    elif isinstance(exact_value, (int, float)):
                        ev = exact_value * np.ones(sd.num_cells * stride)
                    else:
                        ev = exact_value
                    pp.plot_grid(
                        sd,
                        cell_value=approx_value[::stride],
                        figsize=(15, 15),
                    )
                    import matplotlib.pyplot as plt

                    plt.savefig(f"{field.name}_approx.png")
                    pp.plot_grid(
                        sd,
                        cell_value=ev[::stride],
                        figsize=(15, 15),
                    )
                    plt.savefig(f"{field.name}_exact.png")
                    pp.plot_grid(
                        sd,
                        cell_value=(approx_value - ev)[::stride],
                        figsize=(15, 15),
                    )
                    plt.savefig(f"{field.name}_error.png")

            collected_data[field.name] = (error, ref_norm)

        collected_data["time"] = t
        collected_data["cell_diameter"] = sd.cell_diameters().min()
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
        fluid_data = self.mdg.subdomain_data(grid)[pp.PARAMETERS][self.darcy_keyword]

        stiffness = mech_data["fourth_order_tensor"]
        cosserat_parameter = mech_data["cosserat_parameter"]

        mu = stiffness.mu
        lmbda = stiffness.lmbda

        permeability = fluid_data["second_order_tensor"].values[0, 0]

        # Obtain proper measure, e.g., cell volumes for cell-centered quantities and face
        # areas for face-centered quantities.
        if is_cc:
            vol = grid.cell_volumes

            if name == "rotation":
                meas = vol / mu
            elif name == "volumetric_strain":
                meas = vol / mu
            elif name == "displacement":
                meas = vol * mu
            elif name == "pressure":
                meas = vol
        else:
            assert isinstance(grid, pp.Grid)  # to please mypy
            surface_measure = grid.face_areas

            fi, ci, sgn = sps.find(grid.cell_faces)
            fc_cc = grid.face_centers[::, fi] - grid.cell_centers[::, ci]
            dist_fc_cc = np.sqrt(np.sum(fc_cc**2, axis=0))

            def facewise_harmonic_mean(field):
                return 1 / np.bincount(
                    fi, weights=1 / field[ci], minlength=grid.num_faces
                )

            def arithmetic_mean(field):
                return np.bincount(fi, weights=field[ci], minlength=grid.num_faces)

            if name == "total_rotation":
                parameter_weight = arithmetic_mean(cosserat_parameter)
            elif name == "stress":
                parameter_weight = facewise_harmonic_mean(mu)
            elif name == "darcy_flux":
                parameter_weight = facewise_harmonic_mean(permeability)
            else:
                raise ValueError("Unknown field")

            # Distance between neighboring cells
            dist_cc_cc = np.bincount(fi, weights=dist_fc_cc, minlength=grid.num_cells)

            meas = surface_measure * (dist_cc_cc / grid.dim) * parameter_weight

            debug = True

        if not is_scalar:
            meas = meas.repeat(grid.dim)

        # Obtain numerator and denominator to determine the error.
        numerator = np.sqrt(np.sum(meas * np.abs(true_array - approx_array) ** 2))

        denominator = (
            np.sqrt(np.sum(meas * np.abs(true_array) ** 2)) if relative else 1.0
        )

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
        lame_lmbda_base = setup.solid.lame_lambda()
        lame_mu_base = setup.solid.shear_modulus()
        biot_coefficient = setup.solid.biot_coefficient()
        permeability = setup.solid.permeability()
        fluid_compressibility = setup.fluid.compressibility()
        porosity = setup.solid.porosity()

        cosserat_parameter_base = setup.params["cosserat_parameter"]

        pi = sym.pi

        t, x, y = self._symbols()
        all_vars = [x, y]
        # Characteristic function: 1 if x > 0.5 and y > 0.5, 0 otherwise
        char_func = sym.Piecewise((1, ((x > 0.5) & (y > 0.5))), (0, True))

        def make_heterogeneous(v, invert: bool):
            # Helper function to include the heterogeneity into a function.
            if invert:
                return v / ((1 - char_func) + char_func * heterogeneity)
            else:
                return v * ((1 - char_func) + char_func * heterogeneity)

        u = [sym.sin(pi * x) * (1 - y) * y, sym.sin(pi * y) * (1 - x) * x]
        # u = [x, y]

        rot = [x * (1 - x) * sym.sin(pi * y)]
        fluid_p = u[0]
        # rot = [x - y]

        solid_p = u[1]

        q = [-permeability * sym.diff(fluid_p, x), -permeability * sym.diff(fluid_p, y)]

        # Heterogeneous material parameters
        lame_lmbda = make_heterogeneous(lame_lmbda_base, False)
        lame_mu = make_heterogeneous(lame_mu_base, False)
        cosserat_parameter = make_heterogeneous(cosserat_parameter_base, False)
        #  Solid Bulk modulus (heterogeneous)
        K_d = lame_lmbda + (2 / 3) * lame_mu

        # Exact gradient of the displacement
        grad_u = [
            [sym.diff(u[0], x), sym.diff(u[0], y)],
            [sym.diff(u[1], x), sym.diff(u[1], y)],
        ]
        grad_u = [[sym.diff(u[i], var) for var in all_vars] for i in range(self.nd)]
        rot_dim = len(rot)

        grad_rot = [[sym.diff(rot[i], var) for var in all_vars] for i in range(rot_dim)]
        couple_stress = [
            [2 * cosserat_parameter * grad_rot[i][j] for j in range(self.nd)]
            for i in range(len(rot))
        ]

        # Exact elastic stress
        sigma_total = [
            [
                lame_lmbda * solid_p
                + 2 * lame_mu * grad_u[0][0]
                - biot_coefficient * fluid_p,
                lame_mu * (2 * grad_u[0][1] - rot[0]),
            ],
            [
                lame_mu * (2 * grad_u[1][0] + rot[0]),
                lame_lmbda * solid_p
                + 2 * lame_mu * grad_u[1][1]
                - biot_coefficient * fluid_p,
            ],
        ]
        # Mechanics source term
        source_mech = [
            sym.diff(sigma_total[0][0], x) + sym.diff(sigma_total[0][1], y),
            sym.diff(sigma_total[1][0], x) + sym.diff(sigma_total[1][1], y),
        ]
        if self.nd == 2:
            # Need to make this a vector
            source_rot = (
                sym.diff(couple_stress[0][0], x)
                + sym.diff(couple_stress[0][1], y)
                - (sigma_total[1][0] - sigma_total[0][1]) / lame_mu
            ) / 2
        else:
            return NotImplementedError("3D not implemented")

        source_p = sym.diff(u[0], x) + sym.diff(u[1], y) - solid_p

        # Exact divergence of the mass flux
        div_mf = sym.diff(q[0], x) + sym.diff(q[1], y)

        # Exact flow accumulation
        accum_flow = fluid_compressibility * fluid_p + biot_coefficient * solid_p

        # Exact flow source
        source_flow = accum_flow + div_mf

        ## Public attributes
        # Primary variables
        self.u = u  # displacement
        self.rot = rot
        self.solid_pressure = (
            solid_p * lame_lmbda - biot_coefficient * fluid_p
        )  # Solid pressure
        # Secondary variables
        self.sigma_total = sigma_total  # poroelastic (total) stress

        # The 3d expression will be different
        total_rotation = [
            [couple_stress[0][0] / 2 - u[1], couple_stress[0][1] / 2 + u[0]]
        ]

        self.total_rot = total_rotation  # Cosserat couple stress
        self.alt_source_rot = (
            sym.diff(total_rotation[0][0], x)
            + sym.diff(total_rotation[0][1], y)
            - rot[0] / lame_mu
        )

        # Source terms
        self.source_mech = source_mech  # Source term entering the momentum balance
        self.source_rotation = source_rot  # Source term entering the rotation balance
        self.source_p = source_p  # Source term entering the solid pressure balance

        self.source_flow = source_flow
        self.f_pressure = fluid_p  # Fluid pressure

        self.q = q  # Darcy flux

        # Heterogeneous material parameters. Make these available, so that a model can
        # be populated with these parameters.
        self.lame_lmbda = lame_lmbda  # Lamé parameter
        self.lame_mu = lame_mu  # Lamé parameter
        self.cosserat_parameter = cosserat_parameter  # Cosserat parameter

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

    def displacement(self, sd: pp.Grid, time: float) -> np.ndarray:
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
        u_cc: list[np.ndarray] = [u_fun[i](time, *self._cc(sd)) for i in range(self.nd)]

        # Flatten array
        u_flat: np.ndarray = np.asarray(u_cc).ravel("F")

        return u_flat

    def rotation(self, sd: pp.Grid, time: float) -> np.ndarray:
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
        rot_cc = rot_fun(time, *self._cc(sd))

        return rot_cc

    def solid_pressure(self, sd: pp.Grid, time: float) -> np.ndarray:
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
        p_fun = sym.lambdify((t, *x), self.solid_pressure, "numpy")

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
        force_total_fc_old: list[np.ndarray] = [
            # (sigma_xx * n_x + sigma_xy * n_y) * face_area
            sigma_total_fun[0][0](time, *self._fc(sd)) * fn[0]
            + sigma_total_fun[0][1](time, *self._fc(sd)) * fn[1],
            # (sigma_yx * n_x + sigma_yy * n_y) * face_area
            sigma_total_fun[1][0](time, *self._fc(sd)) * fn[0]
            + sigma_total_fun[1][1](time, *self._fc(sd)) * fn[1],
        ]
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
        x, y, t = sym.symbols("x y t")

        # Get list of face indices
        fc = sd.face_centers
        fn = sd.face_normals

        # Lambdify expression

        q_fun: list[Callable] = [
            sym.lambdify((x, y, t), self.q[0], "numpy"),
            sym.lambdify((x, y, t), self.q[1], "numpy"),
        ]

        # Face-centered Darcy fluxes
        q_fc: np.ndarray = (
            q_fun[0](fc[0], fc[1], time) * fn[0] + q_fun[1](fc[0], fc[1], time) * fn[1]
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

        return source_rot

    def solid_pressure_source(self, sd: pp.Grid, time: float) -> np.ndarray:
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


class UnitSquareGrid(pp.ModelGeometry):
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
        super().set_geometry()

        sd = self.mdg.subdomains()[0]
        x, y = sd.nodes[0], sd.nodes[1]
        h = np.min(sd.cell_diameters())

        pert_rate = self.params.get("perturbation", 0.0)
        if self.params.get("h2_perturbation", False):
            pert_rate *= h

        # Nodes to perturb: Not on the boundary, and not at x=0.5 and y=0.5.
        pert_nodes = np.logical_not(
            np.logical_or(np.isin(x, [0, 0.5, 1]), np.isin(y, [0, 0.5, 1]))
        )
        # Set the random seed
        np.random.seed(42)
        # Perturb the nodes
        x[pert_nodes] += pert_rate * h * (np.random.rand(pert_nodes.sum()) - 0.5)
        y[pert_nodes] += pert_rate * h * (np.random.rand(pert_nodes.sum()) - 0.5)

        sd.compute_geometry()

    def set_domain(self) -> None:
        """Set domain."""
        self._domain = nd_cube_domain(2, 1.0)

    def meshing_arguments(self) -> dict[str, float]:
        """Set meshing arguments."""
        default_mesh_arguments = {"cell_size": 0.1}
        return self.params.get("meshing_arguments", default_mesh_arguments)

    def set_fractures(self) -> None:
        """The geometry contains no fractures per se, but we set fractures for simplex
        grids to conform to material heterogeneities. See class documentation for
        details.
        """

        if self.params["grid_type"] == "simplex":
            self._fractures = pp.fracture_sets.orthogonal_fractures_2d(size=1)
            self._fractures = []
        else:
            # No need to do anything for Cartesian grids.
            self._fractures = []

    def meshing_kwargs(self) -> dict:
        """Set meshing arguments."""
        if self.params["grid_type"] == "simplex":
            # Mark the fractures added as constraints (not to be represented as
            # lower-dimensional objects).
            return {}
            return {"constraints": [0, 1]}
        else:
            return {}


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

    def source_solid_pressure(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Solid pressure source."""

        external_sources = pp.ad.TimeDependentDenseArray(
            name="source_solid_pressure",
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


class MBSolutionStrategy(pp.poromechanics.SolutionStrategyPoromechanics):
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

        self.rotation_variable: str = "rotation"
        """Name of the rotation variable."""

        self.volumetric_strain_variable: str = "volumetric_strain"
        """Name of the volumetric strain variable."""
        self.fields = [
            Field("displacement", False, True, True),
            Field("rotation", params["nd"] == 2, True, True),
            Field("volumetric_strain", True, True, True),
            Field("pressure", True, True, True),
        ]

        self.fields.append(Field("stress", False, False, True))
        self.fields.append(Field("total_rotation", params["nd"] == 2, False, True))
        self.fields.append(Field("darcy_flux", True, False, True))

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

        volumetric_strain_vals = np.zeros(num_cells)

        self.equation_system.set_variable_values(
            volumetric_strain_vals,
            [self.volumetric_strain_variable],
            time_step_index=0,
            iterate_index=0,
        )

    def set_materials(self):
        """Set material parameters."""
        super().set_materials()

    def before_nonlinear_loop(self) -> None:
        """Update values of external sources."""
        super().before_nonlinear_loop()

        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time

        # Mechanics source
        mech_source = self.exact_sol.mechanics_source(sd=sd, time=t)
        pp.set_solution_values(
            name="source_mechanics", values=mech_source, data=data, time_step_index=0
        )
        rotation_source = self.exact_sol.rotation_source(sd=sd, time=t)
        pp.set_solution_values(
            name="source_rotation", values=rotation_source, data=data, time_step_index=0
        )
        solid_pressure_source = self.exact_sol.solid_pressure_source(sd=sd, time=t)
        pp.set_solution_values(
            name="source_solid_pressure",
            values=solid_pressure_source,
            data=data,
            time_step_index=0,
        )

        fluid_pressure_source = self.exact_sol.fluid_pressure_source(sd=sd, time=t)
        pp.set_solution_values(
            name="source_fluid_pressure",
            values=fluid_pressure_source,
            data=data,
            time_step_index=0,
        )

    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Collect results."""
        super().after_nonlinear_convergence(solution, errors, iteration_counter)

        sd = self.mdg.subdomains()[0]
        displacement = self.displacement([sd]).value(self.equation_system)

        # pp.plot_grid(sd, cell_value=displacement, figsize=(15, 15))

        debug = []

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

            # Cosserat parameter
            cosserat_parameter = evaluate(self.exact_sol.cosserat_parameter)
            data[pp.PARAMETERS][self.stress_keyword][
                "cosserat_parameter"
            ] = cosserat_parameter

    def _is_time_dependent(self):
        return False

    def _is_nonlinear_problem(self) -> bool:
        """The problem is linear."""
        return False


class EquationsPoromechanics(
    pp.fluid_mass_balance.MassBalanceEquations,
    pp.momentum_balance.ThreeFieldMomentumBalanceEquations,
):
    """Combines mass and momentum balance equations."""

    def set_equations(self):
        """Set the equations for the poromechanics problem.

        Call both parent classes' set_equations methods.

        """
        pp.fluid_mass_balance.MassBalanceEquations.set_equations(self)
        pp.momentum_balance.ThreeFieldMomentumBalanceEquations.set_equations(self)

    def solid_mass_equation(self, subdomains):
        eq = super().solid_mass_equation(subdomains)

        factor = self._biot_coefficient_inv_lambda(subdomains)
        full_eq = eq - self.volume_integral(
            factor * self.pressure(subdomains), subdomains, dim=1
        )
        # eq.name = "solid_mass_poromechanics"
        return full_eq

    def _biot_coefficient_inv_lambda(self, subdomains):
        stiffness = self.stiffness_tensor(subdomains[0])

        inv_lmbda = pp.ad.DenseArray(1 / stiffness.lmbda)
        # Conservation of solid mass
        factor = self.biot_coefficient(subdomains) * inv_lmbda
        return factor

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Mass balance equation for subdomains.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the mass balance equation.

        """
        # Assemble the terms of the mass balance equation.
        accumulation = self.fluid_mass(subdomains)
        flux = self.darcy_flux(subdomains)
        source = self.fluid_source(subdomains)

        # Feed the terms to the general balance equation method.
        eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq.set_name("mass_balance_equation")
        return eq

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
        eff_compressibility = pp.ad.Scalar(
            self.fluid.compressibility()
        ) + self._biot_coefficient_inv_lambda(subdomains) * self.biot_coefficient(
            subdomains
        )

        fluid_contribution = eff_compressibility * self.pressure(subdomains)

        solid_contribution = self._biot_coefficient_inv_lambda(
            subdomains
        ) * self.solid_pressure(subdomains)

        mass = self.volume_integral(
            fluid_contribution + solid_contribution, subdomains, dim=1
        )
        mass.set_name("fluid_mass")
        return mass


class ConstitutiveLaws(
    pp.momentum_balance.ConstitutiveLawsThreeFieldMomentumBalance,
    pp.poromechanics.ConstitutiveLawsPoromechanics,
):
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

    def stress(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Stress operator.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Operator for the stress.

        """
        # Method from constitutive library's LinearElasticRock.
        return self.mechanical_stress(subdomains)


class BoundaryConditions(pp.poromechanics.BoundaryConditionsPoromechanics):
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Pressure values for the Dirichlet boundary condition.

        These values are used for quantities relying on Dirichlet data for pressure on
        the boundary, such as mobility, density or Darcy flux.

        Important:
            Override this method to provide custom Dirichlet boundary data for pressure,
            per boundary grid as a numpy array with numerical values.

        Parameters:
            boundary_grid: Boundary grid to provide values for.

        Returns:
            An array with ``shape(boundary_grid.num_cells,)`` containing the pressure
            values on the provided boundary grid.

        """
        return np.zeros(boundary_grid.num_cells)

    def bc_values_mobility_rho(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Pressure values for the Dirichlet boundary condition.

        These values are used for quantities relying on Dirichlet data for pressure on
        the boundary, such as mobility, density or Darcy flux.

        Important:
            Override this method to provide custom Dirichlet boundary data for pressure,
            per boundary grid as a numpy array with numerical values.

        Parameters:
            boundary_grid: Boundary grid to provide values for.

        Returns:
            An array with ``shape(boundary_grid.num_cells,)`` containing the pressure
            values on the provided boundary grid.

        """
        return np.zeros(boundary_grid.num_cells)


class Variables(
    pp.momentum_balance.VariablesThreeFieldMomentumBalance,
    pp.fluid_mass_balance.VariablesSinglePhaseFlow,
):
    def create_variables(self):
        """Set the variables for the poromechanics problem.

        Call both parent classes' set_variables methods.

        """
        pp.fluid_mass_balance.VariablesSinglePhaseFlow.create_variables(self)
        pp.momentum_balance.VariablesThreeFieldMomentumBalance.create_variables(self)


class Setup(  # type: ignore[misc]
    UnitSquareGrid,
    SourceTerms,
    MBSolutionStrategy,
    DataSaving,
    EquationsPoromechanics,
    ConstitutiveLaws,
    Variables,
    pp.poromechanics.BoundaryConditionsPoromechanics,
):
    pass
