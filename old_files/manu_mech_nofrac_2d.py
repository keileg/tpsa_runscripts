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
                field.is_cc and sd.num_cells > 50 and (field.name == "displacement")
            ):  # or field.name == 'volumetric_strain'):
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

                if field.is_scalar and field.name == "volumetric_strain" and False:
                    pp.plot_grid(
                        sd,
                        cell_value=(exact_value * 0 + 1 * approx_value),
                        figsize=(15, 15),
                    )

                elif False:
                    pp.plot_grid(
                        sd,
                        cell_value=(exact_value * 0 + 1 * approx_value)[::2],
                        figsize=(15, 15),
                    )

                import sympy as sym

                t, x, y = sym.symbols("t x y")
                f = sym.lambdify(
                    (t, x, y),
                    sym.diff(self.exact_sol.u[0], y) - sym.diff(self.exact_sol.u[1], x),
                    "numpy",
                )
                rot = f(0, sd.cell_centers[0], sd.cell_centers[1])

                debug = []

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

        stiffness = mech_data["fourth_order_tensor"]
        cosserat_parameter = mech_data["cosserat_parameter"]

        mu = stiffness.mu
        lmbda = stiffness.lmbda

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

        if False:
            p_base = x * (1 - x) * sym.sin(2 * pi * y)
            p_base = x * (1 - x) * y * (1 - y)
            p = make_heterogeneous(p_base, True)
            # Displacement
            u_base = [p_base, p_base]
            u = [
                make_heterogeneous(u_base[0], True),
                make_heterogeneous(u_base[1], True),
            ]
            rot = sym.sin(2 * pi * y) * sym.sin(2 * pi * x)

        def cosserat_parameter_function(x1, x2):
            return 0

        match setup.params.get("analytical_solution"):

            case "homogeneous":
                pot = sym.sin(2 * pi * x) ** 2 * sym.sin(2 * pi * y) ** 2

                u = [sym.diff(pot, y), -sym.diff(pot, x)]
                # u = [x, y]

                rot = [(sym.diff(u[0], y) - sym.diff(u[1], x))]
                # rot = [x - y]

                solid_p = sym.diff(u[0], x) + sym.diff(u[1], y)
                cosserat_parameter = 0

            case "heterogeneous_lame":

                u_base = [
                    sym.sin(2 * pi * x) * y * (1 - y) * (1 / 2 - y),
                    sym.sin(2 * pi * y) * x * (1 - x) * (1 / 2 - x),
                ]

                u = [
                    make_heterogeneous(u_base[0], True),
                    make_heterogeneous(u_base[1], True),
                ]
                rot = [(sym.diff(u[0], y) - sym.diff(u[1], x))]

                solid_p = sym.diff(u[0], x) + sym.diff(u[1], y)
                cosserat_parameter = 0

            case "cosserat":
                u = [
                    y * (1 - y) * sym.sin(2 * pi * x),
                    x * (1 - x) * sym.sin(2 * pi * y),
                ]

                cosserat_parameter = sym.Min(
                    1, sym.Max(0, sym.Max(3 * x - 1, 3 * y - 1))
                )
                nom = 3
                cosserat_parameter = cosserat_parameter_base * sym.Piecewise(
                    (1, ((x > 2 / nom) | (y > 2 / nom))),
                    (0.0, ((x < 1 / nom) & (y < 1 / nom))),
                    (sym.Max(nom * x - 1, nom * y - 1), True),
                )
                rot = [x * (1 - x) * y * (1 - y)]
                # rot = [sym.sin(2*pi*x) * sym.sin(2 * pi * y)]

                solid_p = sym.diff(u[0], x) + sym.diff(u[1], y)
                # solid_p = sym.sin(2*pi*x) * sym.sin(2 * pi * y)

                def cosserat_parameter_function(x1, x2):
                    return np.minimum(1, np.maximum(0, np.maximum(x1, x2)))

            case _:
                raise NotImplementedError("Unknown analytical solution")

        lame_lmbda = make_heterogeneous(lame_lmbda_base, False)
        lame_mu = make_heterogeneous(lame_mu_base, False)
        # u = [sym.sin(pi * x) * (1 - y) * y, sym.sin(pi * y) * (1 - x) * x]
        # rot = [x * (1 - x) * sym.sin(pi * y)]
        # solid_p = u[1]

        # Heterogeneous material parameters
        #  Solid Bulk modulus (heterogeneous)
        K_d = lame_lmbda + (2 / 3) * lame_mu

        # Exact gradient of the displacement
        grad_u = [
            [sym.diff(u[0], x), sym.diff(u[0], y)],
            [sym.diff(u[1], x), sym.diff(u[1], y)],
        ]
        # grad_u = [[sym.diff(u[i], var) for var in all_vars] for i in range(self.nd)]
        rot_dim = len(rot)

        grad_rot = [[sym.diff(rot[i], var) for var in all_vars] for i in range(rot_dim)]
        couple_stress = [
            [2 * cosserat_parameter * grad_rot[i][j] for j in range(self.nd)]
            for i in range(len(rot))
        ]
        # couple_stress_old = [2 * cosserat_parameter * sym.diff(rot[0], x), 2 * cosserat_parameter * sym.diff(rot[0], y)]

        # Exact elastic stress
        sigma_total = [
            [
                lame_lmbda * solid_p + 2 * lame_mu * grad_u[0][0],
                lame_mu * (2 * grad_u[0][1] - rot[0]),
            ],
            [
                lame_mu * (2 * grad_u[1][0] + rot[0]),
                lame_lmbda * solid_p + 2 * lame_mu * grad_u[1][1],
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
                # 2 * cosserat_parameter * sym.diff(grad_rot[0][0], x)
                # + 2 * cosserat_parameter * sym.diff(grad_rot[0][1], y)
                sym.diff(couple_stress[0][0], x)
                + sym.diff(couple_stress[0][1], y)
                - (sigma_total[1][0] - sigma_total[0][1]) / lame_mu
            ) / 2
        else:
            return NotImplementedError("3D not implemented")

        source_p = sym.diff(u[0], x) + sym.diff(u[1], y) - solid_p

        ## Public attributes
        # Primary variables
        self.u = u  # displacement
        self.rot = [rot[0] * lame_mu]
        self.solid_pressure = solid_p * lame_lmbda  # Solid pressure
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

        # Heterogeneous material parameters. Make these available, so that a model can
        # be populated with these parameters.
        self.lame_lmbda = lame_lmbda  # Lamé parameter
        self.lame_mu = lame_mu  # Lamé parameter
        self.cosserat_parameter_function = cosserat_parameter  # Cosserat parameter
        if False:
            import matplotlib.pyplot as plt

            np.random.seed(0)
            nx = 20
            xn, yn = np.meshgrid(np.linspace(0.5, 0.7, nx), np.linspace(0.5, 0.7, nx))
            xn += 0.2 / nx * np.random.rand(*xn.shape)
            yn += 0.2 / nx * np.random.rand(*yn.shape)
            func = sym.lambdify((x, y), source_mech, "numpy")
            vals = func(xn, yn)[0]
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot_surface(xn, yn, vals)
            plt.show()
        debug = []

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
            return rot_fun(time, *self._cc(sd))
        else:
            return rot_fun(time, *self._fc(sd))

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

        xf = sd.face_centers
        dist = np.sqrt((xf[0] - 2 / 3) ** 2 + (xf[1] - 2 / 3) ** 2)

        ind = np.argsort(dist)

        force_sorted = np.asarray(force_total_fc)[:, ind]

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

        if self.params["grid_type"] == "simplex":
            import shutil

            solution_type = self.params.get("analytical_solution")

            match solution_type:

                case "homogeneous":
                    name = "ex_1"

                case "heterogeneous_lame":
                    name = "ex_2"

                case "cosserat":
                    name = "ex_3"

                case _:
                    raise ValueError("Unknown analytical solution")

            cell_size = self.params["meshing_arguments"]["cell_size"]

            ref_level = int(np.round(np.log2(0.25 / cell_size)))
            full_name = f"partition_{name}_l_{ref_level}.msh"

            shutil.copy("gmsh_frac_file.msh", full_name)

        sd = self.mdg.subdomains()[0]
        x, y = sd.nodes[0], sd.nodes[1]
        h = np.min(sd.cell_diameters())

        pert_rate = self.params.get("perturbation", 0.0)
        if self.params.get("h2_perturbation", False):
            pert_rate *= h

        # Nodes to perturb: Not on the boundary, and not at x=0.5 and y=0.5.
        pert_nodes = np.logical_not(
            np.logical_or(np.isin(x, [0, 1]), np.isin(y, [0, 1]))
        )
        # Set the random seed
        np.random.seed(42)
        # Perturb the nodes
        x[pert_nodes] += pert_rate * h * (np.random.rand(pert_nodes.sum()) - 0.5)
        y[pert_nodes] += pert_rate * h * (np.random.rand(pert_nodes.sum()) - 0.5)

        sd.compute_geometry()

        use_circumcenter = self.params.get("use_circumcenter", False)

        if (
            use_circumcenter
            and isinstance(sd, pp.TriangleGrid)
            and isinstance(self, SetupTpsa)
        ):
            cn = sd.cell_nodes().tocsc()
            ni = cn.indices.reshape((3, sd.num_cells), order="F")

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

            all_sharp = np.logical_and.reduce(
                [angle_0 < 90, angle_1 < 90, angle_2 < 90]
            )

            sd.cell_centers[0, all_sharp] = xc[all_sharp]
            sd.cell_centers[1, all_sharp] = yc[all_sharp]

            debug = True

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

            solution_type = self.params.get("analytical_solution")

            match solution_type:

                case "homogeneous":
                    self._fractures = []

                case "heterogeneous_lame":
                    f_1 = pp.LineFracture(np.array([[1 / 2, 1 / 2], [1, 1 / 2]]))
                    f_2 = pp.LineFracture(np.array([[1, 1 / 2], [1 / 2, 1 / 2]]))

                    self._fractures = [f_1, f_2]

                case "cosserat":
                    f_1 = pp.LineFracture(np.array([[1 / 3, 1 / 3], [0, 1 / 3]]))
                    f_2 = pp.LineFracture(np.array([[0, 1 / 3], [1 / 3, 1 / 3]]))
                    f_3 = pp.LineFracture(np.array([[2 / 3, 2 / 3], [0, 2 / 3]]))
                    f_4 = pp.LineFracture(np.array([[0, 2 / 3], [2 / 3, 2 / 3]]))
                    f_5 = pp.LineFracture(np.array([[1 / 3, 2 / 3], [1 / 3, 2 / 3]]))

                    self._fractures = [f_1, f_2, f_3, f_4, f_5]
                    self._fractures = []

        else:
            # No need to do anything for Cartesian grids.
            self._fractures = []

    def meshing_kwargs(self) -> dict:
        """Set meshing arguments."""
        if self.params["grid_type"] == "simplex":
            # Mark the fractures added as constraints (not to be represented as
            # lower-dimensional objects).
            constraints = np.array([i for i in range(len(self._fractures))], dtype=int)

            return {"constraints": constraints}
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


class BoundaryConditions(
    pp.momentum_balance.BoundaryConditionsThreeFieldMomentumBalance
):

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

    def bc_values_rotation(self, grid: pp.Grid) -> np.ndarray:
        """Rotation values for the Dirichlet boundary condition.

        Parameters:
            boundary_grid: Boundary grid to evaluate values on.

        Returns:
            An array with shape (boundary_grid.num_cells,) containing the rotation values
            on the provided boundary grid.

        """
        return pp.ad.TimeDependentDenseArray(
            name="bc_values_rotation",
            domains=self.mdg.subdomains(),
        )

    def update_all_boundary_conditions(self) -> None:
        """Set values for the rotation and the volumetric strain on boundaries."""
        super().update_all_boundary_conditions()


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

        if isinstance(self, SetupTpsa):
            self.fields.append(Field("volumetric_strain", True, True, False))
            self.fields.append(Field("rotation", params["nd"] == 2, True, True))
            self.fields.append(Field("total_rotation", params["nd"] == 2, False, True))

        self.linear_solver = "iterative"

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
            name="source_mechanics", values=mech_source, data=data, iterate_index=0
        )
        rotation_source = self.exact_sol.rotation_source(sd=sd, time=t)
        pp.set_solution_values(
            name="source_rotation", values=rotation_source, data=data, iterate_index=0
        )
        solid_pressure_source = self.exact_sol.solid_pressure_source(sd=sd, time=t)
        pp.set_solution_values(
            name="source_solid_pressure",
            values=solid_pressure_source,
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

        rot = self.exact_sol.rotation(sd, 0, cc=False)[0]
        tmp = np.zeros(sd.num_faces)
        tmp[boundary_faces] = rot[boundary_faces]
        pp.set_solution_values(
            name="bc_values_rotation",
            values=tmp,
            data=data,
            iterate_index=0,
        )

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

    def after_nonlinear_convergence(self, iteration_counter: int) -> None:
        """Collect results."""
        super().after_nonlinear_convergence(iteration_counter)

        sd = self.mdg.subdomains()[0]
        displacement = self.displacement([sd]).value(self.equation_system)

        stress = self.mechanical_stress([sd]).value(self.equation_system)

        match self.params.get("analytical_solution"):
            case "homogeneous":
                ex = "ex_1"
                data = self.mdg.subdomain_data(sd)
                lmbda = data[pp.PARAMETERS][self.stress_keyword][
                    "fourth_order_tensor"
                ].lmbda
                v = lmbda[0]

                if v == 1:
                    vs = "1e0"
                else:
                    vs = f"1e{int(np.log10(v))}"
                parameter_str = f"_lambda_{vs}"
            case "heterogeneous_lame":
                ex = "ex_2"

                het = self.params["heterogeneity"]
                vs = f"1e{int(np.log10(het))}"
                parameter_str = f"_kappa_{vs}"

            case "cosserat":
                ex = "ex_3"
                parameter_str = "_"
            case _:
                raise NotImplementedError("Unknown analytical solution")

        if False:

            cell_size = self.params["meshing_arguments"]["cell_size"]

            ref_level = int(np.round(np.log2(0.25 / cell_size)))

            method_str = "TPSA" if isinstance(self, SetupTpsa) else "MPSA"

            displacement_str = f"{ex}_{method_str}{parameter_str}_displacement_{ref_level}_{cell_size}.npy"
            stress_str = (
                f"{ex}_{method_str}{parameter_str}_stress_{ref_level}_{cell_size}.npy"
            )

            np.save(displacement_str, displacement)
            np.save(stress_str, stress)

            face_str = f"{ex}_{method_str}_face_centroid_{ref_level}_{cell_size}.npy"
            cell_str = f"{ex}_{method_str}_cell_centroid_{ref_level}_{cell_size}.npy"

            face_area_str = f"{ex}_{method_str}_face_length_{ref_level}_{cell_size}.npy"
            cell_volume_str = (
                f"{ex}_{method_str}_cell_volume_{ref_level}_{cell_size}.npy"
            )
            face_normal_str = (
                f"{ex}_{method_str}_face_normal_{ref_level}_{cell_size}.npy"
            )

            np.save(face_str, sd.face_centers)
            np.save(cell_str, sd.cell_centers)
            np.save(face_area_str, sd.face_areas)
            np.save(cell_volume_str, sd.cell_volumes)
            np.save(face_normal_str, sd.face_normals)

            cell_center_str = (
                f"{ex}_{method_str}_cell_center_{ref_level}_{cell_size}.npy"
            )
            np.save(cell_center_str, sd.cell_centers)
            face_center_str = (
                f"{ex}_{method_str}_face_center_{ref_level}_{cell_size}.npy"
            )
            np.save(face_center_str, sd.face_centers)

            cell_node_str = f"{ex}_{method_str}_cell_node_{ref_level}_{cell_size}.npy"
            cn = sd.cell_nodes().tocsc().indices.reshape((3, sd.num_cells), order="F")
            np.save(cell_node_str, cn)

            face_node_str = f"{ex}_{method_str}_face_node_{ref_level}_{cell_size}.npy"
            fn = sd.face_nodes.tocsc().indices.reshape((2, sd.num_faces), order="F")
            np.save(face_node_str, fn)

            node_str = f"{ex}_{method_str}_node_{ref_level}_{cell_size}.npy"
            np.save(node_str, sd.nodes)

            debug = False

        # pp.plot_grid(sd, cell_value=displacement, figsize=(15, 15))

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
            p_solid_dof = eq_sys.dofs_of([self.solid_pressure(sd)])

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
            nullspace = np.zeros((A_00.shape[0], 2))
            # Translation dofs
            nullspace[::2, 0] = 1
            nullspace[1::2, 1] = 1

            amg_elasticity = pyamg.smoothed_aggregation_solver(A_00, B=nullspace)
            amg_rotation = pyamg.smoothed_aggregation_solver(A_11)
            amg_solid_pressure = pyamg.smoothed_aggregation_solver(A_22)

            def block_preconditioner(r):
                r_0 = r[u_dof]
                r_1 = r[rot_dof]
                r_2 = r[p_solid_dof]
                x = np.zeros_like(r)
                if False:
                    x_0 = amg_elasticity.solve(r_0, tol=1e-5, accel="cg")
                    x_1 = amg_rotation.solve(r_1 - A_00 @ x_0, tol=1e-5)
                    x_2 = amg_solid_pressure.solve(r_2 - A_20 @ x_0, tol=1e-5)

                elif False:
                    x_0 = amg_elasticity.aspreconditioner().matvec(r_0)
                    x_1 = amg_rotation.aspreconditioner().matvec(r_1)
                    x_2 = amg_solid_pressure.aspreconditioner().matvec(r_2)

                else:
                    x_0 = amg_elasticity.aspreconditioner().matvec(r_0)
                    x_1 = amg_rotation.aspreconditioner().matvec(r_1 - A_10 @ x_0)
                    x_2 = amg_solid_pressure.aspreconditioner().matvec(r_2 - A_20 @ x_0)

                x[u_dof] = x_0
                x[rot_dof] = x_1
                x[p_solid_dof] = x_2

                return x

            precond = LinearOperator(A.shape, matvec=block_preconditioner)

            debug = []

            def print_resid(x):
                print(np.linalg.norm(b - A @ x))

            if False:
                x = np.zeros_like(b[p_solid_dof])

                for _ in range(20):
                    x += amg_solid_pressure.solve(b[p_solid_dof] - A_22 @ x)
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
                    M=amg_solid_pressure.aspreconditioner(),
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
                    A, b, tol=1e-10, M=precond, callback=print_resid, x0=x, maxiter=40
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

            cc = self.exact_sol._cc(sd)

            sd.compute_geometry()

            # Set stiffness matrix
            lame_lmbda = evaluate(self.exact_sol.lame_lmbda)
            lame_mu = evaluate(self.exact_sol.lame_mu)
            stiffness = pp.FourthOrderTensor(lmbda=lame_lmbda, mu=lame_mu)
            data[pp.PARAMETERS][self.stress_keyword]["fourth_order_tensor"] = stiffness

            dim = 1 if self.nd == 2 else 3

            pp.set_solution_values(
                name="inv_mu",
                values=1.0 / np.repeat(lame_mu, dim),
                data=data,
                iterate_index=0,
            )

            pp.set_solution_values(
                name="inv_lambda",
                values=1.0 / np.repeat(lame_lmbda, dim),
                data=data,
                iterate_index=0,
            )

            # Cosserat parameter
            cosserat_parameter = evaluate(self.exact_sol.cosserat_parameter_function)
            data[pp.PARAMETERS][self.stress_keyword][
                "cosserat_parameter"
            ] = cosserat_parameter

            sd.cell_centers[0] = cc[0]
            sd.cell_centers[1] = cc[1]


class EquationsMechanicsRealStokes:
    def solid_mass_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:

        discr = self.stress_discretization(subdomains)
        inv_lmbda = self.inv_lambda(subdomains)
        div_mass = pp.ad.Divergence(subdomains, 1)

        assert len(subdomains) == 1
        bc_displacement = discr.bound_mass_displacement() @ self.bc_values_displacement(
            subdomains[0]
        )

        # Conservation of solid mass
        volumetric_strain = self.total_pressure(subdomains)
        solid_mass = div_mass @ (
            discr.mass_displacement() @ self.displacement(subdomains)
            + discr.mass_total_pressure() @ volumetric_strain
            + bc_displacement
        ) - self.source_total_pressure(subdomains)

        if self.solid.lame_lambda() < 1e8:
            # Only add the pressure term for reasonable values of lambda, above the
            # threshold of 1e8, we turn this into a Stokes discretization.
            solid_mass = solid_mass - self.volume_integral(
                inv_lmbda * volumetric_strain, subdomains, dim=1
            )

        solid_mass.set_name("solid_mass_equation")
        return solid_mass


class SetupTpsa(  # type: ignore[misc]
    UnitSquareGrid,
    SourceTerms,
    MBSolutionStrategy,
    DataSaving,
    EquationsMechanicsRealStokes,
    pp.momentum_balance.ThreeFieldMomentumBalanceEquations,
    pp.momentum_balance.ConstitutiveLawsThreeFieldMomentumBalance,
    pp.momentum_balance.VariablesThreeFieldMomentumBalance,
    pp.momentum_balance.SolutionStrategyMomentumBalanceThreeField,
    BoundaryConditions,
    pp.momentum_balance.MomentumBalance,
):
    pass


class SetupMpsa(  # type: ignore[misc]
    UnitSquareGrid,
    SourceTerms,
    MBSolutionStrategy,
    DataSaving,
    # pp.momentum_balance.ThreeFieldMomentumBalanceEquations,
    # pp.momentum_balance.ConstitutiveLawsThreeFieldMomentumBalance,
    # pp.momentum_balance.ConstitutiveLawsMomentumBalance,
    # pp.momentum_balance.VariablesThreeFieldMomentumBalance,
    # pp.momentum_balance.SolutionStrategyMomentumBalanceThreeField,
    BoundaryConditions,
    pp.momentum_balance.MomentumBalance,
):
    pass
