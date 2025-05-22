import porepy as pp
import numpy as np
from collections import namedtuple
import time
from numba import config
from porepy.applications.md_grids.domains import nd_cube_domain
import scipy.sparse as sps
import warnings

config.DISABLE_JIT = True

Field = namedtuple("Field", ["name", "is_scalar", "is_cc", "relative"])


class Geometry:
    def set_domain(self) -> None:
        """Set domain of the problem.

        Defaults to a 2d unit square domain.
        Override this method to define a geometry with a different domain.

        """
        self._domain = nd_cube_domain(3, self.units.convert_units(1.0, "m"))

    def meshing_arguments(self) -> dict[str, float]:
        """Meshing arguments for mixed-dimensional grid generation.

        Returns:
            Meshing arguments compatible with
            :meth:`~porepy.grids.mdg_generation.create_mdg`.

        """

        # Default value of 1/2, scaled by the length unit.
        default_meshing_args: dict[str, float] = {
            "cell_size_x": 1.0 / 10,
            "cell_size_y": 1.0 / 10,
            "cell_size_z": self.params["cell_size_z"],
        }
        default_meshing_args: dict[str, float] = {
            "cell_size_x": self.params["cell_size_z"],
            "cell_size_y": self.params["cell_size_z"],
            "cell_size_z": self.params["cell_size_z"],
        }
        # If meshing arguments are provided in the params, they should already be
        # scaled by the length unit.
        return self.params.get("meshing_arguments", default_meshing_args)


class BoundaryConditions:
    def bc_type_mechanics(self, g: pp.Grid) -> pp.BoundaryConditionVectorial:
        bottom = self.domain_boundary_sides(g).bottom
        top = self.domain_boundary_sides(g).top
        dir_face = np.logical_or(bottom, top)
        dir_face = bottom
        bc = pp.BoundaryConditionVectorial(g, dir_face, "dir")
        return bc

    def bc_values_displacement(self, g: pp.BoundaryGrid | pp.Grid) -> np.ndarray:
        """Displacement values for the Dirichlet boundary condition.

        Parameters:
            boundary_grid: Boundary grid to evaluate values on.

        Returns:
            An array with shape (boundary_grid.num_cells,) containing the displacement
            values on the provided boundary grid.

        """
        bottom = self.domain_boundary_sides(g).bottom
        top = self.domain_boundary_sides(g).top
        dir_face = np.logical_or(bottom, top)
        dir_face = bottom
        val = np.zeros((self.nd, g.num_cells))
        val[0, dir_face] = 0

        return val.ravel("F")

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid | pp.Grid) -> np.ndarray:
        """Stress values for the Neumann boundary condition.

        Parameters:
            boundary_grid: Boundary grid to evaluate values on.

        Returns:
            An array with shape (boundary_grid.num_cells,) containing the stress
            values on the provided boundary grid.

        """
        val = np.zeros((self.nd, boundary_grid.num_cells))
        top = self.domain_boundary_sides(boundary_grid).top
        west = self.domain_boundary_sides(boundary_grid).west
        east = self.domain_boundary_sides(boundary_grid).east
        val[0, top] = 1
        val[2, west] = -1
        val[2, east] = 1

        val *= boundary_grid.cell_volumes

        return val.ravel("F")


class MBSolutionStrategy(pp.momentum_balance.SolutionStrategyMomentumBalance):
    """Solution strategy for the verification setup."""

    def __init__(self, params: dict):
        """Constructor for the class."""
        super().__init__(params)

        self.exact_sol
        """Exact solution object."""

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
            self.fields.append(Field("total_pressure", True, True, False))
            self.fields.append(Field("rotation", True, True, True))

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
                import scipy.sparse.linalg as spla

                tps = spla.factorized(A_22)

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
                    # x_2 = amg_total_pressure.aspreconditioner().matvec(r_2 - A_20 @ x_0)
                    x_2 = total_pressure_solver @ (r_2 - A_20 @ x_0)
                    x_2 = tps(r_2 - A_20 @ x_0)

                x[u_dof] = x_0
                x[rot_dof] = x_1
                x[p_solid_dof] = x_2

                r_0_new = r_0 - A_00 @ x_0
                r_1_new = r_1 - A_10 @ x_0 - A_11 @ x_1
                r_2_new = r_2 - A_20 @ x_0 - A_22 @ x_2

                return x

            precond = LinearOperator(A.shape, matvec=block_preconditioner)

            def print_resid(x):
                # pass
                print(np.linalg.norm(b - A @ x))

            t = time.time()
            x = np.zeros_like(b)
            for _ in range(100):
                x, info = pyamg.krylov.fgmres(
                    A, b, tol=1e-12, M=precond, callback=print_resid, x0=x, maxiter=400
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


class SetupTpsa(  # type: ignore[misc]
    Geometry,
    BoundaryConditions,
    MBSolutionStrategy,
    # DataSaving,
    # DisplacementStress,
    pp.momentum_balance.TpsaMomentumBalanceMixin,
    pp.momentum_balance.MomentumBalance,
):
    pass


for i in range(3):
    nx = 4 * 2**i

    model = SetupTpsa({"cell_size_z": 1.0 / nx})
    pp.run_time_dependent_model(model)

    sd = model.mdg.subdomains()[0]
    u = model.displacement(model.mdg.subdomains()).value(model.equation_system)
    r = model.rotation(model.mdg.subdomains()).value(model.equation_system)
    p = model.total_pressure(model.mdg.subdomains()).value(model.equation_system)

    # print(np.linalg.norm(u[::model.nd] - sd.cell_centers[0]))
    print(np.sqrt(np.sum(sd.cell_volumes * (u[:: model.nd] - sd.cell_centers[2]) ** 2)))

    debug = []
