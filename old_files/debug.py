import porepy as pp
import numpy as np

g = pp.StructuredTriangleGrid([2, 2], [1, 2])
g = pp.CartGrid([4, 4])

# Define the side length of the equilateral triangle
side_length = 1

# Define the height of the equilateral triangle
height = np.sqrt(3) / 2 * side_length

# Define the coordinates for the grid
x = np.arange(0, 4, side_length, dtype=float)
y = np.arange(0, 4, height, dtype=float)

# Create a grid of x and y coordinates
x_coords, y_coords = np.meshgrid(x, y)

x_coords[:] += 0.5 * np.arange(x_coords.shape[0]).reshape(-1, 1)


# Flatten the coordinate grids and stack them into an array of coordinate pairs
coords = np.dstack([x_coords.flatten(), y_coords.flatten()])[0]

# Initialize lists to store the triangle indices
triangles = []

# Generate the indices for the equilateral triangles
for i in range(len(x) - 1):
    for j in range(len(y) - 1):
        # Define the indices for the lower triangle
        lower_triangle = [i + j * len(x), i + 1 + j * len(x), i + (j + 1) * len(x)]
        triangles.append(lower_triangle)

        # Define the indices for the upper triangle
        upper_triangle = [
            i + 1 + j * len(x),
            i + 1 + (j + 1) * len(x),
            i + (j + 1) * len(x),
        ]
        triangles.append(upper_triangle)

g = pp.TriangleGrid(coords.T, np.array(triangles).T)


internal_dofs = np.array([5, 6, 9, 10])
internal_vector_dofs = pp.fvutils.expand_indices_nd(internal_dofs, g.dim)

# g.nodes[:g.dim] += np.random.rand(2, g.num_nodes)

g.compute_geometry()
mu = np.ones(g.num_cells)

C = pp.FourthOrderTensor(mu, mu)

dir_faces = g.get_all_boundary_faces()
bc = pp.BoundaryConditionVectorial(g)
bc.is_neu[:, dir_faces] = False
bc.is_dir[:, dir_faces] = True

data = {
    pp.PARAMETERS: {
        "mechanics": {
            "fourth_order_tensor": C,
            "bc": bc,
            "cosserat_parameter": np.ones(g.num_cells),
        }
    },
    pp.DISCRETIZATION_MATRICES: {"mechanics": {}},
}

tpsa = pp.Tpsa("mechanics")
tpsa.discretize(g, data)

matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["mechanics"]


stress = matrix_dictionary[tpsa.stress_displacement_matrix_key]
stress_rotation = matrix_dictionary[tpsa.stress_rotation_matrix_key]
stress_volumetric_strain = matrix_dictionary[tpsa.stress_volumetric_strain_matrix_key]
rotation_displacement = matrix_dictionary[tpsa.rotation_displacement_matrix_key]
rotation_diffusion = matrix_dictionary[tpsa.rotation_diffusion_matrix_key]
mass_volumetric_strain = matrix_dictionary[tpsa.mass_volumetric_strain_matrix_key]
mass_displacement = matrix_dictionary[tpsa.mass_displacement_matrix_key]

# The following should be zero
u_constant = np.vstack([np.ones(g.num_cells), 2 * np.ones(g.num_cells)]).ravel("F")
rot_constant = np.ones(g.num_cells)
divu_constant = np.zeros(g.num_cells)

gxx = 0.7
gxy = 0.4
gyx = 0.3
gyy = 0.3

import sympy as sym

u_linear = np.vstack([gxx * g.cell_centers[0], gyy * g.cell_centers[1]]).ravel("F")
rot_linear = gxx * g.cell_centers[1] - gyy * g.cell_centers[0]
divu_linear = (gxx + gyy) * np.ones(g.num_cells)

if False:
    u = u_constant
    rot = rot_constant
    divu = divu_constant

    s_u = 0
    s_rot = 0
    s_divu = 0

else:
    u = u_linear
    rot = np.zeros(g.num_cells)
    divu = divu_linear

    s_u = np.vstack([gyy * np.ones(g.num_cells), gxx * np.ones(g.num_cells)]).ravel("F")
    s_rot = -gxx * g.cell_centers[0] + gyy * g.cell_centers[1]
    s_rot = 0
    s_divu = 0


eff_stress = stress @ u + stress_rotation @ rot + stress_volumetric_strain @ divu

eff_rot = rotation_displacement @ u + rotation_diffusion @ rot
eff_divu = mass_displacement @ u + mass_volumetric_strain @ divu

div_scalar = pp.fvutils.scalar_divergence(g)
div_vector = pp.fvutils.vector_divergence(g)

stress_residual = div_vector @ eff_stress - s_u * np.repeat(g.cell_volumes, g.dim)
rotation_residual = div_scalar @ eff_rot - (rot + s_rot) * g.cell_volumes
divu_residual = div_scalar @ eff_divu - (divu + s_divu) * g.cell_volumes


if False:
    assert np.allclose(stress_residual[internal_vector_dofs], 0)
    assert np.allclose(rotation_residual[internal_dofs], 0)
    assert np.allclose(divu_residual[internal_dofs], 0)

debug = []
