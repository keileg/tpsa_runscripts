"""Script to run and plot convergence analysis for 3D problems.

Disclaimer: While this script is provided as a starting point for running and plotting,
it has not been fully cleaned up and documented and will therefore likely contain
confusing and superfluous code, not to mention misleading comments. In particular, an
earlier version of this work contained a model for Cosserat elasticity, which is no
longer included in the code. All references to Cosserat elasticity below should be
ignored.

"""

# The following lines are necessary to avoid conflicts between OpenMP and MKL, which
# can cause the code to run slower than expected. If you know this is not an issue on
# your system, you can remove these lines.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


import model_3d

# import solution_poromechanics
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from collections import namedtuple
import porepy as pp
import pickle
import dataclasses

import matplotlib

# matplotlib.use("Tkagg")
import matplotlib.pyplot as plt
import numpy as np


# For each of the problems considered, the boolean run_{problem} determines if the
# convergence analysis should be run, and plot_{problem} determines if the results
# should be plotted (assumes that the convergence analysis has been run).
run_elasticity = True
plot_elasticity = True

run_heterogeneous = True
plot_heterogenous = True

run_poromechanics = True
plot_poromechanics = True

# Number of grid levels for the convergence analysis.
refinement_levels = 4


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
                model_class=model_3d.SetupTpsa,
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
            fluid = pp.FluidComponent(compressibility=0.01)
            reference_values = pp.ReferenceVariableValues(pressure=0)
            conv_analysis = ConvergenceAnalysis(
                model_class=model_3d.SetupTpsaPoromechanics,
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
    ax.plot([x_min, x_max], [y_min+border, y_min+border], linestyle="-", color="black", linewidth=1.75)
    ax.plot([x_min, x_max], [y_max-border, y_max-border], linestyle="-", color="black", linewidth=1.75)
    ax.plot([x_min+border/2, x_min+border/2], [y_min, y_max], linestyle="-", color="black", linewidth=1.75)
    ax.plot([x_max-border, x_max-border], [y_min, y_max], linestyle="-", color="black", linewidth=1.75)
    # plt.draw()
    # plt.show()
    #fig.patch.set_edgecolor("black")
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
            "lame_lambdas": [1, 1e2, 1e4, 1e10],
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
                        error_str += f"{val[0] / val[1]**0:.5f}, "
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

        plot_and_save(ax, legend_handles, full_stem, "log$_2$($e_{*}$)")
        print(" *************** ")

        # Also plot primary variables
        fig, ax = plt.subplots()
        legend_handles = []
        for j in range(len(res[i])):  # Loop over lambda
            displacement_error = []

            for k in range(len(res[i][j])):
                # Displacement error, scaled by reference value
                displacement_error.append(
                    res[i][j][k]["displacement"][0] / res[i][j][-1]["displacement"][1]
                )

            if params["lame_lambdas"][j] == 1:
                l_val = "1"
            else:
                if params["lame_lambdas"][j] > 1e18:
                    l_val = f"$\infty$"
                elif params["lame_lambdas"][j] > 1e9:
                    l_val = r"$10^{10}$"
                else:
                    l_val = f"1e{int(np.log10(params['lame_lambdas'][j]))}"

            tmp = ax.plot(
                -np.log2(cell_volumes),
                np.log2(displacement_error),
                marker=markers[j],
                color=colors[j],
                label=f"$\lambda$: {l_val}",
                linestyle="-",
            )
            legend_handles.append(tmp[0])

        plot_and_save(ax, [], "primary_variables_" + full_stem, "log$_2$($e$)")


##### Heterogeneous section

heterogeneous_filename_stem = "heterogeneous_3d"

grid_types = ["cartesian"]
grid_types = ["cartesian", "cartesian", "cartesian", "simplex"]
perturbations = [0.0, 0.3, 0.3, 0]
h2_perturbations = [False, False, True, False]
circumcenter = [False, False, False, True]
extrusion = [False, False, False, True]

if run_heterogeneous:
    print("Running heterogeneous elasticity convergence")
    for i in range(len(grid_types)):
        params = {
            "grid_type": grid_types[i],
            "refinement_levels": refinement_levels,
            "cosserat_parameters": [0],
            "lame_lambdas": [1],#, 1e2, 1e4, 1e10],
            "heterogeneity": [1e-4, 1, 1e4],
            "perturbation": perturbations[i],
            "h2_perturbation": h2_perturbations[i],
            "nd": 3,
            "analytical_solution": "heterogeneous_lame",
            "use_circumcenter": circumcenter[i],
            "prismatic_extrusion": extrusion[i],
        }
        elasticity_results = run_convergence_analysis(**params)
        filename = (
            f"{heterogeneous_filename_stem}_{grid_types[i]}_pert_{perturbations[i]}_h2_{h2_perturbations[i]}_circumcenter_{circumcenter[i]}_extrusion_{extrusion[i]}".replace(
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

if plot_heterogenous:
    print("Plotting heterogeneous elasticity convergence")
    for grid_ind in range(len(grid_types)):
        full_stem = f"{heterogeneous_filename_stem}_{grid_types[grid_ind]}_pert_{perturbations[grid_ind]}_h2_{h2_perturbations[grid_ind]}_circumcenter_{circumcenter[grid_ind]}_extrusion_{extrusion[grid_ind]}".replace(
            ".", "-"
        )
        filename = f"{full_stem}.pkl"
        with open(filename, "rb") as f:
            res, params = pickle.load(f)
        j = 0  # There is only one Lame parameter

        colors = ["orange", "blue", "green", "red"]
        markers = ["o", "s", "D", "v"]
        fig, ax = plt.subplots()
        legend_handles = []
        for i in range(len(res)):  # Loop over lambda
            print(f"Heterogeneity: {params['heterogeneity'][i]}")
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
                        "total_pressure",
                        "displacement_stress",                        
                        "stress",
                        "rotation",
                    ]:
                        if key != "stress":
                            error += val[0]
                            error_this_level.append(val[0])
                        error_str += f"{val[0] / val[1]**0:.5f}, "
                        key_list.append(key)
                # print(error)
                if k == 0:
                    print(key_list)
                print(error_str)
                all_errors.append(error / ref_val)
                cell_volumes.append(res[i][j][k]["cell_diameter"])
                error_all_levels.append(error_this_level)

                if params['heterogeneity'][i] == 1:
                    l_val = "1"
                elif params['heterogeneity'][i] < 1e-1:
                    l_val = r"$10^{-4}$"
                else:
                    l_val = f"$10^{int(np.log10(params['heterogeneity'][i]))}$"

            tmp = ax.plot(
                -np.log2(cell_volumes),
                np.log2(all_errors),
                marker=markers[i],
                color=colors[i],
                label=r"$\beta$: " + l_val,
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

        plot_and_save(ax, legend_handles, full_stem, "log$_2$($e_{*}$)")
        print(" *************** ")

        # Also plot primary variables
        fig, ax = plt.subplots()
        legend_handles = []
        for i in range(len(res)):  # Loop over lambda
            displacement_error = []

            for k in range(len(res[i][j])):
                displacement_error.append(
                    res[i][j][k]["displacement"][0] / res[i][j][-1]["displacement"][1]
                )

            if params['heterogeneity'][i] == 1:
                l_val = "1"
            elif params['heterogeneity'][i] < 1e-1:
                l_val = r"$10^{-4}$"
            else:
                l_val = f"$10^{int(np.log10(params['heterogeneity'][i]))}$"

            tmp = ax.plot(
                -np.log2(cell_volumes),
                np.log2(displacement_error),
                marker=markers[i],
                color=colors[i],
                label=r"$\beta$ :" + l_val,
                linestyle="-",
            )
            legend_handles.append(tmp[0])

        plot_and_save(ax, [], "primary_variables_" + full_stem, "log$_2$($e$)")


###### Poromechanics section

poromech_filename_stem = "poromechanics_3d"

grid_types = ["cartesian"]
grid_types = ["cartesian", "cartesian", "cartesian", "simplex"]
perturbations = [0.0, 0.3, 0.3, 0]
h2_perturbations = [False, False, True, False]
circumcenter = [False, False, False, True]
extrusion = [False, False, False, True]
cosserat_parameters = [0]
lame_lambdas = [1]
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
            "permeabilities": [1, 1e-2, 1e-4, 0],
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
                    "darcy_flux"
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
                        'displacement_stress',
                    ]:
                        if key != "stress":
                            error += val[0]
                            errors_this_level.append(val[0])
                            error_str += f"{val[0] / val[1]**0:.5f}, "
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

        plot_and_save(ax, legend_handles, full_stem, "log$_2$($e_{\circ}$)")

        # Also plot primary variables
        fig, ax = plt.subplots()
        legend_handles = []
        i = 0
        for j in range(len(res[i])):  # Loop over Cosserat
            displacement_error = []
            rotation_error = []
            pressure_error = []

            for k in range(len(res[i][j])):
                displacement_error.append(
                    res[i][j][k]["displacement"][0] / res[i][j][-1]["displacement"][1]
                )
                pressure_error.append(
                    res[i][j][k]["pressure"][0] / res[i][j][-1]["pressure"][1]
                )

            if j == 0:
                t1 = ax.plot(
                    -np.log2(cell_volumes[:0]),
                    np.log2(displacement_error[:0]),
                    marker=markers[j],
                    color="k",
                    label="$u$",
                    linestyle="-",
                )
                t3 = ax.plot(
                    -np.log2(cell_volumes[:0]),
                    np.log2(pressure_error[:0]),
                    marker=markers[j],
                    color="k",
                    label="$w$",
                    linestyle=":",
                )
                legend_handles = t1 + t3

            ax.plot(
                -np.log2(cell_volumes),
                np.log2(displacement_error),
                marker=markers[j],
                color=colors[j],
                linestyle="-",
            )
            ax.plot(
                -np.log2(cell_volumes),
                np.log2(pressure_error),
                marker=markers[j],
                color=colors[j],
                linestyle=":",
            )

        plot_and_save(
            ax, legend_handles, "primary_variables_" + full_stem, "log$_2$($e$)"
        )
