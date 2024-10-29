import manu_mech_nofrac_2d
import solution_poromechanics
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from collections import namedtuple
import porepy as pp
import pickle
import dataclasses

import matplotlib

matplotlib.use("Tkagg")
import matplotlib.pyplot as plt
import numpy as np


def run_convergence_analysis(
    grid_type: str,
    refinement_levels: int,
    cosserat_parameters: list[float],
    lame_lambdas: list[float],
    perturbation: float = 0.0,
    h2_perturbation: bool = False,
    nd: int = 2,
):

    all_results = []

    for cos_param in cosserat_parameters:
        cos_results = []
        for lambda_param in lame_lambdas:
            solid = pp.SolidConstants({"lame_lambda": lambda_param})

            conv_analysis = ConvergenceAnalysis(
                model_class=manu_mech_nofrac_2d.Setup,
                model_params={
                    "grid_type": grid_type,
                    "meshing_arguments": {"cell_size": 0.25},
                    "perturbation": perturbation,
                    "h2_perturbation": h2_perturbation,
                    "heterogeneity": 1.0,
                    "cosserat_parameter": cos_param,
                    "material_constants": {"solid": solid},
                    "nd": 2,
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
):

    all_results = []
    # We assume that lambda is fixed to a single value
    assert len(lame_lambdas) == 1
    lambda_param = lame_lambdas[0]

    for cos_param in cosserat_parameters:
        cos_results = []

        for perm in permeabilities:
            solid = pp.SolidConstants({"lame_lambda": lambda_param,
                                      "permeability": perm,
                                      "biot_coefficient": 1})
            fluid = pp.FluidConstants({"pressure": 0,
                                        "compressibility": 0.01})
            conv_analysis = ConvergenceAnalysis(
                model_class=solution_poromechanics.Setup,
                model_params={
                    "grid_type": grid_type,
                    "meshing_arguments": {"cell_size": 0.25},
                    "perturbation": perturbation,
                    "h2_perturbation": h2_perturbation,
                    "heterogeneity": 1.0,
                    "cosserat_parameter": cos_param,
                    "material_constants": {"solid": solid, "fluid": fluid},
                    "nd": 2,
                },
                levels=refinement_levels,
                spatial_refinement_rate=2,
                temporal_refinement_rate=1,
            )

            res = conv_analysis.run_analysis()
            cos_results.append(res)
        all_results.append(cos_results)

    return all_results    

def _add_convergence_lines(ax, fontsize_ticks):
    x_vals = ax.get_xlim()
    dx = x_vals[1] - x_vals[0]
    y_vals = ax.get_ylim()
    dy = y_vals[1] - y_vals[0]
    diff = min(dx, dy)

    x_0 = x_vals[0] + 0.1 * dx
    y_0 = y_vals[0] + 0.1 * dy
    x_1 = x_0 + 0.1 * diff
    y_1 = y_0 + 0.1 * diff
    y_2 = y_0 + 0.2 * diff

    ax.plot([x_0, x_1], [y_1, y_0], color='black')
    ax.plot([x_0, x_1], [y_2 + 0.1 * diff, y_0 + 0.1 * diff], color='black')
    ax.text(x_0 - 0.05 * dx, y_1, "1", fontsize=fontsize_ticks)
    ax.text(x_0 - 0.05 * dx, y_2 + 0.1 * diff, "2", fontsize=fontsize_ticks)    


cosserat_parameters = [1, 1e-2, 1e-4, 1e-6]
lame_lambdas = [1, 1e2, 1e4, 1e6]

cosserat_parameters = [1]
lame_lambdas = [1]

run_elasticity = True
plot_elasticity = True

run_cosserat = False
plot_cosserat = False

run_poromechanics = False
plot_poromechanics = False

refinement_levels = 5

elasticity_filename_stem = "elasticity_2d"

grid_types = ['cartesian']
grid_types = ["cartesian", "cartesian", "cartesian", "simplex"]
perturbations = [0.0, 0.2, 0.2, 0]
h2_perturbations = [False, False, True, False]
if run_elasticity:
    print('Running elasticity convergence')
    for i in range(len(grid_types)):
        params = {
            "grid_type": grid_types[i],
            "refinement_levels": refinement_levels,
            "cosserat_parameters": [0],
            "lame_lambdas": [1, 1e2, 1e4, 1e10],
            "perturbation": perturbations[i],
            "h2_perturbation": h2_perturbations[i],
            "nd": 2,
        }
        elasticity_results = run_convergence_analysis(**params)
        filename = f"{elasticity_filename_stem}_{grid_types[i]}_pert_{perturbations[i]}_h2_{h2_perturbations[i]}".replace('.','-') + ".pkl"

        for m in range(len(elasticity_results)):
            for j in range(len(elasticity_results[m])):
                for k in range(len(elasticity_results[m][j])):
                    elasticity_results[m][j][k] = dataclasses.asdict(elasticity_results[m][j][k])

        with open(filename, "wb") as f:
            pickle.dump([elasticity_results, params], f)

if plot_elasticity:
    print("Plotting elasticity convergence")
    for grid_ind in range(len(grid_types)):
        full_stem = f"{elasticity_filename_stem}_{grid_types[grid_ind]}_pert_{perturbations[grid_ind]}_h2_{h2_perturbations[grid_ind]}".replace('.','-')
        filename = f"{full_stem}.pkl"
        with open(filename, "rb") as f:
            res, params = pickle.load(f)
        i = 0  # There is only one cosserat parameter

        colors = ['orange', 'blue', 'green', 'red']
        markers = ['o', 's', 'D', 'X']
        to_plot = []
        fig, ax = plt.subplots()
        for j in range(len(res[i])):  # Loop over lambda
            print(f"lambda: {params['lame_lambdas'][j]}")
            all_errors = []
            cell_volumes = []
            for k in range(len(res[i][j])):
                error = 0
                ref_val = 0
                error_str = ''
                key_list = []
                for key, val in res[i][j][k].items():
                    if key in ["displacement", "volumetric_strain", "stress", "rotation", 'total_rotation']:
                        error += val[0]
                        ref_val += val[1]
                        error_str += f"{val[0] / val[1]}, "
                        key_list.append(key)
                #print(error)
                if k == 0:
                    print(key_list)
                print(error_str)
                all_errors.append(error / ref_val)
                cell_volumes.append(res[i][j][k]['cell_diameter'])

            if params['lame_lambdas'][j] == 1:
                l_val = '1'
            else:
                if params['lame_lambdas'][j] > 1e8:
                    l_val = f"$\infty$"
                else:    
                    l_val = f"1e{int(np.log10(params['lame_lambdas'][j]))}"

            ax.plot(-np.log(cell_volumes), np.log(all_errors), marker=markers[j],
                     color=colors[j], label=f"$\lambda$: {l_val}")

            print("")

        fontsize_label = 16
        fontsize_ticks = 14
        ax.set_xlabel("-log(cell diameter)", fontsize=fontsize_label)
        ax.set_ylabel("log(error)", fontsize=fontsize_label)
        ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.get_major_locator().set_params(integer=True)

        _add_convergence_lines(ax, fontsize_ticks)

        ax.grid()
        ax.legend()
        # plt.draw()
        plt.show()
        plt.savefig(f"{full_stem}.png", bbox_inches="tight", pad_inches=0)


##### Cosserat section

cosserat_filename_stem = "cosserat_2d"

grid_types = ["cartesian", "cartesian", "cartesian", "simplex"]
perturbations = [0.0, 0.2, 0.2, 0]
h2_perturbations = [False, False, True, False]
cosserat_parameters = [1e-0, 1e-4, 1e-8]
lame_lambdas = [1]
if run_cosserat:
    #
    for i in range(len(grid_types)):
        params = {
            "grid_type": grid_types[i],
            "refinement_levels": refinement_levels,
            "cosserat_parameters": cosserat_parameters,
            "lame_lambdas": lame_lambdas,
            "perturbation": perturbations[i],
            "h2_perturbation": h2_perturbations[i],
            "nd": 2,
        }
        cosserat_results = run_convergence_analysis(**params)
        filename = f"{cosserat_filename_stem}_{grid_types[i]}_pert_{perturbations[i]}_h2_{h2_perturbations[i]}.pkl"

        for i in range(len(cosserat_results)):
            for j in range(len(cosserat_results[i])):
                for k in range(len(cosserat_results[i][j])):
                    cosserat_results[i][j][k] = dataclasses.asdict(cosserat_results[i][j][k])

        with open(filename, "wb") as f:
            pickle.dump([cosserat_results, params], f)

if plot_cosserat:
    for grid_ind in range(len(grid_types)):
        full_stem = f"{cosserat_filename_stem}_{grid_types[grid_ind]}_pert_{perturbations[grid_ind]}_h2_{h2_perturbations[grid_ind]}"
        filename = f"{full_stem}.pkl"
        with open(filename, "rb") as f:
            res, params = pickle.load(f)

        colors = ['orange', 'blue', 'green', 'red']
        markers = ['o', 's', 'D', 'X']
        to_plot = []
        fig, ax = plt.subplots()
        j = 0 # There is only one lambda
        for i in range(len(res)):  # Loop over cosserat
            print(f"cosserat: {params['cosserat_parameters'][i]}")
            all_errors = []
            cell_volumes = []
            for k in range(0, len(res[i][j])):
                error = 0
                ref_val = 0
                error_str = ''
                key_list = []
                for key, val in res[i][j][k].items():
                    if key in ["displacement", 'rotation', "volumetric_strain", "stress", 'total_rotation']:
                        error += val[0]
                        ref_val += val[1]
                        error_str += f"{val[0] / val[1]}, "
                        key_list.append(key)
                #print(error)
                if k == 0:
                    print(key_list)
                print(error_str)
                all_errors.append(error / ref_val)
                cell_volumes.append(res[i][j][k]['cell_diameter'])

            if params['cosserat_parameters'][i] == 1:
                l_val = '1'
            else:
                l_val = f"1e{int(np.log10(np.sqrt(params['cosserat_parameters'][i])))}"

            ax.plot(-np.log(cell_volumes), np.log(all_errors), marker=markers[i],
                     color=colors[i], label=f"$\ell$: {l_val}")

            print("")

        fontsize_label = 16
        fontsize_ticks = 14
        ax.set_xlabel("-log(cell diameter)", fontsize=fontsize_label)
        ax.set_ylabel("log(error)", fontsize=fontsize_label)
        ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.get_major_locator().set_params(integer=True)

        _add_convergence_lines(ax, fontsize_ticks)

        ax.grid()
        ax.legend()
        # plt.draw()
        plt.show()
        plt.savefig(f"{full_stem}.png", bbox_inches="tight", pad_inches=0)


###### Poromechanics section

poromech_filename_stem = "poromechanics_2d"

grid_types = ['cartesian']
grid_types = ["cartesian", "cartesian", "cartesian", "simplex"]
perturbations = [0.0, 0.2, 0.2, 0]
h2_perturbations = [False, False, True, False]
cosserat_parameters = [0]
lame_lambdas = [1]
permeabilities = [1, 1e-2, 1e-4]
if run_poromechanics:
    #
    for i in range(len(grid_types)):
        params = {
            "grid_type": grid_types[i],
            "refinement_levels": refinement_levels,
            "cosserat_parameters": cosserat_parameters,
            "lame_lambdas": lame_lambdas,
            "perturbation": perturbations[i],
            "h2_perturbation": h2_perturbations[i],
            "permeabilities": permeabilities,
            "nd": 2,
        }
        cosserat_results = run_poromech_convergence_analysis(**params)
        filename = f"{poromech_filename_stem}_{grid_types[i]}_pert_{perturbations[i]}_h2_{h2_perturbations[i]}.pkl"

        for i in range(len(cosserat_results)):
            for j in range(len(cosserat_results[i])):
                for k in range(len(cosserat_results[i][j])):
                    cosserat_results[i][j][k] = dataclasses.asdict(cosserat_results[i][j][k])

        with open(filename, "wb") as f:
            pickle.dump([cosserat_results, params], f)

if plot_poromechanics:
    for grid_ind in range(len(grid_types)):
        full_stem = f"{poromech_filename_stem}_{grid_types[grid_ind]}_pert_{perturbations[grid_ind]}_h2_{h2_perturbations[grid_ind]}"
        filename = f"{full_stem}.pkl"
        with open(filename, "rb") as f:
            res, params = pickle.load(f)

        colors = ['orange', 'blue', 'green', 'red']
        markers = ['o', 's', 'D', 'X']
        to_plot = []
        fig, ax = plt.subplots()
        i = 0 # There is only one Cosserat
        for j in range(len(res[i])):  # Loop over cosserat
            print(f"Permeability: {params['permeabilities'][j]}")
            all_errors = []
            cell_volumes = []
            for k in range(0, len(res[i][j])):
                error = 0
                ref_val = 0
                error_str = ''
                key_list = []
                for key, val in res[i][j][k].items():
                    if key in ["displacement", 'rotation', "volumetric_strain", "stress", 'total_rotation', "pressure", "darcy_flux"]:
                        error += val[0]
                        ref_val += val[1]
                        error_str += f"{val[0] / val[1]}, "
                        key_list.append(key)
                #print(error)
                if k == 0:
                    print(key_list)
                print(error_str)
                all_errors.append(error / ref_val)
                cell_volumes.append(res[i][j][k]['cell_diameter'])

            if params['permeabilities'][j] == 1:
                l_val = '1'
            else:
                l_val = f"1e{int(np.log10(params['permeabilities'][j]))}"

            ax.plot(-np.log(cell_volumes), np.log(all_errors), marker=markers[j],
                     color=colors[j], label=f"$\kappa$: {l_val}")

            print("")

        fontsize_label = 16
        fontsize_ticks = 14
        ax.set_xlabel("-log(cell diameter)", fontsize=fontsize_label)
        ax.set_ylabel("log(error)", fontsize=fontsize_label)
        ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.get_major_locator().set_params(integer=True)

        _add_convergence_lines(ax, fontsize_ticks)

        ax.grid()
        ax.legend()
        # plt.draw()
        plt.show()
        plt.savefig(f"{full_stem}.png", bbox_inches="tight", pad_inches=0)


def print_errors_individual_fields(res, fields):
    for i in range(len(res)):
        print(f"cosserat: {cosserat_parameters[i]}")
        for j in range(len(res[i])):
            print(f"lambda: {lame_lambdas[j]}")
            print(" ".join([f for f in fields]))
            for k in range(len(res[i][j])):
                s = []
                for field in fields:
                    s.append(f"{getattr(res[i][j][k], field)} ")
                print(s)
            print("")


def print_error_summed_normed(res, fields):
    for i in range(len(res)):
        print(f"cosserat: {cosserat_parameters[i]}")
        for j in range(len(res[i])):
            print(f"lambda: {lame_lambdas[j]}")
            all_errors = []
            cell_volumes = []
            print("")
            fontsize_label = 16
            fontsize_ticks = 14
            fig, ax = plt.subplots()
            to_plot = []
            for k in range(len(res[i][j])):
                error = 0
                for field in fields:
                    error += getattr(res[i][j][k], field)
                print(error)
                all_errors.append(error)
                cell_volumes.append(res[i][j][k].cell_diameter)

                to_plot.append(-np.log(cell_volumes), np.log(all_errors))
                ax.plot(-np.log(cell_volumes), np.log(all_errors), "-s")

            ax.set_xlabel("-log(cell diameter)", fontsize=fontsize_label)
            ax.set_ylabel("log(error)", fontsize=fontsize_label)
            ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
            ax.grid()
            ax.legend()
            # plt.draw()
            plt.show()
            plt.savefig("tmp.png", bbox_inches="tight", pad_inches=0)
            debug = []


#                     label=f"cosserat: {cosserat_parameters[i]}",
#                      lambda: f"{lame_lambdas[j]}")
# plt.legend()
# plt.yscale('log')
# plt.show()


