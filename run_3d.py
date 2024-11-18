import cosserat_model_3d
#import solution_poromechanics
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from collections import namedtuple
import porepy as pp
import pickle
import dataclasses

import matplotlib

#matplotlib.use("Tkagg")
import matplotlib.pyplot as plt
import numpy as np



def run_convergence_analysis(
    grid_type: str,
    refinement_levels: int,
    cosserat_parameters: list[float],
    lame_lambdas: list[float],
    analytical_solution: str,
    use_cosserat: bool = False,
    perturbation: float = 0.0,
    h2_perturbation: bool = False,
    nd: int = 2,
    use_circumcenter = True,
    prismatic_extrusion = False,
):

    all_results = []

    print(' ')
    print(f' {grid_type}')
    if grid_type == 'cartesian':
        refinement_levels += 1    

    for loc_cos in cosserat_parameters:
        cos_results = []
        for lambda_param in lame_lambdas:
            solid = pp.SolidConstants(lame_lambda= lambda_param,
            biot_coefficient=0)

            conv_analysis = ConvergenceAnalysis(
                model_class=cosserat_model_3d.SetupTpsa,
                model_params={
                    "grid_type": grid_type,
                    "meshing_arguments": {"cell_size": 0.25},
                    "perturbation": perturbation,
                    "h2_perturbation": h2_perturbation,
                    "heterogeneity": 1.0,
                    "cosserat_parameter": loc_cos,
                    "material_constants": {"solid": solid},
                    "nd": 3,
                    "analytical_solution":analytical_solution,
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
    use_circumcenter = True,
    prismatic_extrusion = False,
    analytical_solution: str = "poromechanics",
):

    all_results = []
    # We assume that lambda is fixed to a single value
    assert len(lame_lambdas) == 1
    lambda_param = lame_lambdas[0]

    if grid_type == 'cartesian':
        refinement_levels += 1

    for cos_param in cosserat_parameters:
        cos_results = []

        for perm in permeabilities:
            solid = pp.SolidConstants({"lame_lambda": lambda_param,
                                      "permeability": perm,
                                      "biot_coefficient": 1})
            fluid = pp.FluidConstants({"pressure": 0,
                                        "compressibility": 0.01})
            conv_analysis = ConvergenceAnalysis(
                model_class=cosserat_model_3d.SetupTpsaPoromechanics,
                model_params={
                    "grid_type": grid_type,
                    "meshing_arguments": {"cell_size": 0.25},
                    "perturbation": perturbation,
                    "h2_perturbation": h2_perturbation,
                    "heterogeneity": 1.0,
                    "cosserat_parameter": cos_param,
                    "material_constants": {"solid": solid, "fluid": fluid},
                    "nd": 3,
                    "analytical_solution":analytical_solution,
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

def plot_and_save(ax,legend_handles, file_name, y_label):

    ax.set_xlabel(r"log$_2$($\delta^{-1}$)", fontsize=fontsize_label)
    ax.set_ylabel(y_label, fontsize=fontsize_label)

    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)

    xt = matplotlib.ticker.MaxNLocator(integer=True)
    yt = matplotlib.ticker.MaxNLocator(integer=True)
    ax.xaxis.set_major_locator(xt)
    ax.yaxis.set_major_locator(yt)

    x_vals = ax.get_xlim()
    y_vals = ax.get_ylim()

    xtick_diff = np.diff(ax.get_xticks())[0]
    ytick_diff = np.diff(ax.get_yticks())[0]
    num_minor_bins = 4

    dx_bins = num_minor_bins / xtick_diff
    dy_bins = num_minor_bins / ytick_diff

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

    #_add_convergence_lines(ax, fontsize_label, fontsize_ticks)

    ax.grid(which='major', linewidth=1.5)
    ax.grid(which='minor', linewidth=0.75)
    if len(legend_handles) > 0:
        ax.legend(handles=legend_handles,
            fontsize=fontsize_legend
        )
    # plt.draw()
    #plt.show()
    plt.savefig(f"{file_name}.png", bbox_inches="tight", pad_inches=0)    


run_elasticity = True
plot_elasticity = True

run_cosserat = False
plot_cosserat = False

run_poromechanics = False
plot_poromechanics = False

fontsize_label = 20
fontsize_ticks = 18
fontsize_legend = 16

refinement_levels = 2

elasticity_filename_stem = "elasticity_3d"

grid_types = ["cartesian", "cartesian", "cartesian", "simplex"]
grid_types = ['cartesian']
perturbations = [0.0, 0.3, 0.3, 0]
h2_perturbations = [False, False, True, False]
circumcenter = [False, False, False, True]
extrusion = [False, False, False, True]

if run_elasticity:
    print('Running elasticity convergence')
    for i in range(len(grid_types)):
        params = {
            "grid_type": grid_types[i],
            "refinement_levels": refinement_levels,
            "cosserat_parameters": [0],
            "lame_lambdas": [1],
            "perturbation": perturbations[i],
            "h2_perturbation": h2_perturbations[i],
            "nd": 3,
            "analytical_solution": "homogeneous",
            "use_circumcenter": circumcenter[i],
            "prismatic_extrusion": extrusion[i],            
        }
        elasticity_results = run_convergence_analysis(**params)
        filename = f"{elasticity_filename_stem}_{grid_types[i]}_pert_{perturbations[i]}_h2_{h2_perturbations[i]}_circumcenter_{circumcenter[i]}_extrusion_{extrusion[i]}".replace('.','-') + ".pkl"

        for m in range(len(elasticity_results)):
            for j in range(len(elasticity_results[m])):
                for k in range(len(elasticity_results[m][j])):
                    elasticity_results[m][j][k] = dataclasses.asdict(elasticity_results[m][j][k])

        with open(filename, "wb") as f:
            pickle.dump([elasticity_results, params], f)

if plot_elasticity:
    print("Plotting elasticity convergence")
    for grid_ind in range(len(grid_types)):
        full_stem = f"{elasticity_filename_stem}_{grid_types[grid_ind]}_pert_{perturbations[grid_ind]}_h2_{h2_perturbations[grid_ind]}_circumcenter_{circumcenter[grid_ind]}_extrusion_{extrusion[grid_ind]}".replace('.','-')
        filename = f"{full_stem}.pkl"
        with open(filename, "rb") as f:
            res, params = pickle.load(f)
        i = 0  # There is only one cosserat parameter

        colors = ['orange', 'blue', 'green', 'red']
        markers = ['o', 's', 'D', 'X']
        fig, ax = plt.subplots()
        legend_handles = []
        for j in range(len(res[i])):  # Loop over lambda
            print(f"lambda: {params['lame_lambdas'][j]}")
            all_errors = []
            cell_volumes = []

            error_all_levels = []

            for k in range(len(res[i][j])):
                error = 0
                ref_val = 0
                error_str = ''
                key_list = []
                error_this_level = []
                for key, val in res[i][j][k].items():
                    if key in ["displacement", "total_pressure", "stress", "rotation", 'total_rotation']:
                        error += val[0]
                        ref_val += val[1]
                        error_str += f"{val[0] / val[1]**0:.5f}, "
                        key_list.append(key)
                        error_this_level.append(val[0])
                #print(error)
                if k == 0:
                    print(key_list)
                print(error_str)
                all_errors.append(error / ref_val)
                cell_volumes.append(res[i][j][k]['cell_diameter'])
                error_all_levels.append(error_this_level)

            if params['lame_lambdas'][j] == 1:
                l_val = '1'
            else:
                if params['lame_lambdas'][j] > 1e18:
                    l_val = f"$\infty$"
                else:    
                    l_val = f"1e{int(np.log10(params['lame_lambdas'][j]))}"

            tmp = ax.plot(-np.log2(cell_volumes), np.log2(all_errors), marker=markers[j],
                     color=colors[j], label=f"$\lambda$: {l_val}")
            legend_handles.append(tmp[0])

            arr = np.clip(np.array(error_all_levels), a_min=1e-10, a_max=None)
            print('Log error')
            for row in range(arr.shape[0] - 1):
                print(f"{np.log2(arr[row] / arr[row + 1])}")
            print(' ')
            arr = np.asarray(all_errors)
            print(np.log2(arr[:-1] / arr[1:]))
            print('')
        
        plot_and_save(ax, legend_handles, full_stem, "log$_2$($e_{*}$)")
        print(' *************** ')

        # Also plot primary variables
        fig, ax = plt.subplots()
        legend_handles = []
        for j in range(len(res[i])):  # Loop over lambda
            displacement_error = []

            for k in range(len(res[i][j])):
                displacement_error.append(res[i][j][k]['displacement'][0]/ res[i][j][k]['displacement'][1])

            if params['lame_lambdas'][j] == 1:
                l_val = '1'
            else:
                if params['lame_lambdas'][j] > 1e18:
                    l_val = f"$\infty$"
                else:    
                    l_val = f"1e{int(np.log10(params['lame_lambdas'][j]))}"

            tmp = ax.plot(-np.log2(cell_volumes), np.log2(displacement_error), marker=markers[j],
                     color=colors[j], label=f"$\lambda$: {l_val}", linestyle='-')
            legend_handles.append(tmp[0])
        
        plot_and_save(ax, [], "primary_variables_" + full_stem, "log$_2$($e$)")




##### Cosserat section

cosserat_filename_stem = "cosserat_3d"

grid_types = ["cartesian", "cartesian", "cartesian", "simplex"]
perturbations = [0.0, 0.3, 0.3, 0]
h2_perturbations = [False, False, True, False]
circumcenter = [False, False, False, True]
extrusion = [False, False, False, True]
lame_lambdas = [1]
if run_cosserat:
    #
    for i in range(len(grid_types)):
        params = {
            "grid_type": grid_types[i],
            "refinement_levels": refinement_levels,
            "cosserat_parameters": [1, 1e-4, 1e-8, 0],
            "lame_lambdas": lame_lambdas,
            "perturbation": perturbations[i],
            "h2_perturbation": h2_perturbations[i],
            "nd": 3,
            "analytical_solution": "cosserat",
            "use_circumcenter": circumcenter[i],            
            "prismatic_extrusion": extrusion[i],
        }
        cosserat_results = run_convergence_analysis(**params)
        filename = f"{cosserat_filename_stem}_{grid_types[i]}_pert_{perturbations[i]}_h2_{h2_perturbations[i]}_circumcenter_{circumcenter[i]}_extrusion_{extrusion[i]}.pkl"

        for i in range(len(cosserat_results)):
            for j in range(len(cosserat_results[i])):
                for k in range(len(cosserat_results[i][j])):
                    cosserat_results[i][j][k] = dataclasses.asdict(cosserat_results[i][j][k])

        with open(filename, "wb") as f:
            pickle.dump([cosserat_results, params], f)

if plot_cosserat:
    for grid_ind in range(len(grid_types)):
        full_stem = f"{cosserat_filename_stem}_{grid_types[grid_ind]}_pert_{perturbations[grid_ind]}_h2_{h2_perturbations[grid_ind]}_circumcenter_{circumcenter[grid_ind]}_extrusion_{extrusion[grid_ind]}"
        filename = f"{full_stem}.pkl"
        with open(filename, "rb") as f:
            res, params = pickle.load(f)

        colors = ['orange', 'blue', 'green', 'red']
        markers = ['o', 's', 'D', 'X']
        to_plot = []
        fig, ax = plt.subplots()
        legend_handles = []
        j = 0 # There is only one lambda
        for i in range(len(res)):  # Loop over cosserat
            print(f"cosserat: {params['cosserat_parameters'][i]}")
            all_errors = []
            cell_volumes = []
            error_all_levels = []
            for k in range(0, len(res[i][j])):
                error = 0
                ref_val = 0
                error_str = ''
                key_list = []
                errors_this_level = []
                for key, val in res[i][j][k].items():
                    if key in ["displacement", 'rotation', "total_pressure", "stress", 'total_rotation']:
                        error += val[0]
                        ref_val += val[1]
                        try: 
                            error_str += f"{val[0] / val[1]**1:.5f}, "
                        except ValueError:
                            error_str += 'NaN, '
                        key_list.append(key)
                        errors_this_level.append(val[0]/val[1])
                #print(error)
                if k == 0:
                    print(key_list)
                print(error_str)
                all_errors.append(error / ref_val)
                cell_volumes.append(res[i][j][k]['cell_diameter'])
                error_all_levels.append(errors_this_level)

            if params['cosserat_parameters'][i] == 1:
                l_val = '1'
            else:
                if params['cosserat_parameters'][i] < 1e-10:
                    l_val = '0'
                else:
                    l_val = f"1e{int(np.log10(np.sqrt(params['cosserat_parameters'][i])))}"

            tmp = ax.plot(-np.log2(cell_volumes), np.log2(all_errors), marker=markers[i],
                     color=colors[i], label=f"$\ell$: {l_val}")
            legend_handles += tmp

            arr = np.clip(np.array(error_all_levels), a_min=1e-10, a_max=None)
            print('Log error')
            for row in range(arr.shape[0] - 1):
                print(f"{np.log2(arr[row] / arr[row + 1])}")

            print("")
            arr = np.asarray(all_errors)
            print(np.log2(arr[:-1] / arr[1:]))
            print(' ')

        print(' *************** ')

        plot_and_save(ax, legend_handles, full_stem, "log$_2$($e_{*}$)")

        # Also plot primary variables
        fig, ax = plt.subplots()
        legend_handles = []
        
        for i in range(len(res)):  # Loop over Cosserat
            displacement_error = []
            rotation_error = []

            for k in range(len(res[i][j])):
                displacement_error.append(res[i][j][k]['displacement'][0]/ res[i][j][k]['displacement'][1])
                rotation_error.append(res[i][j][k]['rotation'][0] / res[i][j][k]['rotation'][1])

            if params['cosserat_parameters'][i] == 1:
                l_val = '1'
            else:
                if params['cosserat_parameters'][i] < 1e-10:
                    l_val = '0'
                else:
                    l_val = f"1e{int(np.log10(np.sqrt(params['cosserat_parameters'][i])))}"

            if i == 0:
                t1 = ax.plot(-np.log2(cell_volumes[:0]), np.log2(displacement_error[:0]), marker=markers[j],
                     color='k', label="$u$", linestyle='-')
                t2 = ax.plot(-np.log2(cell_volumes[:0]), np.log2(rotation_error[:0]), marker=markers[j],
                         color='k', label="$r$", linestyle='-.')                     
                legend_handles = t1 + t2
            
            ax.plot(-np.log2(cell_volumes), np.log2(displacement_error), marker=markers[j],
                    color=colors[i],  linestyle='-')
            ax.plot(-np.log2(cell_volumes), np.log2(rotation_error), marker=markers[j],
                        color=colors[i], linestyle='-.')                                     
        
        plot_and_save(ax, legend_handles, "primary_variables_" + full_stem, "log$_2$($e$)")        


###### Poromechanics section

poromech_filename_stem = "poromechanics_3d"

grid_types = ['cartesian', 'cartesian', 'cartesian']
grid_types = ["cartesian", "cartesian", "cartesian", "simplex"]
perturbations = [0.0, 0.3, 0.3, 0]
h2_perturbations = [False, False, True, False]
circumcenter = [False, False, False, True]
extrusion = [False, False, False, True]
cosserat_parameters = [0]
lame_lambdas = [1]
if run_poromechanics:
    #
    for i in range(2, len(grid_types)):
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
                    cosserat_results[i][j][k] = dataclasses.asdict(cosserat_results[i][j][k])

        with open(filename, "wb") as f:
            pickle.dump([cosserat_results, params], f)

if plot_poromechanics:
    for grid_ind in range(len(grid_types)):
        full_stem = f"{poromech_filename_stem}_{grid_types[grid_ind]}_pert_{perturbations[grid_ind]}_h2_{h2_perturbations[grid_ind]}_circumcenter_{circumcenter[grid_ind]}_extrusion_{extrusion[grid_ind]}"
        filename = f"{full_stem}.pkl"
        with open(filename, "rb") as f:
            res, params = pickle.load(f)

        colors = ['orange', 'blue', 'green', 'red']
        markers = ['o', 's', 'D', 'X']
        to_plot = []
        fig, ax = plt.subplots()
        legend_handles = []
        i = 0 # There is only one Cosserat
        for j in range(len(res[i])):  # Loop over cosserat
            print(f"Permeability: {params['permeabilities'][j]}")
            all_errors = []
            cell_volumes = []
            error_all_levels = []
            for k in range(0, len(res[i][j])):
                error = 0
                ref_val = 0
                error_str = ''
                key_list = []
                errors_this_level = []
                for key, val in res[i][j][k].items():
                    if key in ["displacement", 'rotation', "pressure", "total_pressure", "stress", 'total_rotation', "darcy_flux"]:
                        error += val[0]
                        ref_val += val[1]
                        error_str += f"{val[0] / val[1]**0:.5f}, "
                        key_list.append(key)
                        errors_this_level.append(val[0])
                #print(error)
                if k == 0:
                    print(key_list)
                print(error_str)
                all_errors.append(error / ref_val)
                cell_volumes.append(res[i][j][k]['cell_diameter'])
                error_all_levels.append(errors_this_level)

            if params['permeabilities'][j] == 1:
                l_val = '1'
            elif params['permeabilities'][j] < 1e-8:
                l_val = '0'
            else:
                l_val = f"1e{int(np.log10(params['permeabilities'][j]))}"

            tmp = ax.plot(-np.log2(cell_volumes), np.log2(all_errors), marker=markers[j],
                     color=colors[j], label=f"$\kappa$: {l_val}")
            legend_handles += tmp

            arr = np.clip(np.array(error_all_levels), a_min=1e-10, a_max=None)
            print('Log error')
            for row in range(arr.shape[0] - 1):
                print(f"{np.log2(arr[row] / arr[row + 1])}")
            print("")
            arr = np.asarray(all_errors)
            print(np.log2(arr[:-1] / arr[1:]))
            print(' ')

        print(' *************** ')

        plot_and_save(ax, legend_handles, full_stem, "log$_2$($e_{\circ}$)")

        # Also plot primary variables
        fig, ax = plt.subplots()
        legend_handles = []
        i=0
        for j in range(len(res[i])):  # Loop over Cosserat
            displacement_error = []
            rotation_error = []
            pressure_error = []

            for k in range(len(res[i][j])):
                displacement_error.append(res[i][j][k]['displacement'][0]/res[i][j][k]['displacement'][1])
                pressure_error.append(res[i][j][k]['pressure'][0]/res[i][j][k]['pressure'][1])

            if j == 0:
                t1 = ax.plot(-np.log2(cell_volumes[:0]), np.log2(displacement_error[:0]), marker=markers[j],
                     color='k', label="$u$", linestyle='-')
                t3 = ax.plot(-np.log2(cell_volumes[:0]), np.log2(pressure_error[:0]), marker=markers[j],
                         color='k', label="$w$", linestyle=':')                                              
                legend_handles = t1 + t3
            
            ax.plot(-np.log2(cell_volumes), np.log2(displacement_error), marker=markers[j],
                    color=colors[j],  linestyle='-')
            ax.plot(-np.log2(cell_volumes), np.log2(pressure_error), marker=markers[j],
                        color=colors[j], linestyle=':')                  

        plot_and_save(ax, legend_handles, "primary_variables_" + full_stem, "log$_2$($e$)")  


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

                to_plot.append(-np.log2(cell_volumes), np.log2(all_errors))
                ax.plot(-np.log2(cell_volumes), np.log2(all_errors), "-s")

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


