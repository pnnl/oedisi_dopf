import os, sys
import plotter

def plot_result_case(
        casename, alg, 
        time = [30,60,90], coordsys="2D", 
        root = "150", seperator = "    ", annotate=True, 
        ymin = 0.98, ymax = 1.06, 
        ):

    # Directory location
    directory = f"./outputs/{casename}"
    topology_filepath = os.path.join(directory, "topology.json")
    buscoord_filepath = os.path.join(directory, "BusCoords.dat")
    real_voltage_filepath = os.path.join(directory, "voltage_real.feather")
    imag_voltage_filepath = os.path.join(directory, "voltage_imag.feather")

    # Voltage heatmaps of time steps
    fig_filename = os.path.join(directory, f"network_{casename}.png")
    plotter.plot_network(
        topology_filepath,
        buscoord_filepath, 
        real_voltage_filepath, 
        imag_voltage_filepath, 
        root_node=root, sep=seperator,
        time=time, node_size=50, 
        show=False, to_file=fig_filename, 
        suptitle_sfx = f"({alg} Algorithm)", 
        )

    # Voltage trees of time steps
    fig_filename = os.path.join(directory, f"vtree_{casename}.png")
    plotter.plot_voltage_tree(
        topology_filepath,
        buscoord_filepath, 
        real_voltage_filepath, 
        imag_voltage_filepath,
        root_node=root, 
        sep=seperator,
        time=time,
        show=False, to_file=fig_filename,
        lw=2.5, ls='dashed', 
        figsize=(60,15),
        annotate=annotate,
        suptitle_sfx = f"({alg} Algorithm)", 
        coordsys=coordsys,
        ymin = ymin, ymax = ymax, 
        )
    return

if __name__ == "__main__":
    
    case = sys.argv[1]
    alg = "LinDistFlow"
    time = [30,60,90]

    if case == "ieee123":
        root = "150"
        seperator = "    "
        cordsys = "2D"
        annotate=True
        ymin = 0.98
        ymax = 1.06 

    elif case == "small":
        root = "P1UDT942-P1UHS0_1247X"
        seperator = " "
        cordsys = "GEO"
        annotate=False
        ymin = 1.02
        ymax = 1.035

    elif case == "medium":
        root = "P6UDT5293-P6UHS10_1247X"
        seperator = " "
        cordsys = "GEO"
        annotate=False
        ymin = 1.01
        ymax = 1.05
    
    else:
        print("Unknown case")
        sys.exit(0)

    plot_result_case(
            case, alg, 
            time=time, coordsys=cordsys, 
            root = root, seperator = seperator, annotate=annotate, 
            ymin = ymin, ymax = ymax, 
            )