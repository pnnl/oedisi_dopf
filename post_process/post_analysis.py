import os, sys
import plotter

case = sys.argv[1]

if case == "ieee123" or case == "omoo":
    root = "150"
    seperator = "    "
    cordsys = "2D"
    annotate=True
elif case == "small" or case == "omoo_small":
    root = "P1UDT942-P1UHS0_1247X"
    seperator = " "
    cordsys = "GEO"
    annotate=False
elif case == "medium" or case == "omoo_medium":
    root = "P6UDT5293-P6UHS10_1247X"
    seperator = " "
    cordsys = "GEO"
    annotate=False


if case.find("omoo") != -1:
    sfx = "(OMOO Algorithm)"
else:
    sfx = "(LinDistFlow Algorithm)"

directory = f"../outputs/{case}"
topology_filepath = os.path.join(directory, "topology.json")
buscoord_filepath = os.path.join(directory, "BusCoords.dat")
real_voltage_filepath = os.path.join(directory, "voltage_real.feather")
imag_voltage_filepath = os.path.join(directory, "voltage_imag.feather")

time = [30,60,90]
fig_filename = os.path.join(directory, f"network_{case}.png")
plotter.plot_network(
    topology_filepath,
    buscoord_filepath, 
    real_voltage_filepath, 
    imag_voltage_filepath, 
    root_node=root, sep=seperator,
    time=time, node_size=50, 
    show=False, to_file=fig_filename, 
    suptitle_sfx = sfx, 
    )



fig_filename = os.path.join(directory, f"vtree_{case}.png")
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
    suptitle_sfx = sfx, 
    coordsys=cordsys,
    )


time = [75]
fig_filename = os.path.join(directory, f"network_peak_{case}.png")
plotter.plot_network(
    topology_filepath,
    buscoord_filepath, 
    real_voltage_filepath, 
    imag_voltage_filepath, 
    root_node=root, sep=seperator,
    time=time, node_size=200, 
    show=False, to_file=fig_filename, 
    suptitle_sfx = sfx, 
    figsize=(20,20)
    )



fig_filename = os.path.join(directory, f"vtree_peak_{case}.png")
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
    figsize=(20,15),
    annotate=annotate,
    suptitle_sfx = sfx, 
    coordsys=cordsys,
    )