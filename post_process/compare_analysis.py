import os, sys
import plotter


def plot_result_case(
    casename,
    alg,
    time=["07:30", "12:30", "15:30"],
    coordsys="2D",
    root="150",
    seperator="    ",
    annotate=True,
    ymin=0.98,
    ymax=1.06,
):
    # Directory location
    directory = f"../outputs/{casename}"
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
        root_node=root,
        sep=seperator,
        time=time,
        node_size=50,
        show=False,
        to_file=fig_filename,
        suptitle_sfx=f"({alg} Algorithm)",
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
        show=False,
        to_file=fig_filename,
        lw=2.5,
        ls="dashed",
        figsize=(60, 15),
        annotate=annotate,
        suptitle_sfx=f"({alg} Algorithm)",
        coordsys=coordsys,
        ymin=ymin,
        ymax=ymax,
    )
    return


def plot_result_set(
    casename1,
    casename2,
    alg1,
    alg2,
    time=["07:30", "12:30", "15:30"],
    coordsys="2D",
    root="150",
    seperator="    ",
    annotate=True,
    ymin=0.98,
    ymax=1.06,
):
    plot_result_case(
        casename1,
        alg1,
        time=time,
        coordsys=coordsys,
        root=root,
        seperator=seperator,
        annotate=annotate,
        ymin=ymin,
        ymax=ymax,
    )
    plot_result_case(
        casename2,
        alg2,
        time=time,
        coordsys=coordsys,
        root=root,
        seperator=seperator,
        annotate=annotate,
        ymin=ymin,
        ymax=ymax,
    )
    return


def plot_result_comparison(
    casename1,
    casename2,
    time,
    coordsys="2D",
    root="150",
    ymin=0.98,
    ymax=1.06,
    seperator="    ",
    annotate=True,
):
    dir1 = f"../outputs/{casename1}"
    topofile1 = os.path.join(dir1, "topology.json")
    buscordfile1 = os.path.join(dir1, "BusCoords.dat")
    realVfile1 = os.path.join(dir1, "voltage_real.feather")
    imagVfile1 = os.path.join(dir1, "voltage_imag.feather")

    dir2 = f"../outputs/{casename2}"
    topofile2 = os.path.join(dir2, "topology.json")
    buscordfile2 = os.path.join(dir2, "BusCoords.dat")
    realVfile2 = os.path.join(dir2, "voltage_real.feather")
    imagVfile2 = os.path.join(dir2, "voltage_imag.feather")

    fig_filename1 = os.path.join(dir1, f"vtree_peak_{casename1}.png")
    fig_filename2 = os.path.join(dir1, f"network_peak_{casename1}.png")

    plotter.compare_vtree(
        realVfile1,
        realVfile2,
        imagVfile1,
        imagVfile2,
        topofile1,
        topofile2,
        buscordfile1,
        buscordfile2,
        alg1="LinDistFlow",
        alg2="OMOO",
        sep=seperator,
        root_node=root,
        coordsys=coordsys,
        time=time,
        to_file=fig_filename1,
        show=False,
        do_return=False,
        lw=2.5,
        ls="dashed",
        figsize=(40, 18),
        annotate=annotate,
        ymin=ymin,
        ymax=ymax,
    )

    plotter.compare_network(
        realVfile1,
        realVfile2,
        imagVfile1,
        imagVfile2,
        topofile1,
        topofile2,
        buscordfile1,
        buscordfile2,
        alg1="LinDistFlow",
        alg2="OMOO",
        sep=seperator,
        root_node=root,
        coordsys=coordsys,
        time=time,
        to_file=fig_filename2,
        show=False,
        do_return=False,
        vmin=1.0,
        vmax=1.05,
    )

    return


if __name__ == "__main__":
    case = sys.argv[1]

    if case == "ieee123":
        root = "150"
        seperator = "    "
        cordsys = "2D"
        annotate = True
        case1 = "ieee123"
        case2 = "omoo"
        ymin = 0.98
        ymax = 1.06

    elif case == "small":
        root = "P1UDT942-P1UHS0_1247X"
        seperator = " "
        cordsys = "GEO"
        annotate = False
        case1 = "small"
        case2 = "omoo_small"
        ymin = 1.02
        ymax = 1.035

    elif case == "medium":
        root = "P6UDT5293-P6UHS10_1247X"
        seperator = " "
        cordsys = "GEO"
        annotate = False
        case1 = "medium"
        case2 = "omoo_medium"
        ymin = 1.01
        ymax = 1.05

    else:
        print("Unknown case")
        sys.exit(0)

    plot_result_set(
        case1,
        case2,
        alg1="LinDistFlow",
        alg2="OMOO",
        time=["07:30", "12:30", "15:30"],
        coordsys=cordsys,
        root=root,
        seperator=seperator,
        annotate=annotate,
        ymin=ymin,
        ymax=ymax,
    )

    plot_result_comparison(
        case1,
        case2,
        time="11:30",
        coordsys=cordsys,
        root=root,
        seperator=seperator,
        annotate=annotate,
        ymin=ymin,
        ymax=ymax,
    )
