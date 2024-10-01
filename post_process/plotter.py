import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from oedisi.types.data_types import Topology
import networkx as nx
import numpy as np
import json
import pyarrow.feather as feather
from geopy.distance import geodesic
from datetime import datetime
from post_process.errors import errors


def plot_voltages(df_voltages, base_voltages, bus=None, time=0, unit="kV", **kwargs):
    # keyword arguments
    label_fontsize = kwargs.get("fontsize", 25)
    ticklabel_fontsize = label_fontsize - 6
    title_fontsize = label_fontsize + 10

    # common bus or common time: if bus=None, then plot for common time
    fig, ax = plt.subplots(figsize=(20, 10))
    if not bus:
        voltages = df_voltages.iloc[time, :] / base_voltages
        xlabel = "Node number"
        suptitle = f"Voltage magnitude of all nodes at time t={time}"
        x_axis = np.arange(voltages.shape[0]) + 1
        ax.bar(x_axis, voltages, color="seagreen", edgecolor="black")
        ax.set_ylim(0.9, 1.1)
    else:
        voltages = df_voltages[bus] / base_voltages[bus]
        xlabel = "Time"
        suptitle = f"Voltage magnitude time series for bus {bus}"
        x_axis = np.arange(voltages.shape[0])
        ax.plot(x_axis, voltages, "-o", color="royalblue")

    # Formatting
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(f"Voltage Magnitudes ({unit})", fontsize=label_fontsize)
    ax.tick_params(axis="x", labelsize=ticklabel_fontsize)
    ax.tick_params(axis="y", labelsize=ticklabel_fontsize)
    fig.suptitle(suptitle, fontsize=title_fontsize)
    return fig


def extract_topology(topology_file, buscoord_file, sep="    "):
    # Extract topology
    with open(topology_file) as f:
        topology = Topology.parse_obj(json.load(f))
        base_voltage_df = pd.DataFrame(
            {
                "id": topology.base_voltage_magnitudes.ids,
                "value": topology.base_voltage_magnitudes.values,
            }
        )
        base_voltage_df.set_index("id", inplace=True)
        base_voltages = base_voltage_df["value"]
        branch_info, bus_info = extract_info(topology)

    # Extract bus coordinates
    try:
        with open(buscoord_file, "r") as cord_file:
            lines = cord_file.readlines()
    except FileExistsError:
        print(
            f"File {buscoord_file} for bus coordinates. Get this file to generate plots!!!"
        )

    cord = {}
    for line in lines:
        temp = line.strip("\n").split(sep)
        cord[temp[0].upper()] = [float(temp[1]), float(temp[2])]

    return branch_info, bus_info, base_voltages, cord


def extract_info(topology: Topology) -> dict:
    # Get the valid phases for each bus
    buses = list(topology.base_voltage_magnitudes.ids)
    bus_info = {}
    for bus in buses:
        [bus_name, bus_phase] = bus.split(".")
        if bus_name not in bus_info:
            bus_info[bus_name] = [bus_phase]
        else:
            bus_info[bus_name].append(bus_phase)

    branch_info = {}
    from_equip = topology.admittance.from_equipment
    to_equip = topology.admittance.to_equipment

    for fr_eq, to_eq in zip(from_equip, to_equip):
        [from_name, _] = fr_eq.split(".")
        type = "LINE"
        if from_name.find("OPEN") != -1:
            [from_name, _] = from_name.split("_")
            type = "SWITCH"

        [to_name, _] = to_eq.split(".")
        if to_name.find("OPEN") != -1:
            [to_name, _] = to_name.split("_")
            type = "SWITCH"

        if from_name == to_name:
            continue

        key = f"{from_name}_{to_name}"
        key_back = f"{to_name}_{from_name}"

        if key not in branch_info and key_back not in branch_info:
            branch_info[key] = {}
            branch_info[key]["fr_bus"] = ""
            branch_info[key]["to_bus"] = ""
        elif key_back in branch_info:
            continue

        branch_info[key]["type"] = type
        branch_info[key]["fr_bus"] = from_name
        branch_info[key]["to_bus"] = to_name

    return branch_info, bus_info


def dist(cordA, cordB):
    return np.linalg.norm(np.array(cordA) - np.array(cordB))


def geodist(cordA, cordB):
    return geodesic(cordA[::-1], cordB[::-1]).miles


def get_network(
    branch_info, bus_info, pos, root_node="150R", open_buses=[], coordsys="2D"
):
    # construct the networkx graph with only three phase transformers
    network = nx.Graph()
    for brn in branch_info:
        fbus = branch_info[brn]["fr_bus"]
        tbus = branch_info[brn]["to_bus"]
        fbus_phases = bus_info[fbus]
        tbus_phases = bus_info[tbus]
        common_phases = list(set(fbus_phases).intersection(set(tbus_phases)))

        # add switches if they are NOT OPEN
        if branch_info[brn]["type"] == "SWITCH":
            for phase in common_phases:
                u = ".".join([fbus, phase])
                v = ".".join([tbus, phase])
                if (f"{fbus}_OPEN.{phase}" not in open_buses) and (
                    f"{tbus}_OPEN.{phase}" not in open_buses
                ):
                    network.add_edge(u, v)
        else:
            for phase in common_phases:
                u = ".".join([fbus, phase])
                v = ".".join([tbus, phase])
                network.add_edge(u, v)

    # delete nodes if there is no path to the root node
    nodes_to_delete = []
    for n in network.nodes:
        if (
            not nx.has_path(network, source=f"{root_node}.1", target=n)
            and not nx.has_path(network, source=f"{root_node}.2", target=n)
            and not nx.has_path(network, source=f"{root_node}.3", target=n)
        ):
            nodes_to_delete.append(n)
    network.remove_nodes_from(nodes_to_delete)

    # add coordinates of each node and length of each edge to the network as attributes
    for u in network.nodes:
        network.nodes[u]["cord"] = pos[u.split(".")[0]]

    for u, v in network.edges:
        upos = network.nodes[u]["cord"]
        vpos = network.nodes[v]["cord"]
        if coordsys == "2D":
            network[u][v]["length"] = dist(upos, vpos)
        elif coordsys == "GEO":
            network[u][v]["length"] = geodist(upos, vpos)

    # compute distance to root node
    for n in network.nodes:
        phase = n.split(".")[-1]
        r = root_node + f".{phase}"
        if nx.has_path(network, source=r, target=n):
            network.nodes[n]["root_distance"] = nx.shortest_path_length(
                network, source=r, target=n, weight="length"
            )
        else:
            print(f"No path between nodes {r} and {n}")

    return network


def get_time(x):
    return datetime.strptime(x, "%Y-%m-%d %H:%M:%S").time().strftime("%H:%M")


def get_voltage(realV, imagV):
    voltage_real = feather.read_feather(realV)
    voltage_imag = feather.read_feather(imagV)
    df_voltages = np.abs(
        voltage_real.drop("time", axis=1) + 1j * voltage_imag.drop("time", axis=1)
    )
    df_voltages["time"] = voltage_real["time"].apply(get_time)
    return df_voltages.set_index("time")


def get_power(real, imag):
    p = feather.read_feather(real)
    q = feather.read_feather(imag)
    df_power = np.abs(p.drop("time", axis=1) + 1j * q.drop("time", axis=1))
    df_power["time"] = p["time"].apply(get_time)
    return df_power.set_index("time")


def plot_network(
    topology_file,
    buscoord_file,
    realVfile,
    imagVfile,
    root_node="150R",
    sep="    ",
    time=["07:30", "12:30", "15:30"],
    vmin=1.0,
    vmax=1.05,
    to_file=None,
    show=False,
    do_return=False,
    **kwargs,
) -> None:
    # keyword arguments
    figsize = kwargs.get("figsize", (10 * len(time), 10))
    constrained_layout = kwargs.get("constrained_layout", False)
    node_size = kwargs.get("node_size", 50)
    label_fontsize = kwargs.get("fontsize", 25)
    suptitle_sfx = kwargs.get("suptitle_sfx", None)
    ticklabel_fontsize = label_fontsize - 2
    title_fontsize = label_fontsize + 10

    # get voltage data
    df_voltages = get_voltage(realVfile, imagVfile)

    # get open switches in the network
    open_buses = [bus for bus in df_voltages.columns if bus.find("OPEN") != -1]

    # networkx graph
    branch_info, bus_info, base_voltages, cord = extract_topology(
        topology_file, buscoord_file, sep=sep
    )
    network = get_network(
        branch_info, bus_info, cord, root_node=root_node, open_buses=open_buses
    )

    # Plotting
    cmap = plt.cm.plasma
    fig, axs = plt.subplots(
        1, len(time), figsize=figsize, constrained_layout=constrained_layout
    )
    for i, t in enumerate(time):
        voltages = df_voltages.loc[t] / base_voltages
        n_colors = [voltages[n] for n in network.nodes]

        if len(time) > 1:
            ax = axs[i]

        else:
            ax = axs
        ax.set_title(f"Time: {t} hours", fontsize=label_fontsize)

        # Draw the network
        pos = nx.get_node_attributes(network, "cord")
        nx.draw_networkx_nodes(
            network,
            pos,
            ax=ax,
            node_size=node_size,
            node_color=n_colors,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        nx.draw_networkx_edges(network, pos, alpha=0.1, edge_color="k", ax=ax)

    # Colorbar
    cobj = cm.ScalarMappable(cmap=cmap)
    cobj.set_clim(vmin=vmin, vmax=vmax)
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.72, 0.05])
    cbar = fig.colorbar(cobj, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Voltage Magnitude (p.u.)", size=label_fontsize)
    cbar.ax.tick_params(labelsize=ticklabel_fontsize)

    if len(time) > 1:
        suptitle = f"Voltage magnitude heatmaps at {len(time)} different time steps"
    else:
        suptitle = "Voltage magnitude heatmap at a particular time step"

    if suptitle_sfx:
        suptitle = f"{suptitle}  {suptitle_sfx}"
    fig.suptitle(suptitle, fontsize=title_fontsize)

    if to_file:
        fig.savefig(to_file, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    if do_return:
        return fig
    pass


def voltage_tree(network, voltages, ax, title="", coordsys="2D", **kwargs):
    # keyword arguments
    label_fontsize = kwargs.get("fontsize", 40)
    ticklabel_fontsize = label_fontsize - 10
    annotate = kwargs.get("annotate", False)
    ymin = kwargs.get("ymin", 0.98)
    ymax = kwargs.get("ymax", 1.06)

    # plot paramters
    color_choice = {"1": "red", "2": "blue", "3": "seagreen"}
    distance_to_root = nx.get_node_attributes(network, "root_distance")

    # initialize buses to annotate
    bus_annotated = []

    # plot the voltage tree
    for u, v in network.edges:
        u_phase = u.split(".")[-1]
        v_phase = v.split(".")[-1]

        # raise error if there is an edge between different phases
        if u_phase != v_phase:
            raise ValueError("Phases do not match!!!")

        # Plot an edge in the voltage tree
        kwargs_plot = dict(ls=kwargs.get("ls", "dashed"), lw=kwargs.get("lw", 2.5))
        ax.plot(
            [distance_to_root[u], distance_to_root[v]],
            [voltages[u], voltages[v]],
            color=color_choice[u_phase],
            **kwargs_plot,
        )
        ax.set_ylim(ymin, ymax)

        # Annotate for large voltage deviation
        if annotate:
            if abs(voltages[v] - voltages[u]) > 0.01:
                ubus = u.split(".")[0]
                vbus = v.split(".")[0]
                if ubus not in bus_annotated:
                    bus_annotated.append(ubus)
                    ax.annotate(
                        ubus,
                        (distance_to_root[u], voltages[u]),
                        fontsize=label_fontsize - 10,
                    )
                if vbus not in bus_annotated:
                    bus_annotated.append(vbus)
                    ax.annotate(
                        vbus,
                        (distance_to_root[v], voltages[v]),
                        fontsize=label_fontsize - 10,
                    )

    if coordsys == "2D":
        ax.set_xlabel("Feeder length from substation (units)", fontsize=label_fontsize)
    elif coordsys == "GEO":
        ax.set_xlabel("Feeder length from substation (miles)", fontsize=label_fontsize)

    ax.set_ylabel("Voltage at node (p.u.)", fontsize=label_fontsize)
    ax.set_title(title, fontsize=label_fontsize)
    ax.grid(color="k", linestyle="dashed", linewidth=0.2)
    ax.tick_params(axis="x", labelsize=ticklabel_fontsize)
    ax.tick_params(axis="y", labelsize=ticklabel_fontsize)
    return


def plot_voltage_tree(
    topology_file,
    buscoord_file,
    realVfile,
    imagVfile,
    root_node="150R",
    sep="    ",
    coordsys="2D",
    time=[30, 60, 90],
    to_file=None,
    show=False,
    do_return=False,
    **kwargs,
):
    # keyword arguments
    figsize = kwargs.get("figsize", (10 * len(time), 10))
    constrained_layout = kwargs.get("constrained_layout", False)
    label_fontsize = kwargs.get("fontsize", 40)
    title_fontsize = label_fontsize + 10
    suptitle_sfx = kwargs.get("suptitle_sfx", None)

    # get voltage data
    df_voltages = get_voltage(realVfile, imagVfile)

    # get open switches in the network
    open_buses = [bus for bus in df_voltages.columns if bus.find("OPEN") != -1]

    # networkx graph
    branch_info, bus_info, base_voltages, cord = extract_topology(
        topology_file, buscoord_file, sep=sep
    )
    network = get_network(
        branch_info,
        bus_info,
        cord,
        root_node=root_node,
        open_buses=open_buses,
        coordsys=coordsys,
    )

    # Plotting
    fig, axs = plt.subplots(
        1, len(time), figsize=figsize, constrained_layout=constrained_layout
    )

    for i, t in enumerate(time):
        voltages = df_voltages.loc[t] / base_voltages
        if len(time) > 1:
            ax = axs[i]
        else:
            ax = axs
        voltage_tree(
            network, voltages, ax, title=f"Time: {t} hours", coordsys=coordsys, **kwargs
        )

    if len(time) > 1:
        suptitle = f"Voltage trees at {len(time)} different time steps"
    else:
        suptitle = f"Voltage tree at a particular time step"

    if suptitle_sfx:
        suptitle = f"{suptitle}  {suptitle_sfx}"
    fig.suptitle(suptitle, fontsize=title_fontsize)

    if to_file:
        fig.savefig(to_file, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    if do_return:
        return fig
    pass


def plot_opf_voltage_comparison(
    topology_file,
    realVfile,
    imagVfile,
    opfVfile,
    true_volt_units="kV",
    time=["07:30", "12:30", "15:30"],
    to_file=None,
    show=False,
    do_return=False,
    **kwargs,
):
    # get base voltages
    with open(topology_file) as f:
        topology = Topology.parse_obj(json.load(f))
        base_voltage_df = pd.DataFrame(
            {
                "id": topology.base_voltage_magnitudes.ids,
                "value": topology.base_voltage_magnitudes.values,
            }
        )
        base_voltage_df.set_index("id", inplace=True)
        base_voltages = base_voltage_df["value"]

    # get voltage data
    df_true_voltages = get_voltage(realVfile, imagVfile)
    true_voltage_columns = df_true_voltages.columns

    # opf voltage magnitudes
    df_opf_voltages = feather.read_feather(opfVfile)
    df_opf_voltages["time"] = df_opf_voltages["time"].apply(get_time)
    df_opf_voltages = df_opf_voltages.set_index("time")
    opf_voltage_columns = df_opf_voltages.columns

    # find columns which are present in both dataframes
    node_names = [node for node in true_voltage_columns if node in opf_voltage_columns]

    # keyword arguments
    figsize = kwargs.get("figsize", (10 * len(time), 10))
    constrained_layout = kwargs.get("constrained_layout", False)
    label_fontsize = kwargs.get("fontsize", 30)
    legend_fontsize = label_fontsize
    ticklabel_fontsize = label_fontsize - 2
    title_fontsize = label_fontsize + 10
    suptitle_sfx = kwargs.get("suptitle_sfx", None)

    fig, axs = plt.subplots(
        1, len(time), figsize=figsize, constrained_layout=constrained_layout
    )

    for i, hr in enumerate(time):
        if len(time) > 1:
            ax = axs[i]
        else:
            ax = axs

        # pu voltage magnitudes
        if true_volt_units == "pu":
            true_voltages = df_true_voltages.loc[hr]
        else:
            true_voltages = df_true_voltages.loc[hr] / base_voltages

        opf_voltages = df_opf_voltages.loc[hr]
        opf_volt_vals = [opf_voltages[n] for n in node_names]
        true_volt_vals = [true_voltages[n] for n in node_names]
        MAE = errors(true_volt_vals, opf_volt_vals)
        xlabel = "Node number"

        # Plot the comparison
        x_axis = np.arange(len(node_names))
        ax.plot(
            x_axis,
            opf_volt_vals,
            "-o",
            color="crimson",
            lw=2.5,
            label="OPF estimated voltages",
        )
        ax.plot(
            x_axis,
            true_volt_vals,
            "-o",
            color="royalblue",
            lw=2.5,
            label="True voltages",
        )
        # Formatting
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
        ax.set_ylabel(f"Voltage Magnitudes (pu)", fontsize=label_fontsize)
        ax.legend(fontsize=legend_fontsize, markerscale=2)
        ax.tick_params(axis="x", labelsize=ticklabel_fontsize)
        ax.tick_params(axis="y", labelsize=ticklabel_fontsize)
        ax.set_title(f"time={hr} hours, MAE={MAE:0.3f}", fontsize=label_fontsize)

    suptitle = "Voltage validation for DOPF algorithm"
    if suptitle_sfx:
        suptitle = f"{suptitle}  {suptitle_sfx}"
    fig.suptitle(suptitle, fontsize=title_fontsize)

    if to_file:
        fig.savefig(to_file, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    if do_return:
        return fig
    pass


def compare_vtree(
    realVfile1,
    realVfile2,
    imagVfile1,
    imagVfile2,
    topofile1,
    topofile2,
    buscordfile1,
    buscordfile2,
    alg1,
    alg2,
    sep="    ",
    root_node="150",
    coordsys="2D",
    time="11:15",
    to_file=None,
    show=True,
    do_return=False,
    **kwargs,
):
    # keyword arguments
    figsize = kwargs.get("figsize", (40, 20))
    constrained_layout = kwargs.get("constrained_layout", False)
    label_fontsize = kwargs.get("fontsize", 40)
    title_fontsize = label_fontsize + 10

    # get voltage data
    df_voltages1 = get_voltage(realVfile1, imagVfile1)
    df_voltages2 = get_voltage(realVfile2, imagVfile2)

    # get open switches in the network
    open_buses1 = [bus for bus in df_voltages1.columns if bus.find("OPEN") != -1]
    open_buses2 = [bus for bus in df_voltages2.columns if bus.find("OPEN") != -1]

    # networkx graph
    branch_info1, bus_info1, base_voltages1, cord1 = extract_topology(
        topofile1, buscordfile1, sep=sep
    )
    network1 = get_network(
        branch_info1,
        bus_info1,
        cord1,
        root_node=root_node,
        open_buses=open_buses1,
        coordsys=coordsys,
    )
    branch_info2, bus_info2, base_voltages2, cord2 = extract_topology(
        topofile2, buscordfile2, sep=sep
    )
    network2 = get_network(
        branch_info2,
        bus_info2,
        cord2,
        root_node=root_node,
        open_buses=open_buses2,
        coordsys=coordsys,
    )

    # Plotting
    fig, axs = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=constrained_layout
    )

    voltages1 = df_voltages1.loc[time] / base_voltages1
    voltages2 = df_voltages2.loc[time] / base_voltages2

    voltage_tree(
        network1,
        voltages1,
        axs[0],
        title=f"DOPF Algorithm: {alg1}",
        coordsys=coordsys,
        **kwargs,
    )
    voltage_tree(
        network2,
        voltages2,
        axs[1],
        title=f"DOPF Algorithm: {alg2}",
        coordsys=coordsys,
        **kwargs,
    )

    suptitle = f"Voltage tree comparison of networks for two DOPF algorithms, time={time} hours"
    fig.suptitle(suptitle, fontsize=title_fontsize)

    if to_file:
        fig.savefig(to_file, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    if do_return:
        return fig
    pass


def compare_network(
    realVfile1,
    realVfile2,
    imagVfile1,
    imagVfile2,
    topofile1,
    topofile2,
    buscordfile1,
    buscordfile2,
    alg1,
    alg2,
    sep="    ",
    root_node="150",
    coordsys="2D",
    time="11:30",
    vmin=1.0,
    vmax=1.05,
    to_file=None,
    show=True,
    do_return=False,
    **kwargs,
):
    # keyword arguments
    figsize = kwargs.get("figsize", (20, 10))
    constrained_layout = kwargs.get("constrained_layout", False)
    node_size = kwargs.get("node_size", 50)
    label_fontsize = kwargs.get("fontsize", 25)
    ticklabel_fontsize = label_fontsize - 2
    title_fontsize = label_fontsize + 5

    # get voltage data
    df_voltages1 = get_voltage(realVfile1, imagVfile1)
    df_voltages2 = get_voltage(realVfile2, imagVfile2)

    # get open switches in the network
    open_buses1 = [bus for bus in df_voltages1.columns if bus.find("OPEN") != -1]
    open_buses2 = [bus for bus in df_voltages2.columns if bus.find("OPEN") != -1]

    # networkx graph
    branch_info1, bus_info1, base_voltages1, cord1 = extract_topology(
        topofile1, buscordfile1, sep=sep
    )
    network1 = get_network(
        branch_info1,
        bus_info1,
        cord1,
        root_node=root_node,
        open_buses=open_buses1,
        coordsys=coordsys,
    )
    branch_info2, bus_info2, base_voltages2, cord2 = extract_topology(
        topofile2, buscordfile2, sep=sep
    )
    network2 = get_network(
        branch_info2,
        bus_info2,
        cord2,
        root_node=root_node,
        open_buses=open_buses2,
        coordsys=coordsys,
    )

    # Plotting
    cmap = plt.cm.plasma
    fig, axs = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=constrained_layout
    )

    voltages1 = df_voltages1.loc[time, :] / base_voltages1
    n_colors1 = [voltages1[n] for n in network1.nodes]
    voltages2 = df_voltages2.loc[time, :] / base_voltages2
    n_colors2 = [voltages2[n] for n in network2.nodes]

    # Draw the network
    pos1 = nx.get_node_attributes(network1, "cord")
    pos2 = nx.get_node_attributes(network2, "cord")

    nx.draw_networkx_nodes(
        network1,
        pos1,
        ax=axs[0],
        node_size=node_size,
        node_color=n_colors1,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    nx.draw_networkx_edges(network1, pos1, alpha=0.1, edge_color="k", ax=axs[0])
    nx.draw_networkx_nodes(
        network2,
        pos2,
        ax=axs[1],
        node_size=node_size,
        node_color=n_colors2,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    nx.draw_networkx_edges(network2, pos2, alpha=0.1, edge_color="k", ax=axs[1])
    axs[0].set_title(f"DOPF Algorithm: {alg1}", fontsize=label_fontsize)
    axs[1].set_title(f"DOPF Algorithm: {alg2}", fontsize=label_fontsize)

    # Colorbar
    cobj = cm.ScalarMappable(cmap=cmap)
    cobj.set_clim(vmin=vmin, vmax=vmax)
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.72, 0.05])
    cbar = fig.colorbar(cobj, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Voltage Magnitude (p.u.)", size=label_fontsize)
    cbar.ax.tick_params(labelsize=ticklabel_fontsize)

    suptitle = f"Voltage magnitude heatmaps for two DOPF algorithms, time={time} hours"
    fig.suptitle(suptitle, fontsize=title_fontsize)

    if to_file:
        fig.savefig(to_file, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    if do_return:
        return fig
    pass


def plot_curtail(
    curtail_file, pv_systems=None, to_file=None, show=True, do_return=False, **kwargs
):
    df_curtail = feather.read_feather(curtail_file)
    df_curtail["time"] = df_curtail["time"].apply(get_time)
    df_curtail.set_index("time")

    # get the 10 PV Systems with maximum curtailment
    curtail = df_curtail.to_numpy()[:, :-1]
    max_curtail = np.mean(curtail, axis=0)
    max_indices = np.argsort(max_curtail)[::-1]
    sorted_pv_systems = df_curtail.columns.to_numpy()[max_indices]

    if not pv_systems:
        pv_systems = sorted_pv_systems[:10]

    # keyword arguments
    figsize = kwargs.get("figsize", (20, 10))
    constrained_layout = kwargs.get("constrained_layout", False)
    label_fontsize = kwargs.get("fontsize", 25)
    ticklabel_fontsize = label_fontsize - 2
    title_fontsize = label_fontsize + 5
    suptitle_pfx = kwargs.get("suptitle_pfx", "Forecasted ")

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=constrained_layout)
    df_curtail.plot.line(
        y=pv_systems,
        ax=ax,
        marker="^",
        markersize=10,
        linestyle="dashed",
        linewidth=2.0,
    )
    # Formatting
    x_pts = [0, 20, 40, 60, 80]
    xtix = [df_curtail["time"][x] for x in x_pts]
    ax.set_xlabel("Time of day (HH:MM)", fontsize=label_fontsize)
    ax.set_ylabel("PV Curtailment (kW)", fontsize=label_fontsize)
    ax.legend(fontsize=label_fontsize - 4, markerscale=2, ncols=1)
    ax.set_xticks(x_pts, labels=xtix, fontsize=ticklabel_fontsize)
    ax.tick_params(axis="y", labelsize=ticklabel_fontsize)

    suptitle = f"{suptitle_pfx}Curtailment for the PV systems"
    fig.suptitle(suptitle, fontsize=title_fontsize)

    if to_file:
        fig.savefig(to_file, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    if do_return:
        return fig
    pass


def add_data_from_df(
    data_dict,
    df,
    pq="p",
    type="actual",
):
    nodes = [n for n in df.columns.to_list() if n != "time"]
    for k in range(len(df)):
        for n in nodes:
            data_dict["nodes"].append(n)
            data_dict["value"].append(df.iloc[k][n])
            data_dict["pq"].append(pq)
            data_dict["type"].append(type)
    return data_dict


def plot_pv_estimation(
    power_real,
    power_imag,
    power_real_est,
    power_imag_est,
    time=["07:30", "12:30", "15:30"],
    show=True,
    to_file=None,
):
    df_act_p = feather.read_feather(power_real)
    df_act_p["time"] = df_act_p["time"].apply(get_time)
    df_act_q = feather.read_feather(power_imag)
    df_act_q["time"] = df_act_q["time"].apply(get_time)
    df_est_p = feather.read_feather(power_real_est)
    df_est_p["time"] = df_est_p["time"].apply(get_time)
    df_est_q = feather.read_feather(power_imag_est)
    df_est_q["time"] = df_est_q["time"].apply(get_time)

    fig, ax = plt.subplots(layout="constrained")
    df_act_p.plot(kind="bar", ax=ax)
    df_est_p.plot(kind="bar", ax=ax)

    if show:
        plt.show()
    if to_file:
        fig.savefig(to_file, bbox_inches="tight")
