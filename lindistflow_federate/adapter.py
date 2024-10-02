import copy
from pprint import pprint
import numpy as np
import logging
import networkx as nx
from pprint import pprint
from enum import IntEnum
from typing import Tuple
from dataclasses import dataclass, field, asdict
from oedisi.types.data_types import (
    AdmittanceMatrix,
    AdmittanceSparse,
    CommandList,
    EquipmentNodeArray,
    Injection,
    InverterControlList,
    MeasurementArray,
    PowersImaginary,
    PowersReal,
    Topology,
    Incidence,
    VoltagesAngle,
    VoltagesImaginary,
    VoltagesMagnitude,
    VoltagesReal,
)

import bus_update as bu


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


@dataclass
class Branch:
    fr_bus: str
    to_bus: str
    tag: str
    idx: int = 0
    fr_idx: int = 0
    to_idx: int = 0
    phases: list[int] = field(default_factory=lambda: [0] * 3)
    zprim: list[list[list[float]]] = field(
        default_factory=lambda: np.zeros((3, 3, 2)).tolist()
    )
    y: list[list[complex]] = field(
        default_factory=lambda: np.zeros((3, 3), dtype=complex).tolist()
    )


@dataclass
class BranchInfo:
    branches: dict[Branch] = field(default_factory=dict)


@dataclass
class Bus:
    idx: int = 0
    tags: list[str] = field(default_factory=list)
    phases: list[int] = field(default_factory=lambda: [0] * 3)
    base_kv: float = 0.0
    tap_ratio: float = 0.0
    base_pq: list[list[float]] = field(
        default_factory=lambda: np.zeros((3, 2)).tolist()
    )
    base_pv: list[list[float]] = field(
        default_factory=lambda: np.zeros((3, 2)).tolist()
    )
    kv: float = 0.0
    pq: list[list[float]] = field(default_factory=lambda: np.zeros((3, 2)).tolist())
    pv: list[list[float]] = field(default_factory=lambda: np.zeros((3, 2)).tolist())


@dataclass
class BusInfo:
    buses: dict[Bus] = field(default_factory=dict)


def check_radiality(branch_info: BranchInfo, bus_info: BusInfo) -> bool:
    print(len(bus_info.buses), len(branch_info.branches))
    if len(bus_info.buses) - len(branch_info.branches) == 1:
        return True

    logger.debug("Network is not Radial")
    logger.debug(f"Branch: {branch_info.branches.keys()}")
    logger.debug(f"Bus: {bus_info.buses.keys()}")
    return False


def index_info(branch_info: BranchInfo, bus_info: BusInfo) -> (BranchInfo, BusInfo):
    for i, bus in enumerate(bus_info.buses.values()):
        bus.idx = i

    for i, branch in enumerate(branch_info.branches.values()):
        branch.idx = i
        branch.fr_idx = bus_info.buses[branch.fr_bus].idx
        branch.to_idx = bus_info.buses[branch.to_bus].idx

    return (branch_info, bus_info)


def generate_zprim(branch_info: BranchInfo) -> BranchInfo:
    for branch in branch_info.branches.values():
        z = -1 * np.linalg.pinv(branch.y)
        branch.y = []
        for idx, value in np.ndenumerate(z):
            row = idx[0]
            col = idx[1]
            branch.zprim[row][col] = [float(value.real), float(value.imag)]
    return branch_info


def extract_base_voltages(bus_info: BusInfo, voltages: VoltagesMagnitude) -> dict:
    for id, voltage in zip(voltages.ids, voltages.values):
        name, phase = id.split(".", 1)
        phase = int(phase) - 1

        if name not in bus_info.buses:
            continue

        bus_info.buses[name].base_kv = voltage / 1000.0
        bus_info.buses[name].phases[phase] = phase + 1
    return bus_info


def extract_voltages(bus_info: BusInfo, voltages: VoltagesMagnitude) -> dict:
    for id, voltage in zip(voltages.ids, voltages.values):
        name, phase = id.split(".", 1)
        phase = int(phase) - 1

        if name not in bus_info.buses:
            continue

        bus_info.buses[name].kv = voltage / 1000.0
    return bus_info


def pack_voltages(voltages: dict, bus_info: BusInfo, time: int) -> VoltagesMagnitude:
    ids = []
    values = []
    for key, value in voltages.items():
        busid, phase = key.split(".", 1)
        if busid in bus_info.buses:
            ids.append(key)
            values.append(value)
    return VoltagesMagnitude(ids=ids, values=values, time=time)


def pack_powers_real(base: PowersReal, powers: dict, time: int) -> PowersReal:
    ids = []
    eq_ids = []
    values = []
    for id, eq in zip(base.ids, base.equipment_ids):
        if id in powers:
            ids.append(id)
            eq_ids.append(eq)
            values.append(round(powers[id], 6))
    return PowersReal(ids=ids, equipment_ids=eq_ids, values=values, time=time)


def pack_powers_imag(base: PowersImaginary, powers: dict, time: int) -> PowersImaginary:
    ids = []
    eq_ids = []
    values = []
    for id, eq in zip(base.ids, base.equipment_ids):
        if id in powers:
            ids.append(id)
            eq_ids.append(eq)
            values.append(round(powers[id], 6))
    return PowersReal(ids=ids, equipment_ids=eq_ids, values=values, time=time)


def extract_forecast(bus: dict, forecast) -> dict:
    for eq, power in zip(forecast["ids"], forecast["values"]):
        print(eq)
        if "_" in eq:
            [_, name] = eq.rsplit("_", 1)
        else:
            [_, name] = eq.rsplit(".", 1)
        name = name.upper()

        if name not in bus:
            print("NOT IN BUS: ", name)
            continue

        phases = bus[name]["phases"]
        for ph in phases:
            phase = int(ph) - 1
            logger.debug(f"{eq}.{ph} : {power/len(phases)}")
            bus[name]["eqid"] = eq
            bus[name]["pv"][phase][0] = power * 1000 / len(phases)
            bus[name]["pv"][phase][1] = 0.0
    return bus


def extract_powers_real(bus_info: BusInfo, real: PowersReal) -> BusInfo:
    for id, eq, power in zip(real.ids, real.equipment_ids, real.values):
        name, phase = id.split(".", 1)
        phase = int(phase) - 1

        if name not in bus_info.buses:
            continue

        bus_info.buses[name].pq[phase][0] += power * 1000
    return bus_info


def extract_powers_imag(bus_info: BusInfo, imag: PowersImaginary) -> BusInfo:
    for id, eq, power in zip(imag.ids, imag.equipment_ids, imag.values):
        name, phase = id.split(".", 1)
        phase = int(phase) - 1

        if name not in bus_info.buses:
            continue

        bus_info.buses[name].pq[phase][1] += power * 1000
    return bus_info


def extract_base_injection(bus_info: BusInfo, powers: Injection) -> dict:
    real = powers.power_real
    imag = powers.power_imaginary

    for id, eq, power in zip(real.ids, real.equipment_ids, real.values):
        name, phase = id.split(".", 1)
        phase = int(phase) - 1

        if name not in bus_info.buses:
            continue

        bus_info.buses[name].tags.append(eq)
        if "PVSystem" in eq:
            bus_info.buses[name].base_pv[phase][0] += power * 1000
        else:
            bus_info.buses[name].base_pq[phase][0] -= power * 1000

    for id, eq, power in zip(imag.ids, imag.equipment_ids, imag.values):
        name, phase = id.split(".", 1)
        phase = int(phase) - 1

        if name not in bus_info.buses:
            continue

        if "PVSystem" in eq:
            bus_info.buses[name].base_pv[phase][1] += power * 1000
        else:
            bus_info.buses[name].base_pq[phase][1] -= power * 1000
    return bus_info


def extract_transformers(incidences: Incidence) -> (list[str], list[str]):
    xfmrs = []
    to_eq = incidences.to_equipment
    fr_eq = incidences.from_equipment
    ids = incidences.ids
    for fr_eq, to_eq, eq_id in zip(fr_eq, to_eq, ids):
        if "tr" in eq_id or "reg" in eq_id or "xfm" in eq_id:
            if "." in to_eq:
                [to_eq, _] = to_eq.split(".", 1)
            if "." in fr_eq:
                [fr_eq, _] = fr_eq.split(".", 1)
            xfmrs.append(f"{fr_eq}_{to_eq}")
    return xfmrs


def generate_graph(inc: Incidence, slack_bus: str) -> nx.Graph:
    graph = nx.Graph()
    for src, dst, id in zip(inc.from_equipment, inc.to_equipment, inc.ids):
        if "OPEN" in src or "OPEN" in dst:
            continue
        if src == dst:
            continue

        ps = pd = ""
        if "." in src:
            src, ps = src.split(".", 1)
        if "." in dst:
            dst, pd = dst.split(".", 1)

        eq = "LINE"
        if ("sw" in id or "fuse" in id) and "padswitch" not in id:
            eq = "SWITCH"
        if "tr" in id or "reg" in id or "xfm" in id:
            eq = "XFMR"
        graph.add_edge(src, dst, name=f"{src}_{dst}", tag=eq, id=f"{id}")

    for c in nx.connected_components(graph):
        if slack_bus in c:
            return graph.subgraph(c).copy()


def tag_regulators(branch_info: BranchInfo, bus_info: BusInfo) -> BranchInfo:
    for k, branch in branch_info.branches.items():
        if "XFMR" != branch.tag:
            continue

        src = bus_info.buses[branch.fr_bus]
        dst = bus_info.buses[branch.to_bus]

        if round(src.base_kv, 3) == round(dst.base_kv, 3):
            branch.tag == "REG"
    return branch_info


def direct_branch_flows(
    graph: nx.Graph, branch_info: BranchInfo, source: str
) -> BranchInfo:
    dist = nx.single_source_shortest_path(graph, source)

    for k, branch in branch_info.branches.items():
        fr_idx = branch.fr_idx
        to_idx = branch.to_idx
        fr_bus = branch.fr_bus
        to_bus = branch.to_bus
        src = dist[fr_bus]
        dst = dist[to_bus]

        if src > dst:
            branch_info.branches[k].fr_idx = to_idx
            branch_info.branches[k].to_idx = fr_idx
            branch_info.branches[k].fr_bus = to_bus
            branch_info.branches[k].to_bus = fr_bus
    return branch_info


def disconnect_areas(graph: nx.Graph, switches) -> list[nx.Graph]:
    graph.remove_edges_from(switches)

    areas = []
    for c in nx.connected_components(graph):
        areas.append(graph.subgraph(c).copy())
    return areas


def get_switches(graph: nx.Graph):
    switches = []
    for u, v, a in graph.edges(data=True):
        if "SWITCH" == a["tag"]:
            switches.append((u, v, a))
    return switches


def area_disconnects(graph: nx.Graph):
    n_max = 5
    switches = get_switches(graph)
    areas = disconnect_areas(graph, switches)
    area_cnt = [area.number_of_nodes() for area in areas]
    min_n = [area.number_of_nodes() for area in areas]
    min_n.sort(reverse=True)
    min_n = min(min_n[0:n_max])
    z_area = zip(area_cnt, areas)
    z_area = sorted(z_area, key=lambda v: v[0])

    closed = []
    cnt = 0
    for n, area in z_area:
        if n < 2 or n < min_n or cnt > n_max:
            for u, v, a in switches:
                if area.has_node(u) or area.has_node(v):
                    closed.append((u, v, a))
            continue
        cnt += 1

    open = [(u, v, a) for u, v, a in switches if (u, v, a) not in closed]
    return open


def reconnect_area_switches(areas: list[nx.Graph], switches):
    for area in areas:
        for u, v, a in switches:
            if area.has_node(u) or area.has_node(v):
                area.add_edge(u, v, **a)
    return areas


def get_area_source(graph: nx.Graph, slack_bus: str, switches):
    paths = {}
    for u, v, a in switches:
        paths[len(nx.shortest_path(graph, slack_bus, u))] = (u, v, a)
    source = min(paths, key=paths.get)
    return paths[source]


def extract_injection(bus_info: BusInfo, powers: Injection) -> dict:
    real = powers.power_real
    imag = powers.power_imaginary

    for id, eq, power in zip(real.ids, real.equipment_ids, real.values):
        name, phase = id.split(".", 1)
        phase = int(phase) - 1

        if name not in bus_info.buses:
            continue

        bus_info.buses[name].tags.append(eq)
        if "PVSystem" in eq:
            bus_info.buses[name].pv[phase][0] += power * 1000
        else:
            bus_info.buses[name].pq[phase][0] -= power * 1000

    for id, eq, power in zip(imag.ids, imag.equipment_ids, imag.values):
        name, phase = id.split(".", 1)
        phase = int(phase) - 1

        if name not in bus_info.buses:
            continue

        if "PVSystem" in eq:
            bus_info.buses[name].pv[phase][1] += power * 1000
        else:
            bus_info.buses[name].pq[phase][1] -= power * 1000
    return bus_info


def extract_admittance(branch_info: BranchInfo, y: AdmittanceSparse) -> BranchInfo:
    for src, dst, v in zip(y.from_equipment, y.to_equipment, y.admittance_list):
        src, row = src.split(".", 1)
        row = int(row) - 1
        dst, col = dst.split(".", 1)
        col = int(col) - 1

        if src == dst:
            continue

        key = f"{src}_{dst}"
        rev_key = f"{dst}_{src}"
        if key in branch_info.branches:
            branch_info.branches[key].y[row][col] = complex(v[0], v[1])
            branch_info.branches[key].phases[row] = row + 1
            branch_info.branches[key].phases[col] = col + 1
        elif rev_key in branch_info.branches:
            branch_info.branches[rev_key].y[col][row] = complex(v[0], v[1])
            branch_info.branches[rev_key].phases[row] = row + 1
            branch_info.branches[rev_key].phases[col] = col + 1
    return branch_info


def find_primary(graph: nx.DiGraph, bus_info: BusInfo, node: int) -> str:
    while list(graph.predecessors(node)):
        primary = next(graph.predecessors(node))
        if bus_info.buses[primary].base_kv > 0.5:
            return primary
        node = primary
    return None


def get_upstream(branch_info: BranchInfo, node: str) -> str:
    for k, branch in branch_info.branches.items():
        if branch.to_bus == node:
            return branch.fr_bus


def find_branch(branch_info: BranchInfo, src: str, dst: str) -> str:
    for k, branch in branch_info.branches.items():
        if branch.fr_bus == src and branch.to_bus == dst:
            return k


def find_consecutive_phase(connected: list[list[bool]]) -> int:
    connected_row = []
    connected_col = []
    for i in range(3):
        consecutive = 0
        for j in range(3):
            if connected[i][j]:
                consecutive += 1
            else:
                consecutive = 0

            if consecutive >= 2:
                connected_row.append(i)

    for j in range(3):
        consecutive = 0
        for i in range(3):
            if connected[i][j]:
                consecutive += 1
            else:
                consecutive = 0

            if consecutive >= 2:
                connected_col.append(j)

    if connected_col and connected_row:
        raise Exception("hase phase in both row and col of xfmr")

    if connected_row:
        return connected_row[0]

    if connected_col:
        return connected_col[0]


def find_connected_phases(zprim: list[list[list[float]]]) -> list[list[bool]]:
    connected = np.zeros((3, 3), dtype=bool).tolist()
    for i in range(3):
        for j in range(3):
            real = abs(zprim[i][j][0])
            imag = abs(zprim[i][j][1])
            if real > 1e-6 or imag > 1e-6:
                connected[i][j] = True

    return connected


def traverse_secondaries(
    branch_info: BranchInfo, bus_info: BusInfo, node: str, primary: str
) -> list[int]:
    secondaries = [node]
    while node != primary:
        node = get_upstream(branch_info, node)
        secondaries.append(node)

    for k, branch in branch_info.branches.items():
        # make sure first low side of the secondaries isn't in branch
        if branch.fr_bus == primary and branch.to_bus == secondaries[-2]:
            zprim = branch.zprim
            break

    connected_phases = find_connected_phases(zprim)
    connected_phase = find_consecutive_phase(connected_phases)

    if "processed" in branch.tag:
        phases = branch.phases
    else:
        phases = [0] * 3
        phases[connected_phase] = connected_phase + 1

    for i, secondary in enumerate(secondaries):
        if i == len(secondaries) - 1:
            continue

        branch = find_branch(branch_info, secondaries[i + 1], secondary)
        if "processed" in branch_info.branches[branch].tag:
            continue

        branch_info.branches[branch].phases = phases
        est_zprim = np.zeros((3, 3, 2)).tolist()
        est_zprim[connected_phase][connected_phase] = [1e-5, 1e-5]
        branch_info.branches[branch].zprim = est_zprim
        branch_info.branches[branch].tag += ".processed"

        bus_info.buses[secondary].phases = phases
        if secondaries[i + 1] != primary:
            bus_info.buses[secondaries[i + 1]].phases = phases

    return connected_phase


def process_secondary(
    branch_info: BranchInfo, bus_info: BusInfo, node: str, primary: str
) -> (BranchInfo, BusInfo):
    bus = bus_info.buses[node]
    has_phase = [p != 0 for p in bus.phases]

    if all(has_phase):
        return (branch_info, bus_info)

    primary_phase = traverse_secondaries(branch_info, bus_info, node, primary)

    new_pq = np.zeros((3, 2)).tolist()
    new_pv = np.zeros((3, 2)).tolist()
    for pq, pv in zip(bus.pq, bus.pq):
        for ppq, ppv in zip(pq, pv):
            new_pq[primary_phase] += [ppq]
            new_pv[primary_phase] += [ppv]

    bus_info.buses[node].pq = new_pq
    bus_info.buses[node].pv = new_pv

    return (branch_info, bus_info)


def map_secondaries(
    branch_info: BranchInfo, bus_info: BusInfo
) -> (BranchInfo, BusInfo):
    graph = nx.DiGraph()
    branch: Branch
    for branch in branch_info.branches.values():
        graph.add_edge(branch.fr_bus, branch.to_bus)

    secondaries = [
        b
        for b in bus_info.buses.keys()
        if graph.out_degree(b) == 0 and bus_info.buses[b].base_kv < 0.5
    ]

    # Process each leaf node
    bus_data = {k: asdict(v) for k, v in bus_info.buses.items()}
    branch_data = {k: asdict(v) for k, v in branch_info.branches.items()}
    for secondary in secondaries:
        primary_parent = bu.find_primary_parent(secondary, bus_data, graph)

        if primary_parent:
            bu.process_secondary_side(secondary, primary_parent, bus_data, branch_data)

    bus_info.buses = {k: Bus(**v) for k, v in bus_data.items()}
    for k in branch_data.keys():
        if "processed" in branch_data[k]:
            del branch_data[k]["processed"]
    branch_info.branches = {k: Branch(**v) for k, v in branch_data.items()}

    return (branch_info, bus_info)


def extract_info(topology: Topology) -> (BranchInfo, BusInfo, str):
    branch_info = BranchInfo()
    bus_info = BusInfo()
    slack_bus, _ = topology.slack_bus[0].split(".", 1)
    graph = generate_graph(topology.incidences, slack_bus)

    for u, v, a in graph.edges(data=True):
        branch_info.branches[a["name"]] = Branch(fr_bus=u, to_bus=v, tag=a["tag"])
        bus_info.buses[u] = Bus()
        bus_info.buses[v] = Bus()

    branch_info = direct_branch_flows(graph, branch_info, slack_bus)
    branch_info = extract_admittance(branch_info, topology.admittance)
    branch_info = generate_zprim(branch_info)
    bus_info = extract_base_voltages(bus_info, topology.base_voltage_magnitudes)
    bus_info = extract_base_injection(bus_info, topology.injections)
    branch_info = tag_regulators(branch_info, bus_info)
    branch_info, bus_info = index_info(branch_info, bus_info)

    return (branch_info, bus_info, slack_bus)
