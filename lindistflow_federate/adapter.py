import numpy as np
import logging
import networkx as nx
from enum import IntEnum
from typing import Tuple
from dataclasses import dataclass, field
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
    VoltagesReal
)


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
    phases: list[int] = field(default_factory=lambda: [0]*3)
    zprim: list[list[list[float]]] = field(
        default_factory=lambda: np.zeros((3, 3, 2)).tolist())
    y: list[list[complex]] = field(
        default_factory=lambda: np.zeros((3, 3), dtype=complex).tolist())


@dataclass
class BranchInfo:
    branches: dict[Branch] = field(default_factory=dict)


@dataclass
class Bus:
    idx: int = 0
    tags: list[str] = field(default_factory=list)
    phases: list[int] = field(default_factory=lambda: [0]*3)
    base_kv: float = field(default_factory=lambda: [0]*3)
    base_pq: float = field(default_factory=lambda: np.zeros((3, 2)).tolist())
    base_pv: float = field(default_factory=lambda: np.zeros((3, 2)).tolist())
    kv: float = 0.0
    pq: list[list[float]] = field(
        default_factory=lambda: np.zeros((3, 2)).tolist())
    pv: list[list[float]] = field(
        default_factory=lambda: np.zeros((3, 2)).tolist())


@dataclass
class BusInfo:
    buses: dict[Bus] = field(default_factory=dict)


def check_radiality(branch_info: BranchInfo, bus_info: BusInfo) -> bool:
    if len(bus_info.buses)-len(branch_info.branches) == 1:
        return True

    logger.debug("Network is not Radial")
    logger.debug(f"Branch: {branch_info.branches.keys()}")
    logger.debug(f"Bus: {bus_info.buses.keys()}")
    return False


def index_info(
        branch_info: BranchInfo, bus_info: BusInfo) -> (BranchInfo, BusInfo):
    for i, bus in enumerate(bus_info.buses.values()):
        bus.idx = i

    for i, branch in enumerate(branch_info.branches.values()):
        branch.idx = i
        branch.fr_idx = bus_info.buses[branch.fr_bus].idx
        branch.to_idx = bus_info.buses[branch.to_bus].idx

    return (branch_info, bus_info)


def generate_zprim(branch_info: BranchInfo) -> BranchInfo:
    for branch in branch_info.branches.values():
        z = -1*np.linalg.pinv(branch.y)
        for idx, value in np.ndenumerate(z):
            row = idx[0]
            col = idx[1]
            branch.zprim[row][col] = [float(value.real), float(value.imag)]
    return branch_info


def extract_base_kv(bus_info: BusInfo, voltages: VoltagesMagnitude) -> dict:
    for id, voltage in zip(voltages.ids, voltages.values):
        name, phase = id.split(".", 1)
        phase = int(phase) - 1

        if name not in bus_info.buses:
            continue

        bus_info.buses[name].base_kv[phase] = voltage/1000.0
        bus_info.buses[name].phases[phase] = phase+1
    return bus_info


def extract_kv(bus_info: BusInfo, voltages: VoltagesMagnitude) -> dict:
    for id, voltage in zip(voltages.ids, voltages.values):
        name, phase = id.split(".", 1)
        phase = int(phase) - 1

        if name not in bus_info.buses:
            continue

        bus_info.buses[name].kv[phase] = voltage/1000.0
    return bus_info


def pack_voltages(voltages: dict, time: int) -> VoltagesMagnitude:
    ids = []
    values = []
    for key, value in voltages.items():
        for phase, voltage in value.items():
            if voltage == 0.0:
                continue
            id = f"{key}.{phase}"
            ids.append(id)
            values.append(voltage)
    return VoltagesMagnitude(ids=ids, values=values, time=time)


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
            bus[name]["pv"][phase][0] = power*1000/len(phases)
            bus[name]["pv"][phase][1] = 0.0
    return bus


def extract_powers_real(bus_info: BusInfo, real: PowersReal) -> BusInfo:
    for id, eq, power in zip(real.ids, real.equipment_ids, real.values):
        name, phase = id.split(".", 1)
        phase = int(phase) - 1

        if name not in bus_info.buses:
            continue

        bus_info.buses[name].pq[phase][0] += power*1000
    return bus_info


def extract_powers_imag(bus_info: BusInfo, imag: PowersImaginary) -> BusInfo:
    for id, eq, power in zip(imag.ids, imag.equipment_ids, imag.values):
        name, phase = id.split(".", 1)
        phase = int(phase) - 1

        if name not in bus_info.buses:
            continue

        bus_info.buses[name].pq[phase][1] += power*1000
    return bus_info


def extract_injection(bus_info: BusInfo, powers: Injection) -> dict:
    real = powers.power_real
    imag = powers.power_imaginary

    for id, eq, power in zip(real.ids, real.equipment_ids, real.values):
        name, phase = id.split(".", 1)
        phase = int(phase) - 1

        if name not in bus_info.buses:
            continue

        bus_info.buses[name].tag.append(eq)
        if "PVSystem" in eq:
            bus_info.buses[name].base_pv[phase][0] -= power*1000
        else:
            bus_info.buses[name].base_pq[phase][0] += power*1000

    for id, eq, power in zip(imag.ids, imag.equipment_ids, imag.values):
        name, phase = id.split(".", 1)
        phase = int(phase) - 1

        if name not in bus_info.buses:
            continue

        if "PVSystem" in eq:
            bus_info.buses[name].base_pv[phase][1] -= power*1000
        else:
            bus_info.buses[name].base_pq[phase][1] += power*1000
    return bus_info


def extract_admittance(
        branch_info: BranchInfo, y: AdmittanceSparse) -> BranchInfo:
    for src, dst, v in zip(
            y.from_equipment, y.to_equipment, y.admittance_list):
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
            branch_info.branches[key].phases[row] = row+1
            branch_info.branches[key].phases[col] = col+1
        elif rev_key in branch_info.branches:
            branch_info.branches[rev_key].y[col][row] = complex(v[0], v[1])
            branch_info.branches[rev_key].phases[row] = row+1
            branch_info.branches[rev_key].phases[col] = col+1
    return branch_info


def generate_graph(inc: Incidence, slack_bus: str) -> nx.Graph:
    graph = nx.Graph()
    for src, dst, id in zip(inc.from_equipment, inc.to_equipment, inc.ids):
        if "OPEN" in src or "OPEN" in dst:
            continue
        if src == dst:
            continue
        if "." in src:
            src, _ = src.split('.', 1)
        if "." in dst:
            dst, _ = dst.split('.', 1)
        eq = "LINE"
        if "sw" in id or "fuse" in id and "padswitch" not in id:
            eq = "SWITCH"
        if "tr" in id or "reg" in id or "xfm" in id:
            eq = "XFMR"
        graph.add_edge(src, dst, name=f"{src}_{dst}", tag=eq)

    for c in nx.connected_components(graph):
        if slack_bus in c:
            return graph.subgraph(c).copy()


def extract_info(topology: Topology) -> (BranchInfo, BusInfo, str):
    branch_info = BranchInfo()
    bus_info = BusInfo()
    slack_bus, _ = topology.slack_bus[0].split(".", 1)
    graph = generate_graph(topology.incidences, slack_bus)

    for u, v, a in graph.edges(data=True):
        branch_info.branches[a["name"]] = Branch(
            fr_bus=u, to_bus=v, tag=a["tag"])
        bus_info.buses[u] = Bus()
        bus_info.buses[v] = Bus()

    branch_info = extract_admittance(branch_info, topology.admittance)
    branch_info = generate_zprim(branch_info)
    bus_info = extract_base_kv(bus_info, topology.base_voltage_magnitudes)
    bus_info = extract_injection(bus_info, topology.injections)

    return index_info(branch_info, bus_info, slack_bus)
