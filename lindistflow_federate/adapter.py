import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import logging
from enum import IntEnum
from typing import Tuple
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
    VoltagesAngle,
    VoltagesImaginary,
    VoltagesMagnitude,
    VoltagesReal
)


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class Phase(IntEnum):
    A = 1
    B = 2
    C = 3

    def __repr__(self):
        return self.value


def init_branch() -> dict:
    branch = {}
    branch["fr_bus"] = ""
    branch["to_bus"] = ""
    branch["phases"] = []
    branch["zprim"] = np.zeros((3, 3, 2)).tolist()
    branch["y"] = np.zeros((3, 3), dtype=complex)
    return branch


def init_bus() -> dict:
    bus = {}
    bus["phases"] = []
    bus["kv"] = 0
    bus["pq"] = np.zeros((3, 2)).tolist()
    bus["pv"] = np.zeros((3, 2)).tolist()
    return bus


def index_info(branch: dict, bus: dict) -> Tuple[dict, dict]:
    for i, name in enumerate(bus):
        bus[name]["idx"] = i

    for i, name in enumerate(branch):
        branch[name]["idx"] = i
        branch[name]["from"] = bus[branch[name]["fr_bus"]]["idx"]
        branch[name]["to"] = bus[branch[name]["to_bus"]]["idx"]
        y = branch[name]["y"]
        del branch[name]["y"]
        z = -1*np.linalg.pinv(y)
        for idx, value in np.ndenumerate(z):
            row = idx[0]
            col = idx[1]
            branch[name]["zprim"][row][col] = [value.real, value.imag]

    return branch, bus


def extract_voltages(bus: dict, voltages: VoltagesMagnitude) -> dict:
    for id, voltage in zip(voltages.ids, voltages.values):
        [name, _] = id.split('.')
        if name not in bus:
            bus[name] = init_bus()

        bus[name]['kv'] = voltage/1000.0
    return bus


def extract_powers(bus: dict, powers: Injection) -> dict:
    real = powers.power_real
    imag = powers.power_imaginary

    for id, eq, power in zip(real.ids, real.equipment_ids, real.values):
        [name, phase] = id.split('.')
        [type, _] = eq.split('.')
        phase = int(phase) - 1
        if type == "PVSystem":
            bus[name]["eqid"] = eq
            bus[name]["pv"][phase][0] = power*1000
        else:
            bus[name]["eqid"] = eq
            bus[name]["pq"][phase][0] = -power*1000

    for id, eq, power in zip(imag.ids, imag.equipment_ids, imag.values):
        [name, phase] = id.split('.')
        [type, _] = eq.split('.')
        phase = int(phase) - 1
        if type == "PVSystem":
            bus[name]["eqid"] = eq
            bus[name]["pv"][phase][1] = power*1000
        else:
            bus[name]["eqid"] = eq
            bus[name]["pq"][phase][1] = -power*1000
    return bus


def extract_info(topology: Topology) -> Tuple[dict, dict]:
    network = nx.Graph()
    branch_info = {}
    bus_info = {}
    from_equip = topology.admittance.from_equipment
    to_equip = topology.admittance.to_equipment
    admittance = topology.admittance.admittance_list

    for fr_eq, to_eq, y in zip(from_equip, to_equip, admittance):
        [from_name, from_phase] = fr_eq.split('.')
        type = "LINE"
        if from_name.find('OPEN') != -1:
            [from_name, _] = from_name.split('_')
            type = "SWITCH"

        [to_name, to_phase] = to_eq.split('.')
        if to_name.find('OPEN') != -1:
            [to_name, _] = to_name.split('_')
            type = "SWITCH"

        if from_name == to_name:
            continue

        key = f"{from_name}_{to_name}"
        key_back = f"{to_name}_{from_name}"

        if key not in branch_info and key_back not in branch_info:
            branch_info[key] = init_branch()
        elif key_back in branch_info:
            continue

        if from_name not in bus_info:
            bus_info[from_name] = init_bus()

        if to_name not in bus_info:
            bus_info[to_name] = init_bus()

        row = int(from_phase) - 1
        col = int(to_phase) - 1
        branch_info[key]["y"][row][col] = complex(y[0], y[1])

        network.add_edge(from_name, to_name)

        if from_phase not in branch_info[key]['phases']:
            branch_info[key]['phases'].append(from_phase)

        if from_phase not in bus_info[from_name]['phases']:
            bus_info[from_name]['phases'].append(from_phase)

        if to_phase not in bus_info[to_name]['phases']:
            bus_info[to_name]['phases'].append(to_phase)

        branch_info[key]['type'] = type
        branch_info[key]['fr_bus'] = from_name
        branch_info[key]['to_bus'] = to_name

        bus_info = extract_voltages(bus_info, topology.base_voltage_magnitudes)
        bus_info = extract_powers(bus_info, topology.injections)

    # double distance between all nodes
    logger.debug(f"Nodes: {network.number_of_nodes()}")
    logger.debug(f"Edges: {network.number_of_edges()}")
    gcc = network.subgraph(
        sorted(nx.connected_components(network), key=len, reverse=True)[0])
    pos = nx.spring_layout(gcc, seed=123)
    nx.draw_networkx_nodes(gcc, pos, node_size=20)
    nx.draw_networkx_edges(gcc, pos, alpha=0.4)
    nx.draw_networkx_labels(gcc, pos, font_size=2)
    plt.savefig("network.svg")

    return index_info(branch_info, bus_info)
