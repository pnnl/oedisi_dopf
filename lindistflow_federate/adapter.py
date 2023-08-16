import numpy as np
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
    branch["phases"] = []
    branch["zprim"] = np.zeros((3, 3, 2)).tolist()
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

    return branch, bus


def extract_voltages(bus: dict, voltages: VoltagesMagnitude) -> dict:
    for id, voltage in zip(voltages.ids, voltages.values):
        [name, _] = id.split('.')
        if name not in bus:
            bus[name] = init_bus()

        bus[name]['kv'] = voltage
    return bus


def extract_powers(bus: dict, powers: Injection) -> dict:
    real = powers.power_real
    imag = powers.power_imaginary

    for id, eq, power in zip(real.ids, real.equipment_ids, real.values):
        [name, phase] = id.split('.')
        [type, _] = eq.split('.')
        phase = int(phase) - 1
        if type == "PVSystem":
            bus[name]["pv"][phase][0] = power
        else:
            bus[name]["pq"][phase][0] = power

    for id, eq, power in zip(imag.ids, imag.equipment_ids, imag.values):
        [name, phase] = id.split('.')
        [type, _] = eq.split('.')
        phase = int(phase) - 1
        if type == "PVSystem":
            bus[name]["pv"][phase][1] = power
        else:
            bus[name]["pq"][phase][1] = power
    return bus


def extract_info(topology: Topology) -> Tuple[dict, dict]:
    branch_info = {}
    bus_info = {}
    from_equip = topology.admittance.from_equipment
    to_equip = topology.admittance.to_equipment
    admittance = topology.admittance.admittance_list

    for fr_eq, to_eq, y in zip(from_equip, to_equip, admittance):
        [from_name, from_phase] = fr_eq.split('.')
        [to_name, to_phase] = to_eq.split('.')
        from_phase = int(from_phase)
        to_phase = int(to_phase)

        if from_name not in branch_info:
            branch_info[from_name] = init_branch()

        if from_phase not in branch_info[from_name]['phases']:
            branch_info[from_name]['phases'].append(from_phase)

        if to_name not in bus_info:
            bus_info[to_name] = init_bus()

        if to_phase not in bus_info[to_name]['phases']:
            bus_info[to_name]['phases'].append(to_phase)

            z = 1/complex(y[0], y[1])
            row = from_phase - 1
            col = to_phase - 1
            branch_info[from_name]["zprim"][row][col] = [z.real, z.imag]
            branch_info[from_name]["zprim"][col][row] = [z.real, z.imag]

        branch_info[from_name]['type'] = 'LINE'
        branch_info[from_name]['fr_bus'] = from_name
        branch_info[from_name]['to_bus'] = to_name

        bus_info = extract_voltages(bus_info, topology.base_voltage_magnitudes)
        bus_info = extract_powers(bus_info, topology.injections)

    return index_info(branch_info, bus_info)
