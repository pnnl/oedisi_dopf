import numpy as np
import json
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
    Incidence,
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


def convert_id(eqid: str) -> (str, str):
    [bus, phase] = eqid.split('.', 1)
    return (bus, phase)


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
    bus["vmag"] = [0]*3
    bus["pq"] = np.zeros((3, 2)).tolist()
    bus["pv"] = np.zeros((3, 2)).tolist()
    bus["pq_forecast"] = np.zeros((3, 2)).tolist()
    bus["pv_forecast"] = np.zeros((3, 2)).tolist()
    return bus


def index_info(branch: dict, bus: dict) -> (dict, dict):
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

    return (branch, bus)


def extract_base_voltages(bus: dict, voltages: VoltagesMagnitude) -> dict:
    for id, voltage in zip(voltages.ids, voltages.values):
        name, phase = convert_id(id)

        if name not in bus:
            continue

        phase = int(phase) - 1
        bus[name]['kv'] = voltage/1000.0
    return bus


def extract_voltages(bus: dict, voltages: VoltagesMagnitude) -> dict:
    for id, voltage in zip(voltages.ids, voltages.values):
        name, phase = convert_id(id)

        if name not in bus:
            continue

        phase = int(phase) - 1
        bus[name]['vmag'][phase] = voltage
    return bus


def pack_voltages(voltages: dict, time: int) -> VoltagesMagnitude:
    ids = []
    values = []
    for key, value in voltages.items():
        for phase, voltage in value.items():
            if phase == 'A':
                id = f"{key}.1"
            if phase == 'B':
                id = f"{key}.2"
            if phase == 'C':
                id = f"{key}.3"
            ids.append(id)
            values.append(voltage)
    return VoltagesMagnitude(ids=ids, values=values, time=time)


def extract_forecast(bus: dict, forecast) -> dict:
    for eq, power in zip(forecast["ids"], forecast["values"]):
        if "_" in eq:
            [_, name] = eq.rsplit("_", 1)
        else:
            [_, name] = eq.rsplit(".", 1)
        name = name.upper()

        if name not in bus:
            continue
        if "OPEN" in name:
            continue

        phases = bus[name]["phases"]
        for ph in phases:
            phase = int(ph) - 1
            bus[name]["eqid"] = eq
            bus[name]["pv"][phase][0] = power*1000/len(phases)
            bus[name]["pv"][phase][1] = 0.0
    return bus


def extract_powers(bus: dict, real: PowersReal, imag: PowersImaginary) -> dict:
    for id, eq, power in zip(real.ids, real.equipment_ids, real.values):
        name, phase = convert_id(id)

        if name not in bus:
            continue
        if "OPEN" in name:
            continue

        print(id, eq, power)
        phase = int(phase) - 1
        bus[name]["pq"][phase][0] += power*1000

    for id, eq, power in zip(imag.ids, imag.equipment_ids, imag.values):
        name, phase = convert_id(id)

        if name not in bus:
            continue
        if "OPEN" in name:
            continue

        phase = int(phase) - 1
        bus[name]["pq"][phase][1] += power*1000
    return bus


def extract_injection(bus: dict, powers: Injection) -> dict:
    real = powers.power_real
    imag = powers.power_imaginary

    for id, eq, power in zip(real.ids, real.equipment_ids, real.values):
        name, phase = convert_id(id)

        if name not in bus:
            continue
        if "OPEN" in name:
            continue

        [type, _] = eq.split('.')
        phase = int(phase) - 1
        if type == "PVSystem":
            bus[name]["pv_forecast"][phase][0] += power*1000
        else:
            bus[name]["pq_forecast"][phase][0] += power*1000

    for id, eq, power in zip(imag.ids, imag.equipment_ids, imag.values):
        name, phase = convert_id(id)

        if name not in bus:
            continue
        if "OPEN" in name:
            continue

        [type, _] = eq.split('.')
        phase = int(phase) - 1
        if type == "PVSystem":
            bus[name]["pv_forecast"][phase][1] += power*1000
        else:
            bus[name]["pq_forecast"][phase][1] += power*1000
    return bus


class SwitchInfo:
    def __init__(self, name: str, from_bus: str, to_bus: str) -> None:
        self.name = name
        self.from_bus = from_bus
        self.to_bus = to_bus


def extract_switches(incidences: Incidence) -> list[SwitchInfo]:
    switches = []
    from_eq = incidences.from_equipment
    to_eq = incidences.to_equipment
    ids = incidences.ids
    for fr_eq, to_eq, eq_id in zip(from_eq, to_eq, ids):
        if ("sw" in eq_id or "fuse" in eq_id) and "padswitch" not in eq_id:
            if "." in fr_eq:
                [fr_eq, _] = fr_eq.split('.', 1)
            if "." in to_eq:
                [to_eq, _] = to_eq.split('.', 1)
            switches.append(
                SwitchInfo(
                    name=eq_id,
                    from_bus=fr_eq,
                    to_bus=to_eq))

    return switches


def extract_info(topology: Topology) -> (dict, dict):
    branch_info = {}
    bus_info = {}
    switches = extract_switches(topology.incidences)
    fr_buses = [switch.from_bus for switch in switches]
    to_buses = [switch.to_bus for switch in switches]
    from_equip = topology.admittance.from_equipment
    to_equip = topology.admittance.to_equipment
    admittance = topology.admittance.admittance_list

    for fr_eq, to_eq, y in zip(from_equip, to_equip, admittance):
        type = "LINE"
        [from_name, from_phase] = fr_eq.split('.')
        [to_name, to_phase] = to_eq.split('.')

        forward_link = from_name in fr_buses and to_name in to_buses
        reverse_link = to_name in fr_buses and from_name in to_buses
        if forward_link or reverse_link:
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

        if from_phase not in branch_info[key]['phases']:
            branch_info[key]['phases'].append(from_phase)

        row = int(from_phase) - 1
        col = int(to_phase) - 1
        branch_info[key]["y"][row][col] = complex(y[0], y[1])

        if from_phase not in branch_info[key]['phases']:
            branch_info[key]['phases'].append(from_phase)

        if from_phase not in bus_info[from_name]['phases']:
            bus_info[from_name]['phases'].append(from_phase)

        if to_phase not in bus_info[to_name]['phases']:
            bus_info[to_name]['phases'].append(to_phase)

        branch_info[key]['type'] = type
        branch_info[key]['fr_bus'] = from_name.upper()
        branch_info[key]['to_bus'] = to_name.upper()

    base_v = topology.base_voltage_magnitudes
    bus_info = extract_base_voltages(bus_info, base_v)

    return index_info(branch_info, bus_info)
