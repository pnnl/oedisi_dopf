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
    for name in bus:
        bus[name]["phases"] = []
    for id, voltage in zip(voltages.ids, voltages.values):
        [name, phase] = id.split('.')
        if name not in bus:
            continue

        bus[name]['kv'] = voltage/1000.0
        bus[name]['phases'].append(phase)
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


def extract_powers(bus: dict, real: PowersReal, imag: PowersImaginary) -> dict:
    for id, eq, power in zip(real.ids, real.equipment_ids, real.values):
        [name, phase] = id.split('.')

        if name not in bus:
            continue
        [type, _] = eq.split('.')
        phase = int(phase) - 1
        if type == "PVSystem":
            logger.info(f"{id} : {power}")
            bus[name]["eqid"] = eq
            bus[name]["pv"][phase][0] = power*1000
        else:
            bus[name]["eqid"] = eq
            bus[name]["pq"][phase][0] = -power*1000

    for id, eq, power in zip(imag.ids, imag.equipment_ids, imag.values):
        [name, phase] = id.split('.')

        if name not in bus:
            continue

        [type, _] = eq.split('.')
        phase = int(phase) - 1
        if type == "PVSystem":
            bus[name]["eqid"] = eq
            bus[name]["pv"][phase][1] = power*1000
        else:
            bus[name]["eqid"] = eq
            bus[name]["pq"][phase][1] = -power*1000
    return bus


def extract_injection(bus: dict, powers: Injection) -> dict:
    real = powers.power_real
    imag = powers.power_imaginary

    for id, eq, power in zip(real.ids, real.equipment_ids, real.values):
        [name, phase] = id.split('.')

        if name not in bus:
            continue
        if name.find('OPEN') != -1:
            continue

        [type, _] = eq.split('.')
        phase = int(phase) - 1
        if type == "PVSystem":
            bus[name]["eqid"] = eq
            bus[name]["pv"][phase][0] = power*1000
            logger.debug(f"{eq}, {power}")
        else:
            bus[name]["eqid"] = eq
            bus[name]["pq"][phase][0] = -power*1000

    for id, eq, power in zip(imag.ids, imag.equipment_ids, imag.values):
        [name, phase] = id.split('.')

        if name not in bus:
            continue
        if name.find('OPEN') != -1:
            continue

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
    branch_info = {}
    bus_info = {}

    bus_voltages = {id : voltage \
                    for id, voltage in zip(topology.base_voltage_magnitudes.ids, 
                                       topology.base_voltage_magnitudes.values)}
    
    bus_phases = {}
    for id in topology.base_voltage_magnitudes.ids:
        [name, phase] = id.split('.')
        if name not in bus_phases:
            bus_phases[name] = [phase]
        else:
            bus_phases[name].append(phase)

    from_equip = topology.admittance.from_equipment
    to_equip = topology.admittance.to_equipment
    admittance = topology.admittance.admittance_list

    for fr_eq, to_eq, y in zip(from_equip, to_equip, admittance):
        # If voltage of two end nodes are same, the branch is a line or a switch
        # else it is a transformer. For a transformer, count the number of phases
        # If-else condition added
        if bus_voltages[fr_eq] == bus_voltages[to_eq]:
            type = "LINE"

            [from_name, from_phase] = fr_eq.split('.')
            if from_name.find('OPEN') != -1:
                [from_name, _] = from_name.split('_')
                type = "SWITCH"

            [to_name, to_phase] = to_eq.split('.')
            if to_name.find('OPEN') != -1:
                [to_name, _] = to_name.split('_')
                type = "SWITCH"

            if from_name == to_name:
                continue
        else:
            [from_name, from_phase] = fr_eq.split('.')
            [to_name, to_phase] = to_eq.split('.')
            
            # count number of phases for each bus
            fbus_phase_cnt = len(bus_phases[from_name])
            tbus_phase_cnt = len(bus_phases[to_name])
            if fbus_phase_cnt == 3 and tbus_phase_cnt == 3:
                type = "3 PHASE TSFR"
            elif (fbus_phase_cnt == 1 and tbus_phase_cnt==2) or (fbus_phase_cnt == 2 and tbus_phase_cnt==1):
                type = "SPLIT PHASE TSFR"
            elif (fbus_phase_cnt == 1 or tbus_phase_cnt==1):
                type = "1 PHASE TSFR"
            else:
                type = "SOME OTHER TSFR"

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

    return index_info(branch_info, bus_info)


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) == 1:
        case = "ieee123"
    else:
        case = sys.argv[1]
    with open (f"../outputs/{case}/topology.json", 'r') as jsonfile:
        topology = Topology.parse_obj(json.load(jsonfile))
    
    branch_info, bus_info = extract_info(topology)

    for k in branch_info:
        if branch_info[k]["type"] not in ["LINE","SWITCH"]:
            [fbus, tbus] = k.split('_')
            print(branch_info[k]["type"], bus_info[fbus]['kv'],
                  bus_info[tbus]['kv'], bus_info[fbus]["phases"], 
                  bus_info[tbus]["phases"])