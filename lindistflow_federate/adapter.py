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


def init_bus(base_voltage: float) -> dict:
    bus = {}
    bus["phases"] = []
    bus["kv"] = base_voltage
    bus["pq"] = np.zeros((3, 2)).tolist()
    bus["pv"] = np.zeros((3, 2)).tolist()
    return bus


def extract_info(topology: Topology) -> Tuple[dict, dict]:
    branch = init_branch()
    bus = init_bus(2400.0)
    from_equip = topology.admittance.from_equipment
    to_equip = topology.admittance.to_equipment
    admittance = topology.admittance.admittance_list
    logger.info(
        f"from: {len(from_equip)}, to: {len(to_equip)}, admittance: {len(admittance)}")

    for fr_eq, to_eq, y in zip(from_equip, to_equip, admittance):
        [name, phase] = fr_eq.split('.')
        logger.info(name)
        logger.info(phase)
        logger.info(1/complex(y[0], y[1]))
    return branch, bus
