from enum import Enum
import numpy as np
import cvxpy as cp
import math
import logging
import copy
import json
from dataclasses import asdict
from adapter import BranchInfo, Branch, BusInfo, Bus

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

"""
Created on Tue March 30 11:11:21 2021

@author:  Rabayet
"""


"""

Updated:

1. Dist_flag is default False
2. Primary KV level variable is introduced
3. DER maximization without alpha parameter.

4. Hard coded for the split-phase using last char. a,b or c from OpenDSS

5. Q control added
6. Changed BaseS to 1MW and Base kV

"""


class ControlType(Enum):
    WATT = 1
    VAR = 2
    WATT_VAR = 3


def power_balance(A, b, k_frm, k_to, counteq, col, val):
    for k in k_frm:
        A[counteq, col + k] = -1
    for k in k_to:
        A[counteq, col + k] = 1

    A[counteq, val] = -1
    b[counteq] = 0
    return A, b


def convert_pu(
    branch_info: BranchInfo, bus_info: BusInfo
) -> (BranchInfo, BusInfo, float):
    pq_pu = 1 / (100 * 1e6)
    kw_to_va = 1.2
    bus_pu = copy.deepcopy(bus_info)
    branch_pu = copy.deepcopy(branch_info)

    for k, v in bus_info.buses.items():
        bus_pu.buses[k].pq = [[pq * pq_pu for pq in phase] for phase in v.pq]
        bus_pu.buses[k].base_pq = [[pq * pq_pu for pq in phase]
                                   for phase in v.base_pq]
        bus_pu.buses[k].pv = [[pq * pq_pu for pq in phase] for phase in v.pv]
        bus_pu.buses[k].base_pv = [
            [pq * kw_to_va * pq_pu for pq in phase] for phase in v.base_pv
        ]

    for k, v in branch_pu.branches.items():
        if (
            "REG" == v.tag
            or bus_pu.buses[v.fr_bus].base_kv < 1.0
            or bus_pu.buses[v.to_bus].base_kv < 1.0
        ):
            base_kv = 1e6  # impedance will become near zero
        else:
            base_kv = bus_pu.buses[v.fr_bus].base_kv

        z_base = 1 / (base_kv**2 / 100)
        branch_pu.branches[k].zprim = [
            [[e * z_base for e in l1] for l1 in l2] for l2 in v.zprim
        ]

    return (branch_pu, bus_pu, 1 / pq_pu / 1000)


def voltage_cons_pri(
    A,
    b,
    p,
    frm,
    to,
    counteq,
    pii,
    qii,
    pij,
    qij,
    pik,
    qik,
    nbus_ABC,
    nbus_s1s2,
    nbranch_ABC,
    baseZ,
):
    A[counteq, frm] = 1
    A[counteq, to] = -1
    n_flow_ABC = (nbus_ABC * 3 + nbus_s1s2) + (nbus_ABC * 6 + nbus_s1s2 * 2)

    # real power drop
    A[counteq, p + n_flow_ABC + nbranch_ABC * 0] = pii / baseZ
    A[counteq, p + n_flow_ABC + nbranch_ABC * 1] = pij / baseZ
    A[counteq, p + n_flow_ABC + nbranch_ABC * 2] = pik / baseZ

    # reactive power drop
    A[counteq, p + n_flow_ABC + nbranch_ABC * 3] = qii / baseZ
    A[counteq, p + n_flow_ABC + nbranch_ABC * 4] = qij / baseZ
    A[counteq, p + n_flow_ABC + nbranch_ABC * 5] = qik / baseZ
    b[counteq] = 0.0
    return A, b


def voltage_cons_sec(
    A,
    b,
    p,
    frm,
    to,
    counteq,
    p_pri,
    q_pri,
    p_sec,
    q_sec,
    nbus_ABC,
    nbus_s1s2,
    nbranch_ABC,
    nbranch_s1s2,
):
    A[counteq, frm] = 1
    A[counteq, to] = -1
    n_flow_s1s2 = (
        (nbus_ABC * 3 + nbus_s1s2) +
        (nbus_ABC * 6 + nbus_s1s2 * 2) + nbranch_ABC * 6
    )
    # real power drop
    A[counteq, p + n_flow_s1s2] = p_pri + 0.5 * p_sec
    # reactive power drop
    A[counteq, p + n_flow_s1s2 + nbranch_s1s2] = q_pri + 0.5 * q_sec
    b[counteq] = 0.0
    return A, b


def update_ratios(branch_info: BranchInfo, bus_info: BusInfo) -> BusInfo:
    for node in branch_info.branches.values():
        if "REG" == node.tag:
            src = node.fr_bus
            dst = node.to_bus
            v1 = bus_info.buses[src].kv
            v2 = bus_info.buses[dst].kv
            r = v1 / v2

            if v1 < 0.3 or v2 < 0.3:
                continue

            bus_info.buses[src].tap_ratio = 0
            bus_info.buses[dst].tap_ratio = 1 / r

    return bus_info


def solve(branch_info: dict, bus_info: dict, slack_bus: str, relaxed: bool):
    try:
        return optimal_power_flow(branch_info, bus_info, slack_bus, relaxed)

    except:
        return optimal_power_flow(branch_info, bus_info, slack_bus, True)


def optimal_power_flow(
    branch_info: dict, bus_info: dict, slack_bus: str, relaxed: bool
):
    # System's base definition
    BASE_S = 1  # MVA
    PRIMARY_V = 0.12
    branch_pu, bus_pu, kw_converter = convert_pu(branch_info, bus_info)
    bus_pu = update_ratios(branch_pu, bus_pu)

    with open("bus_info_pu.json", "w") as outfile:
        outfile.write(json.dumps(asdict(bus_pu)))

    with open("branch_info_pu.json", "w") as outfile:
        outfile.write(json.dumps(asdict(branch_pu)))

    branches = branch_pu.branches
    buses = bus_pu.buses

    slack_v = max([b.kv for b in buses.values()])
    basekV = buses[slack_bus].base_kv
    baseZ = 1.0
    SOURCE_V = [slack_v / basekV] * 3

    # Find the ABC phase and s1s2 phase triplex line and bus numbers
    nbranch_ABC = 0
    nbus_ABC = 0
    nbranch_s1s2 = 0
    nbus_s1s2 = 0
    mult = 1
    secondary_model = ["TPX_LINE", "SPLIT_PHASE"]
    name = []
    for b_eq in branches:
        if branches[b_eq].tag in secondary_model:
            nbranch_s1s2 += 1
        else:
            nbranch_ABC += 1

    for k, b in buses.items():
        name.append(k)
        if b.base_kv > PRIMARY_V:
            nbus_ABC += 1
        else:
            nbus_s1s2 += 1

    # Number of Optimization Variables
    voltage_count = nbus_ABC * 3 + nbus_s1s2
    injection_count = nbus_ABC * 6 + nbus_s1s2 * 2
    flow_count = nbranch_ABC * 6 + nbranch_s1s2 * 2
    pdg_count = nbus_ABC * 3 + nbus_s1s2
    qdg_count = nbus_ABC * 3 + nbus_s1s2
    variable_number = (
        voltage_count + injection_count + flow_count + pdg_count + qdg_count
    )

    # Number of equality/inequality constraints (Injection equations (ABC) at each bus)
    #    #  Check if this is correct number or not:
    n_bus = nbus_ABC * 3 + nbus_s1s2  # Total Bus Number
    n_branch = nbranch_ABC * 3 + nbranch_s1s2  # Total Branch Number

    constraint_number = 1000 + variable_number + n_bus + 3 * n_bus + n_branch
    A_ineq = np.zeros((constraint_number, variable_number))
    b_ineq = np.zeros(constraint_number)

    x = cp.Variable(variable_number)
    # Initialize the matrices
    P = np.zeros((variable_number, variable_number))
    q_obj_vector = np.zeros(variable_number)
    A_eq = np.zeros((constraint_number, variable_number))
    b_eq = np.zeros(constraint_number)

    # Some extra variable definition for clean code:
    #                      Voltage      PQ_inj     PQ_flow
    state_variable_number = n_bus + 2 * n_bus + 2 * n_branch

    # Q-dg variable starting number
    #                               P_dg
    n_Qdg = state_variable_number + n_bus
    # s = 0

    # # Deciding the starting index for Control Variable
    # if P_control==True:
    #     variable_start_idx = state_variable_number
    # elif Q_control==True:
    #     variable_start_idx = n_Qdg

    # Linear Programming Cost Vector:
    # for k in range(nbus_ABC  + nbus_s1s2):

    for k in range(n_bus):
        q_obj_vector[state_variable_number + k] = -1  # DER max objective

        # q_obj_vector[n_Qdg + k] = -1  # Just Voltage regulation

    # # Define BFM constraints for both real and reactive power: Power flow conservaion
    # Constraint 1: Flow equation

    # sum(Sij) - sum(Sjk) == -sj

    counteq = 0
    for keyb, val_bus in buses.items():
        if keyb == slack_bus:
            continue

        k_frm_3p = []
        k_to_3p = []
        k_frm_1p = []
        k_frm_1pa, k_frm_1pb, k_frm_1pc = [], [], []
        k_frm_1qa, k_frm_1qb, k_frm_1qc = [], [], []
        k_to_1p = []

        # Find bus idx in "from" of branch_sw_data
        ind_frm = 0
        ind_to = 0
        if val_bus.base_kv < PRIMARY_V:
            for key, val_br in branches.items():
                if val_bus.idx == val_br.fr_idx:
                    k_frm_1p.append(ind_frm - nbranch_ABC)

                if val_bus.idx == val_br.to_idx:
                    k_to_1p.append(ind_to - nbranch_ABC)

                ind_to += 1
                ind_frm += 1

            loc = (
                (nbus_ABC * 3 + nbus_s1s2)
                + (nbus_ABC * 6 + nbus_s1s2 * 2)
                + nbranch_ABC * 6
            )
            real_idx = val_bus.idx + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 5
            A_eq, b_eq = power_balance(
                A_eq, b_eq, k_frm_1p, k_to_1p, counteq, loc, real_idx
            )
            counteq += 1

            imag_idx = val_bus.idx + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 5 + nbus_s1s2
            A_eq, b_eq = power_balance(
                A_eq, b_eq, k_frm_1p, k_to_1p, counteq, loc + nbranch_s1s2, imag_idx
            )
            counteq += 1

        else:
            for key, val_br in branches.items():
                if val_bus.idx == val_br.fr_idx:
                    if buses[val_br.to_bus].base_kv > PRIMARY_V:
                        k_frm_3p.append(ind_frm)
                    else:
                        if val_br.phases[0] != 0:
                            k_frm_1pa.append(
                                nbranch_ABC * 6 + ind_frm - nbranch_ABC)
                            k_frm_1qa.append(
                                nbranch_ABC * 3 + ind_frm - nbranch_ABC + nbranch_s1s2
                            )
                        if val_br.phases[1] != 0:
                            k_frm_1pb.append(
                                nbranch_ABC * 5 + ind_frm - nbranch_ABC)
                            k_frm_1qb.append(
                                nbranch_ABC * 2 + ind_frm - nbranch_ABC + nbranch_s1s2
                            )
                        if val_br.phases[2] != 0:
                            k_frm_1pc.append(
                                nbranch_ABC * 4 + ind_frm - nbranch_ABC)
                            k_frm_1qc.append(
                                nbranch_ABC * 1 + ind_frm - nbranch_ABC + nbranch_s1s2
                            )

                if val_bus.idx == val_br.to_idx:
                    if buses[val_br.fr_bus].base_kv > PRIMARY_V:
                        k_to_3p.append(ind_to)
                    else:
                        k_to_1p.append(ind_to - nbranch_ABC)
                ind_to += 1
                ind_frm += 1

            loc = (nbus_ABC * 3 + nbus_s1s2) + (nbus_ABC * 6 + nbus_s1s2 * 2)
            # Finding the kfrms
            k_frm_A = k_frm_3p + k_frm_1pa
            k_frm_B = k_frm_3p + k_frm_1pb
            k_frm_C = k_frm_3p + k_frm_1pc
            k_to_A = k_to_B = k_to_C = k_to_3p

            # Real Power balance equations
            # # Phase A
            pa_idx = val_bus.idx + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 0
            A_eq, b_eq = power_balance(
                A_eq, b_eq, k_frm_A, k_to_A, counteq, loc, pa_idx
            )
            counteq += 1

            # # # Phase B
            pb_idx = val_bus.idx + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 1
            A_eq, b_eq = power_balance(
                A_eq, b_eq, k_frm_B, k_to_B, counteq, loc + nbranch_ABC, pb_idx
            )
            counteq += 1

            # # # Phase C
            pc_idx = val_bus.idx + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 2
            A_eq, b_eq = power_balance(
                A_eq, b_eq, k_frm_C, k_to_C, counteq, loc + nbranch_ABC * 2, pc_idx
            )
            counteq += 1

            k_frm_A = k_frm_3p + k_frm_1qa
            k_frm_B = k_frm_3p + k_frm_1qb
            k_frm_C = k_frm_3p + k_frm_1qc

            # Reactive Power balance equations
            loc = (
                (nbus_ABC * 3 + nbus_s1s2)
                + (nbus_ABC * 6 + nbus_s1s2 * 2)
                + nbranch_ABC * 3
            )
            # Phase A
            qa_idx = val_bus.idx + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 3
            A_eq, b_eq = power_balance(
                A_eq, b_eq, k_frm_A, k_to_A, counteq, loc, qa_idx
            )
            counteq += 1

            # Phase B
            qb_idx = val_bus.idx + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 4
            A_eq, b_eq = power_balance(
                A_eq, b_eq, k_frm_B, k_to_B, counteq, loc + nbranch_ABC, qb_idx
            )
            counteq += 1

            # Phase C
            qc_idx = val_bus.idx + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 5
            A_eq, b_eq = power_balance(
                A_eq, b_eq, k_frm_C, k_to_C, counteq, loc + nbranch_ABC * 2, qc_idx
            )
            counteq += 1

    # Constraint 2: Voltage drop equation:
    # Vj = Vi - Zij Sij* - Sij Zij*

    # For Primary Nodes:
    idx = 0
    v_lim = []
    for k, val_br in branches.items():
        # Not writing voltage constraints for transformers
        if val_br.tag in secondary_model:
            continue

        if "REG" != val_br.tag:
            z = np.asarray(val_br.zprim)
            v_lim.append(val_br.fr_idx)
            v_lim.append(val_br.to_idx)
            # Writing three phase voltage constraints
            # Phase A
            paa, qaa = -2 * z[0, 0][0], -2 * z[0, 0][1]
            pab, qab = (
                -(-z[0, 1][0] + math.sqrt(3) * z[0, 1][1]),
                -(-z[0, 1][1] - math.sqrt(3) * z[0, 1][0]),
            )
            pac, qac = (
                -(-z[0, 2][0] - math.sqrt(3) * z[0, 2][1]),
                -(-z[0, 2][1] + math.sqrt(3) * z[0, 2][0]),
            )
            A_eq, b_eq = voltage_cons_pri(
                A_eq,
                b_eq,
                idx,
                val_br.fr_idx,
                val_br.to_idx,
                counteq,
                paa,
                qaa,
                pab,
                qab,
                pac,
                qac,
                nbus_ABC,
                nbus_s1s2,
                nbranch_ABC,
                baseZ,
            )
            counteq += 1

            # Phase B
            pbb, qbb = -2 * z[1, 1][0], -2 * z[1, 1][1]
            pba, qba = (
                -(-z[0, 1][0] - math.sqrt(3) * z[0, 1][1]),
                -(-z[0, 1][1] + math.sqrt(3) * z[0, 1][0]),
            )
            pbc, qbc = (
                -(-z[1, 2][0] + math.sqrt(3) * z[1, 2][1]),
                -(-z[1, 2][1] - math.sqrt(3) * z[1, 2][0]),
            )
            A_eq, b_eq = voltage_cons_pri(
                A_eq,
                b_eq,
                idx,
                nbus_ABC + val_br.fr_idx,
                nbus_ABC + val_br.to_idx,
                counteq,
                pba,
                qba,
                pbb,
                qbb,
                pbc,
                qbc,
                nbus_ABC,
                nbus_s1s2,
                nbranch_ABC,
                baseZ,
            )
            counteq += 1

            # Phase C
            pcc, qcc = -2 * z[2, 2][0], -2 * z[2, 2][1]
            pca, qca = (
                -(-z[0, 2][0] + math.sqrt(3) * z[0, 2][1]),
                -(-z[0, 2][1] - math.sqrt(3) * z[0, 2][0]),
            )
            pcb, qcb = (
                -(-z[1, 2][0] - math.sqrt(3) * z[1, 2][1]),
                -(-z[0, 2][1] + math.sqrt(3) * z[1, 2][0]),
            )
            A_eq, b_eq = voltage_cons_pri(
                A_eq,
                b_eq,
                idx,
                nbus_ABC * 2 + val_br.fr_idx,
                nbus_ABC * 2 + val_br.to_idx,
                counteq,
                pca,
                qca,
                pcb,
                qcb,
                pcc,
                qcc,
                nbus_ABC,
                nbus_s1s2,
                nbranch_ABC,
                baseZ,
            )
            counteq += 1
        elif "REG" == val_br.tag:
            tap_ratio = buses[val_br.to_bus].tap_ratio
            # Phase A
            A_eq[counteq, val_br.fr_idx] = tap_ratio**2
            A_eq[counteq, val_br.to_idx] = -1
            b_eq[counteq] = 0.0
            counteq += 1

            # Phase B
            A_eq[counteq, nbus_ABC + val_br.fr_idx] = tap_ratio**2
            A_eq[counteq, nbus_ABC + val_br.to_idx] = -1
            b_eq[counteq] = 0.0
            counteq += 1

            # Phase C
            A_eq[counteq, nbus_ABC * 2 + val_br.fr_idx] = tap_ratio**2
            A_eq[counteq, nbus_ABC * 2 + val_br.to_idx] = -1
            b_eq[counteq] = 0.0
            counteq += 1
        idx += 1

    idx = 0
    pq_index = []
    for k, val_br in branches.items():
        # For split phase transformer, we use interlace design
        if val_br.tag not in secondary_model:
            continue

        if val_br.tag == "SPLIT_PHASE":
            pq_index.append(val_br.idx)
            zp = np.asarray(val_br["impedance"])
            zs = np.asarray(val_br["impedance1"])
            v_lim.append(val_br.fr_idx)
            v_lim.append(val_br.to_idx)
            # Writing voltage constraints
            # Phase S1
            p_pri, q_pri = -2 * zp[0], -2 * zp[1]
            p_sec, q_sec = -2 * zs[0], -2 * zs[1]
            if val_br.phase == 1:
                from_bus = val_br.fr_idx
            if val_br.phase == 2:
                from_bus = val_br.fr_idx + nbus_ABC
            if val_br.phase == 3:
                from_bus = val_br.fr_idx + nbus_ABC * 2
            to_bus = val_br.to_idx - nbus_ABC + nbus_ABC * 3

            # Monish
            A_eq, b_eq = voltage_cons_sec(
                A_eq,
                b_eq,
                idx - nbranch_ABC,
                from_bus,
                to_bus,
                counteq,
                p_pri,
                q_pri,
                p_sec,
                q_sec,
                nbus_ABC,
                nbus_s1s2,
                nbranch_ABC,
                baseZ,
            )
            counteq += 1

        else:
            # For triplex line, we assume there is no mutual coupling
            # The impedance of line will be converted into pu here.
            zbase = 120.0 * 120.0 / 15000
            zp = np.asarray(val_br["impedance"])
            v_lim.append(val_br.fr_idx)
            v_lim.append(val_br.to_idx)
            # Writing voltage constraints
            # Phase S1
            p_s1, q_s1 = 0, 0
            p_s2, q_s2 = -2 * zp[0, 0][0] / zbase, -2 * zp[0, 0][1] / zbase
            from_bus = val_br.fr_idx - nbus_ABC + nbus_ABC * 3
            to_bus = val_br.to_idx - nbus_ABC + nbus_ABC * 3

            # Monish
            A_eq, b_eq = voltage_cons_sec(
                A_eq,
                b_eq,
                idx - nbranch_ABC,
                from_bus,
                to_bus,
                counteq,
                p_s1,
                q_s1,
                p_s2,
                q_s2,
                nbus_ABC,
                nbus_s1s2,
                nbranch_ABC,
                baseZ,
            )
            counteq += 1
        idx += 1

    # Constraint 3: Substation voltage definition
    # V_substation = V_source
    source_bus_idx = buses[slack_bus].idx
    A_eq[counteq, source_bus_idx] = 1
    b_eq[counteq] = (SOURCE_V[0]) ** 2
    counteq += 1
    A_eq[counteq, source_bus_idx + nbus_ABC] = 1
    b_eq[counteq] = (SOURCE_V[1]) ** 2
    counteq += 1
    A_eq[counteq, source_bus_idx + nbus_ABC * 2] = 1
    b_eq[counteq] = (SOURCE_V[2]) ** 2
    counteq += 1

    # BFM Power Flow Model ends. Next we define the Control variables:

    # # Make injection a decision variable

    # print("Start: Injection Constraints")

    # P_dg control:
    DG_up_lim = np.zeros((n_bus, 1))
    DG_active_up_lim = np.zeros((n_bus, 1))
    for keyb, val_bus in buses.items():
        if keyb == slack_bus:
            continue

        # Real power injection at a bus
        if val_bus.base_kv > PRIMARY_V:
            # p_inj  + p_gen(control var) =  p_load
            # Phase A Real Power
            A_eq[counteq, nbus_ABC * 3 + nbus_s1s2 +
                 nbus_ABC * 0 + val_bus.idx] = 1
            A_eq[counteq, state_variable_number +
                 nbus_ABC * 0 + val_bus.idx] = 1
            b_eq[counteq] = val_bus.pq[0][0] * BASE_S * mult
            counteq += 1
            # Phase B Real Power
            A_eq[counteq, nbus_ABC * 3 + nbus_s1s2 +
                 nbus_ABC * 1 + val_bus.idx] = 1
            A_eq[counteq, state_variable_number +
                 nbus_ABC * 1 + val_bus.idx] = 1
            b_eq[counteq] = val_bus.pq[1][0] * BASE_S * mult
            counteq += 1
            # Phase C Real Power
            A_eq[counteq, nbus_ABC * 3 + nbus_s1s2 +
                 nbus_ABC * 2 + val_bus.idx] = 1
            A_eq[counteq, state_variable_number +
                 nbus_ABC * 2 + val_bus.idx] = 1
            b_eq[counteq] = val_bus.pq[2][0] * BASE_S * mult
            counteq += 1

            # Q_inj  + Q_gen(control var) =  Q_load
            # Phase A Reactive power
            A_eq[counteq, nbus_ABC * 3 + nbus_s1s2 +
                 nbus_ABC * 3 + val_bus.idx] = 1
            A_eq[counteq, n_Qdg + nbus_ABC * 0 + val_bus.idx] = 1
            b_eq[counteq] = val_bus.pq[0][1] * BASE_S * mult
            counteq += 1
            # Phase B Reactive power
            A_eq[counteq, nbus_ABC * 3 + nbus_s1s2 +
                 nbus_ABC * 4 + val_bus.idx] = 1
            A_eq[counteq, n_Qdg + nbus_ABC * 1 + val_bus.idx] = 1
            b_eq[counteq] = val_bus.pq[1][1] * BASE_S * mult
            counteq += 1
            # Phase C Reactive power
            A_eq[counteq, nbus_ABC * 3 + nbus_s1s2 +
                 nbus_ABC * 5 + val_bus.idx] = 1
            A_eq[counteq, n_Qdg + nbus_ABC * 2 + val_bus.idx] = 1
            b_eq[counteq] = val_bus.pq[2][1] * BASE_S * mult
            counteq += 1

            # DG upper limit set up:
            DG_up_lim[nbus_ABC * 0 +
                      val_bus.idx] = val_bus.base_pv[0][0] * BASE_S
            DG_up_lim[nbus_ABC * 1 +
                      val_bus.idx] = val_bus.base_pv[1][0] * BASE_S
            DG_up_lim[nbus_ABC * 2 +
                      val_bus.idx] = val_bus.base_pv[2][0] * BASE_S

            # DG active limit set up:
            DG_active_up_lim[nbus_ABC * 0 +
                             val_bus.idx] = val_bus.pv[0][0] * BASE_S
            DG_active_up_lim[nbus_ABC * 1 +
                             val_bus.idx] = val_bus.pv[1][0] * BASE_S
            DG_active_up_lim[nbus_ABC * 2 +
                             val_bus.idx] = val_bus.pv[2][0] * BASE_S

        # work on this for the secondary netowrks:
        else:
            A_eq[counteq, nbus_ABC * 3 + nbus_s1s2 +
                 nbus_ABC * 5 + val_bus.idx] = 1
            A_eq[counteq, state_variable_number + val_bus.idx] = 1
            b_eq[counteq] = val_bus.pq[0] * BASE_S * mult
            counteq += 1
            # Reactive power
            A_eq[
                counteq,
                nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 5 + nbus_s1s2 + val_bus.idx,
            ] = 1
            A_eq[counteq, n_Qdg + nbus_ABC * 2 + val_bus.idx] = -1
            b_eq[counteq] = val_bus.pq[1] * BASE_S
            counteq += 1

            DG_up_lim[nbus_ABC * 3 + val_bus.idx] = val_bus.pv[0] * BASE_S

    # Reactive power as a function of real power and inverter rating
    countineq = 0

    # P Q both control:
    for k in range(n_bus):
        A_ineq[countineq, state_variable_number + k] = -1 * math.sqrt(3)
        A_ineq[countineq, n_Qdg + k] = -1
        b_ineq[countineq] = math.sqrt(3) * DG_up_lim[k, 0]
        countineq += 1

        A_ineq[countineq, state_variable_number + k] = math.sqrt(3)
        A_ineq[countineq, n_Qdg + k] = 1
        b_ineq[countineq] = math.sqrt(3) * DG_up_lim[k, 0]
        countineq += 1

        A_ineq[countineq, n_Qdg + k] = -1
        b_ineq[countineq] = (math.sqrt(3) / 2) * DG_up_lim[k, 0]
        countineq += 1

        A_ineq[countineq, n_Qdg + k] = 1
        b_ineq[countineq] = (math.sqrt(3) / 2) * DG_up_lim[k, 0]
        countineq += 1

        A_ineq[countineq, state_variable_number + k] = math.sqrt(3)
        A_ineq[countineq, n_Qdg + k] = -1
        b_ineq[countineq] = math.sqrt(3) * DG_up_lim[k, 0]
        countineq += 1

        A_ineq[countineq, state_variable_number + k] = -1 * math.sqrt(3)
        A_ineq[countineq, n_Qdg + k] = 1
        b_ineq[countineq] = math.sqrt(3) * DG_up_lim[k, 0]
        countineq += 1

        # add active power limit to mppt:
        A_ineq[countineq, state_variable_number + k] = 1
        b_ineq[countineq] = DG_active_up_lim[k, 0]
        countineq += 1

        A_ineq[countineq, state_variable_number + k] = -1
        b_ineq[countineq] = 0.0
        countineq += 1

    # Constraint 3: 0.95^2 <= V <= 1.05^2 (For those nodes where voltage constraint exist)
    # print("Formulating voltage limit constraints")
    v_idxs = list(set(v_lim))
    # # Does the vmin make sense here?
    if relaxed is True:
        vmax = 1.1
        vmin = 0.9
    else:
        vmax = 1.05
        vmin = 0.95

    for k in range(nbus_ABC):
        if k in v_idxs:
            # Upper bound
            A_ineq[countineq, k] = 1
            b_ineq[countineq] = vmax**2
            countineq += 1
            A_ineq[countineq, k + nbus_ABC] = 1
            b_ineq[countineq] = vmax**2
            countineq += 1
            A_ineq[countineq, k + nbus_ABC * 2] = 1
            b_ineq[countineq] = vmax**2
            countineq += 1
            # Lower Bound
            A_ineq[countineq, k] = -1
            b_ineq[countineq] = -(vmin**2)
            countineq += 1
            A_ineq[countineq, k + nbus_ABC] = -1
            b_ineq[countineq] = -(vmin**2)
            countineq += 1
            A_ineq[countineq, k + nbus_ABC * 2] = -1
            b_ineq[countineq] = -(vmin**2)
            countineq += 1

    prob = cp.Problem(
        cp.Minimize(q_obj_vector.T @ x), [A_ineq @
                                          x <= b_ineq, A_eq @ x == b_eq]
    )

    prob.solve(solver=cp.CLARABEL, verbose=True)
    print(prob.solver_stats)
    extra_stats = prob.solver_stats.extra_stats
    if extra_stats is None:
        extra_stats = {}

    opt_gap = 0
    if "info" in extra_stats.keys() and "gap" in extra_stats.keys():
        opt_gap = prob.solver_stats.extra_stats["info"]["gap"]

    fea_gap = 0
    if "info" in extra_stats.keys() and "pres" in extra_stats.keys():
        fea_gap = prob.solver_stats.extra_stats["info"]["pres"]

    stats = {
        "solve_time": prob.solver_stats.solve_time,
        "num_iters": prob.solver_stats.num_iters,
        "optimality_gap": opt_gap,
        "feasibility_gap": fea_gap,
    }

    if prob.status.lower() != "optimal":
        logger.debug("Check for limits. Power flow didn't converge")
        logger.debug(prob.status)
        raise prob.status

    from_bus = []
    to_bus = []
    name = []

    for key, val_br in branches.items():
        from_bus.append(val_br.fr_bus)
        to_bus.append(val_br.to_bus)
        name.append(key)

    i = 0
    mul = 1 / (BASE_S * 1000)
    line_flow = {}
    n_flow_ABC = (nbus_ABC * 3 + nbus_s1s2) + (nbus_ABC * 6 + nbus_s1s2 * 2)

    bus_names = list(buses.keys())
    bus_flows = {}
    for key, val_bus in buses.items():
        pa = x.value[val_bus.idx + voltage_count + nbus_ABC * 0] * kw_converter
        pb = x.value[val_bus.idx + voltage_count + nbus_ABC * 1] * kw_converter
        pc = x.value[val_bus.idx + voltage_count + nbus_ABC * 2] * kw_converter
        qa = x.value[val_bus.idx + voltage_count + nbus_ABC * 3] * kw_converter
        qb = x.value[val_bus.idx + voltage_count + nbus_ABC * 4] * kw_converter
        qc = x.value[val_bus.idx + voltage_count + nbus_ABC * 5] * kw_converter
        bus_flows[f"{key}.1"] = [pa, qa]
        bus_flows[f"{key}.2"] = [pb, qb]
        bus_flows[f"{key}.3"] = [pc, qc]

        if "76" in key:
            print(key, bus_flows[f"{key}.1"],
                  bus_flows[f"{key}.2"], bus_flows[f"{key}.3"])

    n_flow_s1s2 = (
        (nbus_ABC * 3 + nbus_s1s2) +
        (nbus_ABC * 6 + nbus_s1s2 * 2) + nbranch_ABC * 6
    )

    bus_voltage = {}
    for key, val_bus in buses.items():
        base = val_bus.base_kv
        a = math.sqrt(abs(x.value[val_bus.idx])) * base * 1000
        b = math.sqrt(abs(x.value[nbus_ABC + val_bus.idx])) * base * 1000
        c = math.sqrt(abs(x.value[nbus_ABC * 2 + val_bus.idx])) * base * 1000
        bus_voltage[f"{key}.1"] = a
        bus_voltage[f"{key}.2"] = b
        bus_voltage[f"{key}.3"] = c
        i += 1

    P_generation_output = np.zeros((nbus_ABC, 3))
    for k in range(nbus_ABC * 3):
        # injection.append([name[k], '{:.4f}'.format((x.value[k + state_variable_number]))])
        if DG_up_lim[k, 0]:
            if k < nbus_ABC:
                P_generation_output[k, 0] = x.value[k + state_variable_number]
            elif k < (nbus_ABC * 2):
                P_generation_output[k - nbus_ABC, 1] = x.value[
                    k + state_variable_number
                ]
            else:
                P_generation_output[k - (nbus_ABC * 2), 2] = x.value[
                    k + state_variable_number
                ]

    Q_generation_output = np.zeros((nbus_ABC, 3))
    for k in range(nbus_ABC * 3):
        # injection.append([name[k], '{:.4f}'.format((x.value[k + state_variable_number]))])
        if DG_up_lim[k, 0]:
            if k < nbus_ABC:
                Q_generation_output[k, 0] = x.value[k + n_Qdg]
            elif k < (nbus_ABC * 2):
                Q_generation_output[k - nbus_ABC, 1] = x.value[k + n_Qdg]
            else:
                Q_generation_output[k - (nbus_ABC * 2), 2] = x.value[k + n_Qdg]
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    opf_control_variable = {}
    for key, val_bus in buses.items():
        opf_control_variable[key] = {}
        opf_control_variable[key]["Pdg_gen"] = {}
        opf_control_variable[key]["Qdg_gen"] = {}
        opf_control_variable[key]["Pdg_gen"]["A"] = (
            x.value[val_bus.idx + state_variable_number] * kw_converter
        )
        opf_control_variable[key]["Pdg_gen"]["B"] = (
            x.value[nbus_ABC + val_bus.idx +
                    state_variable_number] * kw_converter
        )
        opf_control_variable[key]["Pdg_gen"]["C"] = (
            x.value[nbus_ABC * 2 + val_bus.idx +
                    state_variable_number] * kw_converter
        )
        opf_control_variable[key]["Qdg_gen"]["A"] = x.value[val_bus.idx + n_Qdg]
        opf_control_variable[key]["Qdg_gen"]["B"] = x.value[
            nbus_ABC + val_bus.idx + n_Qdg
        ]
        opf_control_variable[key]["Qdg_gen"]["C"] = x.value[
            nbus_ABC * 2 + val_bus.idx + n_Qdg
        ]

    return bus_voltage, bus_flows, opf_control_variable, stats
