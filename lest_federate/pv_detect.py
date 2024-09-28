import numpy as np
import math
import logging
from oedisi.types.data_types import (
    VoltagesMagnitude,
    PowersReal,
    PowersImaginary
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

"""
Author: Rounak Meyur

Adapted from code written by Shiva Poudel and Monish Mukherjee

Description: Builds a matrix to relate the linear voltage magnitude estimates of all nodes to the
real and reactive power injections at the nodes.
"""


def power_balance_relation(
        A: np.ndarray,
        k_from: list,
        k_to: list,
        col_offset: int,
        j: int
):
    for k in k_from:
        A[j, col_offset + k] = 1
    for k in k_to:
        A[j, col_offset + k] = -1

    return A


def voltage_cons_pri(
        Z: np.ndarray,
        A: np.ndarray,
        idx, frm, to,
        pii, qii, pij, qij, pik, qik, baseZ,
        nbranch_ABC: int,
        brn_offset: int,
        bus_offset: int
):
    """
    Z: The matrix which relates voltage difference to power flows
    A: The matrix which relates voltage difference to voltages
    idx: The entry index for the branch
    """
    A[idx + brn_offset, frm + bus_offset] = 1
    A[idx + brn_offset, to + bus_offset] = -1

    # real power drop
    Z[idx + brn_offset, idx + nbranch_ABC * 0] = pii / baseZ
    Z[idx + brn_offset, idx + nbranch_ABC * 1] = pij / baseZ
    Z[idx + brn_offset, idx + nbranch_ABC * 2] = pik / baseZ
    # reactive power drop
    Z[idx + brn_offset, idx + nbranch_ABC * 3] = qii / baseZ
    Z[idx + brn_offset, idx + nbranch_ABC * 4] = qij / baseZ
    Z[idx + brn_offset, idx + nbranch_ABC * 5] = qik / baseZ

    return Z, A


def base_voltage_dict2(bus_info: dict) -> (list[float], list[str]):
    values = [bus['base_kv'] for bus in bus_info.values()]
    ids = []
    for key, bus in bus_info.items():
        for phase in range(3):
            phase += 1
            ids.append(f"{key}.{phase}")
    return (values, ids)


def get_hmat(
        bus_info: dict,
        branch_info: dict,
        source_bus: str,
        SBASE: float = 1e6
):
    slack_index = list(bus_info.keys()).index(source_bus)

    # System's base definition
    PRIMARY_V = 0.12
    # SBASE = 100.0  # in MVA
    _SBASE = SBASE/1e6  # in MVA

    # Find the ABC phase and s1s2 phase triplex line and bus numbers
    nbranch_ABC = 0
    nbus_ABC = 0
    nbranch_s1s2 = 0
    nbus_s1s2 = 0
    secondary_model = ['TPX_LINE', 'SPLIT_PHASE']
    name = []
    for b_eq in branch_info:
        if branch_info[b_eq]['tag'] in secondary_model:
            nbranch_s1s2 += 1
        else:
            nbranch_ABC += 1

    for b_eq in bus_info:
        name.append(b_eq)
        if bus_info[b_eq]['base_kv'] > PRIMARY_V:
            nbus_ABC += 1
        else:
            nbus_s1s2 += 1

    # Number of equality/inequality constraints (Injection equations (ABC) at each bus)
    #    #  Check if this is correct number or not:
    n_bus = nbus_ABC * 3 + nbus_s1s2  # Total Bus Number
    n_branch = nbranch_ABC * 3 + nbranch_s1s2  # Total Branch Number

    A1 = np.zeros(shape=(2 * (nbus_ABC * 3 + nbus_s1s2),
                  2 * (nbranch_ABC * 3 + nbranch_s1s2)))

    # # Define BFM constraints for both real and reactive power: Power flow conservaion
    # Constraint 1: Flow equation

    # sum(Sij) - sum(Sjk) == -sj

    for keyb, val_bus in bus_info.items():
        if keyb != source_bus:
            k_frm_3p = []
            k_to_3p = []
            k_frm_1p = []
            k_frm_1pa, k_frm_1pb, k_frm_1pc = [], [], []
            k_frm_1qa, k_frm_1qb, k_frm_1qc = [], [], []
            k_to_1p = []

            # Find bus idx in "from" of branch_sw_data
            ind_frm = 0
            ind_to = 0
            if val_bus['base_kv'] < PRIMARY_V:
                # if the bus is a part of a split phase transformer
                for key, val_br in branch_info.items():
                    if val_bus['idx'] == val_br['fr_idx']:
                        k_frm_1p.append(ind_frm - nbranch_ABC)

                    if val_bus['idx'] == val_br['to_idx']:
                        k_to_1p.append(ind_to - nbranch_ABC)
                    ind_to += 1
                    ind_frm += 1

            else:
                # iterate through all the branches and find all whose 'fr_idx' or 'to_idx' bus match the current bus
                for key, val_br in branch_info.items():
                    if val_bus['idx'] == val_br['fr_idx']:
                        if bus_info[val_br['to_bus']]['base_kv'] > PRIMARY_V:
                            k_frm_3p.append(ind_frm)
                        else:
                            if key[-1] == 'a':
                                k_frm_1pa.append(
                                    nbranch_ABC * 6 + ind_frm - nbranch_ABC)
                                k_frm_1qa.append(
                                    nbranch_ABC * 3 + ind_frm - nbranch_ABC + nbranch_s1s2)
                            if key[-1] == 'b':
                                k_frm_1pb.append(
                                    nbranch_ABC * 5 + ind_frm - nbranch_ABC)
                                k_frm_1qb.append(
                                    nbranch_ABC * 2 + ind_frm - nbranch_ABC + nbranch_s1s2)
                            if key[-1] == 'c':
                                k_frm_1pc.append(
                                    nbranch_ABC * 4 + ind_frm - nbranch_ABC)
                                k_frm_1qc.append(
                                    nbranch_ABC * 1 + ind_frm - nbranch_ABC + nbranch_s1s2)

                    if val_bus['idx'] == val_br['to_idx']:
                        if bus_info[val_br['fr_bus']]['base_kv'] > PRIMARY_V:
                            k_to_3p.append(ind_to)
                        else:
                            k_to_1p.append(ind_to - nbranch_ABC)
                    ind_to += 1
                    ind_frm += 1

                loc = 0
                # Finding the kfrms and ktos for the branches
                k_frm_A = k_frm_3p + k_frm_1pa
                k_frm_B = k_frm_3p + k_frm_1pb
                k_frm_C = k_frm_3p + k_frm_1pc
                k_to_A = k_to_B = k_to_C = k_to_3p

                # Real Power balance equations
                # Phase A
                A1 = power_balance_relation(
                    A1, k_frm_A, k_to_A,
                    loc + nbranch_ABC * 0,
                    val_bus['idx'] + nbus_ABC * 0
                )
                # Phase B
                A1 = power_balance_relation(
                    A1, k_frm_B, k_to_B,
                    loc + nbranch_ABC * 1,
                    val_bus['idx'] + nbus_ABC * 1
                )
                # Phase C
                A1 = power_balance_relation(
                    A1, k_frm_C, k_to_C,
                    loc + nbranch_ABC * 2,
                    val_bus['idx'] + nbus_ABC * 2
                )

                # Finding the k_froms and k_tos for the branches
                k_frm_A = k_frm_3p + k_frm_1qa
                k_frm_B = k_frm_3p + k_frm_1qb
                k_frm_C = k_frm_3p + k_frm_1qc

                # Reactive Power balance equations
                # Phase A
                A1 = power_balance_relation(
                    A1, k_frm_A, k_to_A,
                    loc + nbranch_ABC * 3,
                    val_bus['idx'] + nbus_ABC * 3
                )
                # Phase B
                A1 = power_balance_relation(
                    A1, k_frm_B, k_to_B,
                    loc + nbranch_ABC * 4,
                    val_bus['idx'] + nbus_ABC * 4
                )
                # Phase C
                A1 = power_balance_relation(
                    A1, k_frm_C, k_to_C,
                    loc + nbranch_ABC * 5,
                    val_bus['idx'] + nbus_ABC * 5
                )

    # Constraint 2: Voltage drop equation:
    # Vj = Vi - Zij Sij* - Sij Zij*
    A2 = np.zeros(shape=(n_branch, 2 * n_branch))
    Av = np.zeros(shape=(n_branch, n_bus))

    # For Primary Nodes:
    idx = 0
    v_lim = []
    for k, val_br in branch_info.items():
        # compute base impedance
        basekV = bus_info[val_br['to_bus']]['base_kv']
        baseZ = (basekV ** 2) / (_SBASE)

        # Not writing voltage constraints for transformers
        if val_br['tag'] not in secondary_model:
            z = np.asarray(val_br['zprim'])
            v_lim.append(val_br['fr_idx'])
            v_lim.append(val_br['to_idx'])
            # Writing three phase voltage constraints
            # Phase A
            paa, qaa = -2 * z[0, 0][0], -2 * z[0, 0][1]
            pab, qab = -(- z[0, 1][0] + math.sqrt(3) * z[0, 1][1]), -(
                - z[0, 1][1] - math.sqrt(3) * z[0, 1][0])
            pac, qac = -(- z[0, 2][0] - math.sqrt(3) * z[0, 2][1]), -(
                - z[0, 2][1] + math.sqrt(3) * z[0, 2][0])
            A2, Av = voltage_cons_pri(
                A2, Av,
                idx, val_br['fr_idx'], val_br['to_idx'],
                paa, qaa, pab, qab, pac, qac, baseZ,
                nbranch_ABC, nbranch_ABC * 0, nbus_ABC * 0)

            # Phase B
            pbb, qbb = -2 * z[1, 1][0], -2 * z[1, 1][1]
            pba, qba = -(- z[0, 1][0] - math.sqrt(3) * z[0, 1][1]), -(
                - z[0, 1][1] + math.sqrt(3) * z[0, 1][0])
            pbc, qbc = -(- z[1, 2][0] + math.sqrt(3) * z[1, 2][1]), -(
                - z[1, 2][1] - math.sqrt(3) * z[1, 2][0])
            A2, Av = voltage_cons_pri(
                A2, Av,
                idx, val_br['fr_idx'], val_br['to_idx'],
                pba, qba, pbb, qbb, pbc, qbc, baseZ,
                nbranch_ABC, nbranch_ABC * 1, nbus_ABC * 1)

            # Phase C
            pcc, qcc = -2 * z[2, 2][0], -2 * z[2, 2][1]
            pca, qca = -(- z[0, 2][0] + math.sqrt(3) * z[0, 2][1]), -(
                - z[0, 2][1] - math.sqrt(3) * z[0, 2][0])
            pcb, qcb = -(- z[1, 2][0] - math.sqrt(3) * z[1, 2][1]), -(
                - z[0, 2][1] + math.sqrt(3) * z[1, 2][0])
            A2, Av = voltage_cons_pri(
                A2, Av,
                idx, val_br['fr_idx'], val_br['to_idx'],
                pca, qca, pcb, qcb, pcc, qcc, baseZ,
                nbranch_ABC, nbranch_ABC * 2, nbus_ABC * 2)

        idx += 1

    ########################################################################################################################
    # va,vb,vc are phase voltage vectors for non slack buses
    # v0a, v0b, v0c are phase voltages for slack bus
    # fpa, fpb, fpc are real power flow vectors for each phase
    # fqa, fqb, fqc are reactive power flow vectors for each phase
    # pa, pb, pc are real power injections at non slack buses
    # qa, qb, qc are reactive power injections at non slack buses

    # We write the vector expressions
    # vn = [va.T, vb.T, vc.T]
    # v0 = [v0a, v0b, v0c]
    # v = [v0.T vn.T]
    # fp = [fpa.T, fpb.T, fpc.T]
    # fq = [fqa.T, fqb.T, fqc.T]
    # f = [fp.T, fq.T]
    # pn = [pa.T, pb.T, pc.T]
    # qn = [qa.T, qb.T, qc.T]
    # p = [pn.T, qn.T]

    # The power balance equations are
    # p_all = A1 @ f
    # Remove rows of A1 corresponding to the slack nodes to get the square matrix H22 (for a radial system)
    # p = H22 @ f
    # f = H22_inv @ p

    # The lindistflow equations are
    # v_delta = -A2 @ f
    # where v_delta is the vector of voltage differences along the branches
    # v_delta = Av @ v = [Av0 Avr] @ [v0.T vn.T] = (Av0 @ v0) + (Avr @ vn)
    # A2 @ f = (Av0 @ v0) + (Avr @ vn)
    # vn = -(Avr_inv @ Av0) @ v0 - (Avr_inv @ A2) @ f

    # Denote the following
    # H11 = -(Avr_inv @ Av0)
    # H12 = -(Avr_inv @ A2)
    ########################################################################################################################
    slack_node_idx = [slack_index, slack_index +
                      nbus_ABC, slack_index + 2 * nbus_ABC]
    slack_node_idx_pq = slack_node_idx + [slack_index + n_bus, slack_index + n_bus + nbus_ABC,
                                          slack_index + n_bus + 2 * nbus_ABC]
    Av0 = Av[:, slack_node_idx]
    Avr = np.delete(Av, slack_node_idx, axis=1)
    Avr_inv = np.linalg.inv(Avr)

    A_inc = Avr.transpose()
    # I = np.eye(A_inc.shape[1])
    # O = np.zeros((A_inc.shape[1], A_inc.shape[1]))

    H11 = - (Avr_inv @ Av0)
    H12 = - (Avr_inv @ A2)
    H13 = np.zeros(shape=(H11.shape[0], H12.shape[1]))

    H22 = np.delete(A1, slack_node_idx_pq, axis=0)
    H23 = np.identity(H22.shape[0])
    H21 = np.zeros(shape=(H22.shape[0], H11.shape[1]))

    H22_inv = np.linalg.inv(H22)
    H_linear = np.hstack((H11, H12 @ H22_inv))

    # construct the H matrix
    H1 = np.hstack((H11, H12, H13))
    H2 = np.hstack((H21, -H22, H23))
    H = np.vstack((H1, H2))

    return H_linear, A_inc


def get_pq2(
        bus_info: dict,
        source_bus: str,
        # SBASE: float = 100.0e6
        SBASE: float = 1e6
):
    n_bus = len(bus_info) - 1
    pq = np.zeros(shape=(6 * n_bus,))
    count = 0
    for keyb, val_bus in bus_info.items():
        if keyb != source_bus:
            # Real power injection at a bus
            # Phase A Real Power
            pq[count + n_bus * 0] = val_bus['pq'][0][0]
            # Phase B Real Power
            pq[count + n_bus * 1] = val_bus['pq'][1][0]
            # Phase C Real Power
            pq[count + n_bus * 2] = val_bus['pq'][2][0]

            # Phase A Reactive Power
            pq[count + n_bus * 3] = val_bus['pq'][0][1]
            # Phase B Reactive Power
            pq[count + n_bus * 4] = val_bus['pq'][1][1]
            # Phase C Reactive Power
            pq[count + n_bus * 5] = val_bus['pq'][2][1]

            count += 1

    pq_load = np.zeros(shape=(6 * n_bus,))
    count = 0
    for keyb, val_bus in bus_info.items():
        if keyb != source_bus:
            # Real power injection at a bus
            # Phase A Real Power
            pq_load[count + n_bus * 0] = val_bus['base_pq'][0][0]
            # Phase B Real Power
            pq_load[count + n_bus * 1] = val_bus['base_pq'][1][0]
            # Phase C Real Power
            pq_load[count + n_bus * 2] = val_bus['base_pq'][2][0]

            # Phase A Reactive Power
            pq_load[count + n_bus * 3] = val_bus['base_pq'][0][1]
            # Phase B Reactive Power
            pq_load[count + n_bus * 4] = val_bus['base_pq'][1][1]
            # Phase C Reactive Power
            pq_load[count + n_bus * 5] = val_bus['base_pq'][2][1]

            count += 1
    return pq/(SBASE), -1*pq_load/(SBASE)


def get_pq(
        bus_info: dict,
        source_bus: str,
        SBASE: float = 1e6
):
    n_bus = len(bus_info) - 1
    pq = np.zeros(shape=(6 * n_bus,))
    count = 0
    for keyb, val_bus in bus_info.items():
        if keyb != source_bus:
            # Real power injection at a bus
            # Phase A Real Power
            pq[count + n_bus * 0] = val_bus['pq'][0][0]
            # Phase B Real Power
            pq[count + n_bus * 1] = val_bus['pq'][1][0]
            # Phase C Real Power
            pq[count + n_bus * 2] = val_bus['pq'][2][0]

            # Reactive power load at a bus
            # Phase A Reactive Power
            pq[count + n_bus * 3] = val_bus['pq'][0][1]
            # Phase B Reactive Power
            pq[count + n_bus * 4] = val_bus['pq'][1][1]
            # Phase C Reactive Power
            pq[count + n_bus * 5] = val_bus['pq'][2][1]

            count += 1
    return pq / (SBASE)


def get_pq_forecast(
        bus_info: dict,
        source_bus: str,
        SBASE: float = 1e6
):
    n_bus = len(bus_info) - 1
    pq = np.zeros(shape=(6 * n_bus,))
    count = 0
    for keyb, val_bus in bus_info.items():
        if keyb != source_bus:
            # Real power injection at a bus
            # Phase A Real Power
            pq[count + n_bus * 0] = val_bus['pv'][0][0] + \
                val_bus['pq_forecast'][0][0]
            # Phase B Real Power
            pq[count + n_bus * 1] = val_bus['pv'][1][0] + \
                val_bus['pq_forecast'][1][0]
            # Phase C Real Power
            pq[count + n_bus * 2] = val_bus['pv'][2][0] + \
                val_bus['pq_forecast'][2][0]

            # Phase A Reactive Power
            pq[count + n_bus * 3] = val_bus['pv'][0][1] + \
                val_bus['pq_forecast'][0][1]
            # Phase B Reactive Power
            pq[count + n_bus * 4] = val_bus['pv'][1][1] + \
                val_bus['pq_forecast'][1][1]
            # Phase C Reactive Power
            pq[count + n_bus * 5] = val_bus['pv'][2][1] + \
                val_bus['pq_forecast'][2][1]

            count += 1
    return -1*pq / (SBASE)


def get_pv(
        bus_info: dict,
        source_bus: str,
        SBASE: float = 1e6
):
    n_bus = len(bus_info) - 1
    pv = np.zeros(shape=(6 * n_bus,))
    count = 0
    for keyb, val_bus in bus_info.items():
        if keyb != source_bus:
            # Real power generation at a bus
            # Phase A Real Power
            pv[count + n_bus * 0] = val_bus['pv'][0][0]
            # Phase B Real Power
            pv[count + n_bus * 1] = val_bus['pv'][1][0]
            # Phase C Real Power
            pv[count + n_bus * 2] = val_bus['pv'][2][0]

            # Reactive power generation at a bus
            # Phase A Reactive Power
            pv[count + n_bus * 3] = val_bus['pv'][0][1]
            # Phase B Reactive Power
            pv[count + n_bus * 4] = val_bus['pv'][1][1]
            # Phase C Reactive Power
            pv[count + n_bus * 5] = val_bus['pv'][2][1]

            count += 1
    return pv / (SBASE)


def get_v(
        bus_info: dict,
        source_bus: str
):
    n_bus = len(bus_info)
    v = np.zeros(shape=(3 * n_bus,))
    slack_index = []
    count = 0
    for keyb, val_bus in bus_info.items():
        if keyb == source_bus:
            slack_index = [count, count + n_bus, count + 2 * n_bus]

        vmag = [0.0]*3
        kv = val_bus['kv']
        if kv == 0:
            kv = val_bus['base_kv']

        for p in val_bus['phases']:
            if p != 0:
                vmag[p-1] = val_bus['kv']

        v[count + n_bus * 0] = vmag[0]
        v[count + n_bus * 1] = vmag[1]
        v[count + n_bus * 2] = vmag[2]
        count += 1
    return v, slack_index


def get_vbase(
        bus_info: dict
):
    n_bus = len(bus_info)
    v = np.zeros(shape=(3 * n_bus,))
    for keyb, val_bus in bus_info.items():
        idx = val_bus["idx"]
        v[idx + n_bus * 0] = val_bus['base_kv']*1000.0
        v[idx + n_bus * 1] = val_bus['base_kv']*1000.0
        v[idx + n_bus * 2] = val_bus['base_kv']*1000.0
    return v


def get_vbase2(
        bus_info: dict,
        voltages: VoltagesMagnitude
):
    n_bus = len(bus_info)
    v = np.zeros(shape=(3 * n_bus,))
    ids = voltages.ids
    vals = voltages.values

    for keyb, val_bus in bus_info.items():
        idx = val_bus["idx"]
        v[idx + n_bus *
            0] = vals[ids.index(f"{keyb}.1")] if f"{keyb}.1" in ids else 1.0
        v[idx + n_bus *
            1] = vals[ids.index(f"{keyb}.2")] if f"{keyb}.2" in ids else 1.0
        v[idx + n_bus *
            2] = vals[ids.index(f"{keyb}.3")] if f"{keyb}.3" in ids else 1.0

    return v


def get_nodes(bus_info: dict) -> list:
    n_bus = len(bus_info)
    nodes = ["" for x in range(3*n_bus)]
    for key, bus in bus_info.items():
        idx = bus["idx"]
        nodes[idx + n_bus * 0] = f"{key}.1"
        nodes[idx + n_bus * 1] = f"{key}.2"
        nodes[idx + n_bus * 2] = f"{key}.3"
    return nodes


def map_values(nodes: list, values: list) -> dict:
    assert (len(nodes) == len(values))
    return {nodes[idx]: values[idx] for idx in range(len(nodes))}


def run_dsse(
    bus_info: dict,
    branch_info: dict,
    config: dict,
    source_bus: str,
    base_s: float
) -> (dict, dict):
    H_check, A_inc = get_hmat(bus_info, branch_info, source_bus, SBASE=base_s)

    # pq = get_pq(bus_info, source_bus, SBASE=base_s)
    # pq_load = get_pq_forecast(bus_info, source_bus, SBASE=base_s)
    pq, pq_load = get_pq2(bus_info, source_bus, SBASE=base_s)
    vmag, vslack = get_v(bus_info, source_bus)
    values, ids = base_voltage_dict2(bus_info)
    base_v = get_vbase(bus_info)

    # compute per unit voltage magnitudes
    vmag_pu = vmag / base_v

    z = np.hstack((np.identity(len(vslack)), np.zeros(
        shape=(len(vslack), pq.shape[0]))))
    for i in range(len(vslack)):
        H_check = np.insert(H_check, vslack[i], z[i, :], axis=0)

    ###################### Forming the big H matrix #############################
    small_o = np.zeros((A_inc.shape[1], A_inc.shape[1]))
    small_I = np.identity(A_inc.shape[1])

    H1 = np.hstack(
        (H_check, np.zeros((H_check.shape[0], A_inc.shape[1])), np.zeros((H_check.shape[0], A_inc.shape[1]))))
    H2 = np.hstack(
        (np.zeros((A_inc.shape[1], len(vslack))), -small_I, small_o, small_I, small_o))
    H3 = np.hstack(
        (np.zeros((A_inc.shape[1], len(vslack))), small_o, -small_I, small_o, small_I))

    # added for Pinj and Qinj measurements:
    H4 = np.hstack(
        (np.zeros((A_inc.shape[1], len(vslack))), small_I, small_o, small_o, small_o))
    H5 = np.hstack(
        (np.zeros((A_inc.shape[1], len(vslack))), small_o, small_I, small_o, small_o))

    H = np.vstack((H1, H2, H3, H4, H5))

    logger.debug(np.linalg.matrix_rank(H))

    # Make it Automatic::
    pq_forcast = pq_load

    # get the node list to sort the outputs in plotting order
    nodes = get_nodes(bus_info)
    node_ids = ids

    nodes_ord = [nodes.index(nd) for nd in node_ids]

    v_meas_true = np.square(vmag_pu[nodes_ord])
    measurement_all = np.concatenate((v_meas_true, pq_forcast, pq))
    # measurement_all = H @ x_check_all

    v_lin_all = measurement_all[:H_check.shape[0]]
    pl_lin_all = measurement_all[H_check.shape[0]: H_check.shape[0] + A_inc.shape[1]]
    ql_lin_all = measurement_all[H_check.shape[0] +
                                 A_inc.shape[1]: H_check.shape[0] + (2 * A_inc.shape[1])]
    p_inj_all = measurement_all[H_check.shape[0] +
                                (2 * A_inc.shape[1]): H_check.shape[0] + (3 * A_inc.shape[1])]
    q_inj_all = measurement_all[H_check.shape[0] + (3 * A_inc.shape[1]):]

    V_W = np.array([1/(config['v_sigma']**2)]*len(vmag_pu))
    V_W[vslack] = 1e7*V_W[vslack]  # shouldn't be needed
    Pl_W = np.array([1 / (config['l_sigma'] ** 2)] * len(pq_load))
    Pinj_W = np.array([1 / (config['i_sigma'] ** 2)] * len(pq))

    v_sigma = config['v_sigma']
    l_sigma = config['l_sigma']
    i_sigma = config['i_sigma']

    V_W = np.array([1 / (v_sigma ** 2)]
                   * (A_inc.shape[1] + len(vslack)))
    V_W[vslack] = V_W[vslack]*1e5
    Pl_W = np.array([1 / (l_sigma ** 2)] * A_inc.shape[1] * 2)
    Pinj_W = np.array([1 / (i_sigma ** 2)] * A_inc.shape[1] * 2)
    W_array = np.hstack((V_W, Pl_W, Pinj_W))
    W = np.diag(W_array)
    W = W/1e3
    # W = W @ np.linalg.inv(W)

    Z_meas = np.hstack(
        (v_lin_all, pl_lin_all, ql_lin_all, p_inj_all, q_inj_all))
    G = H.transpose() @ W @ H
    G_inv = np.linalg.inv(G)

    x_est = G_inv @ H.transpose() @ W @ Z_meas
    v_sub_est = np.sqrt(x_est[:len(vslack)])
    p_inj_est = x_est[len(vslack): len(vslack) + (1 * A_inc.shape[1])]
    q_inj_est = x_est[len(vslack) + (1 * A_inc.shape[1]): len(vslack) + (2 * A_inc.shape[1])]
    Ppv_inj_est = x_est[len(vslack) + (2 * A_inc.shape[1]): len(vslack) + (3 * A_inc.shape[1])]
    Qpv_inj_est = x_est[len(vslack) + (3 * A_inc.shape[1]):]

    p = Ppv_inj_est * base_s / 1e3
    q = Qpv_inj_est * base_s / 1e3

    nodes = get_nodes(bus_info)
    nodes = [node for i, node in enumerate(nodes) if i not in vslack]
    powers_real = map_values(nodes, p)
    powers_imag = map_values(nodes, q)

    return (powers_real, powers_imag)
