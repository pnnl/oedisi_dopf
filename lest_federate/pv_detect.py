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


def get_hmat(
        bus_info: dict,
        branch_info: dict,
        source_bus: str,
        SBASE: float = 1e6
):
    slack_index = list(bus_info.keys()).index(source_bus)

    # System's base definition
    PRIMARY_V = 0.12
    _SBASE = SBASE / 1e6

    # Find the ABC phase and s1s2 phase triplex line and bus numbers
    nbranch_ABC = 0
    nbus_ABC = 0
    nbranch_s1s2 = 0
    nbus_s1s2 = 0
    secondary_model = ['TPX_LINE', 'SPLIT_PHASE']
    name = []
    for b_eq in branch_info:
        if branch_info[b_eq]['type'] in secondary_model:
            nbranch_s1s2 += 1
        else:
            nbranch_ABC += 1

    for b_eq in bus_info:
        name.append(b_eq)
        if bus_info[b_eq]['kv'] > PRIMARY_V:
            nbus_ABC += 1
        else:
            nbus_s1s2 += 1

    # Number of equality/inequality constraints (Injection equations (ABC) at each bus)
    #    #  Check if this is correct number or not:
    n_bus = nbus_ABC * 3 + nbus_s1s2  # Total Bus Number
    n_branch = nbranch_ABC * 3 + nbranch_s1s2  # Total Branch Number

    # Constraint 2: Voltage drop equation:
    # Vj = Vi - Zij Sij* - Sij Zij*
    A2 = np.zeros(shape=(n_branch, 2 * n_branch))
    A1 = np.zeros(shape=(n_branch, n_bus))

    # For Primary Nodes:
    idx = 0
    v_lim = []
    for k, val_br in branch_info.items():
        # compute base impedance
        basekV = bus_info[val_br['to_bus']]['kv']
        baseZ = (basekV ** 2) / (_SBASE)

        # Not writing voltage constraints for transformers
        if val_br['type'] not in secondary_model:
            z = np.asarray(val_br['zprim'])
            v_lim.append(val_br['from'])
            v_lim.append(val_br['to'])
            # Writing three phase voltage constraints
            # Phase A
            paa, qaa = -2 * z[0, 0][0], -2 * z[0, 0][1]
            pab, qab = -(- z[0, 1][0] + math.sqrt(3) * z[0, 1][1]), -(
                - z[0, 1][1] - math.sqrt(3) * z[0, 1][0])
            pac, qac = -(- z[0, 2][0] - math.sqrt(3) * z[0, 2][1]), -(
                - z[0, 2][1] + math.sqrt(3) * z[0, 2][0])
            A2, A1 = voltage_cons_pri(
                A2, A1,
                idx, val_br['from'], val_br['to'],
                paa, qaa, pab, qab, pac, qac, baseZ,
                nbranch_ABC, nbranch_ABC * 0, nbus_ABC * 0)

            # Phase B
            pbb, qbb = -2 * z[1, 1][0], -2 * z[1, 1][1]
            pba, qba = -(- z[0, 1][0] - math.sqrt(3) * z[0, 1][1]), -(
                - z[0, 1][1] + math.sqrt(3) * z[0, 1][0])
            pbc, qbc = -(- z[1, 2][0] + math.sqrt(3) * z[1, 2][1]), -(
                - z[1, 2][1] - math.sqrt(3) * z[1, 2][0])
            A2, A1 = voltage_cons_pri(
                A2, A1,
                idx, val_br['from'], val_br['to'],
                pba, qba, pbb, qbb, pbc, qbc, baseZ,
                nbranch_ABC, nbranch_ABC * 1, nbus_ABC * 1)

            # Phase C
            pcc, qcc = -2 * z[2, 2][0], -2 * z[2, 2][1]
            pca, qca = -(- z[0, 2][0] + math.sqrt(3) * z[0, 2][1]), -(
                - z[0, 2][1] - math.sqrt(3) * z[0, 2][0])
            pcb, qcb = -(- z[1, 2][0] - math.sqrt(3) * z[1, 2][1]), -(
                - z[0, 2][1] + math.sqrt(3) * z[1, 2][0])
            A2, A1 = voltage_cons_pri(
                A2, A1,
                idx, val_br['from'], val_br['to'],
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
    # p_all = A @ f
    # Remove rows of A corresponding to the slack nodes to get the square matrix H22 (for a radial system)
    # p = H22 @ f
    # f = H22_inv @ p

    # The lindistflow equations are
    # v_delta = -A2 @ f
    # where v_delta is the vector of voltage differences along the branches
    # v_delta = A1 @ v = [A1_slack A1_red] @ [v0.T vn.T] = (A1_slack @ v0) + (A1_red @ vn)
    # - A2 @ f = (A1_slack @ v0) + (A1_red @ vn)
    # vn = -(Avr_inv @ Av0) @ v0 - (Avr_inv @ A2) @ f

    # Denote the following
    # H11 = -(Avr_inv @ Av0)
    # H12 = -(Avr_inv @ A2)
    ########################################################################################################################
    slack_node_idx = [slack_index, slack_index +
                      nbus_ABC, slack_index + 2 * nbus_ABC]
    A1_slack = A1[:, slack_node_idx]
    A1_red = np.delete(A1, slack_node_idx, axis=1)
    A1r_inv = np.linalg.inv(A1_red)

    # get incidence matrix as transpose of Avr
    # A_inc is nbus by nbranch
    A_inc = A1_red.T

    # get the voltage relation with power injections
    H11 = - (A1r_inv @ A1_slack)
    H12 = - (A1r_inv @ A2)
    H22 = np.kron(np.eye(2, dtype=int), A_inc)

    H22_inv = np.linalg.inv(H22)
    H_linear = np.hstack((H11, H12 @ H22_inv))

    # add rows to the H_linear matrix for the slack bus voltages
    # since we will be using them as measurement in our estimation
    z = np.hstack((
        np.identity(len(slack_node_idx)),
        np.zeros(shape=(len(slack_node_idx), H22.shape[0]))
    ))
    for i in range(len(slack_node_idx)):
        H_linear = np.insert(H_linear, slack_node_idx[i], z[i, :], axis=0)

    ###################### Forming the big H matrix #############################
    small_o = np.zeros((A_inc.shape[1], A_inc.shape[1]))
    small_I = np.identity(A_inc.shape[1])

    # voltage of all nodes as a function of slack node voltage, injections and PV generations
    H1 = np.hstack((
        H_linear,                                           # slack node voltage, injections
        # real PV generation
        np.zeros((H_linear.shape[0], A_inc.shape[1])),
        # reactive PV generation
        np.zeros((H_linear.shape[0], A_inc.shape[1]))
    ))

    # real load forecast as a function of slack node voltage, injections and PV generations
    H2 = np.hstack((
        # slack node voltage
        np.zeros((A_inc.shape[1], len(slack_node_idx))),
        -small_I, small_o,                                  # real and reactive injections
        # real and reactive PV generation
        small_I, small_o
    ))
    # reactive load forecast as a function of slack node voltage, injections and PV generations
    H3 = np.hstack((
        # slack node voltage
        np.zeros((A_inc.shape[1], len(slack_node_idx))),
        small_o, -small_I,                                  # real and reactive injections
        # real and reactive PV generation
        small_o, small_I
    ))

    # Pinj and Qinj measurements as functions of slack node voltage, injections and PV generations:
    H4 = np.hstack((
        np.zeros((A_inc.shape[1], len(slack_node_idx))),
        small_I, small_o,
        small_o, small_o
    ))
    H5 = np.hstack((
        np.zeros((A_inc.shape[1], len(slack_node_idx))),
        small_o, small_I,
        small_o, small_o
    ))

    # stack them all to get the big H matrix
    H = np.vstack((H1, H2, H3, H4, H5))

    return H, A_inc


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
            pq[count + n_bus * 0] = val_bus['pv'][0][0] + val_bus['pq'][0][0]
            # Phase B Real Power
            pq[count + n_bus * 1] = val_bus['pv'][1][0] + val_bus['pq'][1][0]
            # Phase C Real Power
            pq[count + n_bus * 2] = val_bus['pv'][2][0] + val_bus['pq'][2][0]

            # Phase A Reactive Power
            pq[count + n_bus * 3] = val_bus['pv'][0][1] + val_bus['pq'][0][1]
            # Phase B Reactive Power
            pq[count + n_bus * 4] = val_bus['pv'][1][1] + val_bus['pq'][1][1]
            # Phase C Reactive Power
            pq[count + n_bus * 5] = val_bus['pv'][2][1] + val_bus['pq'][2][1]

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
            # Real power load at a bus
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
        v[count + n_bus * 0] = val_bus['vmag'][0]
        v[count + n_bus * 1] = val_bus['vmag'][1]
        v[count + n_bus * 2] = val_bus['vmag'][2]
        count += 1
    return v, slack_index


def get_vbase(
        bus_info: dict
):
    n_bus = len(bus_info)
    v = np.zeros(shape=(3 * n_bus,))
    count = 0
    for keyb, val_bus in bus_info.items():
        v[count + n_bus * 0] = val_bus['kv']*1000.0
        v[count + n_bus * 1] = val_bus['kv']*1000.0
        v[count + n_bus * 2] = val_bus['kv']*1000.0
        count += 1
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


def map_values(
    p: list, q: list, bus_info: dict, source_bus: str, slack_idx: list[int]
) -> (dict, dict):
    p_map = {}
    q_map = {}

    nodes = get_nodes(bus_info)
    for key, bus in bus_info.items():
        if key == source_bus:
            continue

        for phase in bus['phases']:
            phase = int(phase)-1
            name = f"{key}.{phase+1}"
            idx = nodes.index(name)

            # bus_pq = bus['pq'][phase]
            # pv = p[idx]*1000 >= tol
            pv = True
            p_map[name] = 0
            q_map[name] = 0
            if pv:
                if idx < len(p):
                    p_map[name] = p[idx]
                    q_map[name] = q[idx]

    return (p_map, q_map)


def run_dsse(
    bus_info: dict,
    branch_info: dict,
    config: dict,
    source_bus: str,
    base_s: float
) -> (dict, dict):
    H, A_inc = get_hmat(bus_info, branch_info, source_bus, SBASE=base_s)

    pq = get_pq(bus_info, source_bus, SBASE=base_s)
    pq_load = get_pq_forecast(bus_info, source_bus, SBASE=base_s)
    vmag, vslack = get_v(bus_info, source_bus)
    base_v = get_vbase(bus_info)

    print("Slack Idx: ", vslack)

    # compute per unit voltage magnitudes
    vmag_pu = vmag / base_v

    # estimation:
    V_W = np.array([1/(config['v_sigma']**2)]*len(vmag_pu))
    V_W[vslack] = 1e7*V_W[vslack]  # shouldn't be needed
    Pl_W = np.array([1 / (config['l_sigma'] ** 2)] * len(pq_load))
    Pinj_W = np.array([1 / (config['i_sigma'] ** 2)] * len(pq))
    W_array = np.hstack((V_W, Pl_W, Pinj_W))
    W = np.diag(W_array)

    Z_meas = np.hstack((vmag_pu, pq_load, pq))
    G = H.T @ W @ H
    G_inv = np.linalg.inv(G)
    x_est = G_inv @ H.T @ W @ Z_meas

    v_sub_est = np.sqrt(x_est[:len(vslack)])
    Ppv_inj_est = x_est[len(vslack) + (2 * A_inc.shape[1])
                            : len(vslack) + (3 * A_inc.shape[1])]
    Qpv_inj_est = x_est[len(vslack) + (3 * A_inc.shape[1]):]

    # p_inj_est = x_est[len(vslack): len(vslack)+int(len(pq)/2)]
    # q_inj_est = x_est[len(vslack) + int(len(pq)/2): len(vslack) + len(pq)]
    # Ppv_inj_est = x_est[len(vslack) + len(pq): len(vslack) + len(pq) + int(len(pq_load)/2)]
    # Qpv_inj_est = x_est[len(vslack) + len(pq) + int(len(pq_load)/2):]

    p = Ppv_inj_est * base_s / 1e3
    q = Qpv_inj_est * base_s / 1e3

    (powers_real, powers_imag) = map_values(
        p, q, bus_info, source_bus, vslack)

    return (powers_real, powers_imag)
