import math
import numpy as np
from typing import Tuple


################# PER UNIT Computation ########################
# We assume the substation node voltage to be the base voltage
# and the base MVA is 1 MVA. With this assumption, we have the
# base voltage of each component to be their rated voltage (irr-
# -espective of the transformers).

Sbase = 1.0 # base MVA is assumed to be 1 MVA


###############################################################
# TASK: Functions to compute the Z matrices for a branch
# The input for each function will be the Z-matrix for the branch
# If it is a multi-phase branch, the input will be a matrix

def make_Z_3ph_line(info : dict) -> Tuple(np.ndarray, np.ndarray):
    # get the z values of the line
    zprim = info["zprim"]
    Vbase = info["base"][0] / 1000.0 # base voltage in kV
    Zbase = (Vbase ** 2) / Sbase
    
    # get the resistance and reactance matrices
    Z = np.asarray(zprim) / Zbase
    R = Z[:, :, 0]
    X = Z[:, :, 1]

    # get the K matrices to multiply the R and X matrices
    Kr = np.array(
        [[-2, 1, 1], 
         [1, -2, 1], 
         [1, 1, -2]]
        )
    Kx = np.array(
        [[0, - math.sqrt(3), math.sqrt(3)], 
         [math.sqrt(3), 0, - math.sqrt(3)], 
         [- math.sqrt(3), math.sqrt(3), 0]]
        )
    
    # get the H matrices
    HP = np.multiply(R,Kr) + np.multiply(X,Kx)
    HQ = np.multiply(X,Kr) - np.multiply(R,Kx)
    return (HP, HQ)


###############################################################
# Zmatrix : Single phase service transformer
def vdrop_coefficients_split_phase(info : dict) -> Tuple(float, float):
    """
    Z matrix for a split phase transformer. The transformer is single 
    phase.
    """

    # Compute the base impedances of both sides of the transformer
    Vbase_prim = info["base"][0] / 1000.0
    Vbase_sec = info["base"][1] / 1000.0
    Zbase_prim = Vbase_prim / Sbase
    Zbase_sec = Vbase_sec / Sbase

    # Get the primary and secondary pu impedances
    zprim = np.asarray(info["zprim"]) / Zbase_prim
    zsec = np.asarray(info["zsec"]) / Zbase_sec

    # Drop in voltage per unit power injection
    p_pri, q_pri = -2 * zprim[0], -2 * zprim[1]
    p_sec, q_sec = -2 * zsec[0], -2 * zsec[1]
    hp = p_pri + (0.5 * p_sec)
    hq = q_pri + (0.5 * q_sec)
    return hp, hq

def constraint_vector_split_phase(
        info : dict, 
        variable_count : dict,
        ) -> np.ndarray:
    # extract from the variable_count dictionary 
    nbus_abc = variable_count["nbus_ABC"]
    nbus_s1s2 = variable_count["nbus_s1s2"]
    nbranch_abc = variable_count["nbranch_ABC"]
    nbranch_s1s2 = variable_count["nbranch_s1s2"]

    # initialize the vector for the constraint with all zeros
    num_variables = (3 * nbus_abc) + nbus_s1s2 + (6 * nbus_abc) + (2 * nbus_s1s2) + (6 * nbranch_abc) + (2 * nbranch_s1s2)
    a = np.zeros(shape=(num_variables,))

    # get the voltage drop coefficients
    hp, hq = vdrop_coefficients_split_phase(info)
    flow_variable_start_index = (3 * nbus_abc) + nbus_s1s2 + (6 * nbus_abc) + (2 * nbus_s1s2) + (6 * nbranch_abc)
    a[flow_variable_start_index + info["idx"]] = - hp                   # real power drop
    a[flow_variable_start_index + nbranch_s1s2 + info["idx"]] = - hq    # reactive power drop

    # from and to bus ids
    tbus_id = info["to"]
    a[tbus_id + (2*nbus_abc)] = -1
    fbus_id = info["from"]
    if info["phases"] == ["a"] or info["phases"] == ["1"]:
        a[fbus_id] = 1
    elif info["phases"] == ["b"] or info["phases"] == ["2"]:
        a[fbus_id + nbus_abc] = 1
    elif info["phases"] == ["c"] or info["phases"] == ["3"]:
        a[fbus_id + (2*nbus_abc)] = 1
    else:
        raise ValueError(f"Invalid phase {info['phases']} in the info dictionary!!!")
    
    return