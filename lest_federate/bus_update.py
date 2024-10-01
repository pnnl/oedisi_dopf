import json
import networkx as nx
import numpy as np


def find_consecutive_true(matrix):
    connected_phase_row = []
    connected_phase_col = []

    # Check rows for consecutive True values
    for i, row in enumerate(matrix):
        consecutive = 0
        for j in range(len(row)):
            if row[j]:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= 2:
                connected_phase_row.append(i)
                # print(f"Consecutive True found in row {i} starting at column {j - 1}")

    # Check columns for consecutive True values
    for j in range(len(matrix[0])):
        consecutive = 0
        for i in range(len(matrix)):
            if matrix[i][j]:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= 2:
                connected_phase_col.append(j)
                # print(f"Consecutive True found in column {j} starting at row {i - 1}")

    if connected_phase_row and connected_phase_col:
        raise Exception("has both row and col value in xfmr.")

    con_ph = None
    if connected_phase_row:
        con_ph = connected_phase_row[0]
    elif connected_phase_col:
        con_ph = connected_phase_col[0]

    return con_ph


def find_connected_phase(zprim, threshold):
    # Create a 3x3 boolean matrix
    bool_matrix = []

    for row in zprim:
        bool_row = []
        for element in row:
            # Check if any part of the complex value (real or imaginary) is greater than the threshold
            if abs(element[0]) > threshold or abs(element[1]) > threshold:
                bool_row.append(True)
            else:
                bool_row.append(False)
        bool_matrix.append(bool_row)

    # Find consecutive True values in rows and columns
    _connected_phase = find_consecutive_true(bool_matrix)

    return _connected_phase


def find_immediate_parent_node(branch_data, current_bus: str):
    for branch_id, branch in branch_data.items():
        if branch["to_bus"] == str(current_bus):
            immediate_parent_bus = branch["fr_bus"]

    return immediate_parent_bus


def find_branch_key(branch_data, fr_bus, to_bus):
    for branch_id, branch in branch_data.items():
        if branch["fr_bus"] == str(fr_bus) and branch["to_bus"] == str(to_bus):
            branch_key = branch_id
            break
        else:
            branch_key = []
    return branch_key


def correct_secondary_network(branch_data, bus_data, target_bus, leaf_bus):
    # Find parent node of secondary bus:
    leaf_parent_walk = [leaf_bus]
    current_bus = leaf_bus
    im_parent = current_bus
    while im_parent != target_bus:
        im_parent = find_immediate_parent_node(branch_data, current_bus)
        leaf_parent_walk.append(im_parent)
        current_bus = im_parent

    # Iterate through the branches
    for branch_id, branch in branch_data.items():
        if (
            branch["fr_bus"] == str(target_bus)
            and branch["to_bus"] == leaf_parent_walk[-2]
        ):
            zprim = branch.get("zprim", None)
            break

    connected_phase_temp = find_connected_phase(zprim, threshold=1e-6)

    if "processed" in branch:
        connected_phase = [i - 1 for i in branch["phases"] if i != 0][0]
    else:
        connected_phase = connected_phase_temp

    # Correct branch and bus phase till the secondary side of the xfmr
    correct_xfmr_phases = [0, 0, 0]
    correct_xfmr_phases[connected_phase] = int(connected_phase) + 1

    for walk in range(len(leaf_parent_walk) - 1):
        current_br = find_branch_key(
            branch_data, leaf_parent_walk[1 + walk], leaf_parent_walk[0 + walk]
        )
        if "processed" in branch_data[current_br]:
            pass
        else:
            branch_data[current_br]["phases"] = correct_xfmr_phases

            corrected_zprim = list([[[0, 0], [0, 0], [0, 0]] for a in zprim])
            corrected_zprim[connected_phase][connected_phase] = [
                1e-5,
                1e-5,
            ]  # estimating secondary Zimpedances as a low value
            branch_data[current_br]["zprim"] = corrected_zprim
            branch_data[current_br]["processed"] = True

        bus_data[leaf_parent_walk[0 + walk]]["phases"] = correct_xfmr_phases
        if leaf_parent_walk[1 + walk] != target_bus:
            bus_data[leaf_parent_walk[1 + walk]]["phases"] = correct_xfmr_phases

    ####

    return connected_phase


# Function to find the first high voltage parent node


def find_primary_parent(leaf_bus, bus_data, G):
    current_bus = leaf_bus
    while list(G.predecessors(current_bus)):
        parent = next(G.predecessors(current_bus))
        if bus_data[parent]["base_kv"] > 0.5:
            return parent
        current_bus = parent
    return None


# Function to update PQ values to the same phase


def process_secondary_side(leaf_bus, parent_bus, bus_data, branch_data):
    leaf_pq = bus_data[leaf_bus]["pq"]
    leaf_pv = bus_data[leaf_bus]["pv"]
    leaf_base_pq = bus_data[leaf_bus]["base_pq"]
    leaf_base_pv = bus_data[leaf_bus]["base_pv"]
    # parent_NZ_phase_idx = [idx for idx, val in enumerate(parent_phases) if val != 0]

    leaf_phases = bus_data[leaf_bus]["phases"]
    leaf_phase_bool = [p != 0 for p in leaf_phases]

    if sum(leaf_phase_bool) == 3:
        pass
    else:
        parent_phases = correct_secondary_network(
            branch_data, bus_data, parent_bus, leaf_bus
        )

        # Assign all PQ of the leaf node to the corresponding phases of the parent node
        # Create an empty PQ for three phases
        new_pq = [[0.0, 0.0] for _ in range(3)]
        new_pv = [[0.0, 0.0] for _ in range(3)]
        new_base_pq = [[0.0, 0.0] for _ in range(3)]
        new_base_pv = [[0.0, 0.0] for _ in range(3)]

        leaf_pq_sum = [0, 0]
        leaf_pv_sum = [0, 0]
        leaf_base_pq_sum = [0, 0]
        leaf_base_pv_sum = [0, 0]
        for i, phase in enumerate(leaf_phases):
            leaf_pq_sum = [x + y for x, y in zip(leaf_pq[i], leaf_pq_sum)]
            leaf_pv_sum = [x + y for x, y in zip(leaf_pv[i], leaf_pv_sum)]

            leaf_base_pq_sum = [
                x + y for x, y in zip(leaf_base_pq[i], leaf_base_pq_sum)
            ]
            leaf_base_pv_sum = [
                x + y for x, y in zip(leaf_base_pv[i], leaf_base_pv_sum)
            ]

        new_pq[parent_phases] = leaf_pq_sum
        new_pv[parent_phases] = leaf_pv_sum
        new_base_pq[parent_phases] = leaf_base_pq_sum
        new_base_pv[parent_phases] = leaf_base_pv_sum
        # Update the leaf node's PQ
        bus_data[leaf_bus]["pq"] = new_pq
        bus_data[leaf_bus]["pv"] = new_pv
        bus_data[leaf_bus]["base_pq"] = new_base_pq
        bus_data[leaf_bus]["base_pv"] = new_base_pv

    # Function to update the direction of the branch based on distance from root


def update_branch_direction_based_on_root(
    branch_data, root_node="P1UDT942-P1UHS0_1247X"
):
    G = nx.Graph()

    for branch_id, branch in branch_data.items():
        from_bus = branch["fr_bus"]
        to_bus = branch["to_bus"]
        G.add_edge(from_bus, to_bus)

    # Use BFS to calculate the shortest path from the root to all nodes
    distances_from_root = nx.single_source_shortest_path_length(G, root_node)

    for branch_id, branch in branch_data.items():
        from_bus = branch["fr_bus"]
        to_bus = branch["to_bus"]
        from_idx = branch["fr_idx"]
        to_idx = branch["to_idx"]

        # Get distances of from_bus and to_bus from the root node
        from_distance = distances_from_root[from_bus]
        to_distance = distances_from_root[to_bus]

        # If the from_bus is farther from the root than to_bus, swap them
        if from_distance > to_distance:
            # Swap the buses
            branch_data[branch_id]["fr_bus"] = to_bus
            branch_data[branch_id]["to_bus"] = from_bus
            branch_data[branch_id]["fr_idx"] = to_idx
            branch_data[branch_id]["to_idx"] = from_idx
            print(f"Swapped {branch_id}: from {from_bus} to {
                  to_bus} (now {to_bus} to {from_bus})")
        else:
            print(f"Kept {branch_id}: from {from_bus} to {to_bus} (no change)")
