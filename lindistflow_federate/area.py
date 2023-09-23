
"""
Created on Sun March 10 12:58:46 2023
@author: poud579 & Rabayet
"""

import networkx as nx
import copy
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


def graph_process(branch_info: dict):
    graph = nx.Graph()
    for b in branch_info:
        graph.add_edge(branch_info[b]['fr_bus'],
                       branch_info[b]['to_bus'])
    return graph


def graph_process(branch_info: dict):
    for b in branch_info:
        graph.add_edge(branch_info[b]['fr_bus'],
                       branch_info[b]['to_bus'])
    return graph


def area_info(branch_info: dict, bus_info: dict, source_bus: str):

    G = graph_process(branch_info)

    # Find area between the switches
    for e in edge:
        G.remove_edge(e[0], e[1])

    area_info_swt = {'area_cen': {}}

    # area_info_swt['area_cen']['edges'] = [] # include switches that are opened i.e., [['54', '94'], ['151', '300']]
    area_info_swt['area_cen']['edges'] = open_switches
    area_info_swt['area_cen']['source_bus'] = sourcebus
    area_info_swt['area_cen']['vsrc'] = v_source
    edge = area_info_swt['area_cen']['edges']
    area_source_bus = area_info_swt['area_cen']['source_bus']
    G_area = copy.deepcopy(G)
    branch_sw_data_area_cen, bus_info_area_cen = area_info(G_area, edge, branch_sw_xfmr, bus_info, sourcebus,
                                                           area_source_bus)

    areas_info = {'bus_info': {}, 'branch_info': {}}
    areas_info['bus_info']['area_cen'] = bus_info_area_cen
    areas_info['branch_info']['area_cen'] = branch_sw_data_area_cen

    # List of sub-graphs. The one that has no sourcebus is the disconnected one
    sp_graph = list(nx.connected_components(G))
    for k in sp_graph:
        if sourcebus == area_source_bus:
            if sourcebus in k:
                area = k
                break
        else:
            if sourcebus not in k:
                area = k
                break

    bus_info_area_i = {}
    idx = 0
    sump = 0
    sumq = 0
    sumpv = 0
    for key, val_bus in bus_info.items():
        if key in area:
            if bus_info[key]['kv'] > primary_kv_level:
                bus_info_area_i[key] = {}
                bus_info_area_i[key]['idx'] = idx
                bus_info_area_i[key]['phases'] = bus_info[key]['phases']
                bus_info_area_i[key]['kv'] = bus_info[key]['kv']
                bus_info_area_i[key]['s_rated'] = (
                    bus_info[key]['pv'][0][0] + bus_info[key]['pv'][1][0] + bus_info[key]['pv'][2][0])
                bus_info_area_i[key]['pv'] = [
                    [pv[0] * mult_pv, pv[1] * mult_pv] for pv in bus_info[key]['pv']]
                bus_info_area_i[key]['pq'] = [
                    [pq[0] * mult_load, pq[1] * mult_load] for pq in bus_info[key]['pq']]
                sump += bus_info_area_i[key]['pq'][0][0]
                sump += bus_info_area_i[key]['pq'][1][0]
                sump += bus_info_area_i[key]['pq'][2][0]
                sumq += bus_info_area_i[key]['pq'][0][1]
                sumq += bus_info_area_i[key]['pq'][1][1]
                sumq += bus_info_area_i[key]['pq'][2][1]
                sumpv += bus_info_area_i[key]['pv'][0][0]
                sumpv += bus_info_area_i[key]['pv'][1][0]
                sumpv += bus_info_area_i[key]['pv'][2][0]
                idx += 1

    for key, val_bus in bus_info.items():
        if key in area:
            if bus_info[key]['kv'] < primary_kv_level:
                bus_info_area_i[key] = {}
                bus_info_area_i[key]['idx'] = idx
                bus_info_area_i[key]['phases'] = bus_info[key]['phases']
                bus_info_area_i[key]['kv'] = bus_info[key]['kv']
                bus_info_area_i[key]['pv'] = [
                    i * mult_sec_pv for i in bus_info[key]['pv']]
                bus_info_area_i[key]['pq'] = [
                    i * mult_load for i in bus_info[key]['pq']]
                bus_info_area_i[key]['s_rated'] = (bus_info[key]['pv'][0])
                sump += bus_info_area_i[key]['pq'][0]
                sumq += bus_info_area_i[key]['pq'][1]
                sumpv += bus_info_area_i[key]['pv'][0]
                idx += 1
    idx = 0

    secondary_model = ['SPLIT_PHASE', 'TPX_LINE']
    branch_sw_data_area_i = {}
    nor_open = ['sw7', 'sw8']
    for key, val_bus in branch_sw_data.items():
        if val_bus['fr_bus'] in bus_info_area_i and val_bus['to_bus'] in bus_info_area_i:
            if branch_sw_data[key]['type'] not in secondary_model and key not in nor_open:
                branch_sw_data_area_i[key] = {}
                branch_sw_data_area_i[key]['idx'] = idx
                branch_sw_data_area_i[key]['type'] = branch_sw_data[key]['type']
                branch_sw_data_area_i[key]['from'] = bus_info_area_i[branch_sw_data[key]
                                                                     ['fr_bus']]['idx']
                branch_sw_data_area_i[key]['to'] = bus_info_area_i[branch_sw_data[key]
                                                                   ['to_bus']]['idx']
                branch_sw_data_area_i[key]['fr_bus'] = branch_sw_data[key]['fr_bus']
                branch_sw_data_area_i[key]['to_bus'] = branch_sw_data[key]['to_bus']
                branch_sw_data_area_i[key]['phases'] = branch_sw_data[key]['phases']
                if branch_sw_data[key]['type'] == 'SPLIT_PHASE':
                    branch_sw_data_area_i[key]['impedance'] = branch_sw_data[key]['impedance']
                    branch_sw_data_area_i[key]['impedance1'] = branch_sw_data[key]['impedance1']
                else:
                    branch_sw_data_area_i[key]['zprim'] = branch_sw_data[key]['zprim']
                idx += 1
    idx = 0
    for key, val_bus in branch_sw_data.items():
        if val_bus['fr_bus'] in bus_info_area_i and val_bus['to_bus'] in bus_info_area_i:
            if branch_sw_data[key]['type'] in secondary_model:
                branch_sw_data_area_i[key] = {}
                branch_sw_data_area_i[key]['idx'] = idx
                branch_sw_data_area_i[key]['type'] = branch_sw_data[key]['type']
                branch_sw_data_area_i[key]['from'] = bus_info_area_i[branch_sw_data[key]
                                                                     ['fr_bus']]['idx']
                branch_sw_data_area_i[key]['to'] = bus_info_area_i[branch_sw_data[key]
                                                                   ['to_bus']]['idx']
                branch_sw_data_area_i[key]['fr_bus'] = branch_sw_data[key]['fr_bus']
                branch_sw_data_area_i[key]['to_bus'] = branch_sw_data[key]['to_bus']
                branch_sw_data_area_i[key]['phases'] = branch_sw_data[key]['phases']
                if branch_sw_data[key]['type'] == 'SPLIT_PHASE':
                    branch_sw_data_area_i[key]['impedance'] = branch_sw_data[key]['impedance']
                    branch_sw_data_area_i[key]['impedance1'] = branch_sw_data[key]['impedance1']
                else:
                    branch_sw_data_area_i[key]['impedance'] = branch_sw_data[key]['zprim']
                idx += 1
    return branch_sw_data_area_i, bus_info_area_i


def check_network_radiality(branch_info_cen, bus_info_cen, bus_info):
    if not len(bus_info_cen)-len(branch_info_cen) == 1:
        # if not len(bus_info_cen) - len(bus_info) == 0:
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("                !!!!  ERROR  !!!!                   ")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Stopping simulation prematurely")
        print("------------------------------------------------------------")
        print(" ->>> Network is NOT Radial. PLease check the network data")
        print("------------------------------------------------------------")
        exit()


def check_network_PowerFlow(branch_info_cen, bus_info_cen, Base_kV, agent_source_bus, agent_source_bus_idx, vsrc, solver_name):
    # Power Flow Check:
    print('checking power flow')
    pf_flag = 1
    P_control = 1
    Q_control = 0

    print_LineFlows_Voltage = 0

    bus_voltage_area_cen, flow_area_cen, Control_variables_dict, kw_converter = _solve_lindist_OPF(branch_info_cen, bus_info_cen,
                                                                                                   Base_kV, agent_source_bus,
                                                                                                   agent_source_bus_idx, vsrc,
                                                                                                   pf_flag,
                                                                                                   solver_name, P_control, Q_control,
                                                                                                   print_LineFlows_Voltage, print_result=False)

    maxV = 0
    minV = 5
    for key, val in bus_voltage_area_cen.items():
        node_voltA = bus_voltage_area_cen[key]['A']
        node_voltB = bus_voltage_area_cen[key]['B']
        node_voltC = bus_voltage_area_cen[key]['C']
        node_max = max(node_voltA, node_voltB, node_voltC)
        node_min = min(node_voltA, node_voltB, node_voltC)
        if node_max > maxV:
            maxV = node_max
        if node_min < minV:
            minV = node_min

    if maxV > 1.2 or minV < 0.8:
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("                !!!!  ERROR  !!!!                   ")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Stopping simulation prematurely")
        print(
            "----------------------------------------------------------------------------")
        print(" ->>> Network is not power flow feasible. Update the network parameter")
        print(
            "----------------------------------------------------------------------------")
        exit()
    else:
        all_voltage_plot(bus_info_cen, bus_voltage_area_cen,
                         title="Voltage: Base Power Flow", plot_node_voltage=1)
