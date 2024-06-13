import copy
import os
import math
import json
import pathlib

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np


class ModelUtil:

    def __init__(self, dss):
        self.bus_info = {}
        self.dss = dss
        # Need to change this if coordinates are defined with csv format
        f = open('Buscoords.dss', 'r')
        bus_coor = {}
        for line in f:
            values = line.split(' ')
            if len(values) > 1:
                bus_name = values[0].upper()
                x_coor = float(values[1])
                y_coor = float(values[2])
                bus_coor[bus_name] = {'x': x_coor, 'y': y_coor}

        print('Total buses: {}'.format(len(dss.Circuit.AllBusNames())))
        count = 0
        for bus in dss.Circuit.AllBusNames():
            self.dss.Circuit.SetActiveBus(bus)
            count += len(self.dss.Bus.Nodes())
            self.bus_info[bus.upper()] = {'nodes': self.dss.Bus.Nodes(), 'basekV': self.dss.Bus.kVBase(), 'loc': []}
            if bus.upper() in bus_coor:
                loc = [bus_coor[bus.upper()]['x'], bus_coor[bus.upper()]['y']]
                self.bus_info[bus.upper()]['loc'] = loc

        self.lines_df = self.dss.utils.lines_to_dataframe()
        self.branch_info = {}
        for ind in self.lines_df.index:
            bus1 = self.lines_df['Bus1'][ind].split('.')[0]
            bus2 = self.lines_df['Bus2'][ind].split('.')[0]
            self.branch_info[ind] = {'bus1': bus1.upper(), 'bus2': bus2.upper()}

    def find_network_graph(self):
        flag = self.dss.Topology.First()
        G = nx.Graph()
        count = 0
        power_delivery_elements = []
        while flag > 0:
            # print(self.dss.Topology.BranchName())
            # print(self.dss.CktElement.BusNames())
            power_delivery_elements.append(self.dss.Topology.BranchName().split('.')[0])
            bus1 = self.dss.CktElement.BusNames()[0].split('.')[0].upper()
            bus2 = self.dss.CktElement.BusNames()[1].split('.')[0].upper()
            if bus1 != bus2:
                G.add_edge(bus1.upper(), bus2.upper())
            if 'Transformer' in self.dss.Topology.BranchName():
                self.branch_info[self.dss.Topology.BranchName()] = {'bus1': bus1, 'bus2': bus2}
            if 'Reactor' in self.dss.Topology.BranchName():
                self.branch_info[self.dss.Topology.BranchName()] = {'bus1': bus1, 'bus2': bus2}
            # if 'Line' in self.dss.Topology.BranchName():
            #     self.branch_info[self.dss.Topology.BranchName()] = {'bus1': bus1, 'bus2': bus2}
            count += 1
            # print(self.dss.Topology.BranchName())
            flag = self.dss.Topology.Next()
        print('Edges: {} and Nodes: {}'.format(G.number_of_edges(), G.number_of_nodes()))
        print('Total PDEs: {}'.format(count))
        print('Types of Elements: ', set(power_delivery_elements))
        return G

    def grid_voltage_profile(self):
        node_names = self.dss.Circuit.AllNodeNames()
        nodeA_names = []
        nodeB_names = []
        nodeC_names = []
        for node in node_names:
            if ".1" in node:
                nodeA_names.append(node)
            elif ".2" in node:
                nodeB_names.append(node)
            elif ".3" in node:
                nodeC_names.append(node)

        bus_voltages = {}
        bus_A = []
        bus_B = []
        bus_C = []
        phases = [1, 2, 3]
        for p in phases:
            for idx, voltage in enumerate(self.dss.Circuit.AllNodeVmagPUByPhase(p)):
                if p == 1:
                    bus_A.append(voltage)
                    bus_voltages[nodeA_names[idx]] = voltage
                elif p == 2:
                    bus_B.append(voltage)
                    bus_voltages[nodeB_names[idx]] = voltage
                elif p == 3:
                    bus_C.append(voltage)
                    bus_voltages[nodeC_names[idx]] = voltage

        # Min, Max voltage and profile plot
        print('\n..........Voltage Min-Max.............')
        print(max(bus_A), max(bus_B), max(bus_C))
        print(min(bus_A), min(bus_B), min(bus_C))
        plt.scatter(range(len(bus_A)), bus_A, facecolors='none', edgecolors='r')
        plt.scatter(range(len(bus_B)), bus_B, facecolors='none', edgecolors='g')
        plt.scatter(range(len(bus_C)), bus_C, facecolors='none', edgecolors='b')
        n_nodes = max(len(bus_A), len(bus_B), len(bus_C))

        plt.ylim([0.9, 1.1])
        plt.xlabel('Bus Index')
        plt.ylabel('Voltage (p.u.)')
        plt.legend(['Phase-A', 'Phase-B', 'Phase-C'])
        plt.plot(np.ones(n_nodes) * 1.05, 'r--')
        plt.plot(np.ones(n_nodes) * 0.95, 'r--')
        plt.show()

        # Extract line flows/currents. Example is shown for a line connected to source bus
        print('\n..........Substation Flow.............')
        line_flow_sensors = ['sb8_p14uhs13_1247_402']
        for line in line_flow_sensors:
            element = 'Line.' + line
            self.dss.Circuit.SetActiveElement(element)
            print(complex(self.dss.CktElement.Powers()[0], self.dss.CktElement.Powers()[1]))
            print(complex(self.dss.CktElement.Powers()[2], self.dss.CktElement.Powers()[3]))
            print(complex(self.dss.CktElement.Powers()[4], self.dss.CktElement.Powers()[5]))

    def voltage_violations_check(self):
        nodeA_voltages = self.dss.Circuit.AllNodeVmagPUByPhase(1)
        nodeB_voltages = self.dss.Circuit.AllNodeVmagPUByPhase(2)
        nodeC_voltages = self.dss.Circuit.AllNodeVmagPUByPhase(3)
        vmax = [max(nodeA_voltages), max(nodeB_voltages), max(nodeC_voltages)]
        violation = 0
        if max(vmax) > 1.05:
            violation = 1

        return violation

    def plot_circuit(self):
        G = self.find_network_graph()
        for branch in self.branch_info:
            try:
                point1 = self.bus_info[self.branch_info[branch]['bus1']]['loc']
                point2 = self.bus_info[self.branch_info[branch]['bus2']]['loc']
                bus1 = self.branch_info[branch]['bus1'].upper()
                bus2 = self.branch_info[branch]['bus2'].upper()
                x_values = [point1[0], point2[0]]
                y_values = [point1[1], point2[1]]
                e = (bus1, bus2)
                if G.has_edge(*e):
                    if 'Transformer' not in branch:
                        plt.plot(x_values, y_values, 'k-')
                    else:
                        plt.plot(x_values, y_values, 'g-')
                else:
                    # print(branch)
                    plt.plot(x_values, y_values, 'r-')
            except:
                pass
        plt.plot(-121.98832075802551, 37.26119909978503, 'bs')
        plt.axis('off')
        plt.show()
