from lindistflow_federate.adapter import (
    generate_graph,
    area_dissconnects,
    disconnect_areas,
    reconnect_area_switches,
    get_area_source,
)
from pprint import pprint
from oedisi.types.data_types import Topology
from oedisi.componentframework.system_configuration import (
    WiringDiagram,
    Component,
    Link,
)
import networkx as nx
import copy
import os
import json
import sys


ROOT = os.getcwd()
OUTPUTS = f"{ROOT}/outputs"
SCENARIO_DIR = f"{ROOT}/scenario/"


def get_topology(path: str) -> Topology:
    assert os.path.exists(path), "need to generate topology from base scenario"

    with open(path) as f:
        topology = Topology.parse_obj(json.load(f))

    return topology


def get_system(path: str) -> WiringDiagram:
    with open(path) as f:
        system: WiringDiagram = WiringDiagram.parse_obj(json.load(f))
    return system


if __name__ == "__main__":
    scenario = sys.argv[1]

    path = f"{OUTPUTS}/{scenario}/topology.json"
    topology = get_topology(path)
    slack_bus, _ = topology.slack_bus[0].split(".", 1)

    G = generate_graph(topology.incidences, slack_bus)
    print("Total: ", G)
    graph = copy.deepcopy(G)
    graph2 = copy.deepcopy(G)
    boundaries = area_dissconnects(graph)
    areas = disconnect_areas(graph2, boundaries)
    areas = reconnect_area_switches(areas, boundaries)

    for area in areas:
        print(area)
        src = []
        for u, v, a in boundaries:
            if area.has_edge(u, v):
                print("\t", a["id"])
                src.append((u, v, a))
        su, sv, sa = get_area_source(G, slack_bus, src)
        print("\tsource: ", sa["id"])

    path = f"{SCENARIO_DIR}/{scenario}/system.json"
    system = get_system(path)

    feeder: Component = [c for c in system.components if c.name == "feeder"][0]
    feeder.parameters["topology_output"] = f"{
        OUTPUTS}/admm_{scenario}/topology.json"

    pprint(feeder)
