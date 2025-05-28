from admm_federate.adapter import (
    generate_graph,
    area_disconnects,
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
ALGO = "admm"
NAME = ""
OUTPUTS = ""
SCENARIOS = ""

SMART_DS = {
    "SFO/P1U": "p1uhs0_1247/p1uhs0_1247--p1udt942",
    "SFO/P6U": "p6uhs10_1247/p6uhs10_1247--p6udt5293",
    "SFO/P9U": "p9uhs16_1247/p9uhs16_1247--p9udt12866",
}
MODELS = ["ieee123", "SFO/P1U", "SFO/P6U", "SFO/P9U"]
LEVELS = ["low", "medium", "high", "extreme"]


def generate_feeder(MODEL: str, LEVEL: str, OUTPUTS: str) -> Component:
    if "ieee" in MODEL:
        smart_ds = False
        base = "gadal_ieee123"
        profiles = f"{base}/profiles"
        opendss = f"{base}/qsts"
        file = "opendss/master.dss"
    else:
        smart_ds = True
        base = f"SMART-DS/v1.0/2018/{MODEL}"
        scenario = f"scenarios/solar_{
            LEVEL}_batteries_none_timeseries"
        profiles = f"{base}/profiles"
        opendss = f"{base}/{scenario}/opendss/{SMART_DS[MODEL]}"
        file = "opendss/Master.dss"

    return Component(
        name="feeder",
        type="Feeder",
        parameters={
            "use_smartds": smart_ds,
            "use_sparse_admittance": True,
            "profile_location": profiles,
            "opendss_location": opendss,
            "feeder_file": file,
            "start_date": "2018-05-01 11:30:00",
            "number_of_timesteps": 5,
            "run_freq_sec": 900,
            "topology_output": f"{OUTPUTS}/topology.json",
            "buscoords_output": f"{OUTPUTS}/Buscoords.dat",
        },
    )


def generate_recorder(port: str, src: str, OUTPUTS: str) -> (Component, Link):
    name = f"recorder_{port}_{src}"
    file = f"{port}_{src}"
    if src == "feeder":
        name = f"recorder_{port}"
        file = port

    component = Component(
        name=name,
        type="Recorder",
        parameters={
            "feather_filename": f"{OUTPUTS}/{file}.feather",
            "csv_filename": f"{OUTPUTS}/{file}.csv",
        },
    )
    link = Link(
        source=src, source_port=port, target=component.name, target_port="subscription"
    )
    return (component, link)


def generate_sensor(port: str, src: str) -> (Component, Link):
    if "power_real" in port:
        file = "sensors/real_ids.json"
    if "power_imag" in port:
        file = "sensors/reactive_ids.json"
    if "voltage" in port:
        file = "sensors/voltage_ids.json"
    component = Component(
        name=f"sensor_{port}",
        type="Sensor",
        parameters={
            "additive_noise_stddev": 0.01,
            "multiplicative_noise_stddev": 0.001,
            "measurement_file": f"../{src}/{file}",
        },
    )
    link = Link(
        source=src, source_port=port, target=component.name, target_port="subscription"
    )
    return (component, link)


def link_feeder(system: WiringDiagram, feeder: Component) -> None:
    port = "voltage_real"
    component, link = generate_recorder(port, feeder.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    port = "voltage_imag"
    component, link = generate_recorder(port, feeder.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    port = "power_real"
    component, link = generate_recorder(port, feeder.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    port = "power_imag"
    component, link = generate_recorder(port, feeder.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    port = "available_power"
    component, link = generate_recorder(port, feeder.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)


def link_hub_voltage(system: WiringDiagram, hub: Component, src: int) -> None:
    system.links.append(
        Link(source=f"{ALGO}_{src}", source_port="area_v",
             target=f"{hub.name}", target_port=f"area_v{src}")
    )
    system.links.append(
        Link(source=f"{hub.name}", source_port=f"area_v{src}",
             target=f"{ALGO}_{src}", target_port="area_v")
    )


def link_hub_power(system: WiringDiagram, hub: Component, src: int) -> None:
    system.links.append(
        Link(source=f"{ALGO}_{src}", source_port="area_p",
             target=f"{hub.name}", target_port=f"area_p{src}")
    )
    system.links.append(
        Link(source=f"{hub.name}", source_port=f"area_p{src}",
             target=f"{ALGO}_{src}", target_port="area_p")
    )
    system.links.append(
        Link(source=f"{ALGO}_{src}", source_port="area_q",
             target=f"{hub.name}", target_port=f"area_q{src}")
    )
    system.links.append(
        Link(source=f"{hub.name}", source_port=f"area_q{src}",
             target=f"{ALGO}_{src}", target_port="area_q")
    )


def link_hub_control(system: WiringDiagram, hub: Component, src: int) -> None:
    system.links.append(
        Link(source=f"{ALGO}_{src}", source_port="area_c",
             target=f"{hub.name}", target_port=f"area_c{src}")
    )


def link_algo(system: WiringDiagram, algo: Component, feeder: Component) -> None:
    port = "voltage_real"
    system.links.append(
        Link(source=feeder.name, source_port=port,
             target=algo.name, target_port=port)
    )

    port = "voltage_imag"
    system.links.append(
        Link(source=feeder.name, source_port=port,
             target=algo.name, target_port=port)
    )

    port = "power_real"
    system.links.append(
        Link(source=feeder.name, source_port=port,
             target=algo.name, target_port=port)
    )

    port = "power_imag"
    system.links.append(
        Link(source=feeder.name, source_port=port,
             target=algo.name, target_port=port)
    )

    port = "voltage_mag"
    component, link = generate_recorder(port, algo.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    port = "voltage_angle"
    component, link = generate_recorder(port, algo.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    port = "power_mag"
    component, link = generate_recorder(port, algo.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    port = "power_angle"
    component, link = generate_recorder(port, algo.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    port = "solver_stats"
    component, link = generate_recorder(port, algo.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    port = "estimated_power"
    component, link = generate_recorder(port, algo.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    port = "available_power"
    system.links.append(
        Link(source=feeder.name, source_port=port,
             target=algo.name, target_port=port)
    )

    port = "injections"
    system.links.append(
        Link(source=feeder.name, source_port=port,
             target=algo.name, target_port=port)
    )

    port = "topology"
    system.links.append(
        Link(source=feeder.name, source_port=port,
             target=algo.name, target_port=port)
    )


def generate(MODEL: str, LEVEL: str) -> None:
    OUTPUTS = f"{ROOT}/outputs/lindistflow/{MODEL}"
    SCENARIOS = f"{ROOT}/scenarios/{ALGO}/{MODEL}"

    if "ieee" not in MODEL:
        OUTPUTS = f"{OUTPUTS}/{LEVEL}"
        SCENARIOS = f"{SCENARIOS}/{LEVEL}"

    path = f"{OUTPUTS}/topology.json"
    topology = get_topology(path)
    slack_bus, _ = topology.slack_bus[0].split(".", 1)

    G = generate_graph(topology.incidences, slack_bus)
    print("Total: ", G)
    graph = copy.deepcopy(G)
    graph2 = copy.deepcopy(G)
    boundaries = area_disconnects(graph)
    areas = disconnect_areas(graph2, boundaries)
    areas = reconnect_area_switches(areas, boundaries)

    system = WiringDiagram(name=f"{ALGO}_{MODEL}", components=[], links=[])
    feeder = generate_feeder(MODEL, LEVEL, OUTPUTS)
    system.components.append(feeder)

    link_feeder(system, feeder)

    switch_map = {}
    source_map = {}
    for i, area in enumerate(areas):
        src = []
        switches = []
        for u, v, a in boundaries:
            if area.has_edge(u, v):
                switches.append(a["id"])
                src.append((u, v, a))
        su, sv, sa = get_area_source(G, slack_bus, src)
        switch_map[i] = switches
        source_map[i] = sa["id"]

    sub_areas = {}
    for area, switches in switch_map.items():
        area_set = set()
        for a, s in switch_map.items():
            if any(switch in s for switch in switches):
                if a != area:
                    area_set.add(a)
        sub_areas[area] = area_set

    for k, v in sub_areas.items():
        algo = Component(
            name=f"{ALGO}_{k}",
            type="OptimalPowerFlow",
            parameters={
                "deltat": 0.1,
                "relaxed": False,
                "control_type": "real",
                "switches": switch_map[k],
                "source": source_map[k]
            },
        )
        system.components.append(algo)
        link_algo(system, algo, feeder)

    hub_voltage = Component(
        name="hub_voltage",
        type="VoltageHub",
        parameters={
            "max_iter": 10,
        },
    )
    system.components.append(hub_voltage)

    hub_power = Component(
        name="hub_power",
        type="PowerHub",
        parameters={
            "max_iter": 10,
        },
    )
    system.components.append(hub_power)

    hub_control = Component(
        name="hub_control",
        type="ControlHub",
        parameters={
            "max_iter": 10,
        },
    )
    system.components.append(hub_control)

    for k, v in sub_areas.items():
        print(k, v)
        link_hub_voltage(system, hub_voltage, k)
        link_hub_power(system, hub_power, k)
        link_hub_control(system, hub_control, k)

    if not os.path.exists(SCENARIOS):
        os.makedirs(SCENARIOS)

    if not os.path.exists(OUTPUTS):
        os.makedirs(OUTPUTS)

    with open(f"{SCENARIOS}/system.json", "w") as f:
        f.write(system.json())

    check = WiringDiagram.parse_file(f"{SCENARIOS}/system.json")

    components = {}
    for c in system.components:
        name = c.name
        print("Linking Component: ", name)
        if "hub" in name:
            components[c.type] = f"{name}/component_definition.json"
            continue

        if "_" in name:
            name, _ = c.name.split("_", 1)
            components[c.type] = f"{
                name}_federate/component_definition.json"

        components[c.type] = f"{name}_federate/component_definition.json"

    with open(f"{SCENARIOS}/components.json", "w") as f:
        f.write(json.dumps(components))


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
    if len(sys.argv) == 2:
        model = sys.argv[1]
        for level in LEVELS:
            print("generating: ", model, level)
            generate(model, level)
        exit()

    for model in MODELS:
        for level in LEVELS:
            print("generating: ", model, level)
            generate(model, level)
