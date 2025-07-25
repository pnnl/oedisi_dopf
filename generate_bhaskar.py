import os
import sys
import json
from oedisi.componentframework.system_configuration import (
    WiringDiagram,
    Component,
    Link,
)
from typing import Tuple


ROOT = os.getcwd()
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
        scenario = f"scenarios/solar_{LEVEL}_batteries_none_timeseries"
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
            "number_of_timesteps": 24*5,
            "run_freq_sec": 900,
            "topology_output": f"{OUTPUTS}/topology.json",
            "buscoords_output": f"{OUTPUTS}/Buscoords.dat",
        },
    )


def generate_recorder(port: str, src: str, OUTPUTS: str) -> Tuple[Component, Link]:
    component = Component(
        name=f"recorder_{port}",
        type="Recorder",
        parameters={
            "feather_filename": f"{OUTPUTS}/{port}.feather",
            "csv_filename": f"{OUTPUTS}/{port}.csv",
        },
    )
    link = Link(
        source=src, source_port=port, target=component.name, target_port="subscription"
    )
    return (component, link)





def generate(MODEL: str, LEVEL: str) -> None:
    OUTPUTS = f"{ROOT}/outputs/bhaskar"
    SCENARIOS = f"{ROOT}/scenarios/bhaskar"

    if "ieee" not in MODEL:
        OUTPUTS = f"{OUTPUTS}/{LEVEL}"
        SCENARIOS = f"{SCENARIOS}/{LEVEL}"

    system = WiringDiagram(name=f"bhaskar_{MODEL}", components=[], links=[])
    feeder = generate_feeder(MODEL, LEVEL, OUTPUTS)
    system.components.append(feeder)

    

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

    port = "voltage_mag"
    component, link = generate_recorder(port, feeder.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    port = "voltage_angle"
    component, link = generate_recorder(port, feeder.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    port = "available_power"
    component, link = generate_recorder(port, feeder.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    if not os.path.exists(SCENARIOS):
        os.makedirs(SCENARIOS)

    if not os.path.exists(OUTPUTS):
        os.makedirs(OUTPUTS)

    with open(f"{SCENARIOS}/system.json", "w") as f:
        f.write(system.json(indent=2))

    check = WiringDiagram.parse_file(f"{SCENARIOS}/system.json")

    components = {}
    for c in system.components:
        name = c.name
        if "_" in name:
            name, _ = c.name.split("_", 1)
        components[c.type] = f"{name}_federate/component_definition.json"

    with open(f"{SCENARIOS}/components.json", "w") as f:
        f.write(json.dumps(components))


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
