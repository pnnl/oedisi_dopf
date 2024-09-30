import os
import sys
import json
from oedisi.componentframework.system_configuration import (
    WiringDiagram, Component, Link
)


ROOT = os.getcwd()
ALGO = "lest"
NAME = ""
OUTPUTS = ""
SCENARIOS = ""

SMART_DS = {
    "SFO/P1U": "p1uhs0_1247/p1uhs0_1247--p1udt942",
    "SFO/P6U": "p6uhs10_1247/p6uhs10_1247--p6udt5293",
    "SFO/P9U": "p9uhs16_1247/p9uhs16_1247--p9udt12866"
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
            "start_date": "2018-05-01 04:00:00",
            "number_of_timesteps": 4,
            "run_freq_sec": 4*3600,
            "topology_output": f"{OUTPUTS}/topology.json",
            "buscoords_output": f"{OUTPUTS}/Buscoords.dat"
        }
    )


def generate_recorder(port: str, src: str, OUTPUTS: str) -> (Component, Link):
    component = Component(
        name=f"recorder_{port}",
        type="Recorder",
        parameters={
            "feather_filename": f"{OUTPUTS}/{port}.feather",
            "csv_filename": f"{OUTPUTS}/{port}.csv",
        }
    )
    link = Link(
        source=src,
        source_port=port,
        target=component.name,
        target_port="subscription"
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
            "measurement_file": f"../{src}/{file}"
        }
    )
    link = Link(
        source=src,
        source_port=port,
        target=component.name,
        target_port="subscription"
    )
    return (component, link)


def generate(MODEL: str, LEVEL: str) -> None:
    OUTPUTS = f"{ROOT}/outputs/{ALGO}/{MODEL}"
    SCENARIOS = f"{ROOT}/scenarios/{ALGO}/{MODEL}"

    if "ieee" not in MODEL:
        OUTPUTS = f"{OUTPUTS}/{LEVEL}"
        SCENARIOS = f"{SCENARIOS}/{LEVEL}"

    system = WiringDiagram(name=f"{ALGO}_{MODEL}", components=[], links=[])
    feeder = generate_feeder(MODEL, LEVEL, OUTPUTS)
    system.components.append(feeder)

    algo = Component(
        name="lest",
        type="Estimator",
        parameters={
            "tol": 1e3,
            "v_sigma": 0.02,
            "l_sigma": 0.001,
            "i_sigma": 0.001,
            "deltat": 0.1
        }
    )
    system.components.append(algo)

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

    component, link = generate_sensor(port, feeder.name)
    system.components.append(component)
    system.links.append(link)

    system.links.append(Link(
        source=component.name,
        source_port="publication",
        target=algo.name,
        target_port=port
    ))

    port = "power_imag"
    component, link = generate_recorder(port, feeder.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    component, link = generate_sensor(port, feeder.name)
    system.components.append(component)
    system.links.append(link)

    system.links.append(Link(
        source=component.name,
        source_port="publication",
        target=algo.name,
        target_port=port
    ))

    port = "voltage_mag"
    component, link = generate_sensor(port, feeder.name)
    system.components.append(component)
    system.links.append(link)

    system.links.append(Link(
        source=component.name,
        source_port="publication",
        target=algo.name,
        target_port=port
    ))

    component, link = generate_recorder(port, algo.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    port = "voltage_angle"
    system.links.append(Link(
        source=feeder.name,
        source_port=port,
        target=algo.name,
        target_port=port
    ))

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

    ctx = "injections"
    system.links.append(Link(
        source=feeder.name,
        source_port=ctx,
        target=algo.name,
        target_port=ctx
    ))

    ctx = "topology"
    system.links.append(Link(
        source=feeder.name,
        source_port=ctx,
        target=algo.name,
        target_port=ctx
    ))

    if not os.path.exists(SCENARIOS):
        os.makedirs(SCENARIOS)

    with open(f"{SCENARIOS}/system.json", 'w') as f:
        f.write(system.json())

    check = WiringDiagram.parse_file(f"{SCENARIOS}/system.json")

    components = {}
    for c in system.components:
        name = c.name
        if "_" in name:
            name, _ = c.name.split("_", 1)
        components[c.type] = f"{name}_federate/component_definition.json"

    with open(f"{SCENARIOS}/components.json", 'w') as f:
        f.write(json.dumps(components))


if __name__ == "__main__":
    for model in MODELS:
        for level in LEVELS:
            generate(model, level)
