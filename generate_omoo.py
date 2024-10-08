import os
import sys
import json
from oedisi.componentframework.system_configuration import (
    WiringDiagram,
    Component,
    Link,
)


ROOT = os.getcwd()
ALGO = "omoo"
MODEL = ""
PV_LEVEL = "low"
ES_LEVEL = "none"
NAME = ""
OUTPUTS = ""
SCENARIOS = ""

SMART_DS = {
    "SFO/P1U": "p1uhs0_1247/p1uhs0_1247--p1udt942",
    "SFO/P6U": "p6uhs10_1247/p6uhs10_1247--p6udt5293",
    "SFO/P9U": "p9uhs16_1247/p9uhs16_1247--p9udt12866",
}

LEVELS = ["low", "medium", "high", "extreme"]


def generate_feeder() -> Component:
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
            PV_LEVEL}_batteries_{ES_LEVEL}_timeseries"
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
            "start_date": "2018-05-01 12:00:00",
            "number_of_timesteps": 1,
            "run_freq_sec": 900,
            "topology_output": f"{OUTPUTS}/topology.json",
            "buscoords_output": f"{OUTPUTS}/Buscoords.dat",
        },
    )


def generate_recorder(port: str, ctx: str, src: str) -> (Component, Link):
    component = Component(
        name=f"recorder_{port}_{ctx}",
        type="Recorder",
        parameters={
            "feather_filename": f"{OUTPUTS}/{port}_{ctx}.feather",
            "csv_filename": f"{OUTPUTS}/{port}_{ctx}.csv",
        },
    )
    if "sensor" in src:
        port = "publication"
    link = Link(
        source=src, source_port=port, target=component.name, target_port="subscription"
    )
    return (component, link)


def generate_sensor(port: str, ctx: str, src: str) -> (Component, Link):
    if "powers_real" in port:
        file = "real_ids.json"
    if "powers_imag" in port:
        file = "reactive_ids.json"
    if "voltage" in port:
        file = "voltage_ids.json"
    component = Component(
        name=f"sensor_{port}_{ctx}",
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("generate_lindistflow.py <model> [pv penetration] [es penetration]\n")
        print("\tex. generate_indistflow.py ieee123")
        print("\tex. generate_indistflow.py SFO/P1U extreme low\n")
        print("Available Smart-DS Models:")
        print("\t", list(SMART_DS.keys()), "\n")
        print("Available Penetration levels:")
        print("\t", LEVELS, "\n")
        exit()

    MODEL = sys.argv[1]
    if len(sys.argv) == 3:
        if sys.argv[2] not in LEVELS:
            print("PV_LEVEL must be: ", LEVELS)
            exit()
        PV_LEVEL = sys.argv[2]

    if len(sys.argv) == 4:
        lvl = sys.argv[3]
        if lvl not in LEVELS and lvl != "none":
            print("ES_LEVEL must be none or: ", LEVELS)
            exit()
        ES_LEVEL = sys.argv[3]

    OUTPUTS = f"{ROOT}/outputs/{ALGO}/{MODEL}"
    SCENARIOS = f"{ROOT}/scenarios/{ALGO}/{MODEL}"

    if "ieee" not in MODEL:
        OUTPUTS = f"{OUTPUTS}/{PV_LEVEL}/{ES_LEVEL}"
        SCENARIOS = f"{SCENARIOS}/{PV_LEVEL}/{ES_LEVEL}"

    system = WiringDiagram(name=f"{ALGO}_{MODEL}", components=[], links=[])
    feeder = generate_feeder()
    system.components.append(feeder)

    algo = Component(
        name=ALGO,
        type="OptimalPowerFlow",
        parameters={"algorithm_parameters": {"tol": 1e-5}},
    )
    system.components.append(algo)

    for subject in ["voltages", "powers"]:
        port = f"{subject}_real"
        component, link = generate_recorder(port, "ref", feeder.name)
        system.components.append(component)
        system.links.append(link)

        system.links.append(
            Link(
                source=feeder.name, source_port=port, target=algo.name, target_port=port
            )
        )

        component, link = generate_recorder(port, "algo", algo.name)
        system.components.append(component)
        system.links.append(link)

        port = f"{subject}_imag"
        component, link = generate_recorder(port, "ref", feeder.name)
        system.components.append(component)
        system.links.append(link)

        system.links.append(
            Link(
                source=feeder.name, source_port=port, target=algo.name, target_port=port
            )
        )

        component, link = generate_recorder(port, "algo", algo.name)
        system.components.append(component)
        system.links.append(link)

    ctx = "pv_set"
    system.links.append(
        Link(source=algo.name, source_port=ctx, target=feeder.name, target_port=ctx)
    )

    ctx = "available_power"
    system.links.append(
        Link(source=feeder.name, source_port=ctx, target=algo.name, target_port=ctx)
    )

    ctx = "injections"
    system.links.append(
        Link(source=feeder.name, source_port=ctx, target=algo.name, target_port=ctx)
    )

    ctx = "topology"
    system.links.append(
        Link(source=feeder.name, source_port=ctx, target=algo.name, target_port=ctx)
    )

    if not os.path.exists(SCENARIOS):
        os.makedirs(SCENARIOS)

    with open(f"{SCENARIOS}/system.json", "w") as f:
        f.write(system.json())

    check = WiringDiagram.parse_file(f"{SCENARIOS}/system.json")

    components = {}
    for c in system.components:
        name = c.name
        if "_" in name:
            name, _ = c.name.split("_", 1)
        components[c.type] = f"{name}_federate/component_definition.json"

    with open(f"{SCENARIOS}/components.json", "w") as f:
        f.write(json.dumps(components))
