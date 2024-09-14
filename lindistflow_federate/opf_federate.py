import logging
import helics as h
import json
from pathlib import Path
from datetime import datetime
from oedisi.types.data_types import (
    CommandList,
    Command,
    PowersImaginary,
    PowersReal,
    Injection,
    Topology,
    VoltagesMagnitude,
    MeasurementArray,
)
import adapter
import lindistflow
from dataclasses import asdict
from area import area_info, check_network_radiality
import xarray as xr

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


def xarray_to_dict(data):
    """Convert xarray to dict with values and ids for JSON serialization."""
    coords = {key: list(data.coords[key].data) for key in data.coords.keys()}
    return {"values": list(data.data), **coords}


class StaticConfig(object):
    name: str
    deltat: float
    control_type: lindistflow.ControlType
    pf_flag: bool


class Subscriptions(object):
    voltages_mag: VoltagesMagnitude
    injections: Injection
    topology: Topology
    pv_forecast: list


class OPFFederate(object):
    def __init__(self) -> None:
        self.sub = Subscriptions()
        self.load_static_inputs()
        self.load_input_mapping()
        self.initilize()
        self.load_component_definition()
        self.register_subscription()
        self.register_publication()

    def load_component_definition(self) -> None:
        path = Path(__file__).parent / "component_definition.json"
        with open(path, "r", encoding="UTF-8") as file:
            self.component_config = json.load(file)

    def load_input_mapping(self):
        path = Path(__file__).parent / "input_mapping.json"
        with open(path, "r", encoding="UTF-8") as file:
            self.inputs = json.load(file)

    def load_static_inputs(self):
        self.static = StaticConfig()
        path = Path(__file__).parent / "static_inputs.json"
        with open(path, "r", encoding="UTF-8") as file:
            config = json.load(file)

        self.static.name = config["name"]
        self.static.deltat = config["deltat"]
        self.static.control_type = lindistflow.ControlType(
            config["control_type"])
        self.static.pf_flag = config["pf_flag"]

    def initilize(self) -> None:
        self.info = h.helicsCreateFederateInfo()
        self.info.core_name = self.static.name
        self.info.core_type = h.HELICS_CORE_TYPE_ZMQ
        self.info.core_init = "--federates=1"

        h.helicsFederateInfoSetTimeProperty(
            self.info, h.helics_property_time_delta, self.static.deltat
        )

        self.fed = h.helicsCreateValueFederate(self.static.name, self.info)

    def register_subscription(self) -> None:
        self.sub.topology = self.fed.register_subscription(
            self.inputs["topology"], "")
        self.sub.voltages_mag = self.fed.register_subscription(
            self.inputs["voltages_magnitude"], "")
        self.sub.injections = self.fed.register_subscription(
            self.inputs["injections"], "")
        self.sub.available_power = self.fed.register_subscription(
            self.inputs["available_power"], "")

    def register_publication(self) -> None:
        self.pub_commands = self.fed.register_publication(
            "change_commands", h.HELICS_DATA_TYPE_STRING, ""
        )

        self.pub_voltages = self.fed.register_publication(
            "opf_voltages_magnitude", h.HELICS_DATA_TYPE_STRING, ""
        )

    def get_set_points(self, control, bus_info, conversion):
        setpoint = {}
        for key, val in control.items():
            if key in bus_info:
                bus = bus_info[key]
                if 'eqid' in bus:
                    eqid = bus['eqid']
                    [eq_type, _] = eqid.split('.')
                    if eq_type == "PVSystem":
                        sp = lindistflow.ignore_phase(val)*conversion
                        setpoint[eqid] = 0.0 if sp < 0.1 else sp
        return setpoint

    def run(self) -> None:
        logger.info(f"Federate connected: {datetime.now()}")
        self.fed.enter_executing_mode()
        granted_time = h.helicsFederateRequestTime(
            self.fed, h.HELICS_TIME_MAXTIME)

        while granted_time < h.HELICS_TIME_MAXTIME:

            if not self.sub.voltages_mag.is_updated():
                granted_time = h.helicsFederateRequestTime(
                    self.fed, h.HELICS_TIME_MAXTIME
                )
                continue

            topology: Topology = Topology.parse_obj(self.sub.topology.json)
            branch_info, bus_info, slack_bus = adapter.extract_info(topology)

            injections = Injection.parse_obj(self.sub.injections.json)
            bus_info = adapter.extract_injection(bus_info, injections)

            voltages_mag = VoltagesMagnitude.parse_obj(
                self.sub.voltages_mag.json)
            bus_info = adapter.extract_voltages(bus_info, voltages_mag)

            time = voltages_mag.time
            logger.debug(f"Timestep: {time}")

            with open("bus_info.json", "w") as outfile:
                outfile.write(json.dumps(asdict(bus_info)))

            with open("branch_info.json", "w") as outfile:
                outfile.write(json.dumps(asdict(branch_info)))

            assert (adapter.check_radiality(branch_info, bus_info))

            p_inj = MeasurementArray.parse_obj(
                self.sub.available_power.json)
            for id, v in zip(p_inj.ids, p_inj.values):
                logger.debug(f"{id},  {v}")

            voltages, power_flow, control, conversion = lindistflow.solve(
                branch_info, bus_info, slack_bus, self.static.control_type, self.static.pf_flag)
            real_setpts = self.get_set_points(control, bus_info, conversion)

            # get the control commands for the feeder federate
            commands = []
            for key, val in control.items():
                if key not in bus_info.buses:
                    continue

                bus = bus_info.buses[key]
                if all([pv == 0.0 for pv in bus.pv]):
                    continue

                setpoint = max([abs(v) for v in val])*conversion
                if setpoint < 0.1:
                    continue

                pv = [tag for tag in bus.tags if "PVSystem" in tag]
                ctrl = self.static.control_type
                for tag in pv:
                    if ctrl == lindistflow.ControlType.WATT:
                        commands.append((tag, setpoint, 0))
                    elif ctrl == lindistflow.ControlType.VAR:
                        commands.append((tag, 0, setpoint))
                    elif ctrl == lindistflow.ControlType.WATT_VAR:
                        continue

            if commands:
                self.pub_commands.publish(
                    json.dumps(commands)
                )

            pub_mags = adapter.pack_voltages(voltages, time)
            self.pub_voltages.publish(
                pub_mags.json()
            )

        self.stop()

    def stop(self) -> None:
        h.helicsFederateDisconnect(self.fed)
        h.helicsFederateFree(self.fed)
        h.helicsCloseLibrary()
        logger.info(f"Federate disconnected: {datetime.now()}")


if __name__ == "__main__":
    fed = OPFFederate()
    fed.run()
