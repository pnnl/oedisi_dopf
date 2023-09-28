import logging
import helics as h
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple
from oedisi.types.data_types import (
    AdmittanceMatrix,
    AdmittanceSparse,
    CommandList,
    Command,
    EquipmentNodeArray,
    Injection,
    InverterControlList,
    MeasurementArray,
    PowersImaginary,
    PowersReal,
    Topology,
    VoltagesAngle,
    VoltagesImaginary,
    VoltagesMagnitude,
    VoltagesReal
)
import adapter
import lindistflow
from area import area_info

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class StaticConfig(object):
    name: str
    deltat: float
    control_type: lindistflow.ControlType
    pf_flag: bool


class Subscriptions(object):
    voltages_real: VoltagesReal
    voltages_imag: VoltagesImaginary
    topology: Topology


class EchoFederate(object):
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
        self.sub.voltages_real = self.fed.register_subscription(
            self.inputs["voltages_real"], "")
        self.sub.voltages_imag = self.fed.register_subscription(
            self.inputs["voltages_imag"], "")

    def register_publication(self) -> None:
        self.pub_commands = self.fed.register_publication(
            "change_commands", h.HELICS_DATA_TYPE_STRING, ""
        )

    def run(self) -> None:
        logger.info(f"Federate connected: {datetime.now()}")
        self.fed.enter_executing_mode()
        granted_time = h.helicsFederateRequestTime(
            self.fed, h.HELICS_TIME_MAXTIME)

        while granted_time < h.HELICS_TIME_MAXTIME:

            if not self.sub.topology.is_updated():
                granted_time = h.helicsFederateRequestTime(
                    self.fed, h.HELICS_TIME_MAXTIME
                )
                continue

            topology = Topology.parse_obj(self.sub.topology.json)
            [branch_info, bus_info] = adapter.extract_info(topology)
            with open('branch_info.json', 'w', encoding='UTF-8') as f:
                f.write(json.dumps(branch_info))

            with open('bus_info.json', 'w', encoding='UTF-8') as f:
                f.write(json.dumps(bus_info))

            slack = topology.slack_bus[0]
            [slack_bus, phase] = slack.split('.')

            area_branch, area_bus = area_info(
                branch_info, bus_info, slack_bus)

            real = VoltagesReal.parse_obj(self.sub.voltages_real.json)
            logger.info(real)

            imag = VoltagesImaginary.parse_obj(self.sub.voltages_imag.json)
            logger.info(imag)

            voltages, power_flow, control, converter = lindistflow.optimal_power_flow(
                area_branch, area_bus, slack_bus, self.static.control_type, self.static.pf_flag)

            commands = []
            for key, val in control.items():
                if key in bus_info:
                    bus = bus_info[key]
                    if 'eqid' in bus_info[key]:
                        eqid = bus['eqid']
                        if 'PVSystem' in eqid:
                            logger.info(key)
                            setpoint = lindistflow.ignore_phase(val)
                            logger.info(setpoint)

                            if self.static.control_type == lindistflow.ControlType.WATT:
                                value = setpoint/1000.0
                                commands.append(
                                    Command(obj_name=eqid, obj_property='kVA', val=value))
                            elif self.static.control_type == lindistflow.ControlType.VAR:
                                value = setpoint/1000.0
                                commands.append(
                                    Command(obj_name=eqid, obj_property='kVA', val=value))
                            elif self.static.control_type == lindistflow.ControlType.WATT_VAR:
                                value = setpoint/1000.0
                                commands.append(
                                    Command(obj_name=eqid, obj_property='kVA', val=value))

            logger.info(commands)
            self.pub_commands.publish(
                CommandList(__root__=commands).json()
            )
        self.stop()

    def stop(self) -> None:
        h.helicsFederateDisconnect(self.fed)
        h.helicsFederateFree(self.fed)
        h.helicsCloseLibrary()
        logger.info(f"Federate disconnected: {datetime.now()}")


if __name__ == "__main__":
    fed = EchoFederate()
    fed.run()
