import area
import pv_detect
import adapter
import logging
import helics as h
import json
from pathlib import Path
from datetime import datetime
from oedisi.types.data_types import (
    Topology,
    PowersReal,
    PowersImaginary,
    Injection,
    VoltagesMagnitude,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class StaticConfig(object):
    name: str
    tol: float
    v_sigma: float
    l_sigma: float
    i_sigma: float
    deltat: float


class Subscriptions(object):
    voltages_mag: VoltagesMagnitude
    powers_real: PowersReal
    powers_imag: PowersImaginary
    topology: Topology


class EstimatorFederate(object):
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
        self.static.tol = config["tol"]
        self.static.v_sigma = config["v_sigma"]
        self.static.l_sigma = config["l_sigma"]
        self.static.i_sigma = config["i_sigma"]
        self.static.deltat = config["deltat"]

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
            self.inputs["voltages_mag"], "")
        self.sub.powers_real = self.fed.register_subscription(
            self.inputs["powers_real"], "")
        self.sub.powers_imag = self.fed.register_subscription(
            self.inputs["powers_imag"], "")

    def register_publication(self) -> None:
        self.pub_powers_real = self.fed.register_publication(
            "powers_real", h.HELICS_DATA_TYPE_STRING, ""
        )

        self.pub_powers_imag = self.fed.register_publication(
            "powers_imag", h.HELICS_DATA_TYPE_STRING, ""
        )

        self.pub_injections = self.fed.register_publication(
            "injections", h.HELICS_DATA_TYPE_STRING, ""
        )

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
            [branch_info, bus_info] = adapter.extract_info(topology)

            injections: Injection = Injection.parse_obj(topology.injections)

            bus_info = adapter.extract_injection(bus_info, injections)

            slack = topology.slack_bus[0]
            [slack_bus, phase] = slack.split('.')

            voltages_mag = VoltagesMagnitude.parse_obj(
                self.sub.voltages_mag.json)
            bus_info = adapter.extract_voltages(bus_info, voltages_mag)

            time = voltages_mag.time
            logger.debug(time)

            # get the available power in real time
            powers_real: PowersReal
            powers_real = PowersReal.parse_obj(self.sub.powers_real.json)

            powers_imag: PowersImaginary
            powers_imag = PowersImaginary.parse_obj(self.sub.powers_imag.json)
            bus_info = adapter.extract_powers(
                bus_info, powers_real, powers_imag)

            with open("bus_info_oedisi_ieee123.json", "w") as outfile:
                outfile.write(json.dumps(bus_info))

            with open("branch_info_oedisi_ieee123.json", "w") as outfile:
                outfile.write(json.dumps(branch_info))

            assert (area.check_network_radiality(
                bus=bus_info, branch=branch_info))

            base_s = 1e6
            (p, q) = pv_detect.run_dsse(
                bus_info, branch_info, self.static.__dict__, slack_bus, base_s)

            real_inj: PowersReal = injections.power_real
            for val, id, eq in zip(real_inj.values, real_inj.ids, real_inj.equipment_ids):
                if "PVSystem" in eq:
                    if id in p:
                        print(eq, id, val, p[id])

            self.pub_powers_real.publish(
                powers_real.json()
            )
            self.pub_powers_imag.publish(
                powers_imag.json()
            )

        self.stop()

    def stop(self) -> None:
        h.helicsFederateDisconnect(self.fed)
        h.helicsFederateFree(self.fed)
        h.helicsCloseLibrary()
        logger.info(f"Federate disconnected: {datetime.now()}")


if __name__ == "__main__":
    fed = EstimatorFederate()
    fed.run()
