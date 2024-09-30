import area
import pv_detect
import adapter
import logging
import helics as h
import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from oedisi.types.data_types import (
    Topology,
    VoltagesMagnitude,
    VoltagesAngle,
    VoltagesReal,
    VoltagesImaginary,
    PowersMagnitude,
    PowersAngle,
    PowersReal,
    PowersImaginary,
    Injection,
    VoltagesMagnitude,
    MeasurementArray,
    EquipmentNodeArray
)
import xarray as xr
import numpy as np
from pprint import pprint

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


def eqarray_to_xarray(eq: EquipmentNodeArray):
    return xr.DataArray(
        eq.values,
        dims=('ids',),
        coords={
            "ids": eq.ids,
            "equipment_ids": ("ids", eq.equipment_ids),
        },
    )


def measurement_to_xarray(eq: MeasurementArray):
    return xr.DataArray(eq.values, coords={"ids": eq.ids})


def xarray_to_dict(data):
    """Convert xarray to dict with values and ids for JSON serialization."""
    coords = {key: list(data.coords[key].data) for key in data.coords.keys()}
    return {"values": list(data.data), **coords}


def xarray_to_eqarray(data):
    """Convert xarray to dict with values and ids for JSON serialization."""
    coords = {key: list(data.coords[key].data) for key in data.coords.keys()}
    return {"values": list(data.data), **coords}


def xarray_to_powers_cart(data, **kwargs):
    """Conveniently turn xarray into PowersReal and PowersImaginary."""
    real = PowersReal(**xarray_to_dict(data.real), **kwargs)
    imag = PowersImaginary(**xarray_to_dict(data.imag), **kwargs)
    return real, imag


def xarray_to_voltages_cart(data, **kwargs):
    """Conveniently turn xarray into PowersReal and PowersImaginary."""
    real = VoltagesReal(**xarray_to_dict(data.real), **kwargs)
    imag = VoltagesImaginary(**xarray_to_dict(data.imag), **kwargs)
    return real, imag


def xarray_to_powers_pol(data, **kwargs):
    """Conveniently turn xarray into PowersReal and PowersImaginary."""
    mag = PowersMagnitude(**xarray_to_eqarray(np.abs(data)), **kwargs)
    ang = PowersAngle(
        **xarray_to_eqarray(np.arctan2(data.imag, data.real)), **kwargs)
    return mag, ang


def xarray_to_voltages_pol(data, **kwargs):
    """Conveniently turn xarray into PowersReal and PowersImaginary."""
    mag = VoltagesMagnitude(**xarray_to_dict(np.abs(data)), **kwargs)
    ang = VoltagesAngle(
        **xarray_to_dict(np.arctan2(data.imag, data.real)), **kwargs)
    return mag, ang


class StaticConfig(object):
    name: str
    tol: float
    v_sigma: float
    l_sigma: float
    i_sigma: float
    deltat: float


class Subscriptions(object):
    voltages_mag: VoltagesMagnitude
    voltages_angle: VoltagesAngle
    powers_real: PowersReal
    powers_imag: PowersImaginary
    topology: Topology
    injections: Injection


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
        self.sub.injections = self.fed.register_subscription(
            self.inputs["injections"], "")
        self.sub.voltages_mag = self.fed.register_subscription(
            self.inputs["voltage_mag"], "")
        self.sub.voltages_angle = self.fed.register_subscription(
            self.inputs["voltage_angle"], "")
        self.sub.powers_real = self.fed.register_subscription(
            self.inputs["power_real"], "")
        self.sub.powers_imag = self.fed.register_subscription(
            self.inputs["power_imag"], "")

    def register_publication(self) -> None:
        self.pub_voltages_mag = self.fed.register_publication(
            "voltage_mag", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_voltages_angle = self.fed.register_publication(
            "voltage_angle", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_powers_mag = self.fed.register_publication(
            "power_mag", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_powers_angle = self.fed.register_publication(
            "power_angle", h.HELICS_DATA_TYPE_STRING, ""
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

            topology = Topology.parse_obj(self.sub.topology.json)
            branch_info, bus_info, slack_bus = adapter.extract_info(topology)

            injections = Injection.parse_obj(self.sub.injections.json)
            bus_info = adapter.extract_base_injection(
                bus_info, injections)

            voltages_mag = VoltagesMagnitude.parse_obj(
                self.sub.voltages_mag.json)

            voltages_angle = VoltagesAngle.parse_obj(
                self.sub.voltages_angle.json)
            bus_info = adapter.extract_voltages(
                bus_info, voltages_mag)

            time = voltages_mag.time
            logger.debug(f"Timestep: {time}")

            # get the available power in real time
            powers_real = PowersReal.parse_obj(self.sub.powers_real.json)
            bus_info = adapter.extract_powers_real(bus_info, powers_real)

            powers_imag = PowersImaginary.parse_obj(self.sub.powers_imag.json)
            bus_info = adapter.extract_powers_imag(bus_info, powers_imag)

            branch_info, bus_info = adapter.map_secondaries(
                branch_info, bus_info)

            with open("bus_info.json", "w") as outfile:
                outfile.write(json.dumps(asdict(bus_info)))

            with open("branch_info.json", "w") as outfile:
                outfile.write(json.dumps(asdict(branch_info)))

            assert (adapter.check_radiality(branch_info, bus_info))
            bus_data = {k: asdict(v) for k, v in bus_info.buses.items()}
            branch_data = {k: asdict(v)
                           for k, v in branch_info.branches.items()}
            base_s = 1e6
            p, q = pv_detect.run_dsse(
                bus_data, branch_data, self.static.__dict__, slack_bus, base_s)

            power_real = adapter.pack_powers_real(powers_real, p, time)
            power_imag = adapter.pack_powers_imag(powers_imag, p, time)

            power = eqarray_to_xarray(
                power_real) + 1j*eqarray_to_xarray(power_imag)

            power_mag, power_ang = xarray_to_powers_pol(power)
            power_mag.time = time
            power_ang.time = time

            self.pub_voltages_mag.publish(
                voltages_mag.json()
            )
            self.pub_voltages_angle.publish(
                voltages_angle.json()
            )
            self.pub_powers_mag.publish(
                power_mag.json()
            )
            self.pub_powers_angle.publish(
                power_ang.json()
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
