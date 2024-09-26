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
    PowersMagnitude,
    PowersAngle,
    Injection,
    Topology,
    VoltagesReal,
    VoltagesImaginary,
    VoltagesMagnitude,
    VoltagesAngle,
    MeasurementArray,
    EquipmentNodeArray
)
import adapter
import lindistflow
from dataclasses import asdict
from area import area_info, check_network_radiality
import xarray as xr
from pprint import pprint
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


def eqarray_to_xarray(eq: EquipmentNodeArray):
    return xr.DataArray(
        eq.values,
        dims=("eqnode",),
        coords={
            "equipment_ids": ("eqnode", eq.equipment_ids),
            "ids": ("eqnode", eq.ids),
        },
    )


def measurement_to_xarray(eq: MeasurementArray):
    return xr.DataArray(eq.values, coords={"ids": eq.ids})


def xarray_to_dict(data):
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
    mag = PowersMagnitude(**xarray_to_dict(np.abs(data)), **kwargs)
    ang = PowersAngle(
        **xarray_to_dict(np.arctan2(data.imag, data.real)), **kwargs)
    return mag, ang


def xarray_to_voltages_pol(data, **kwargs):
    """Conveniently turn xarray into PowersReal and PowersImaginary."""
    mag = VoltagesMagnitude(**xarray_to_dict(np.abs(data)), **kwargs)
    ang = VoltagesAngle(
        **xarray_to_dict(np.arctan2(data.imag, data.real)), **kwargs)
    return mag, ang


class StaticConfig(object):
    name: str
    deltat: float
    control_type: str
    relaxed: bool


class Subscriptions(object):
    powers_real: PowersReal
    powers_imag: PowersImaginary
    voltages_real: VoltagesReal
    voltages_imag: VoltagesImaginary
    available_power: PowersReal
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
        self.static.control_type = config["control_type"]
        self.static.relaxed = config["relaxed"]

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
        self.sub.powers_imag = self.fed.register_subscription(
            self.inputs["powers_imag"], "")
        self.sub.powers_real = self.fed.register_subscription(
            self.inputs["powers_real"], "")
        self.sub.voltages_imag = self.fed.register_subscription(
            self.inputs["voltages_imag"], "")
        self.sub.voltages_real = self.fed.register_subscription(
            self.inputs["voltages_real"], "")
        self.sub.injections = self.fed.register_subscription(
            self.inputs["injections"], "")
        self.sub.available_power = self.fed.register_subscription(
            self.inputs["available_power"], "")

    def register_publication(self) -> None:
        self.pub_pv_set = self.fed.register_publication(
            "pv_set", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_powers_real = self.fed.register_publication(
            "powers_real", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_powers_imag = self.fed.register_publication(
            "powers_imag", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_voltages_real = self.fed.register_publication(
            "voltages_real", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_voltages_imag = self.fed.register_publication(
            "voltages_imag", h.HELICS_DATA_TYPE_STRING, ""
        )

    def get_set_points(self, control: dict, bus_info: adapter.BusInfo, conversion: float) -> dict[complex]:
        setpoints = {}
        for key, val in control.items():
            if key in bus_info.buses:
                bus = bus_info.buses[key]
                for tag in set(bus.tags):
                    if "PVSystem" in tag:
                        p = max([p for p in val["Pdg_gen"].values()])
                        q = max([q for q in val["Qdg_gen"].values()])
                        setpoints[key] = (p + 1j*q)*conversion
        return setpoints

    def run(self) -> None:
        logger.info(f"Federate connected: {datetime.now()}")
        self.fed.enter_executing_mode()
        granted_time = h.helicsFederateRequestTime(
            self.fed, h.HELICS_TIME_MAXTIME)

        while granted_time < h.HELICS_TIME_MAXTIME:

            if not self.sub.injections.is_updated():
                granted_time = h.helicsFederateRequestTime(
                    self.fed, h.HELICS_TIME_MAXTIME
                )
                continue

            topology: Topology = Topology.parse_obj(self.sub.topology.json)
            branch_info, bus_info, slack_bus = adapter.extract_info(topology)

            injections = Injection.parse_obj(self.sub.injections.json)
            bus_info = adapter.extract_injection(bus_info, injections)

            powers_real = PowersReal.parse_obj(
                self.sub.powers_real.json)
            powers_imag = PowersImaginary.parse_obj(
                self.sub.powers_imag.json)
            powers = eqarray_to_xarray(
                powers_real) + 1j*eqarray_to_xarray(powers_imag)

            voltages_real = VoltagesReal.parse_obj(
                self.sub.voltages_real.json)
            voltages_imag = VoltagesImaginary.parse_obj(
                self.sub.voltages_imag.json)
            voltages = measurement_to_xarray(
                voltages_real) + 1j*measurement_to_xarray(voltages_imag)

            voltages_mag, voltages_ang = xarray_to_voltages_pol(voltages)
            bus_info = adapter.extract_voltages(
                bus_info, voltages_mag)

            time = voltages_real.time
            logger.debug(f"Timestep: {time}")

            branch_info, bus_info = adapter.map_secondaries(
                branch_info, bus_info)

            with open("bus_info.json", "w") as outfile:
                outfile.write(json.dumps(asdict(bus_info)))

            with open("branch_info.json", "w") as outfile:
                outfile.write(json.dumps(asdict(branch_info)))

            assert (adapter.check_radiality(branch_info, bus_info))

            p_inj = MeasurementArray.parse_obj(
                self.sub.available_power.json)

            mode = self.static.control_type
            relaxed = self.static.relaxed
            v_mag, p_real, control, conversion = lindistflow.solve(
                branch_info, bus_info, slack_bus, relaxed)
            real_setpts = self.get_set_points(control, bus_info, conversion)

            # get the control commands for the feeder federate
            commands = []
            for eq, val in real_setpts.items():
                if abs(val) < 1e-6:
                    continue

                print(eq, val)

                commands.append((eq, val.real, val.imag))

            if commands:
                self.pub_pv_set.publish(
                    json.dumps(commands)
                )

            v_mag = adapter.pack_voltages(v_mag, bus_info, time)
            voltages = measurement_to_xarray(
                v_mag)*np.exp(1j*measurement_to_xarray(voltages_ang))
            voltages_real, voltages_imag = xarray_to_voltages_cart(voltages)
            voltages_real.time = time
            voltages_imag.time = time
            self.pub_voltages_real.publish(
                voltages_real.json()
            )
            self.pub_voltages_imag.publish(
                voltages_imag.json()
            )

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
    fed = OPFFederate()
    fed.run()
