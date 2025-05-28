import logging
import helics as h
import json
from pathlib import Path
from datetime import datetime
from oedisi.types.data_types import (
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
    EquipmentNodeArray,
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
        dims=("ids",),
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
    switches: list[str]
    source: str


class Subscriptions(object):
    powers_real: PowersReal
    powers_imag: PowersImaginary
    voltages_real: VoltagesReal
    voltages_imag: VoltagesImaginary
    available_power: PowersReal
    injections: Injection
    topology: Topology
    pv_forecast: list
    area_v: VoltagesMagnitude
    area_p: PowersReal
    area_q: PowersImaginary


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
        self.static.switches = config["switches"]
        self.static.source = config["source"]

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
            self.inputs["power_imag"], "")
        self.sub.powers_real = self.fed.register_subscription(
            self.inputs["power_real"], "")
        self.sub.voltages_imag = self.fed.register_subscription(
            self.inputs["voltage_imag"], "")
        self.sub.voltages_real = self.fed.register_subscription(
            self.inputs["voltage_real"], "")
        self.sub.injections = self.fed.register_subscription(
            self.inputs["injections"], "")
        self.sub.available_power = self.fed.register_subscription(
            self.inputs["available_power"], "")
        self.sub.area_v = self.fed.register_subscription(
            self.inputs["area_v"], "")
        self.sub.area_v = self.fed.register_subscription(
            self.inputs["area_p"], "")
        self.sub.area_v = self.fed.register_subscription(
            self.inputs["area_q"], "")

    def register_publication(self) -> None:
        self.pub_pv_set = self.fed.register_publication(
            "pv_set", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_solver_stats = self.fed.register_publication(
            "solver_stats", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_estimated_power = self.fed.register_publication(
            "estimated_power", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_powers_mag = self.fed.register_publication(
            "power_mag", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_powers_angle = self.fed.register_publication(
            "power_angle", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_voltages_mag = self.fed.register_publication(
            "voltage_mag", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_voltages_angle = self.fed.register_publication(
            "voltage_angle", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_admm_v.append(self.fed.register_publication(
            "area_v", h.HELICS_DATA_TYPE_STRING, "")
        )
        self.pub_admm_p.append(self.fed.register_publication(
            "area_p", h.HELICS_DATA_TYPE_STRING, "")
        )
        self.pub_admm_q.fed.register_publication(
            "area_q", h.HELICS_DATA_TYPE_STRING, ""
        )

    def get_set_points(self, control: dict, bus_info: adapter.BusInfo) -> dict[complex]:
        setpoints = {}
        for key, val in control.items():
            if key in bus_info.buses:
                bus = bus_info.buses[key]
                for tag in set(bus.tags):
                    if "PVSystem" in tag:
                        p = max([p for p in val["Pdg_gen"].values()])
                        q = max([q for q in val["Qdg_gen"].values()])
                        setpoints[tag] = p + 1j * q
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

            powers_real = PowersReal.parse_obj(self.sub.powers_real.json)
            powers_imag = PowersImaginary.parse_obj(self.sub.powers_imag.json)
            powers = eqarray_to_xarray(powers_real) + 1j * eqarray_to_xarray(
                powers_imag
            )

            if self.sub.area_v0 is not None:
                if self.sub.area_v0.is_updated():
                    print(VoltagesMagnitude.parse_obj(self.sub.area_v0.json))
            if self.sub.area_v1 is not None:
                if self.sub.area_v1.is_updated():
                    print(VoltagesMagnitude.parse_obj(self.sub.area_v1.json))
            if self.sub.area_v2 is not None:
                if self.sub.area_v2.is_updated():
                    print(VoltagesMagnitude.parse_obj(self.sub.area_v2.json))
            if self.sub.area_v3 is not None:
                if self.sub.area_v3.is_updated():
                    print(VoltagesMagnitude.parse_obj(self.sub.area_v3.json))
            if self.sub.area_v4 is not None:
                if self.sub.area_v4.is_updated():
                    print(VoltagesMagnitude.parse_obj(self.sub.area_v4.json))

            voltages_real = VoltagesReal.parse_obj(self.sub.voltages_real.json)
            voltages_imag = VoltagesImaginary.parse_obj(
                self.sub.voltages_imag.json)
            voltages = measurement_to_xarray(
                voltages_real
            ) + 1j * measurement_to_xarray(voltages_imag)

            time = voltages_real.time
            logger.debug(f"Timestep: {time}")

            voltages_mag, voltages_ang = xarray_to_voltages_pol(voltages)
            voltages_mag.time = time
            voltages_ang.time = time

            bus_info = adapter.extract_voltages(bus_info, voltages_mag)

            branch_info, bus_info = adapter.map_secondaries(
                branch_info, bus_info)

            with open("bus_info.json", "w") as outfile:
                outfile.write(json.dumps(asdict(bus_info)))

            with open("branch_info.json", "w") as outfile:
                outfile.write(json.dumps(asdict(branch_info)))

            assert adapter.check_radiality(branch_info, bus_info)

            p_inj = MeasurementArray.parse_obj(self.sub.available_power.json)

            relaxed = self.static.relaxed
            v_mag, pq, control, stats = lindistflow.solve(
                branch_info, bus_info, slack_bus, relaxed
            )
            real_setpts = self.get_set_points(control, bus_info)

            p = {k: p[0] for k, p in pq.items()}
            q = {k: p[1] for k, p in pq.items()}

            # get the control commands for the feeder federate
            commands = []
            for eq, val in real_setpts.items():
                if abs(val) < 1e-6:
                    continue

                commands.append((eq, val.real, val.imag))

            if commands:
                self.pub_pv_set.publish(json.dumps(commands))

            v_mag = adapter.pack_voltages(v_mag, bus_info, time)
            power_real = adapter.pack_powers_real(powers_real, p, time)
            power_imag = adapter.pack_powers_imag(powers_real, q, time)

            power = eqarray_to_xarray(power_real) + \
                1j * eqarray_to_xarray(power_imag)

            power_mag, power_ang = xarray_to_powers_pol(power)
            power_mag.time = time
            power_ang.time = time

            solver_stats = MeasurementArray(
                ids=list(stats.keys()),
                values=list(stats.values()),
                time=time,
                units="s",
            )

            est_power = MeasurementArray(
                ids=list(real_setpts.keys()),
                values=list(real_setpts.values()),
                time=time,
                units="W",
            )

            self.pub_voltages_mag.publish(v_mag.json())
            self.pub_voltages_angle.publish(voltages_ang.json())

            self.pub_estimated_power.publish(est_power.json())
            self.pub_solver_stats.publish(solver_stats.json())
            self.pub_powers_mag.publish(power_mag.json())
            self.pub_powers_angle.publish(power_ang.json())

        self.stop()

    def stop(self) -> None:
        h.helicsFederateDisconnect(self.fed)
        h.helicsFederateFree(self.fed)
        h.helicsCloseLibrary()
        logger.info(f"Federate disconnected: {datetime.now()}")


if __name__ == "__main__":
    fed = OPFFederate()
    fed.run()
