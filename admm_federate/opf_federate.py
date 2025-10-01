import copy
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
    CommandList,
    Command
)
import adapter
import lindistflow
from dataclasses import asdict
from area import area_info, check_network_radiality
import xarray as xr
from pprint import pprint
import numpy as np
import time

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
    max_itr: int
    control_type: str
    relaxed: bool
    switches: list[str]
    source: str
    config: lindistflow.ADMMConfig


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
    area_branch: adapter.BranchInfo = None
    area_bus: adapter.BusInfo = None
    area_v: VoltagesMagnitude
    area_p: PowersReal
    area_q: PowersImaginary

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
        self.static.max_itr = config["max_itr"]
        self.static.deltat = 1
        self.static.control_type = config["control_type"]
        self.static.relaxed = config["relaxed"]
        self.static.switches = config["switches"]
        self.static.source = config["source"]
        self.static.config = lindistflow.ADMMConfig()
        self.static.config.relaxed = config["relaxed"]
        self.static.config.rho_vup = config["rho_vup"]
        self.static.config.rho_sup = config["rho_sup"]
        self.static.config.rho_vdn = config["rho_vdn"]
        self.static.config.rho_sdn = config["rho_sdn"]

    def initilize(self) -> None:
        self.info = h.helicsCreateFederateInfo()
        self.info.core_name = self.static.name
        self.info.core_type = h.HELICS_CORE_TYPE_ZMQ
        self.info.core_init = "--federates=1"

        # h.helicsFederateInfoSetTimeProperty(self.info, h.helics_property_time_delta, self.static.deltat)

        self.fed = h.helicsCreateValueFederate(self.static.name, self.info)
        # h.helicsFederateSetFlagOption(self.fed, h.helics_flag_slow_responding, True)
        h.helicsFederateSetTimeProperty(
            self.fed, h.HELICS_PROPERTY_TIME_PERIOD, 1)

        # h.helicsFederateSetTimeProperty(self.fed, h.HELICS_PROPERTY_TIME_OFFSET, 0.1)
        # h.helicsFederateSetFlagOption(self.fed, h.HELICS_FLAG_UNINTERRUPTIBLE, True)

    def register_subscription(self) -> None:
        self.sub.topology = self.fed.register_subscription(
            self.inputs["topology"], "")
        self.sub.injections = self.fed.register_subscription(
            self.inputs["injections"], "")
        self.sub.powers_imag = self.fed.register_subscription(
            self.inputs["power_imag"], "")
        self.sub.powers_real = self.fed.register_subscription(
            self.inputs["power_real"], "")
        self.sub.voltages_imag = self.fed.register_subscription(
            self.inputs["voltage_imag"], "")
        self.sub.voltages_real = self.fed.register_subscription(
            self.inputs["voltage_real"], "")
        self.sub.area_v = self.fed.register_subscription(
            self.inputs["sub_v"], "")
        self.sub.area_p = self.fed.register_subscription(
            self.inputs["sub_p"], "")
        self.sub.area_q = self.fed.register_subscription(
            self.inputs["sub_q"], "")

    def register_publication(self) -> None:
        self.pub_pv_set = self.fed.register_publication(
            "pub_c", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_solver_stats = self.fed.register_publication(
            "solver_stats", h.HELICS_DATA_TYPE_STRING, ""
        )
#        self.pub_powers_mag = self.fed.register_publication(
#            "power_mag", h.HELICS_DATA_TYPE_STRING, ""
#        )
#        self.pub_powers_angle = self.fed.register_publication(
#            "power_angle", h.HELICS_DATA_TYPE_STRING, ""
#        )
#        self.pub_voltages_mag = self.fed.register_publication(
#            "voltage_mag", h.HELICS_DATA_TYPE_STRING, ""
#        )
#        self.pub_voltages_angle = self.fed.register_publication(
#            "voltage_angle", h.HELICS_DATA_TYPE_STRING, ""
#        )
        self.pub_admm_v = self.fed.register_publication(
            "pub_v", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_admm_p = self.fed.register_publication(
            "pub_p", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_admm_q = self.fed.register_publication(
            "pub_q", h.HELICS_DATA_TYPE_STRING, ""
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

    def first_pub(self):
        logger.info("running first publication for admm")
        area_v = VoltagesMagnitude(ids=[], values=[], time=0)
        area_p = PowersReal(ids=[], equipment_ids=[], values=[], time=0)
        area_q = PowersImaginary(
            ids=[], equipment_ids=[], values=[], time=0)

        self.pub_admm_v.publish(area_v.json())
        self.pub_admm_p.publish(area_p.json())
        self.pub_admm_q.publish(area_q.json())

    def itr_pub(self):
        topology: Topology = Topology.parse_obj(self.sub.topology.json)
        branch_info, bus_info, slack_bus = adapter.extract_info(topology)
        self.static.config.slack = slack_bus

        G = adapter.generate_graph(topology.incidences, slack_bus)
        graph = copy.deepcopy(G)
        graph2 = copy.deepcopy(G)
        boundaries = adapter.area_disconnects(graph)
        boundary = []
        shared = []
        if self.static.source == slack_bus:
            source = (slack_bus, 0)
        for u, v, a in boundaries:
            if a["id"] in self.static.switches and not a["id"] == self.static.source:
                boundary.append((u, v, a))
                shared.append(v)

            if a["id"] == self.static.source:
                boundary.append((u, v, a))
                shared.append(u)
                source = (u, v)

        areas = adapter.disconnect_areas(graph2, boundary)
        areas = adapter.reconnect_area_switches(
            areas, boundary)

        logger.debug(f"shared: {shared}")
        # TODO: check if switchs exist for this area
        if self.area_branch == None:
            ids = [a["id"] for _, _, a in boundary]
            for area in areas:
                area_branch, area_bus = adapter.generate_area_info(
                    area, topology, source[0], ids)
                if area_branch != None and area_bus != None:
                    self.area_branch = area_branch
                    self.area_bus = area_bus
                    self.area_graph = area

        voltages_real = VoltagesReal.parse_obj(self.sub.voltages_real.json)
        voltages_imag = VoltagesImaginary.parse_obj(
            self.sub.voltages_imag.json)
        voltages = measurement_to_xarray(
            voltages_real
        ) + 1j * measurement_to_xarray(voltages_imag)

        voltages_mag, voltages_ang = xarray_to_voltages_pol(voltages)
        t = voltages_real.time
        voltages_mag.time = t
        voltages_ang.time = t

        injections = Injection.parse_obj(self.sub.injections.json)
        bus_info = adapter.extract_injection(
            self.area_bus, injections)

        bus_info = adapter.extract_voltages(
            bus_info, voltages_mag)

        if self.sub.area_q.is_updated():
            p = PowersImaginary.parse_obj(self.sub.area_q.json)
            if p.values:
                logger.debug("Updating Area Reactive Power")
                p = adapter.filter_boundary_power_imag(shared, p)
                p = adapter.replace_boundary_power_imag(self.area_q, p)
                bus_info = adapter.extract_powers_imag(bus_info, p)

        if self.sub.area_p.is_updated():
            p = PowersReal.parse_obj(self.sub.area_p.json)
            if p.values:
                logger.debug("Updating Area Real Power")
                p = adapter.filter_boundary_power_real(shared, p)
                p = adapter.replace_boundary_power_real(self.area_p, p)
                bus_info = adapter.extract_powers_real(bus_info, p)

        if self.sub.area_v.is_updated():
            vmag = VoltagesMagnitude.parse_obj(self.sub.area_v.json)
            if vmag.values:
                logger.debug("Updating Area Voltages")
                vmag = adapter.filter_boundary_voltage(shared, vmag)
                vmag = adapter.replace_boundary_voltage(self.area_v, vmag)
                bus_info = adapter.extract_voltages(
                    bus_info, vmag)

        child_buses = []
        for u, v, a in boundary:
            if a["id"] != self.static.source and a["id"] in self.static.switches:
                child_buses.append(v)

        branch_info, bus_info = adapter.map_secondaries(
            self.area_branch, bus_info)

        child_info = adapter.BusInfo()
        parent = adapter.Bus()
        for k, v in bus_info.buses.items():
            if k == source[0]:
                logger.debug(f"parent bus :{v}")
                parent = v
            if k in child_buses:
                logger.debug(f"child buses {k}:{v}")
                child_info.buses[k] = v

        assert adapter.check_radiality(branch_info, bus_info)

        self.static.config.source_bus = source[0]
        self.static.config.source_line = adapter.get_edge_name(
            self.area_graph, source[0])
        self.static.config.relaxed = self.static.relaxed
        v_mag, pq, control, stats = lindistflow.solve(
            branch_info, bus_info, child_info, self.static.config
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

        power_real = PowersReal(
            ids=list(p.keys()), values=list(p.values()), equipment_ids=list(p.keys()), time=t)
        # power_real = adapter.pack_powers_real(injections.power_real, p, t)
        self.area_p = adapter.filter_boundary_power_real(shared, power_real)

        logger.debug(power_real)
        logger.debug(self.area_p)

        stats["sdn"] = 0
        for k, v in child_info.buses.items():
            for phase in v.phases:
                real = p[f"{k}.{phase}"]
                imag = q[f"{k}.{phase}"]
                stats["sdn"] += real**2 + imag**2

        kvup = 0
        for k, v in v_mag.items():
            bus, phase = k.split(".", 1)
            if bus == source[0]:
                kvup += v/1000
        stats["vup"] = abs(bus_info.buses[source[0]].kv - kvup/3)

        solver_stats = MeasurementArray(
            ids=list(stats.keys()),
            values=list(stats.values()),
            time=t,
            units="s",
        )

        # self.pub_voltages_mag.publish(v_mag.json())
        # self.pub_voltages_angle.publish(voltages_ang.json())
        # self.pub_powers_mag.publish(power_mag.json())
        # self.pub_powers_angle.publish(power_ang.json())

        v_mag = adapter.pack_voltages(v_mag, bus_info, t)
        self.area_v = adapter.filter_boundary_voltage(shared, v_mag)

        power_real = PowersReal(
            ids=list(p.keys()), values=list(p.values()), equipment_ids=list(p.keys()), time=t)
        # power_real = adapter.pack_powers_real(injections.power_real, p, t)
        self.area_p = adapter.filter_boundary_power_real(shared, power_real)
        logger.debug(p)

        power_imag = PowersImaginary(
            ids=list(q.keys()), values=list(q.values()), equipment_ids=list(p.keys()), time=t)
        # power_imag = adapter.pack_powers_imag(topology.injections.power_imaginary, q, t)
        self.area_q = adapter.filter_boundary_power_imag(shared, power_imag)

        self.pub_admm_p.publish(self.area_p.json())
        self.pub_admm_q.publish(self.area_q.json())
        self.pub_admm_v.publish(self.area_v.json())
        logger.debug(solver_stats)
        logger.debug(commands)
        self.pub_solver_stats.publish(solver_stats.json())
        self.pub_pv_set.publish(json.dumps(commands))

    def run(self) -> None:
        logger.info(f"Federate connected: {datetime.now()}")
        itr_need = h.helics_iteration_request_iterate_if_needed
        itr_stop = h.helics_iteration_request_no_iteration
        h.helicsFederateEnterExecutingMode(self.fed)
        logger.info(f"Federate executing: {datetime.now()}")

        # setting up time properties
        update_interval = int(h.helicsFederateGetTimeProperty(
            self.fed, h.HELICS_PROPERTY_TIME_PERIOD))

        granted_time = 0
        logger.debug("Step 0: Starting Time/Iter loop")
        while granted_time < h.HELICS_TIME_MAXTIME:
            request_time = granted_time + update_interval
            logger.debug("Step 1: published initial values for iteration")
            itr_flag = itr_need
            self.first_pub()
            itr = 0
            while True:
                logger.debug(f"Step 2: Requesting time {request_time}")
                granted_time, itr_status = h.helicsFederateRequestTimeIterative(
                    self.fed, request_time, itr_flag)
                logger.info(f"\tgranted time = {granted_time}")
                logger.info(f"\titr status = {itr_status}")

                logger.debug("Step 3: checking if next step")
                if itr_status == h.helics_iteration_result_next_step:
                    logger.debug(f"\titr next: {itr}")
                    itr_flag = itr_stop
                    break

                itr += 1
                logger.debug("Step 4: update iteration")
                logger.info(f"\titr: {itr}")

                logger.debug("Step 5: checking max itr count")
                if itr >= self.static.max_itr:
                    logger.debug("\t reached max itr")
                    itr_flag = itr_stop
                    continue

                logger.debug("Step 6: run solution solution")
                self.itr_pub()
                itr_flag = itr_need

        self.stop()

    def stop(self) -> None:
        h.helicsFederateDisconnect(self.fed)
        h.helicsFederateFree(self.fed)
        h.helicsCloseLibrary()
        logger.info(f"Federate disconnected: {datetime.now()}")


if __name__ == "__main__":
    fed = OPFFederate()
    fed.run()
