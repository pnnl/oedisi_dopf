import copy
import logging
import helics as h
import json
from pathlib import Path
from datetime import datetime
from oedisi.types.common import BrokerConfig
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
import networkx as nx
import math

from pydantic import BaseModel
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


class ComponentParameters(BaseModel):
    t_steps: int
    max_itr: int
    control_type: str
    switches: list[str]
    source_bus: str
    source_line: str
    relaxed: bool
    rho_sup: float
    rho_vup: float
    rho_sdn: float
    rho_vdn: float
    vup_tol: float
    sdn_tol: float


class StaticConfig(object):
    name: str
    vup_tol: float
    sdn_tol: float
    deltat: float
    max_itr: int
    t_steps: int
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
    converged: bool = False
    parent_bus: str = ""
    parent_line: str = ""
    child_buses: [str] = []
    shared_buses: [str] = []
    switch_buses: [str] = []
    shared_lines: {str, str} = {}
    area_graph: nx.Graph = None
    area_branch: adapter.BranchInfo = None
    area_bus: adapter.BusInfo = None
    parent_info: adapter.BusInfo = None
    child_info: adapter.BusInfo = None
    area_v: VoltagesMagnitude
    area_p: PowersReal
    area_q: PowersImaginary

    def __init__(self, broker_config) -> None:
        self.sub = Subscriptions()
        self.load_static_inputs()
        self.load_input_mapping()
        self.initilize(broker_config)
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
        self.static.vup_tol = config["vup_tol"]
        self.static.sdn_tol = config["sdn_tol"]
        self.static.deltat = config["deltat"]
        self.static.max_itr = config["max_itr"]
        self.static.t_steps = config["number_of_timesteps"]
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

    def initilize(self, broker_config) -> None:
        self.info = h.helicsCreateFederateInfo()
        self.info.core_name = self.static.name
        self.info.core_type = h.HELICS_CORE_TYPE_ZMQ
        self.info.core_init = "--federates=1"

        # h.helicsFederateInfoSetTimeProperty(self.info, h.helics_property_time_delta, self.static.deltat)

        h.helicsFederateInfoSetBroker(self.info, broker_config.broker_ip)
        h.helicsFederateInfoSetBrokerPort(self.info, broker_config.broker_port)

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

    def bus_to_branch_power(self, buses: dict) -> dict:
        branches = {}
        for k, v in buses.items():
            bus, phase = k.split(".", 1)
            if bus in self.shared_lines.keys():
                name = f"{self.shared_lines[bus]}.{phase}"
                branches[name] = v
        return branches

    def init_area(self):
        topology: Topology = Topology.parse_obj(self.sub.topology.json)
        branch_info, bus_info, slack_bus = adapter.extract_info(topology)
        self.static.config.slack = slack_bus

        G = adapter.generate_graph(topology.incidences, slack_bus)
        graph = copy.deepcopy(G)
        graph2 = copy.deepcopy(G)
        boundaries = adapter.area_disconnects(graph)

        upstream = []
        if self.static.source == slack_bus:
            self.parent_bus = slack_bus
            self.shared_buses.append(slack_bus)

        boundary = []
        for u, v, a in boundaries:
            if a["id"] in self.static.switches and not a["id"] == self.static.source:
                boundary.append((u, v, a))
                self.child_buses.append(v)
                self.shared_buses.append(v)
                self.switch_buses.append(u)
                self.switch_buses.append(v)
                self.shared_lines[u] = f"{u}_{v}"
                self.shared_lines[v] = f"{v}_{u}"

            if a["id"] == self.static.source:
                boundary.append((u, v, a))
                self.parent_bus = u
                self.shared_buses.append(u)
                self.switch_buses.append(u)
                self.switch_buses.append(v)
                self.parent_line = f"{u}_{v}"
                self.shared_lines[u] = f"{u}_{v}"
                self.shared_lines[v] = f"{v}_{u}"

        areas = adapter.disconnect_areas(graph2, boundary)
        areas = adapter.reconnect_area_switches(
            areas, boundary)

        ids = [a["id"] for _, _, a in boundary]
        for area in areas:
            area_branch, area_bus = adapter.generate_area_info(
                area, topology, self.parent_bus, ids)
            if area_branch is not None and area_bus is not None:
                for u, v in area.edges(self.parent_bus):
                    if self.parent_line == "":
                        self.parent_line = f"{u}_{v}"
                self.area_branch = area_branch
                self.area_bus = area_bus
                self.area_graph = area

        self.child_info = adapter.BusInfo()
        for k in self.child_buses:
            if k in self.area_bus.buses:
                self.child_info.buses[k] = copy.deepcopy(
                    self.area_bus.buses[k])
                self.child_info.buses[k].pv = np.zeros((3, 2)).tolist()

        self.parent_info = adapter.BusInfo()
        self.parent_info.buses[self.parent_bus] = copy.deepcopy(
            self.area_bus.buses[self.parent_bus])
        self.parent_info.buses[self.parent_bus].pv = np.zeros((3, 2)).tolist()

        logger.debug("Parent Info")
        logger.debug(f"\tbus = {self.parent_bus}")
        logger.debug(f"\tline = {self.parent_line}")
        logger.debug("Child Info")
        logger.debug(f"\tbuses = {self.child_buses}")
        logger.debug("Shared Info")
        logger.debug(f"\tbuses = {self.shared_buses}")

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
        self.area_v = VoltagesMagnitude(ids=[], values=[], time=0)
        self.area_p = PowersReal(ids=[], equipment_ids=[], values=[], time=0)
        self.area_q = PowersImaginary(
            ids=[], equipment_ids=[], values=[], time=0)

        self.pub_admm_v.publish(self.area_v.json())
        self.pub_admm_p.publish(self.area_p.json())
        self.pub_admm_q.publish(self.area_q.json())

    def itr_pub(self):
        if self.area_graph is None:
            self.init_area()

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
            copy.deepcopy(self.area_bus), injections)

        bus_info = adapter.extract_voltages(
            bus_info, voltages_mag)

        branch_info, bus_info = adapter.map_secondaries(
            copy.deepcopy(self.area_branch), bus_info)

        with open("bus_info.json", "w") as outfile:
            outfile.write(json.dumps(asdict(bus_info)))

        with open("branch_info.json", "w") as outfile:
            outfile.write(json.dumps(asdict(branch_info)))

        p_err = 0
        q_err = 0
        v_err = 0
        p = PowersReal.parse_obj(self.sub.area_p.json)
        if p.values and self.area_p.values:
            logger.debug("Updating Area Active Power")
            p = adapter.filter_boundary_power_real(self.shared_buses, p)
            p, p_err = adapter.update_boundary_power_real(p, self.static.name)
            self.parent_info = adapter.extract_powers_real(
                self.parent_info, p, True)
            self.child_info = adapter.extract_powers_real(
                self.child_info, p, True)

        q = PowersImaginary.parse_obj(self.sub.area_q.json)
        if p.values and self.area_q.values:
            logger.debug("Updating Area Reactive Power")
            q = adapter.filter_boundary_power_imag(self.shared_buses, q)
            q, q_err = adapter.update_boundary_power_imag(q, self.static.name)
            self.parent_info = adapter.extract_powers_imag(
                self.parent_info, q, True)
            self.child_info = adapter.extract_powers_imag(
                self.child_info, q, True)

        vmag = VoltagesMagnitude.parse_obj(self.sub.area_v.json)
        if vmag.values and self.area_v.values:
            logger.debug("Updating Area Voltages")
            vmag = adapter.filter_boundary_voltage(self.switch_buses, vmag)
            vmag, v_err = adapter.update_boundary_voltage(self.area_v, vmag)
            self.parent_info = adapter.extract_voltages(
                self.parent_info, vmag)
            self.child_info = adapter.extract_voltages(
                self.child_info, vmag)

        with open("bus_info_updated.json", "w") as outfile:
            outfile.write(json.dumps(asdict(bus_info)))

        with open("branch_info_updated.json", "w") as outfile:
            outfile.write(json.dumps(asdict(branch_info)))

        assert adapter.check_radiality(branch_info, bus_info)

        self.static.config.source_bus = self.parent_bus
        self.static.config.source_line = adapter.get_edge_name(
            self.area_graph, self.parent_bus)
        self.static.config.relaxed = self.static.relaxed
        v_mag, branch_pq, aux_pq, control, stats = lindistflow.solve(
            branch_info, bus_info, self.child_info, self.parent_info, self.static.config
        )
        real_setpts = self.get_set_points(control, bus_info)

        bp = {k: p[0] for k, p in branch_pq.items()}
        bq = {k: p[1] for k, p in branch_pq.items()}

        p = {k: p[0] for k, p in aux_pq.items()}
        p = self.bus_to_branch_power(p)

        q = {k: p[1] for k, p in aux_pq.items()}
        q = self.bus_to_branch_power(q)

        # replace branch flows with aux loads if they exist
        for k, v in bp.items():
            if k in p:
                bp[k] = p[k]
                bq[k] = q[k]

        # CAPTURE STATS FOR PUB
        power_real = PowersReal(
            ids=list(bp.keys()), values=list(bp.values()), equipment_ids=list(bp.keys()), time=t)
        self.area_p = copy.deepcopy(adapter.filter_line_power_real(
            self.shared_buses, power_real, self.static.name))

        power_imag = PowersImaginary(
            ids=list(bq.keys()), values=list(bq.values()), equipment_ids=list(bq.keys()), time=t)
        self.area_q = copy.deepcopy(adapter.filter_line_power_imag(
            self.shared_buses, power_imag, self.static.name))

        vmag = adapter.pack_voltages(v_mag, bus_info, t)
        self.area_v = copy.deepcopy(
            adapter.filter_boundary_voltage(self.switch_buses, vmag))

        self.pub_admm_p.publish(self.area_p.json())
        self.pub_admm_q.publish(self.area_q.json())
        self.pub_admm_v.publish(self.area_v.json())

        # SET COMMANDS FOR PUB
        commands = []
        for eq, val in real_setpts.items():
            if abs(val) < 1e-6:
                continue

            commands.append((eq, val.real, val.imag))
        self.pub_pv_set.publish(json.dumps(commands))

        # CAPTURE STATS FOR PUB
        stats["admm_iteration"] = self.itr
        stats["vup"] = v_err
        stats["sdn"] = math.sqrt(p_err**2 + q_err**2)
        logger.debug(f"Errors : {stats['vup']}, {stats['sdn']}")

        if all([v_err != 0, p_err != 0, q_err != 0]):
            v_settled = stats["vup"] <= self.static.vup_tol
            p_settled = stats["sdn"] <= self.static.sdn_tol
            if v_settled and p_settled:
                logger.debug("Converged")
                self.converged = True

        solver_stats = MeasurementArray(
            ids=list(stats.keys()),
            values=list(stats.values()),
            time=t,
            units="s",
        )
        self.pub_solver_stats.publish(solver_stats.json())

        # self.pub_voltages_mag.publish(v_mag.json())
        # self.pub_voltages_angle.publish(voltages_ang.json())
        # self.pub_powers_mag.publish(power_mag.json())
        # self.pub_powers_angle.publish(power_ang.json())

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
        while granted_time <= self.static.t_steps:
            request_time = granted_time + update_interval
            logger.debug("Step 1: published initial values for iteration")
            itr_flag = itr_need
            self.first_pub()
            self.converged = False
            self.itr = 0
            while True:
                logger.debug(f"Step 2: Requesting time {request_time}")
                granted_time, itr_status = h.helicsFederateRequestTimeIterative(
                    self.fed, request_time, itr_flag)
                logger.info(f"\tgranted time = {granted_time}")
                logger.info(f"\titr status = {itr_status}")

                logger.debug("Step 3: checking if next step")

                if itr_status == h.helics_iteration_result_next_step:
                    logger.debug(f"\titr next: {self.itr}")
                    itr_flag = itr_stop
                    break

                if self.converged:
                    logger.debug(f"\tconverged: {self.itr}")
                    itr_flag = itr_stop
                    break

                self.itr += 1
                logger.debug("Step 4: update iteration")
                logger.info(f"\titr: {self.itr}")

                logger.debug("Step 5: checking max itr count")
                if self.itr >= self.static.max_itr:
                    logger.debug("\t reached max itr")
                    itr_flag = itr_stop
                    break

                logger.debug("Step 6: run solution solution")
                self.itr_pub()
                itr_flag = itr_need

        logger.debug("FINISHED")
        self.stop()

    def stop(self) -> None:
        h.helicsFederateDisconnect(self.fed)
        h.helicsFederateFree(self.fed)
        h.helicsCloseLibrary()
        logger.info(f"Federate disconnected: {datetime.now()}")


def run_simulator(broker_config: BrokerConfig) -> None:
    #    schema = ComponentParameters.schema_json(indent=2)
    #    with open("./admm_federate/admm_schema.json", "w") as f:
    #        f.write(schema)
    #
    sfed = OPFFederate(broker_config)
    sfed.run()


if __name__ == "__main__":
    run_simulator(BrokerConfig(broker_ip="127.0.0.1"))
