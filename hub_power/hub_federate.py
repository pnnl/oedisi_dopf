import logging
import helics as h
import json
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
from oedisi.types.common import BrokerConfig
from oedisi.types.data_types import (
    PowersReal,
    PowersImaginary,
    MeasurementArray,
    EquipmentNodeArray,
)
import xarray as xr

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


def xarray_to_eqarray(data):
    """Convert xarray to dict with values and ids for JSON serialization."""
    coords = {key: list(data.coords[key].data) for key in data.coords.keys()}
    return {"values": list(data.data), **coords}


def xarray_to_powers_cart(data, **kwargs):
    """Conveniently turn xarray into PowersReal and PowersImaginary."""
    real = PowersReal(**xarray_to_dict(data.real), **kwargs)
    imag = PowersImaginary(**xarray_to_dict(data.imag), **kwargs)
    return real, imag


class ComponentParameters(BaseModel):
    name: str
    max_itr: int
    t_steps: int


class StaticConfig(object):
    name: str
    max_itr: int
    t_steps: int


class Subscriptions(object):
    p0: PowersReal
    p1: PowersReal
    p2: PowersReal
    p3: PowersReal
    p4: PowersReal
    q0: PowersImaginary
    q1: PowersImaginary
    q2: PowersImaginary
    q3: PowersImaginary
    q4: PowersImaginary


class HubFederate(object):
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
        self.static.max_itr = config["max_itr"]
        self.static.t_steps = config["number_of_timesteps"]

    def initilize(self, broker_config) -> None:
        self.info = h.helicsCreateFederateInfo()
        self.info.core_name = self.static.name
        self.info.core_type = h.HELICS_CORE_TYPE_ZMQ
        self.info.core_init = "--federates=1"

        h.helicsFederateInfoSetBroker(self.info, broker_config.broker_ip)
        h.helicsFederateInfoSetBrokerPort(self.info, broker_config.broker_port)

        # h.helicsFederateInfoSetTimeProperty(self.info, h.helics_property_time_delta, 0.01)
        self.fed = h.helicsCreateValueFederate(self.static.name, self.info)
        # h.helicsFederateSetFlagOption(self.fed, h.helics_flag_slow_responding, True)
        h.helicsFederateSetTimeProperty(
            self.fed, h.HELICS_PROPERTY_TIME_PERIOD, 1)

    def register_subscription(self) -> None:
        self.sub.p0 = self.fed.register_subscription(
            self.inputs["sub_p0"], ""
        )
        self.sub.p1 = self.fed.register_subscription(
            self.inputs["sub_p1"], ""
        )
        self.sub.p2 = self.fed.register_subscription(
            self.inputs["sub_p2"], ""
        )
        self.sub.p3 = self.fed.register_subscription(
            self.inputs["sub_p3"], ""
        )
        self.sub.p4 = self.fed.register_subscription(
            self.inputs["sub_p4"], ""
        )
        self.sub.q0 = self.fed.register_subscription(
            self.inputs["sub_q0"], ""
        )
        self.sub.q1 = self.fed.register_subscription(
            self.inputs["sub_q1"], ""
        )
        self.sub.q2 = self.fed.register_subscription(
            self.inputs["sub_q2"], ""
        )
        self.sub.q3 = self.fed.register_subscription(
            self.inputs["sub_q3"], ""
        )
        self.sub.q4 = self.fed.register_subscription(
            self.inputs["sub_q4"], ""
        )

    def register_publication(self) -> None:
        self.pub_area_p = []
        for i in range(6):
            self.pub_area_p.append(self.fed.register_publication(
                f"pub_p{i}", h.HELICS_DATA_TYPE_STRING, "")
            )

        self.pub_area_q = []
        for i in range(6):
            self.pub_area_q.append(self.fed.register_publication(
                f"pub_q{i}", h.HELICS_DATA_TYPE_STRING, "")
            )

    def publish_imag(self):
        all_q = PowersImaginary(ids=[], equipment_ids=[], values=[], time=0)
        if self.sub.q0.is_updated():
            logger.debug("area 1 uqdated")
            q = PowersImaginary.parse_obj(self.sub.q0.json)
            all_q.time = q.time
            all_q.values += q.values
            all_q.ids += q.ids
            all_q.equipment_ids += q.equipment_ids

        if self.sub.q1.is_updated():
            logger.debug("area 2 uqdated")
            q = PowersImaginary.parse_obj(self.sub.q1.json)
            all_q.time = q.time
            all_q.values += q.values
            all_q.ids += q.ids
            all_q.equipment_ids += q.equipment_ids

        if self.sub.q2.is_updated():
            logger.debug("area 3 uqdated")
            q = PowersImaginary.parse_obj(self.sub.q2.json)
            all_q.time = q.time
            all_q.values += q.values
            all_q.ids += q.ids
            all_q.equipment_ids += q.equipment_ids

        if self.sub.q3.is_updated():
            logger.debug("area 4 uqdated")
            q = PowersImaginary.parse_obj(self.sub.q3.json)
            all_q.time = q.time
            all_q.values += q.values
            all_q.ids += q.ids
            all_q.equipment_ids += q.equipment_ids

        if self.sub.q4.is_updated():
            logger.debug("area 5 uqdated")
            q = PowersImaginary.parse_obj(self.sub.q4.json)
            all_q.time = q.time
            all_q.values += q.values
            all_q.ids += q.ids
            all_q.equipment_ids += q.equipment_ids

        logger.debug("All Imag Power")
        for eq, v in zip(all_q.equipment_ids, all_q.values):
            logger.debug(f"{eq} = {v}")

        for area in range(6):
            self.pub_area_q[area].publish(all_q.json())

    def publish_real(self):
        all_p = PowersReal(ids=[], equipment_ids=[], values=[], time=0)
        if self.sub.p0.is_updated():
            logger.debug("area 1 updated")
            p = PowersReal.parse_obj(self.sub.p0.json)
            all_p.time = p.time
            all_p.values += p.values
            all_p.ids += p.ids
            all_p.equipment_ids += p.equipment_ids

        if self.sub.p1.is_updated():
            logger.debug("area 2 updated")
            p = PowersReal.parse_obj(self.sub.p1.json)
            all_p.time = p.time
            all_p.values += p.values
            all_p.ids += p.ids
            all_p.equipment_ids += p.equipment_ids

        if self.sub.p2.is_updated():
            logger.debug("area 3 updated")
            p = PowersReal.parse_obj(self.sub.p2.json)
            all_p.time = p.time
            all_p.values += p.values
            all_p.ids += p.ids
            all_p.equipment_ids += p.equipment_ids

        if self.sub.p3.is_updated():
            logger.debug("area 4 updated")
            p = PowersReal.parse_obj(self.sub.p3.json)
            all_p.time = p.time
            all_p.values += p.values
            all_p.ids += p.ids
            all_p.equipment_ids += p.equipment_ids

        if self.sub.p4.is_updated():
            logger.debug("area 5 updated")
            p = PowersReal.parse_obj(self.sub.p4.json)
            all_p.time = p.time
            all_p.values += p.values
            all_p.ids += p.ids
            all_p.equipment_ids += p.equipment_ids

        logger.debug("All Real Power")
        for eq, v in zip(all_p.equipment_ids, all_p.values):
            logger.debug(f"{eq} = {v}")

        for area in range(6):
            self.pub_area_p[area].publish(all_p.json())

    def run(self) -> None:
        logger.info(f"Federate connected: {datetime.now()}")
        itr_need = h.helics_iteration_request_iterate_if_needed
        itr_stop = h.helics_iteration_request_no_iteration
        h.helicsFederateEnterExecutingMode(self.fed)

        update_interval = int(h.helicsFederateGetTimeProperty(
            self.fed, h.HELICS_PROPERTY_TIME_PERIOD))

        logger.debug(f"update interval: {update_interval}")
        granted_time = 0
        logger.debug("Step 0: Starting Time/Itr Loops")
        while granted_time <= self.static.t_steps:
            request_time = granted_time + update_interval
            logger.debug("Step 1: published initial values for iteration")
            itr_flag = itr_need
            self.publish_real()
            self.publish_imag()

            itr = 0
            while True:
                logger.debug(f"Step 2: Requesting time {request_time}")
                granted_time, itr_status = h.helicsFederateRequestTimeIterative(
                    self.fed, request_time, itr_flag)
                logger.debug(f"\tgranted time = {granted_time}")
                logger.debug(f"\titr status = {itr_status}")

                logger.debug("Step 3: checking if next step")
                if itr_status == h.helics_iteration_result_next_step:
                    logger.debug(f"\titr next: {itr}")
                    itr_flag = itr_stop
                    break

                itr += 1
                logger.info(f"\titer: {itr}")

                logger.debug("Step 4: checking if max iterations")
                if itr >= self.static.max_itr:
                    logger.debug(f"\tmax iteration reached")
                    itr_flag = itr_stop
                    continue

                logger.debug("Step 5: publishing updates")
                self.publish_real()
                self.publish_imag()
                itr_flag = itr_need

        self.stop()

    def stop(self) -> None:
        h.helicsFederateDisconnect(self.fed)
        h.helicsFederateFree(self.fed)
        h.helicsCloseLibrary()
        logger.info(f"Federate disconnected: {datetime.now()}")


def run_simulator(broker_config: BrokerConfig):
    #    schema = ComponentParameters.schema_json(indent=2)
    #    with open("./hub_power/hub_power_schema.json", "w") as f:
    #        f.write(schema)
    #
    sfed = HubFederate(broker_config)
    sfed.run()


if __name__ == "__main__":
    run_simulator(BrokerConfig(broker_ip="0.0.0.0"))
