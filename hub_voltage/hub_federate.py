import logging
import helics as h
import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from oedisi.types.data_types import (
    VoltagesMagnitude,
    PowersReal,
    PowersImaginary,
    MeasurementArray,
    EquipmentNodeArray,
    CommandList
)
import xarray as xr
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


def xarray_to_eqarray(data):
    """Convert xarray to dict with values and ids for JSON serialization."""
    coords = {key: list(data.coords[key].data) for key in data.coords.keys()}
    return {"values": list(data.data), **coords}


def xarray_to_powers_cart(data, **kwargs):
    """Conveniently turn xarray into PowersReal and PowersImaginary."""
    real = PowersReal(**xarray_to_dict(data.real), **kwargs)
    imag = PowersImaginary(**xarray_to_dict(data.imag), **kwargs)
    return real, imag


class StaticConfig(object):
    name: str
    max_itr: int


class Subscriptions(object):
    v0: VoltagesMagnitude
    v1: VoltagesMagnitude
    v2: VoltagesMagnitude
    v3: VoltagesMagnitude
    v4: VoltagesMagnitude


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
        self.static.max_itr = config["max_itr"]

    def initilize(self) -> None:
        self.info = h.helicsCreateFederateInfo()
        self.info.core_name = self.static.name
        self.info.core_type = h.HELICS_CORE_TYPE_ZMQ
        self.info.core_init = "--federates=1"

        h.helicsFederateInfoSetTimeProperty(
            self.info, h.helics_property_time_delta, 0.001)
        self.fed = h.helicsCreateValueFederate(self.static.name, self.info)
        h.helicsFederateSetFlagOption(
            self.fed, h.helics_flag_slow_responding, True)

    def register_subscription(self) -> None:
        self.sub.v0 = self.fed.register_subscription(
            self.inputs["sub_v0"], ""
        )
        self.sub.v1 = self.fed.register_subscription(
            self.inputs["sub_v1"], ""
        )
        self.sub.v2 = self.fed.register_subscription(
            self.inputs["sub_v2"], ""
        )
        self.sub.v3 = self.fed.register_subscription(
            self.inputs["sub_v3"], ""
        )
        self.sub.v4 = self.fed.register_subscription(
            self.inputs["sub_v4"], ""
        )

    def register_publication(self) -> None:
        self.pub_area_voltages = []
        for i in range(6):
            self.pub_area_voltages.append(self.fed.register_publication(
                f"pub_v{i}", h.HELICS_DATA_TYPE_STRING, "")
            )

    def publish_all(self):
        all_v = VoltagesMagnitude(ids=[], values=[], time=0)
        if self.sub.v0.is_updated():
            logger.debug("area 1 updated")
            v = VoltagesMagnitude.parse_obj(self.sub.v0.json)
            logger.debug(v)
            all_v.time = v.time
            all_v.values += v.values
            all_v.ids += v.ids

        if self.sub.v1.is_updated():
            logger.debug("area 2 updated")
            v = VoltagesMagnitude.parse_obj(self.sub.v1.json)
            logger.debug(v)
            all_v.time = v.time
            all_v.values += v.values
            all_v.ids += v.ids

        if self.sub.v2.is_updated():
            logger.debug("area 3 updated")
            v = VoltagesMagnitude.parse_obj(self.sub.v2.json)
            logger.debug(v)
            all_v.time = v.time
            all_v.values += v.values
            all_v.ids += v.ids

        if self.sub.v3.is_updated():
            logger.debug("area 4 updated")
            v = VoltagesMagnitude.parse_obj(self.sub.v3.json)
            logger.debug(v)
            all_v.time = v.time
            all_v.values += v.values
            all_v.ids += v.ids

        if self.sub.v4.is_updated():
            logger.debug("area 5 updated")
            v = VoltagesMagnitude.parse_obj(self.sub.v4.json)
            logger.debug(v)
            all_v.time = v.time
            all_v.values += v.values
            all_v.ids += v.ids

        for area in range(6):
            self.pub_area_voltages[area].publish(all_v.json())

    def run(self) -> None:
        logger.info(f"Federate connected: {datetime.now()}")
        itr_flag = h.helics_iteration_request_iterate_if_needed
        itr_skip = h.HELICS_ITERATION_REQUEST_ITERATE_IF_NEEDED
        h.helicsFederateEnterExecutingMode(self.fed)

        granted_time = 0
        while granted_time < h.HELICS_TIME_MAXTIME:
            logger.debug(f"granted time: {granted_time}")
            self.publish_all()
            itr = 0
            while True:
                granted_time, itr_status = h.helicsFederateRequestTimeIterative(
                    self.fed, h.HELICS_TIME_MAXTIME, itr_flag)

                if itr_status == h.helics_iteration_result_next_step:
                    logger.debug(f"itr next: {granted_time}")
                    break

                itr += 1
                logger.info(f"iter: {itr}")

                if itr >= self.static.max_itr:
                    logger.debug(f"itr max: {granted_time}")
                    continue

                self.publish_all()

            logger.debug("EXITING ITER LOOP")
        self.stop()

    def stop(self) -> None:
        h.helicsFederateDisconnect(self.fed)
        h.helicsFederateFree(self.fed)
        h.helicsCloseLibrary()
        logger.info(f"Federate disconnected: {datetime.now()}")


if __name__ == "__main__":
    fed = EstimatorFederate()
    fed.run()
