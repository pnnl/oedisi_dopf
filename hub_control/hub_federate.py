import logging
import helics as h
import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from oedisi.types.data_types import (
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


class StaticConfig(object):
    name: str


class Subscriptions(object):
    area_c0: CommandList
    area_c1: CommandList
    area_c2: CommandList
    area_c3: CommandList
    area_c4: CommandList


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

    def initilize(self) -> None:
        self.info = h.helicsCreateFederateInfo()
        self.info.core_name = self.static.name
        self.info.core_type = h.HELICS_CORE_TYPE_ZMQ
        self.info.core_init = "--federates=1"
        h.helicsFederateInfoSetTimeProperty(
            self.info, h.helics_property_time_delta, 0.001)
        self.fed = h.helicsCreateValueFederate(self.static.name, self.info)

    def register_subscription(self) -> None:
        self.sub.area_c0 = self.fed.register_subscription(
            self.inputs["area_c0"], ""
        )
        self.sub.area_c1 = self.fed.register_subscription(
            self.inputs["area_c1"], ""
        )
        self.sub.area_c2 = self.fed.register_subscription(
            self.inputs["area_c2"], ""
        )
        self.sub.area_c3 = self.fed.register_subscription(
            self.inputs["area_c3"], ""
        )
        self.sub.area_c4 = self.fed.register_subscription(
            self.inputs["area_c4"], ""
        )

    def register_publication(self) -> None:
        self.pub_commands = self.fed.register_publication(
            "pv_set", h.HELICS_DATA_TYPE_STRING, "")

    def run(self) -> None:
        logger.info(f"Federate connected: {datetime.now()}")
        self.fed.enter_executing_mode()
        granted_time = h.helicsFederateRequestTime(
            self.fed, h.HELICS_TIME_MAXTIME)

        while granted_time < h.HELICS_TIME_MAXTIME:
            # each published voltage is sent to all areas immediatly because
            # some areas may be waiting on their neighbors input to run
            if self.sub.area_c0.is_updated():
                c = CommandList.parse_obj(
                    self.sub.area_c0.json
                )
                self.pub_pv_set.publish(c.json())

            if self.sub.area_c1.is_updated():
                c = CommandList.parse_obj(
                    self.sub.area_c1.json
                )
                self.pub_pv_set.publish(c.json())

            if self.sub.area_c2.is_updated():
                c = CommandList.parse_obj(
                    self.sub.area_c2.json
                )
                self.pub_pv_set.publish(c.json())

            if self.sub.area_c3.is_updated():
                c = CommandList.parse_obj(
                    self.sub.area_c3.json
                )
                self.pub_pv_set.publish(c.json())

            if self.sub.area_c4.is_updated():
                c = CommandList.parse_obj(
                    self.sub.area_c4.json
                )
                self.pub_pv_set.publish(c.json())

            if self.sub.area_c5.is_updated():
                c = CommandList.parse_obj(
                    self.sub.area_c5.json
                )
                self.pub_pv_set.publish(c.json())

        self.stop()

    def stop(self) -> None:
        h.helicsFederateDisconnect(self.fed)
        h.helicsFederateFree(self.fed)
        h.helicsCloseLibrary()
        logger.info(f"Federate disconnected: {datetime.now()}")


if __name__ == "__main__":
    fed = EstimatorFederate()
    fed.run()
