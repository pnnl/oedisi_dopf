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

    def run(self) -> None:
        logger.info(f"Federate connected: {datetime.now()}")
        self.fed.enter_executing_mode()
        granted_time = h.helicsFederateRequestTime(
            self.fed, h.HELICS_TIME_MAXTIME)

        logger.info(f"Granted Time: {granted_time}")
        updated = [False]*5
        while granted_time < h.HELICS_TIME_MAXTIME:
            # each published voltage is sent to all areas immediatly because
            # some areas may be waiting on their neighbors input to run
            if self.sub.v0.is_updated():
                updated[0] = True
                v = VoltagesMagnitude.parse_obj(
                    self.sub.v0.json
                )

                for area in range(6):
                    self.pub_area_voltages[area].publish(v.json())

            if self.sub.v1.is_updated():
                updated[1] = True
                v = VoltagesMagnitude.parse_obj(
                    self.sub.v1.json
                )

                for area in range(6):
                    self.pub_area_voltages[area].publish(v.json())

            if self.sub.v2.is_updated():
                updated[2] = True
                v = VoltagesMagnitude.parse_obj(
                    self.sub.v2.json
                )

                for area in range(6):
                    self.pub_area_voltages[area].publish(v.json())

            if self.sub.v3.is_updated():
                updated[3] = True
                v = VoltagesMagnitude.parse_obj(
                    self.sub.v3.json
                )

                for area in range(6):
                    self.pub_area_voltages[area].publish(v.json())

            if self.sub.v4.is_updated():
                updated[4] = True
                v = VoltagesMagnitude.parse_obj(
                    self.sub.v4.json
                )

                for area in range(6):
                    self.pub_area_voltages[area].publish(v.json())

            if not all(updated):
                granted_time = h.helicsFederateRequestTime(
                    self.fed, h.HELICS_TIME_MAXTIME)
            logger.info(f"No Updates: {granted_time}")
        self.stop()

    def stop(self) -> None:
        h.helicsFederateDisconnect(self.fed)
        h.helicsFederateFree(self.fed)
        h.helicsCloseLibrary()
        logger.info(f"Federate disconnected: {datetime.now()}")


if __name__ == "__main__":
    fed = EstimatorFederate()
    fed.run()
