import logging
import helics as h
import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from oedisi.types.data_types import (
    MeasurementArray,
    EquipmentNodeArray,
    CommandList,
    Command
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
    t_steps: int


class Subscriptions(object):
    c0: CommandList
    c1: CommandList
    c2: CommandList
    c3: CommandList
    c4: CommandList


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
        self.static.t_steps = config["number_of_timesteps"]

    def initilize(self) -> None:
        self.info = h.helicsCreateFederateInfo()
        self.info.core_name = self.static.name
        self.info.core_type = h.HELICS_CORE_TYPE_ZMQ
        self.info.core_init = "--federates=1"

        # h.helicsFederateInfoSetTimeProperty(self.info, h.helics_property_time_delta, 0.01)
        self.fed = h.helicsCreateValueFederate(self.static.name, self.info)
        # h.helicsFederateSetFlagOption(self.fed, h.helics_flag_slow_responding, True)
        h.helicsFederateSetTimeProperty(
            self.fed, h.HELICS_PROPERTY_TIME_PERIOD, 1)

    def register_subscription(self) -> None:
        self.sub.c0 = self.fed.register_subscription(
            self.inputs["sub_c0"], ""
        )
        self.sub.c1 = self.fed.register_subscription(
            self.inputs["sub_c1"], ""
        )
        self.sub.c2 = self.fed.register_subscription(
            self.inputs["sub_c2"], ""
        )
        self.sub.c3 = self.fed.register_subscription(
            self.inputs["sub_c3"], ""
        )
        self.sub.c4 = self.fed.register_subscription(
            self.inputs["sub_c4"], ""
        )

    def register_publication(self) -> None:
        self.pub_commands = self.fed.register_publication(
            "pv_set", h.HELICS_DATA_TYPE_STRING, "")

    def run(self) -> None:
        logger.info(f"Federate connected: {datetime.now()}")
        h.helicsFederateEnterExecutingMode(self.fed)
        logger.info(f"Federate executing: {datetime.now()}")

        # setting up time properties
        update_interval = h.helicsFederateGetTimeProperty(
            self.fed, h.HELICS_PROPERTY_TIME_PERIOD)
        logger.debug(f"update interval: {update_interval}")

        commands = []
        granted_time = 0
        logger.debug("Step 0: Starting Time loop")
        while granted_time <= self.static.t_steps:
            request_time = granted_time + update_interval
            logger.debug(f"Step 1: Requesting Time {request_time}")
            granted_time = h.helicsFederateRequestTime(self.fed, request_time)
            logger.debug(f"\tgranted time = {granted_time}")

            logger.debug("Step 2: collect all commands")
            commands = []
            if self.sub.c0.is_updated():
                logger.debug("\tarea 1: updated")
                control = self.sub.c0.json
                logger.debug(control)
                for c in control:
                    commands.append(c)

            if self.sub.c1.is_updated():
                logger.debug("\tarea 2: updated")
                control = self.sub.c1.json
                logger.debug(control)
                for c in control:
                    commands.append(c)

            if self.sub.c2.is_updated():
                logger.debug("\tarea 3: updated")
                control = self.sub.c2.json
                logger.debug(control)
                for c in control:
                    commands.append(c)

            if self.sub.c3.is_updated():
                logger.debug("\tarea 4: updated")
                control = self.sub.c3.json
                logger.debug(control)
                for c in control:
                    commands.append(c)

            if self.sub.c4.is_updated():
                logger.debug("\tarea 5: updated")
                control = self.sub.c4.json
                logger.debug(control)
                for c in control:
                    commands.append(c)
            logger.debug("Step 3: Commands collected")
            logger.info(commands)

            logger.debug("Step 4: Publishing commands")
            self.pub_commands.publish(json.dumps(commands))

        self.stop()

    def stop(self) -> None:
        h.helicsFederateDisconnect(self.fed)
        h.helicsFederateFree(self.fed)
        h.helicsCloseLibrary()
        logger.info(f"Federate disconnected: {datetime.now()}")


if __name__ == "__main__":
    fed = EstimatorFederate()
    fed.run()
