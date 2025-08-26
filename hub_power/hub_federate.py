import logging
import helics as h
import json
from pathlib import Path
from datetime import datetime
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


class StaticConfig(object):
    name: str


class Subscriptions(object):
    area_p0: PowersReal
    area_p1: PowersReal
    area_p2: PowersReal
    area_p3: PowersReal
    area_p4: PowersReal
    area_q0: PowersImaginary
    area_q1: PowersImaginary
    area_q2: PowersImaginary
    area_q3: PowersImaginary
    area_q4: PowersImaginary


class HubFederate(object):
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
        self.sub.area_p0 = self.fed.register_subscription(
            self.inputs["area_p0"], ""
        )
        self.sub.area_p1 = self.fed.register_subscription(
            self.inputs["area_p1"], ""
        )
        self.sub.area_p2 = self.fed.register_subscription(
            self.inputs["area_p2"], ""
        )
        self.sub.area_p3 = self.fed.register_subscription(
            self.inputs["area_p3"], ""
        )
        self.sub.area_p4 = self.fed.register_subscription(
            self.inputs["area_p4"], ""
        )
        self.sub.area_q0 = self.fed.register_subscription(
            self.inputs["area_q0"], ""
        )
        self.sub.area_q1 = self.fed.register_subscription(
            self.inputs["area_q1"], ""
        )
        self.sub.area_q2 = self.fed.register_subscription(
            self.inputs["area_q2"], ""
        )
        self.sub.area_q3 = self.fed.register_subscription(
            self.inputs["area_q3"], ""
        )
        self.sub.area_q4 = self.fed.register_subscription(
            self.inputs["area_q4"], ""
        )

    def register_publication(self) -> None:
        self.pub_area_p = []
        for i in range(6):
            self.pub_area_p.append(self.fed.register_publication(
                f"area_p{i}", h.HELICS_DATA_TYPE_STRING, "")
            )

        self.pub_area_q = []
        for i in range(6):
            self.pub_area_q.append(self.fed.register_publication(
                f"area_q{i}", h.HELICS_DATA_TYPE_STRING, "")
            )

    def run(self) -> None:
        logger.info(f"Federate connected: {datetime.now()}")
        self.fed.enter_executing_mode()
        granted_time = h.helicsFederateRequestTime(
            self.fed, h.HELICS_TIME_MAXTIME)

        while granted_time < h.HELICS_TIME_MAXTIME:
            # each published power is sent to all areas immediatly because
            # some areas may be waiting on their neighbors input to run
            if self.sub.area_p0.is_updated():
                p = PowersReal.parse_obj(
                    self.sub.area_p0.json
                )

                for area in range(6):
                    self.pub_area_p[area].publish(p.json())

            if self.sub.area_p1.is_updated():
                p = PowersReal.parse_obj(
                    self.sub.area_p1.json
                )

                for area in range(6):
                    self.pub_area_p[area].publish(p.json())

            if self.sub.area_p2.is_updated():
                p = PowersReal.parse_obj(
                    self.sub.area_p2.json
                )

                for area in range(6):
                    self.pub_area_p[area].publish(p.json())

            if self.sub.area_p3.is_updated():
                p = PowersReal.parse_obj(
                    self.sub.area_p3.json
                )

                for area in range(6):
                    self.pub_area_p[area].publish(p.json())

            if self.sub.area_p4.is_updated():
                p = PowersReal.parse_obj(
                    self.sub.area_p4.json
                )

                for area in range(6):
                    self.pub_area_p[area].publish(p.json())

            if self.sub.area_p5.is_updated():
                p = PowersReal.parse_obj(
                    self.sub.area_p5.json
                )

                for area in range(6):
                    self.pub_area_p[area].publish(p.json())

            if self.sub.area_q0.is_updated():
                q = PowersImaginary.parse_obj(
                    self.sub.area_q0.json
                )

                for area in range(6):
                    self.pub_area_q[area].publish(q.json())

            if self.sub.area_q1.is_updated():
                q = PowersImaginary.parse_obj(
                    self.sub.area_q1.json
                )

                for area in range(6):
                    self.pub_area_q[area].publish(q.json())

            if self.sub.area_q2.is_updated():
                q = PowersImaginary.parse_obj(
                    self.sub.area_q2.json
                )

                for area in range(6):
                    self.pub_area_q[area].publish(q.json())

            if self.sub.area_q3.is_updated():
                q = PowersImaginary.parse_obj(
                    self.sub.area_q3.json
                )

                for area in range(6):
                    self.pub_area_q[area].publish(q.json())

            if self.sub.area_q4.is_updated():
                q = PowersImaginary.parse_obj(
                    self.sub.area_q4.json
                )

                for area in range(6):
                    self.pub_area_q[area].publish(q.json())

            if self.sub.area_q5.is_updated():
                q = PowersImaginary.parse_obj(
                    self.sub.area_q5.json
                )

                for area in range(6):
                    self.pub_area_q[area].publish(q.json())
        self.stop()

    def stop(self) -> None:
        h.helicsFederateDisconnect(self.fed)
        h.helicsFederateFree(self.fed)
        h.helicsCloseLibrary()
        logger.info(f"Federate disconnected: {datetime.now()}")


if __name__ == "__main__":
    fed = HubFederate()
    fed.run()
