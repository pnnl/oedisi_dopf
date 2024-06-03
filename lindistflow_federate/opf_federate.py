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
    Injection,
    Topology,
    VoltagesMagnitude,
    MeasurementArray,
)
import adapter
import lindistflow
from area import area_info
import xarray as xr

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

def xarray_to_dict(data):
    """Convert xarray to dict with values and ids for JSON serialization."""
    coords = {key: list(data.coords[key].data) for key in data.coords.keys()}
    return {"values": list(data.data), **coords}


class StaticConfig(object):
    name: str
    deltat: float
    control_type: lindistflow.ControlType
    pf_flag: bool


class Subscriptions(object):
    voltages_mag: VoltagesMagnitude
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
        self.static.control_type = lindistflow.ControlType(
            config["control_type"])
        self.static.pf_flag = config["pf_flag"]

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
        self.sub.voltages_mag = self.fed.register_subscription(
            self.inputs["voltages_magnitude"], "")
        self.sub.injections = self.fed.register_subscription(
            self.inputs["injections"], "")
        # Optional subscription: PV forecast
        self.sub.pv_forecast = self.fed.register_subscription(
            self.inputs["pv_forecast"], "")
        self.sub.pv_forecast.set_default("[]")
        self.sub.pv_forecast.option["CONNECTION_OPTIONAL"] = True

    def register_publication(self) -> None:
        self.pub_commands = self.fed.register_publication(
            "change_commands", h.HELICS_DATA_TYPE_STRING, ""
        )

        self.pub_voltages = self.fed.register_publication(
            "opf_voltages_magnitude", h.HELICS_DATA_TYPE_STRING, ""
        )

        self.pub_delta_setpt = self.fed.register_publication(
            "delta_setpoint", h.HELICS_DATA_TYPE_STRING, ""
        )

        self.pub_curtail_forecast = self.fed.register_publication(
            "forecast_curtail", h.HELICS_DATA_TYPE_STRING, ""
        )
    
    def get_set_points(self, control, bus_info, conversion):
        setpoint = {}
        for key, val in control.items():
            if key in bus_info:
                bus = bus_info[key]
                if 'eqid' in bus:
                    eqid = bus['eqid']
                    [eq_type, _] = eqid.split('.')
                    if eq_type == "PVSystem":
                        sp = lindistflow.ignore_phase(val)*conversion
                        setpoint[eqid] = 0.0 if sp < 0.1 else sp
        return setpoint

    def run(self) -> None:
        logger.info(f"Federate connected: {datetime.now()}")
        self.fed.enter_executing_mode()
        granted_time = h.helicsFederateRequestTime(
            self.fed, h.HELICS_TIME_MAXTIME)

        grab_forecast_flag = False
        time_ctr = -1

        while granted_time < h.HELICS_TIME_MAXTIME:

            if not self.sub.voltages_mag.is_updated():
                granted_time = h.helicsFederateRequestTime(
                    self.fed, h.HELICS_TIME_MAXTIME
                )
                continue

            topology = Topology.parse_obj(self.sub.topology.json)
            [branch_info, bus_info] = adapter.extract_info(topology)

            slack = topology.slack_bus[0]
            [slack_bus, phase] = slack.split('.')

            area_branch, area_bus = area_info(
                branch_info, bus_info, slack_bus)

            voltages_mag = VoltagesMagnitude.parse_obj(
                self.sub.voltages_mag.json)

            area_bus = adapter.extract_voltages(area_bus, voltages_mag)

            
            # evaluate the forecasted PV set points and forecasted curtailment
            if not grab_forecast_flag:
                pv_forecast = self.sub.pv_forecast.json
                forecast_setp = {}
                forecast_curt = {}
                for k, forecast in enumerate(pv_forecast):
                    logger.info(f"Forecasting for time step {k}")
                    
                    forecast_generation = json.loads(forecast)
                    dict_forecast_gen = dict(zip(
                        forecast_generation["ids"], 
                        forecast_generation["values"]
                        ))
                    
                    # insert forecasted generation values to the PV injection vector
                    area_bus = adapter.extract_forecast(
                        area_bus, 
                        forecast_generation
                        )
                    
                    # perform forecast LinDistFlow
                    voltages, power_flow, forecast_control, conv = lindistflow.optimal_power_flow(
                        area_branch, area_bus, slack_bus, 
                        self.static.control_type, self.static.pf_flag
                        )
                    
                    # compute the forecatsed set points
                    forecast_setpts = self.get_set_points(
                        forecast_control, 
                        area_bus, conv
                        )
                    
                    # make outputs ready for publishing
                    for eqid in forecast_setpts:
                        setpt = forecast_setpts[eqid]
                        curt = dict_forecast_gen[eqid] - setpt

                        # add the forecasted set points
                        if eqid not in forecast_setp:
                            forecast_setp[eqid] = [setpt]
                        else:
                            forecast_setp[eqid].append(setpt)

                        # add the forecasted curtailments
                        if eqid not in forecast_curt:
                            forecast_curt[eqid] = [curt]
                        else:
                            forecast_curt[eqid].append(curt)
                        
                grab_forecast_flag = True
                
                

            time = voltages_mag.time
            logger.info(time)

            injection = Injection.parse_obj(self.sub.injections.json)
            area_bus = adapter.extract_injection(area_bus, injection)

            voltages, power_flow, control, conversion = lindistflow.optimal_power_flow(
                area_branch, area_bus, slack_bus, self.static.control_type, self.static.pf_flag)
            real_setpts = self.get_set_points(control, area_bus, conversion)
            
            # Compute the delta change in setpoints and publish
            time_ctr += 1
            pveq_id = []
            delta_setpt = []
            forecast_curtail = []
            for eq_id in real_setpts:
                pveq_id.append(eq_id)
                delta_setpt.append(real_setpts[eq_id]-forecast_setp[eq_id][time_ctr])
                forecast_curtail.append(forecast_curt[eq_id][time_ctr])
            delta_sp = xr.DataArray(delta_setpt, coords={"ids": pveq_id})
            fore_curt = xr.DataArray(forecast_curtail, coords={"ids": pveq_id})
            self.pub_delta_setpt.publish(
                MeasurementArray(
                    **xarray_to_dict(delta_sp),time=time,
                    units="kW",
                    ).json()
                )
            self.pub_curtail_forecast.publish(
                MeasurementArray(
                    **xarray_to_dict(fore_curt),time=time,
                    units="kW",
                    ).json()
                )
            
            
            # get the control commands for the feeder federate
            commands = []
            for key, val in control.items():
                if key in area_bus:
                    bus = area_bus[key]
                    if 'eqid' in bus:
                        eqid = bus['eqid']
                        [type, _] = eqid.split('.')
                        if type == "PVSystem":
                            setpoint = lindistflow.ignore_phase(val)*conversion
                            if setpoint < 0.1:
                                continue

                            if self.static.control_type == lindistflow.ControlType.WATT:
                                commands.append((eqid, setpoint, 0))
                            elif self.static.control_type == lindistflow.ControlType.VAR:
                                commands.append((eqid, 0, setpoint))
                            elif self.static.control_type == lindistflow.ControlType.WATT_VAR:
                                # todo
                                pass

            if commands:
                self.pub_commands.publish(
                    json.dumps(commands)
                )

            pub_mags = adapter.pack_voltages(voltages, time)
            self.pub_voltages.publish(
                pub_mags.json()
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
