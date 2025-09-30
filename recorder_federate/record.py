import logging
import helics as h
import numpy as np
from pydantic import BaseModel
import pandas as pd
from typing import List
import json
import csv
import pyarrow as pa
from datetime import datetime
from oedisi.types.data_types import MeasurementArray

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class Recorder:
    def __init__(self, name, feather_filename, csv_filename, input_mapping):
        self.rng = np.random.default_rng(12345)
        deltat = 0.01

        # Create Federate Info object that describes the federate properties #
        fedinfo = h.helicsCreateFederateInfo()
        fedinfo.core_name = name
        fedinfo.core_type = h.HELICS_CORE_TYPE_ZMQ
        fedinfo.core_init = "--federates=1"
        logger.debug(name)

        # h.helicsFederateInfoSetTimeProperty(fedinfo, h.helics_property_time_delta, deltat)

        self.vfed = h.helicsCreateValueFederate(name, fedinfo)
        # h.helicsFederateSetFlagOption(self.vfed, h.helics_flag_slow_responding, True)
        h.helicsFederateSetTimeProperty(
            self.vfed, h.HELICS_PROPERTY_TIME_PERIOD, 1)
        # h.helicsFederateSetFlagOption(self.vfed, h.helics_flag_wait_for_current_time_update, True)

        logger.info("Value federate created")

        # Register the publication #
        self.sub = self.vfed.register_subscription(
            input_mapping["subscription"], "")
        self.feather_filename = feather_filename
        self.csv_filename = csv_filename

    def run(self):
        # Enter execution mode #
        # self.vfed.enter_initializing_mode()
        h.helicsFederateEnterExecutingMode(self.vfed)
        logger.info("Entering execution mode")

        # setting up time properties
        update_interval = int(h.helicsFederateGetTimeProperty(
            self.vfed, h.HELICS_PROPERTY_TIME_PERIOD))

        start = True
        granted_time = 0
        logger.debug("Step 0: Starting Time Loop")
        with pa.OSFile(self.feather_filename, "wb") as sink:
            writer = None
            while granted_time < h.HELICS_TIME_MAXTIME:
                request_time = granted_time + update_interval
                logger.debug(f"Step 1: Requesting Time {request_time}")
                granted_time = h.helicsFederateRequestTime(
                    self.vfed, request_time)
                logger.debug(f"\tgranted time = {granted_time}")

                # Check that the data is a MeasurementArray type
                logger.debug("Step 2: updating measurments")
                logger.debug(f"is valid: {self.sub.is_valid()}")
                measurement = MeasurementArray.parse_obj(self.sub.json)

                measurement_dict = {
                    key: value
                    for key, value in zip(measurement.ids, measurement.values)
                }
                measurement_dict["time"] = measurement.time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                if start:
                    schema_elements = [(key, pa.float64())
                                       for key in measurement.ids]
                    schema_elements.append(("time", pa.string()))
                    schema = pa.schema(schema_elements)
                    writer = pa.ipc.new_file(sink, schema)
                    start = False
                cnt = 0

                logger.debug("Step 3: writing measurements")
                writer.write_batch(
                    pa.RecordBatch.from_pylist([measurement_dict]))

                # Egranted_time, itr_status = h.helicsFederateRequestTimeIterative(
                # Eself.vfed, h.HELICS_TIME_MAXTIME, itr_flag)

            if writer is not None:
                writer.close()

            logger.info("end time: " + str(datetime.now()))
        data = pd.read_feather(self.feather_filename)
        data.to_csv(self.csv_filename, header=True, index=False)
        self.destroy()

    def destroy(self):
        logger.info("Federate disconnected")
        h.helicsFederateDisconnect(self.vfed)
        h.helicsFederateFree(self.vfed)
        h.helicsCloseLibrary()


if __name__ == "__main__":
    with open("static_inputs.json") as f:
        config = json.load(f)
        name = config["name"]
        feather_path = config["feather_filename"]
        csv_path = config["csv_filename"]

    with open("input_mapping.json") as f:
        input_mapping = json.load(f)

    sfed = Recorder(name, feather_path, csv_path, input_mapping)
    sfed.run()
