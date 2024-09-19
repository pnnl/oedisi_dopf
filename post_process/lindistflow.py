import sys
import os
import pandas as pd
import numpy as np

from oedisi.types.data_types import (
    Topology,
    VoltagesReal,
    VoltagesImaginary
)

ROOT = os.getcwd()
ALGO = "lindistflow"
MODEL = ""
OUT_DIR = ""


def get_voltage() -> pd.DataFrame:
    pass


def get_topology() -> (pd.DataFrame, pd.DataFrame):
    path = f"{OUT_DIR}/topology.json"
    topology = Topology.parse_file(path)
    print(topology)
    base_voltage_magnitudes = np.array(topology.base_voltage_magnitudes.values)


if __name__ == "__main__":
    MODEL = sys.argv[1]
    OUT_DIR = f"{ROOT}/outputs/{ALGO}/{MODEL}"
    get_topology()
