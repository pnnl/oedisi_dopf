import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np


def extract_complex(real_path: str, imag_path) -> (list[datetime], np.ndarray):
    real = pd.read_feather(real_path)
    imag = pd.read_feather(imag_path)
    c = np.empty([len(real), 2], dtype=np.complex64)
    c.real = real.drop("time", axis=1)
    c.imag = imag.drop("time", axis=1)
    return (real["time"], c)


if __name__ == "__main__":
    assert sys.argv.count() > 2
    scenario = sys.argv[1]

    root = os.getcwd()
    outputs = f"{root}/outputs/scenario/{scenario}"
    file_ref_voltage = f"{root}"
    ref_voltage = extract_complex()
