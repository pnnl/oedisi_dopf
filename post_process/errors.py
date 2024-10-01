import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.feather as feather
from datetime import datetime
from oedisi.types.data_types import Topology
import json
import sys


def get_time(x):
    return datetime.strptime(x, "%Y-%m-%d %H:%M:%S").time().strftime("%H:%M")


def get_voltage(realV, imagV):
    voltage_real = feather.read_feather(realV)
    voltage_imag = feather.read_feather(imagV)
    df_voltages = np.abs(
        voltage_real.drop("time", axis=1) + 1j * voltage_imag.drop("time", axis=1)
    )
    df_voltages["time"] = voltage_real["time"].apply(get_time)
    return df_voltages.set_index("time")


def errors(true_voltages, opf_voltages):
    true_mag = np.abs(true_voltages)
    nonzero_parts = true_mag != 0.0
    MAE = np.mean(
        np.array(np.abs(true_mag - opf_voltages) / true_mag)[nonzero_parts] * 100
    )
    return MAE


def error_table(true_voltages, opf_voltages):
    error_table = []
    for i, t in enumerate(true_voltages.index):
        MAE = errors(true_voltages.iloc[i, :], opf_voltages.iloc[i, :])
        error_table.append({"t": t, "MAE": MAE})
    return pd.DataFrame(error_table)


def plot_errors(err_table):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(err_table["t"], err_table["MAE"])
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=20)
    ax.set_xlabel("Time (15 minute)", fontsize=18)
    ax.set_xticks(err_table["t"][::10], err_table["t"][::10], rotation=-25, fontsize=10)
    return fig


if __name__ == "__main__":
    case_name = sys.argv[1]

    realVfile = f"outputs/{case_name}/voltage_real.feather"
    imagVfile = f"outputs/{case_name}/voltage_imag.feather"
    opfVfile = f"outputs/{case_name}/opf_voltage_mag.feather"
    topofile = f"outputs/{case_name}/topology.json"

    with open(topofile) as f:
        topology = Topology.parse_obj(json.load(f))
        base_voltage_df = pd.DataFrame(
            {
                "id": topology.base_voltage_magnitudes.ids,
                "value": topology.base_voltage_magnitudes.values,
            }
        )
        base_voltage_df.set_index("id", inplace=True)
        base_voltages = base_voltage_df["value"]

    df_true_voltages = get_voltage(realVfile, imagVfile) / base_voltages
    true_voltage_columns = df_true_voltages.columns

    df_OPF_voltages = feather.read_feather(opfVfile)
    df_OPF_voltages["time"] = df_OPF_voltages["time"].apply(get_time)
    df_OPF_voltages = df_OPF_voltages.set_index("time")
    opf_voltage_columns = df_OPF_voltages.columns

    node_names = [node for node in true_voltage_columns if node in opf_voltage_columns]
    true_voltages = df_true_voltages[df_true_voltages.columns.intersection(node_names)]
    OPF_voltages = df_OPF_voltages[df_OPF_voltages.columns.intersection(node_names)]

    err_table = error_table(true_voltages, OPF_voltages)

    MAE = errors(true_voltages, OPF_voltages)
    fig_error = plot_errors(err_table)
    fig_error.suptitle(
        f"Mean Absolute Voltage Magnitude Errors MAE={MAE:0.3f}", fontsize=22
    )
    fig_error.savefig(f"outputs/{case_name}/opf_approx_errors.png")
