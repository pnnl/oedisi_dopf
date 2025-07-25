import os
import sys
import numpy as np
import pandas as pd
import opendssdirect as dss
from pprint import pprint


def init_paths(data: str, model: str) -> str:
    if not os.path.exists(data):
        print("data location doesn't exist")
        exit()

    feature = os.path.abspath(f"./features/{model}")
    os.makedirs(feature, exist_ok=True)
    return feature


def init_opendss(path: str) -> None:
    os.chdir(path)
    files = os.listdir()
    master = "Master.dss"
    if master not in files:
        master = "master.dss"

    dss.Text.Command(f'Redirect {master}')
    dss.Text.Command(f'Compile {master}')


# split dataframe by heterogeneous characteristic
def seperate_data(data: pd.DataFrame, info: str) -> list[pd.DataFrame]:
    filtered = []
    unique = data[info].unique()
    for v in unique:
        filtered.append(data[data[info] == v])
    return filtered


def export_features(data: pd.DataFrame, output: str) -> None:
    features = pd.DataFrame()
    for df in data:
        for _, row in df.iterrows():
            if "" == row['Yearly']:
                continue
            dss.LoadShape.Name(row['Yearly'])
            features[row['Name']] = dss.LoadShape.PMult()
        ctx = f"{df['Name'].iloc[0]}_{df['Name'].iloc[-1]}"
        features.to_csv(f"{output}/{ctx}.csv")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Need path to input model and output name")
        print("example: python feature_extraction.py ./data/ieee123/opendss ieee123")
        exit()

    data_dir = os.path.abspath(sys.argv[1])
    model = sys.argv[2]
    feature_dir = init_paths(data_dir, model)

    init_opendss(data_dir)

    loads = dss.utils.loads_to_dataframe()
    split_loads = seperate_data(loads, 'kV')

    homogeneous_loads = []
    for split in split_loads:
        homogeneous_loads.append(seperate_data(split, 'kW'))

    for data in homogeneous_loads:
        export_features(data, feature_dir)

    # pvs = dss.utils.pvsystems_to_dataframe()
