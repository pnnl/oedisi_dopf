#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 11:23:32 2025

@author: mitr284

Script to Anonymise OpenDSS files using Differential Privacy
"""

import pandas as pd
import os
import opendssdirect as dss
import uuid
import random
import hashlib
import argparse


def init_paths(args) -> str:
    input = os.path.abspath(f"{args.data}/opendss")
    if not os.path.exists(input):
        print("data location doesn't exist")
        exit()

    return input


def init_opendss(path: str):
    os.chdir(path)

    dss.Text.Command("Clear")
    dss.Text.Command("Redirect master.dss")
    dss.Text.Command('Solve')


def clear_folder(folder_path: str):
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory.")
        return

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Removed file: {item_path}")


def replace_in_file(filepath: str, old_string: str, new_string: str):
    try:
        # Read the entire content of the file
        with open(filepath, 'r') as file:
            file_content = file.read()

        # Perform the replacement
        modified_content = file_content.replace(old_string, new_string)

        # Write the modified content back to the file
        with open(filepath, 'w') as file:
            file.write(modified_content)

    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


def anonymize_name(original_names):
    return {name: f"anon_{uuid.uuid4().hex[:8]}" for name in original_names}


def hash_name(name, salt="dp_salt"):
    return hashlib.sha256((salt + name).encode()).hexdigest()[:10]


def dp_randomized_response(name, all_names, epsilon=5.0):
    p = (1 + (2.71828 ** epsilon)) / (len(all_names) + (2.71828 ** epsilon))
    if random.random() < p:
        return hash_name(name)  # pseudonymized true name
    else:
        return hash_name(random.choice(all_names))  # pseudonymized random name


def add_noise(value, noise_level=0.05):
    noisy_value = value + random.uniform(-noise_level*value, noise_level*value)

    return abs(noisy_value)


def replace_with_anonymized(df, original_col, anon_col):
    return df.drop(
        columns=[original_col]).rename(columns={anon_col: original_col})


def write_dss_from_dataframe_auto(df, filename, object_type, object_name_column):

    property_columns = [col for col in df.columns if col != object_name_column]

    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            object_name = row[object_name_column]
            f.write(f"new {object_type}.{object_name} ")
            for property_name in property_columns:
                value = row[property_name]
                f.write(f"{property_name}={value} ")
            f.write("\n")


def create_master(path: str, **dss_files):
    master_path = f"{path}/master.dss"
    circuit = dss.Circuit.Name()

    with open(master_path, 'w') as f:
        f.write("Clear\n")
        f.write(
            f"New Circuit.{circuit} basekv=4.16 Bus1=150 pu=1.00 R1=0 X1=0.0001 R0=0 X0=0.0001\n")

        for label, filename in dss_files.items():
            f.write(f"Redirect {filename}\n")


def rename_loads(path: str):
    for name in dss.Loads.AllNames():
        old_name = f"Load.{name}"
        new_name = f"Load.{name}NEW"
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
                print(f"Removed file: {item_path}")

        replace_in_file()


'''
# Anonymize lists
anonymized_buses = [dp_randomized_response(name, bus_names) for name in bus_names]
anonymized_components = [dp_randomized_response(name, component_names) for name in component_names]
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Anonymize OpenDSS data.")
    parser.add_argument(
        "--data", help="path to folder that containes the models opendss folder e.g ./private_data")
    args = parser.parse_args()
    data = init_paths(args)
    init_opendss(data)

    '''-----Gather Load data----'''
    df_load = dss.utils.loads_to_dataframe()

    load_data = []

    if dss.Loads.First() > 0:

        while True:
            load_data.append({
                "name": dss.Loads.Name(),
                "bus": dss.CktElement.BusNames()[0],
                "kW": dss.Loads.kW(),
                "kvar": dss.Loads.kvar(),
                "phases": dss.Loads.Phases(),
            })

            if dss.Loads.Next() == 0:
                break

    load_df = pd.DataFrame(load_data)

    df_load.loc[:, 'bus'] = load_df['bus'].values

    ''' --- Gather Transformer data ---'''
    df_transformer = dss.utils.transformers_to_dataframe()

    transformer_data = []
    if dss.Transformers.First() > 0:
        while True:
            transformer_data.append({
                "name": dss.Transformers.Name(),
                "buses": dss.CktElement.BusNames()[0],
                "kVs": dss.Transformers.kV(),
                "kVAs": dss.Transformers.kVA(),
                "num_windings": dss.Transformers.NumWindings(),
            })
            if dss.Transformers.Next() == 0:
                break

    transformer_df = pd.DataFrame(transformer_data)
    df_transformer.loc[:, 'bus'] = transformer_df['buses'].values

    '''---Gather Line Data---'''

    df_line = dss.utils.lines_to_dataframe()

    line_data = []
    if dss.Lines.First() > 0:
        while True:
            line_data.append({
                "name": dss.Lines.Name(),
                "bus1": dss.Lines.Bus1(),
                "bus2": dss.Lines.Bus2(),
                "length": dss.Lines.Length(),
                "phases": dss.Lines.Phases(),
                "line_code": dss.Lines.LineCode(),
            })
            if dss.Lines.Next() == 0:
                break

    line_df = pd.DataFrame(line_data)
    df_line.loc[:, 'bus1'] = line_df['bus1'].values
    df_line.loc[:, 'bus2'] = line_df['bus2'].values

    '''--- Gather Capacitor Data ---'''

    df_capacitor = dss.utils.capacitors_to_dataframe()

    capacitor_data = []
    if dss.Capacitors.First() > 0:
        while True:
            capacitor_data.append({
                "name": dss.Capacitors.Name(),
                "bus": dss.CktElement.BusNames()[0],
                "kvar": dss.Capacitors.kvar(),
                "kv": dss.Capacitors.kV(),

            })
            if dss.Capacitors.Next() == 0:
                break

    capacitor_df = pd.DataFrame(capacitor_data)

    df_capacitor.loc[:, 'bus'] = capacitor_df['bus'].values

    ''' --- Gather PV data ---'''
    df_pv = dss.utils.pvsystems_to_dataframe()

    pv_data = []
    if dss.PVsystems.First() > 0:
        while True:
            pv_data.append({
                "name": dss.PVsystems.Name(),
                "bus": dss.CktElement.BusNames()[0],
                "kvar": dss.PVsystems.kvar(),
                "kW": dss.PVsystems.kW(),
            })
            if dss.PVsystems.Next() == 0:
                break

    pv_df = pd.DataFrame(pv_data)
    df_pv.loc[:, 'bus'] = pv_df['bus'].values

    df_regulators = dss.utils.regcontrols_to_dataframe()

    # print(df_regulators)

    bus_list = pd.concat([
        df_pv['bus'],
        df_transformer["bus"].explode(),
        df_capacitor['bus'],
        df_line["bus1"],
        df_line["bus2"],
        df_load["bus"],
    ]).unique()

    name_list = pd.concat([
        df_load['Name'],
        df_transformer["Name"].explode(),
        df_capacitor["Name"],
        df_line["Name"],
        df_pv["Name"]
    ]).unique()

    # name_mapping = anonymize_name(bus_list)
    # name_mapping_2 = anonymize_name(name_list)

    epsilon = 1

    anonymized_bus_list = [dp_randomized_response(
        name, bus_list, epsilon) for name in bus_list]

    df_anonymized_bus = pd.DataFrame({
        'original_bus': bus_list,
        'anonymized_bus': anonymized_bus_list
    })

    bus_mapping = dict(
        zip(df_anonymized_bus['original_bus'], df_anonymized_bus['anonymized_bus']))

    '''---- Mapping Bus Names ---'''

    df_load['anom_bus'] = df_load['bus'].map(bus_mapping)
    df_transformer["anom_buses"] = df_transformer["bus"].map(bus_mapping)
    df_capacitor["anom_bus"] = df_capacitor["bus"].map(bus_mapping)
    df_line["anom_bus1"] = df_line["bus1"].map(bus_mapping)
    df_line["anom_bus2"] = df_line["bus2"].map(bus_mapping)
    df_pv["anom_bus"] = df_pv["bus"].map(bus_mapping)

    anonymized_name_list = [dp_randomized_response(
        name, name_list, epsilon) for name in name_list]

    df_anonymized_name = pd.DataFrame({
        'original_name': name_list,
        'anonymized_name': anonymized_name_list
    })

    name_mapping = dict(
        zip(df_anonymized_name['original_name'], df_anonymized_name['anonymized_name']))

    '''---- Anonymizing  Names ---'''
    df_load['anom_name'] = df_load['Name'].map(name_mapping)
    df_transformer["anom_name"] = df_transformer["Name"].map(name_mapping)
    df_capacitor["anom_name"] = df_capacitor["Name"].map(name_mapping)
    df_line["anom_name"] = df_line["Name"].map(name_mapping)
    df_pv["anom_name"] = df_pv["Name"].map(name_mapping)

    df_load['anom_kW'] = df_load['kW'].apply(add_noise)
    df_load['anom_kvar'] = df_load['kvar'].apply(add_noise)

    df_capacitor['anom_kvar'] = df_capacitor['kvar'].apply(add_noise)
    df_capacitor['anom_kV'] = df_capacitor['kV'].apply(add_noise)

    df_line['anom_Length'] = df_line['Length'].apply(add_noise)

    df_pv['anom_kW'] = df_pv['kW'].apply(add_noise)
    df_pv['anom_kvar'] = df_pv['kvar'].apply(add_noise)

    df_load = replace_with_anonymized(df_load, 'bus', 'anom_bus')
    df_load = replace_with_anonymized(df_load, 'Name', 'anom_name')
    df_load = replace_with_anonymized(df_load, 'kW', 'anom_kW')
    df_load = replace_with_anonymized(df_load, 'kvar', 'anom_kvar')

    df_transformer = replace_with_anonymized(
        df_transformer, 'bus', 'anom_buses')
    df_transformer = replace_with_anonymized(
        df_transformer, 'Name', 'anom_name')

    df_capacitor = replace_with_anonymized(df_capacitor, 'Name', 'anom_name')
    df_capacitor = replace_with_anonymized(df_capacitor, 'bus', 'anom_bus')
    df_capacitor = replace_with_anonymized(df_capacitor, 'kV', 'anom_kV')
    df_capacitor = replace_with_anonymized(df_capacitor, 'kvar', 'anom_kvar')

    df_pv = replace_with_anonymized(df_pv, 'Name', 'anom_name')
    df_pv = replace_with_anonymized(df_pv, 'bus', 'anom_bus')
    df_pv = replace_with_anonymized(df_pv, 'kW', 'anom_kW')
    df_pv = replace_with_anonymized(df_pv, 'kvar', 'anom_kvar')

    df_line = replace_with_anonymized(df_line, 'bus1', 'anom_bus1')
    df_line = replace_with_anonymized(df_line, 'bus2', 'anom_bus2')
    df_line = replace_with_anonymized(df_line, 'Name', 'anom_name')
    df_line = replace_with_anonymized(df_line, 'Length', 'anom_Length')

    ''' ---Converting dataframe back to .dss ---'''

    clear_folder(data)

    write_dss_from_dataframe_auto(
        df_pv, f"{data}/PVSystem.dss", "PVSystem", "Name")
    write_dss_from_dataframe_auto(
        df_load, f"{data}/Load.dss", "Load", "Name")
    write_dss_from_dataframe_auto(
        df_transformer, f"{data}/Transformer.dss", "Transformer", "Name")
    write_dss_from_dataframe_auto(
        df_line, f"{data}/Line.dss", "Line", "Name")
    write_dss_from_dataframe_auto(
        df_capacitor, f"{data}/Capacitor.dss", "Capacitor", "Name")

    ''' ---Creating a new master.dss ---'''

    create_master(data, PV="PVSystem.dss", Transformer='Transformer.dss', Load="Load.dss",
                  Capacitor="Capacitor.dss", Line="Line.dss")
