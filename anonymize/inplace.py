import os
import shutil
import opendssdirect as dss
import argparse
import uuid
import random
import hashlib
import csv
import re
from pprint import pprint


def init_paths(args) -> str:
    input = os.path.abspath(f"{args.data}/opendss")
    if not os.path.exists(input):
        print("data location doesn't exist")
        exit()

    return input


def init_opendss(path: str):
    os.chdir(path)

    master = "master.dss"
    if not os.path.exists(master):
        master = "Master.dss"

    dss.Text.Command("Clear")
    dss.Text.Command(f"Redirect {master}")
    dss.Text.Command(f"Compile {master}")
    dss.Text.Command('Solve')


def add_noise(value: float, noise_level: float = 0.05) -> float:
    noisy_value = value + random.uniform(-noise_level*value, noise_level*value)

    return abs(noisy_value)


def anonymize_name(original_names):
    return {name: f"anon_{uuid.uuid4().hex[:8]}" for name in original_names}


def hash_name(name, salt="dp_salt"):
    return hashlib.sha256((salt + name).encode()).hexdigest()[:10]


def dp_randomized_response(name, all_names, epsilon=5.0):
    return hash_name(name)  # pseudonymized true name

    # skipping for now as these names need to be unique
    p = (1 + (2.71828 ** epsilon)) / (len(all_names) + (2.71828 ** epsilon))
    if random.random() < p:
        return hash_name(name)  # pseudonymized true name
    else:
        return hash_name(random.choice(all_names))  # pseudonymized random name


def clear_tree(tree_path: str):
    if os.path.exists(tree_path):
        try:
            shutil.rmtree(tree_path)
            os.mkdir(tree_path)
            print(f"Directory '{
                  tree_path}' and its contents removed successfully.")
        except OSError as e:
            print(f"Error: Could not remove directory '{tree_path}': {e}")
    else:
        print(f"Directory '{tree_path}' does not exist.")


def clear_folder(folder_path: str):
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory.")
        return

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Removed file: {item_path}")


def rename_file(old_file: str, new_file: str):
    try:
        os.rename(old_file, new_file)
    except FileNotFoundError:
        print(f"Error: File '{old_file}' not found.")
    except PermissionError:
        print(f"Error: Permission denied. Unable to rename '{
              old_file}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def regex_bus(filepath: str, replace: dict, regex: str):
    try:
        # Read the entire content of the file
        with open(filepath, 'r') as file:
            file_content = file.read()

        # Function to replace found values with dict values
        def replace_values(match):
            base = match.group(0)
            value = match.group(1)
            return f"{base}{replace[value]}"

        # Perform replacement in file content
        updated_content = re.sub(regex, replace_values, file_content)
        with open(filepath, "w") as file:
            file.write(updated_content)

    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


def regex_xfmr(filepath: str, replace: dict, regex: str):
    try:
        # Read the entire content of the file
        with open(filepath, 'r') as file:
            file_content = file.read()

        # Function to replace found values with dict values
        def replace_values(match):
            values = match.group(1).split(",")
            replaced_values = [
                replace.get(value.strip(), value.strip()) for value in values
            ]
            return "Buses=[" + ", ".join(replaced_values) + "]"

        # Perform replacement in file content
        updated_content = re.sub(regex, replace_values, file_content)
        with open(filepath, "w") as file:
            file.write(updated_content)

    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


def replace_in_file(filepath: str, replace: dict):
    try:
        # Read the entire content of the file
        with open(filepath, 'r') as file:
            file_content = file.read()

        # Perform the replacement
        for k, v in replace.items():
            file_content = file_content.replace(k, v)

        # Write the modified content back to the file
        with open(filepath, 'w') as file:
            file.write(file_content)

    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


def anon_xfmrs(replace: dict) -> dict:
    for name in dss.Transformers.AllNames():
        dss.Transformers.Name(name)
        n = dss.Transformers.NumWindings()
        for i in range(1, n+1):
            dss.Transformers.Wdg(i)
            kva = dss.Transformers.kVA()
            dss.Transformers.kVA(add_noise(kva))
        old_name = f"Transformer.{name}"
        new_name = f"Transformer.{dp_randomized_response(
            name, dss.Transformers.AllNames())}"
        replace[old_name] = new_name
        replace[old_name.replace(".", "=")] = new_name.replace(".", "=")
    return replace


def anon_regctrl(replace: dict) -> dict:
    for name in dss.RegControls.AllNames():
        old_name = f"RegControl.{name}"
        new_name = f"RegControl.{dp_randomized_response(
            name, dss.Transformers.AllNames())}"
        replace[old_name] = new_name
    return replace


def anon_caps(replace: dict) -> dict:
    for name in dss.Capacitors.AllNames():
        dss.Capacitors.Name(name)
        kvar = dss.Capacitors.kvar()
        print(name, kvar)
        dss.Capacitors.kvar(add_noise(kvar))
        old_name = f"Capacitor.{name}"
        new_name = f"Capacitor.{dp_randomized_response(
            name, dss.Capacitors.AllNames())}"
        replace[old_name] = new_name
    return replace


def anon_pvs(replace: dict) -> dict:
    for name in dss.PVsystems.AllNames():
        dss.PVsystems.Name(name)
        kva = dss.PVsystems.kVARated()
        dss.PVsystems.kVARated(add_noise(kva))
        pmpp = dss.PVsystems.Pmpp()
        dss.PVsystems.Pmpp(add_noise(pmpp))
        old_name = f"PVsystem.{name}"
        new_name = f"PVsystem.{dp_randomized_response(
            name, dss.PVsystems.AllNames())}"
        replace[old_name] = new_name
    return replace


def anon_loads(replace: dict) -> dict:
    for name in dss.Loads.AllNames():
        dss.Loads.Name(name)
        kw = dss.Loads.kW()
        dss.Loads.kW(add_noise(kw))
        kvar = dss.Loads.kvar()
        dss.Loads.kvar(add_noise(kvar))
        old_name = f"Load.{name}"
        new_name = f"Load.{dp_randomized_response(name, dss.Loads.AllNames())}"
        replace[old_name] = new_name
    return replace


def anon_buses(replace: dict) -> dict:
    for name in dss.Circuit.AllBusNames():
        anon = dp_randomized_response(name, dss.Circuit.AllBusNames())
        replace[name] = anon
    return replace


def anon_lines(replace: dict) -> dict:
    for name in dss.Lines.AllNames():
        old_name = f"Line.{name}"
        new_name = f"Line.{dp_randomized_response(
            name, dss.Lines.AllNames())}"
        replace[old_name] = new_name
    return replace


def anon_loadshapes(replace: dict) -> dict:
    cnt = 0
    for name in dss.LoadShape.AllNames():
        old_name = f"Loadshape.{name}"
        new_name = f"Loadshape.profile_{cnt}"
        cnt += 1

        _, anon = new_name.split(".", 1)
        replace[old_name] = new_name
        replace[f"Yearly={name}"] = f"Yearly={anon}"
    return replace


def replace_all(path: str, replace: dict):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            replace_in_file(item_path, replace)


def regex_buses(path: str, replace: dict, regex: str):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            regex_bus(item_path, replace, regex)


def overwrite_shapes(path: str, rename: dict):
    with open(f"{path}/LoadShape.dss", 'w') as file:
        for name in dss.LoadShape.AllNames():
            if "Default" == name:
                continue

            dss.LoadShape.Name(name)
            obj, name = rename[f"Loadshape.{name}"].split(".", 1)

            profile = f"../profiles/{name}.csv"
            npts = dss.LoadShape.Npts()
            hr = dss.LoadShape.HrInterval()

            with open(profile, 'w', newline='') as f:
                writer = csv.writer(f)

                # Write multiple rows at once
                values = [[v] for v in dss.LoadShape.PMult()]
                writer.writerows(values)

            mult = f"(file={profile})"
            file.write(
                f"New Loadshape.{name} npts={npts} interval={hr} mult={mult}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Anonymize OpenDSS data.")
    parser.add_argument(
        "--data", help="path to folder that containes the models opendss folder e.g ./private_data")

    # Load OpenDSS
    args = parser.parse_args()
    dss_folder = init_paths(args)
    init_opendss(dss_folder)
    exit()

    # reload model to make sure names align
    anon_names = anon_loadshapes({})
    anon_names = anon_caps(anon_names)
    anon_names = anon_loads(anon_names)
    anon_names = anon_lines(anon_names)
    anon_names = anon_pvs(anon_names)
    anon_names = anon_xfmrs(anon_names)
    anon_names = anon_regctrl(anon_names)

    # remove old files and save clean OpenDSS formatted files
    clear_folder(dss_folder)
    dss.Circuit.Save(dss_folder, 0)

    # save updated kva, kw, kvar values and then rename
    replace_all(dss_folder, anon_names)

    # move all profiles into a single folder and overwrite
    clear_tree(f"{args.data}/profiles")
    overwrite_shapes(dss_folder, anon_names)

    # finally regex replace any buses in array format
    buses = anon_buses({})
    regex = r"Bus+\d=(.*?)(?=\s|\.)"
    regex_buses(dss_folder, buses, regex)

    phase_buses = {}
    for k, v in buses.items():
        phase_buses[k] = v
        phase_buses[f"{k}.1"] = f"{v}.1"
        phase_buses[f"{k}.2"] = f"{v}.2"
        phase_buses[f"{k}.3"] = f"{v}.3"
    regex = r"Buses=\[([^\]]+)\]"
    xfmr_file = f"{dss_folder}/Transformer.dss"
    regex_xfmr(xfmr_file, phase_buses, regex)
