import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def init_paths(data: str, model: str) -> str:
    if not os.path.exists(data):
        print("data location doesn't exist")
        exit()

    output = os.path.abspath(f"./output/{model}")
    os.makedirs(output, exist_ok=True)
    return output


def prepair_data(data: np.array, seq_len):
    split = []
    for i in range(0, len(data) - seq_len):
        rolling_window = data[i:i + seq_len]
        split.append(rolling_window)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(split))
    data = []
    for i in range(len(split)):
        data.append(split[idx[i]])

    return np.asarray(data)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Need path to input model and output name")
        print("example: python main.py ./features/ieee123 ieee123")
        exit()

    data_dir = os.path.abspath(sys.argv[1])
    model = sys.argv[2]
    output_dir = init_paths(data_dir, model)

    profiles = os.listdir(data_dir)
    print(profiles)

    i = len(profiles)-1
    df = pd.read_csv(f"{data_dir}/{profiles[i]}", index_col=0)
    real = df.to_numpy()
    day = 96
    window = 7*day
    real = real[:window-1]
    n_seq = 24
    data = prepair_data(real, n_seq)
    print("real: ", real.shape)
    print("data: ", data.shape)

    synth = np.load(f"{output_dir}/synth_raw.npy")
    print("synth: ", synth.shape)

    fig, ax = plt.subplots(
        layout="constrained",
        figsize=(8, 3),
    )
    ax.grid()
    time = range(n_seq)
    obs = np.random.randint(len(synth))

    ax.plot(time, data[obs], '--', label=df.columns)
    ax.plot(time, synth[obs], 'o-', label=df.columns)
    ax.legend()
    plt.savefig(f"{output_dir}/compare_profiles.png", dpi=400)
    plt.close()
