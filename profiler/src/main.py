import os
import sys
import tsgm
import numpy as np
import pandas as pd


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
    if len(sys.argv) != 2:
        print("Need model name")
        print("example: python main.py ieee123")
        exit()

    model = sys.argv[1]
    data_dir = os.path.abspath(f"./features/{model}")
    output_dir = init_paths(data_dir, model)

    profiles = os.listdir(data_dir)

    for i in len(profiles):
        print("Generating: ", profiles[i])
        df = pd.read_csv(f"{data_dir}/{profiles[i]}", index_col=0)
        real = df.to_numpy()

        n_seq = 24
        window = 7*n_seq
        real = real[:window-1]
        data = prepair_data(real, n_seq)

        print("real: ", real.shape)
        print("data: ", data.shape)
        _, _, n_feat = data.shape

        # scaler = tsgm.utils.TSFeatureWiseScaler()
        # scaled_data = scaler.fit_transform(data)

        model = tsgm.models.timeGAN.TimeGAN(
            seq_len=n_seq,
            n_features=n_feat,
            hidden_dim=4*n_feat,
        )
        model.compile()

        model.fit(
            data=data,
            epochs=1000,
        )

        synth = model.generate(len(data))
        np.save(f"{output_dir}/{profiles[i]}.npy", synth)
