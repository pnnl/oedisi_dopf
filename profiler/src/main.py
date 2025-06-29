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

    n_seq = 24
    data = prepair_data(real[:95], n_seq)
    _, _, n_feat = data.shape
    print("real: ", real.shape)
    print("data: ", data.shape)

    scaler = tsgm.utils.TSFeatureWiseScaler()
    scaled_data = scaler.fit_transform(data)

    model = tsgm.models.timeGAN.TimeGAN(
        seq_len=n_seq,
        n_features=n_feat,
        hidden_dim=4*n_feat,
    )
    model.compile()

    model.fit(
        data=scaled_data,
        epochs=10,
    )

    synth = model.generate(96)
    np.save(f"{output_dir}/synth_raw.npy", synth)
    exit()

    synthetic = np.reshape(synthetic, (real.shape))
    new_profile = scaler.inverse_transform(synthetic)
    df = pd.DataFrame(new_profile, columns=df.columns)
    df.to_csv(f"{output_dir}/{profiles[i]}")
