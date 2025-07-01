import os
import sys
import tsgm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "lines.linewidth": 1,
})


def init_paths(data: str, output: str, model: str) -> str:
    if not os.path.exists(data):
        print("features location doesn't exist")
        exit()

    if not os.path.exists(output):
        print("output location doesn't exist")
        exit()

    metrics = os.path.abspath(f"./metrics/{model}")
    os.makedirs(metrics, exist_ok=True)
    return metrics


def plot_profile(real: pd.DataFrame, synth: pd.DataFrame, output: str) -> None:
    min_per_year = 365*24*60
    min_per_day = 24*60
    interval = int(min_per_year/len(real))
    window = int(min_per_day/interval)
    time = range(window)
    n, m = np.shape(real)
    n2, m2 = np.shape(synth)
    print("Real: ", n, m)
    print("Synth: ", n2, m2)

    cols = 2
    fig, ax = plt.subplots(
        layout="constrained",
        figsize=(8, 3),
        ncols=cols,
        nrows=1,
        sharey=True,
    )
    ax[0].grid()
    ax[1].grid()

    for j in range(m):
        ax[0].plot(range(96), real.iloc[:96, j])
        ax[1].plot(range(96), synth.iloc[:96, j])
    fig.tight_layout()
    plt.savefig(f"{output}/compare_profiles.png", dpi=400)
    plt.close()


def plot_tsne(real: pd.DataFrame, synth: pd.DataFrame, output: str) -> None:
    min_per_year = 365*24*60
    min_per_day = 24*60
    interval = int(min_per_year/len(real))
    window = int(min_per_day/interval)

    real = real.to_numpy()
    synth = synth.to_numpy()

    real = np.array(np.split(real, len(real)/window))
    synth = np.array(np.split(synth, len(synth)/window))

    scaler = tsgm.utils.TSFeatureWiseScaler()
    norm_r = scaler.fit_transform(real)
    norm_s = scaler.fit_transform(synth)

    path = f"{output}/tsne.pdf"
    tsgm.utils.visualize_tsne_unlabeled(
        norm_r, norm_s, path=path, perplexity=10, markersize=100, alpha=0.5)


def print_metrics(real: pd.DataFrame, synth: pd.DataFrame) -> None:
    min_per_year = 365*24*60
    min_per_day = 24*60
    interval = int(min_per_year/len(real))
    window = int(min_per_day/interval)

    real = real.to_numpy()
    synth = synth.to_numpy()

    real = np.array(np.split(real, len(real)/window))
    synth = np.array(np.split(synth, len(synth)/window))

    scaler = tsgm.utils.TSFeatureWiseScaler()
    norm_r = scaler.fit_transform(real)
    norm_s = scaler.fit_transform(synth)

    mmd_metric = tsgm.metrics.MMDMetric()
    print(mmd_metric(norm_r, norm_s))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Need path to input model and output name")
        print("example: python feature_extraction.py ./data/ieee123/opendss ieee123")
        exit()

    model = sys.argv[1]
    features_dir = os.path.abspath(f"./features/{model}")
    output_dir = os.path.abspath(f"./output/{model}")
    metrics_dir = init_paths(features_dir, output_dir, model)

    features = os.listdir(features_dir)
    output = os.listdir(output_dir)
    print(features, output)

    i_feat = len(features)-1
    i_out = len(output)-1
    real = pd.read_csv(f"{features_dir}/{features[i_feat]}", index_col=0)
    synth = pd.read_csv(f"{output_dir}/{output[i_out]}", index_col=0)
    plot_profile(real, synth, metrics_dir)
    plot_tsne(real, synth, metrics_dir)
    print_metrics(real, synth)
