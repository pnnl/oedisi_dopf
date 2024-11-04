import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_data(path: str) -> np.ndarray:
    data = pd.read_csv(path, header=None)
    return data.astype(float).to_numpy()


def get_complex_data(path: str) -> np.ndarray:
    data = pd.read_csv(path, header=None).apply(
        lambda s: s.str.replace("i", "j"))
    return data.astype(complex).to_numpy()


if __name__ == "__main__":
    cwd = os.getcwd()
    init = f"{cwd}/builds/ekf/SFO/P6U/low/ekf/output/Results/init/"
    res = f"{cwd}/builds/ekf/SFO/P6U/low/ekf/output/Results/1/"
    ypu_file = os.path.abspath(f"{init}/Ypu.csv")
    f_file = os.path.abspath(f"{init}/F.csv")
    j_file = os.path.abspath(f"{res}/J.csv")
    s1_file = os.path.abspath(f"{res}/S1.csv")
    s2_file = os.path.abspath(f"{res}/S2.csv")
    s3_file = os.path.abspath(f"{res}/S3.csv")
    p_file = os.path.abspath(f"{res}/P.csv")
    ppre_file = os.path.abspath(f"{res}/Ppre.csv")
    q_file = os.path.abspath(f"{res}/Q.csv")
    r_file = os.path.abspath(f"{res}/R.csv")

    ypu = get_complex_data(ypu_file)
    f = get_data(f_file)
    j = get_data(j_file)
    p = get_data(p_file)
    s1r = get_data(s1_file)
    s2r = get_data(s2_file)
    s3r = get_data(s3_file)
    p = get_data(p_file)
    p = get_data(p_file)
    ppre = get_data(ppre_file)
    q = get_data(q_file)
    r = get_data(r_file)
    f = get_data(f_file)

    s1 = j.T
    val = np.allclose(s1, s1r)
    print("s1 = ", val)
    s2 = ppre@s1
    val = np.allclose(s2, s2r)
    print("s2 = ", val)
    s3 = j@s2
    val = np.allclose(s3, s3r)
    print("s3 = ", val)
    supd = s3 + r

    s = j@ppre@j.T + r
    val = np.allclose(s, supd)
    print("s = ", val)
    print("rank = ", np.linalg.pinv(s))

    print(s.diagonal())
    sd = np.nonzero(s.diagonal())
    print("zeros: ", len(sd[0]))

    y = s.diagonal()
    y = y/y
    x = list(range(len(y)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(s, mode="markers"))
    fig.update_layout()
    fig.show()
