import pandas as pd
import matplotlib.pyplot as plt


def _plot(A, legend, scale="linear"):
    fig, ax1 = plt.subplots(1)
    ax1.plot(
        range(1, len(A) + 1),
        A,
        color="green",
        marker="o",
        linestyle="dashed",
        linewidth=2,
        markersize=12,
    )
    ax1.legend([legend])
    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.tick_params(axis="x", labelsize=12)
    ax1.tick_params(axis="y", labelsize=12)

    # ax1.set_ylim(bottom = -1e4, top =1e5 ,auto=False)

    # plt.xlim([1,len(A)+1])
    # plt.ylim([-2e5, 1e4])
    try:
        ax1.set_yscale(scale)
        plt.show()
    except:
        print("Please insert correct scale string")
        exit()


df = pd.read_csv("pareto_front.csv", header=0, index_col=0)

cost_p = df["pcost"]
feas_gap = df["gap"]
pr_res = df["pres"]

_plot(-cost_p * 1e5, legend="Objective Value (kvar)")
_plot(feas_gap, legend="Feasibility Gap", scale="log")
_plot(pr_res, legend="Primal Residual", scale="log")


# print(df)
