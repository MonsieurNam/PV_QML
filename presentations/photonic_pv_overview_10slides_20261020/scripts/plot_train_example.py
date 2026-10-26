"""Render measured training observations with CNSPlots; no fitted forecasts."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import pandas as pd
import cnsplots as cns

project = Path(__file__).resolve().parents[1]
data = pd.read_csv(project / "sources/nist_train_example.csv")
assert len(data) == 25 and data.minutes_from_issue.min() == -60 and data.minutes_from_issue.max() == 60
out = project / "images"
out.mkdir(exist_ok=True)
with cns.settings.context(figure_dpi=144, savefig_dpi=144, savefig_bbox="tight", savefig_transparent=False, font_sans_serif=("Arial", "DejaVu Sans")):
    cns.figure(width=1800, height=650)
    fig = plt.gcf()
    fig.set_facecolor("#14263D")
    ax = plt.gca()
    ax.set_facecolor("#14263D")
    cns.lineplot(data=data.loc[data.minutes_from_issue.le(0)], x="minutes_from_issue", y="ac_power_kw", estimator=None, errorbar=None, ax=ax, color="#80C7C2", linewidth=2.8, marker="o", markersize=4, label="Available history")
    cns.lineplot(data=data.loc[data.minutes_from_issue.ge(0)], x="minutes_from_issue", y="ac_power_kw", estimator=None, errorbar=None, ax=ax, color="#E7EDF4", linewidth=2.2, linestyle="--", label="Later observations")
    ax.axvline(0, color="#80C7C2", linewidth=1.5, linestyle=":")
    ax.axvspan(10, 15, color="#E1BA82", alpha=0.3)
    target = data.loc[data.minutes_from_issue.eq(15), "ac_power_kw"].item()
    ax.plot([15], [target], marker="o", color="#E1BA82", markersize=9)
    ax.annotate("Target bin\n12:10–12:15", xy=(12.5, 61), xytext=(29, 68), fontsize=30, color="#E1BA82", ha="left", va="center", arrowprops={"arrowstyle":"-", "color":"#E1BA82", "linewidth":1.3})
    ax.text(-2, 69, "Issue\n12:00", color="#80C7C2", fontsize=30, ha="right", va="center")
    ax.set_xlim(-60, 60)
    ax.set_ylim(0, 75)
    ax.xaxis.set_major_locator(FixedLocator([-60, -30, 0, 30, 60]))
    ax.xaxis.set_major_formatter(FixedFormatter(["11:00", "11:30", "12:00", "12:30", "13:00"]))
    ax.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75]))
    ax.set_xlabel("1 June 2015 · fixed EST (UTC−05:00)", fontsize=30, color="#E7EDF4", labelpad=12)
    ax.set_ylabel("AC power (kW)", fontsize=32, color="#E7EDF4", labelpad=12)
    ax.tick_params(axis="both", colors="#E7EDF4", labelsize=30, pad=7)
    for spine in ax.spines.values():
        spine.set_color("#E7EDF4")
    ax.grid(axis="y", color="#E7EDF4", alpha=0.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.26), ncol=2, fontsize=30, labelcolor="#E7EDF4", handlelength=2.4, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.23, top=0.79)
    cns.savefig(out / "nist_train_example.png")
    cns.savefig(out / "nist_train_example.svg")
    plt.close(fig)
print("Measured training example exported: PNG + editable SVG; no forecast; final dimensions follow tight-bbox export.")
