#!/usr/bin/env python3

# The code here is tailored for reproducing the figures in the article.
# See the --help output for instructions.

import argparse
import io
import math
import numpy as np
import pandas as pd
import tomli
import matplotlib.pyplot as plt

START_DATA = 6  # first data column in the csv (earlier entries are table spec)
HAVI_KEYS = 71053459  # number of keys in the havi dataset

class Style:
    # BODY_WIDTH roughly reflects the line width in our paper
    BODY_WIDTH = 4.80143904801 # in
    LINESTYLES = { 16: ":", 32: "--", 64: "-" }
    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def row(self, r):
        return {
            "label": str(r.pbs) + "x" + str(r.rw),
            "color": self.COLORS[self.BS[r.table].index((r.pbs, r.sbs))],
            "linestyle": self.LINESTYLES[r.rw],
        }

    def __init__(self, data):
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 7,
            "figure.figsize": (self.BODY_WIDTH, self.BODY_WIDTH * 2/3),
            "figure.titlesize": "large",
            "figure.labelsize": "large",
            "text.usetex": True,
            })

        def df_tups(df):
            return list(df.itertuples(index=False, name=None))

        _BS = data[["table", "pbs", "sbs"]].drop_duplicates()
        self.BS = {
            "cuckoo": df_tups(_BS[_BS.table == "cuckoo"][["pbs", "sbs"]]),
            "iceberg": df_tups(_BS[_BS.table == "iceberg"][["pbs", "sbs"]])
        }

class Config:
    def __init__(self, frontmatter, data, kind):
        self.fill_ratios = np.array([float(x) for x in data.columns.values[START_DATA:]])
        self.pos_ratios = sorted(
                [r for r in pd.unique(data.positive_ratio) if not math.isnan(r)]
        )
        self.style = Style(data)

        # Havi has no front matter (the table is always the same size)
        if kind == "havi":
            return

        self.p_log_entries = frontmatter["p_log_entries"]
        self.s_log_entries = frontmatter["s_log_entries"]
        self.entries = {"cuckoo" : 2**self.p_log_entries,
                        "iceberg": 2**self.p_log_entries + 2**self.s_log_entries}

def init(f, kind):
    s = f.read()
    csv = ""
    conf = {}
    if kind == "rates":
        # Read front matter
        front, csv = s.split("###", maxsplit=1)
        front = "".join([ ln.removeprefix("#") for ln in front.splitlines(True)])
        conf = tomli.loads(front)
    else:
        csv = s

    # Read data
    data = pd.read_csv(io.StringIO(csv), comment="#", skipinitialspace=True)
    if "1.0" not in data.columns:
        data["1.0"] = np.NaN
    data = (data.groupby(["table","operation","positive_ratio","rw","pbs","sbs"],
                         sort=False, dropna=False)
            .mean(numeric_only=True).reset_index())

    return Config(conf, data, kind), data

def putax(conf, data, table, ax):
    putdata = data[(data.operation == "put") & (data.table == table)]
    ax.set_xlim(min(conf.fill_ratios), max(conf.fill_ratios))
    for _, row in putdata.iterrows():
        times = row[START_DATA:].values / 1000 # s
        ax.plot(conf.fill_ratios, conf.fill_ratios * conf.entries[table] / times,
                **conf.style.row(row))

def findax(conf, data, table, axs):
    finddata = data[(data.operation == "find") & (data.table == table)]
    for i, pr in enumerate(conf.pos_ratios):
        axs[i].set_xlim(min(conf.fill_ratios), max(conf.fill_ratios))
        d = finddata[finddata.positive_ratio == pr]
        for _, row in d.iterrows():
            times = row[START_DATA:] / 1000 # s
            ln, = axs[i].plot(conf.fill_ratios,
                             (conf.entries[table] / 2) / times,
                             **conf.style.row(row))
            if i != 0:
                ln.set_label("")

def fopax(conf, data, table, axs):
    fopdata = data[(data.operation == "fop") & (data.table == table)]
    for i, pr in enumerate(conf.pos_ratios[0:3]):
        axs[i].set_xlim(min(conf.fill_ratios), max(conf.fill_ratios))
        d = fopdata[fopdata.positive_ratio == pr]
        for _, row in d.iterrows():
            times = row[START_DATA:] / 1000 # s
            ln, = axs[i].plot(conf.fill_ratios,
                          conf.entries[table] / times, **conf.style.row(row))
            if i != 0:
                ln.set_label("")

def putfig_article(conf, data):
    fig, ax = plt.subplot_mosaic(
        "CI..", sharey=True,layout="constrained",
        figsize=(conf.style.BODY_WIDTH, conf.style.BODY_WIDTH * 1/3))
    putax(conf, data, "cuckoo", ax["C"])
    ax["C"].set_title("Cuckoo")
    putax(conf, data, "iceberg", ax["I"])
    ax["I"].set_title("Iceberg")
    ax["I"].set_ylim(bottom=0)
    fig.supylabel("Insertions (key/s)")
    fig.supxlabel("Final fill factor", x=0.3)

    # The put figure contains the legend, which is the same for all other figs
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               ncol=3, title="(Primary) bucket size x slot size", loc="right")
    return fig

def findfig_article(conf, data):
    fig, ax = plt.subplot_mosaic(
            "abcd\nABCD", sharey=True,layout="constrained",
            figsize=(conf.style.BODY_WIDTH, conf.style.BODY_WIDTH * 1/3 * 1.7))
    cax = [ax["a"], ax["b"], ax["c"], ax["d"]]
    iax = [ax["A"], ax["B"], ax["C"], ax["D"]]
    findax(conf, data, "cuckoo", cax)
    findax(conf, data, "iceberg", iax)
    cax[0].set_ylabel("Cuckoo")
    iax[0].set_ylabel("Iceberg")
    cax[0].set_ylim(bottom=0)
    iax[0].set_ylim(bottom=0)
    fig.suptitle("Positive query ratio")
    fig.supxlabel("Fill factor")
    fig.supylabel("Lookups (key/s)")
    for i, a in enumerate(cax):
        a.set_title(str(conf.pos_ratios[i]))
    return fig

def fopfig_article(conf, data):
    fig, ax = plt.subplot_mosaic(
        "abc.\nABC.", layout="constrained",
        figsize=(conf.style.BODY_WIDTH, conf.style.BODY_WIDTH * 1/3 * 1.7))
    cax = [ax["a"], ax["b"], ax["c"]]
    iax = [ax["A"], ax["B"], ax["C"]]
    fopax(conf, data, "cuckoo", cax)
    fopax(conf, data, "iceberg", iax)
    cax[0].set_ylabel("Cuckoo")
    iax[0].set_ylabel("Iceberg")

    for i in range(1,3):
        cax[i].sharey(cax[0])
        iax[i].sharey(iax[0])
        cax[i].tick_params(labelleft=False)
        iax[i].tick_params(labelleft=False)

    for a in ax.values():
        a.ticklabel_format(axis="y", scilimits = (9,9))

    fig.suptitle("Before fill factor")
    fig.supxlabel("After fill factor")
    fig.supylabel("Find-or-puts (key/s)")
    for i, a in enumerate(cax):
        a.set_title(str(conf.pos_ratios[i]))
    return fig

def haviax(conf, data, table, ax):
    havi = data[(data.operation == "havi") & (data.table == table)]
    ax.set_xlim(min(conf.fill_ratios), max(conf.fill_ratios))
    for _, row in havi.iterrows():
        times = row[START_DATA:].values / 1000 # s
        ax.plot(conf.fill_ratios, conf.fill_ratios * HAVI_KEYS / times,
                     **conf.style.row(row))

def fopfig_havi(conf, data):
    fig, ax = plt.subplot_mosaic(
            "CI..",layout="constrained",
            figsize=(conf.style.BODY_WIDTH, conf.style.BODY_WIDTH * 1/3))
    haviax(conf, data, "cuckoo", ax["C"])
    ax["C"].set_title("Cuckoo")
    haviax(conf, data, "iceberg", ax["I"])
    ax["I"].set_title("Iceberg")
    fig.supylabel("Throughput (key/s)")
    fig.supxlabel("Processed ratio", x=0.3)
    for a in ax.values():
        a.ticklabel_format(axis="y", scilimits = (9,9))
    return fig

ap = argparse.ArgumentParser(
        description="""\
Create benchmark figures

Generates figures used in the article.
Use --use-tex to make prettier figures using LaTeX (requires LaTeX)\
""")

ap.add_argument("--kind",
                choices=["rates", "havi"],
                default="rates",
                help="kind of benchmark data to analyze (default: rates)")
ap.add_argument("data",
                type=argparse.FileType("r"),
                help="output file to analyze")
ap.add_argument("-t", "--file-type",
                default="pdf",
                help="file type for figures (as file extension)")
ap.add_argument("--use-tex",
                action="store_true",
                help="use LaTeX for generating figure text")
args = ap.parse_args()

conf, data = init(args.data, args.kind)
plt.rcParams.update({"text.usetex": args.use_tex})

if args.kind == "rates":
    putfig_article(conf, data).savefig("put." + args.file_type)
    findfig_article(conf, data).savefig("find." + args.file_type)
    fopfig_article(conf, data).savefig("fop." + args.file_type)
elif args.kind == "havi":
    fopfig_havi(conf, data).savefig("havi." + args.file_type)
