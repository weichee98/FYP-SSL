import os
import sys
import matplotlib.pyplot as plt

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

import ABIDE
import ADHD


def plot_ABIDE_num_subjects(output_path):
    df = ABIDE.load_meta_df()
    df["DX_GROUP"] = df["DX_GROUP"].apply(lambda x: "Diseased" if x == 1 else "Control")
    num_subjects = df.groupby(["SITE_ID", "DX_GROUP"]).size().unstack()

    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    num_subjects.plot.bar(stacked=True, ax=ax, rot=45)
    ax.set_ylabel("Number of Subjects")
    ax.set_xlabel("Sites")
    
    ax.set_axisbelow(True)
    plt.grid(True, axis='y', linewidth=1., linestyle="dashed")
    plt.tight_layout()
    f.savefig(output_path)


def plot_ADHD_num_subjects(output_path):
    df = ADHD.load_meta_df()
    dx = [
        "Typically Developing Children (TDC)",
        "ADHD-Combined",
        "ADHD-Hyperactive/Impulsive",
        "ADHD-Inattentive"
    ]
    df["DX"] = df["DX"].apply(lambda x: dx[x])
    num_subjects = df.groupby(["SITE_NAME", "DX"]).size().unstack()

    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    num_subjects.plot.bar(stacked=True, ax=ax, rot=45, legend=True)
    ax.set_ylabel("Number of Subjects")
    ax.set_xlabel("Sites")
    
    ax.set_axisbelow(True)
    plt.grid(True, axis='y', linewidth=1., linestyle="dashed")
    plt.legend(title="DX", bbox_to_anchor=(0.57, 1), loc="upper center")
    plt.tight_layout()
    f.savefig(output_path)


if __name__ == "__main__":
    # plot_ABIDE_num_subjects("num_subjects_ABIDE.png")
    plot_ADHD_num_subjects("num_subjects_ADHD.png")