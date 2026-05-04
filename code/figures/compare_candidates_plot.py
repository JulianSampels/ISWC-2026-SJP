import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib for PGF/Sans-Serif to match thesis style
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'text.usetex': False,
    'pgf.rcfonts': False,
    'font.size': 11,
})

def load_metrics_csv(path):
    df = pd.read_csv(path)
    metric_map = {row["metric"]: float(row["value"]) for _, row in df.iterrows()}
    return metric_map

def collect_results(results_root, method_key):
    rows = []
    for name in sorted(os.listdir(results_root), key=lambda x: int(x) if x.isdigit() else float('inf')):
        run_dir = os.path.join(results_root, name)
        if not os.path.isdir(run_dir):
            continue
        metrics_path = os.path.join(run_dir, f"{method_key}_candidates_metrics.csv")
        if not os.path.isfile(metrics_path):
            continue
        m = load_metrics_csv(metrics_path)
        rows.append({
            "run": name,
            "total_size": m["total candidate size"],
            "avg_size_head": m["average candidate size (head)"],
            "norm_size_triple": m["normalised candidate size (triple)"],
            "coverage_macro": m["coverage_macro"],
            "density_macro": m["density_macro"],
            "n_triples": m["n_triples"],
            "n_heads": m["n_heads"]
        })
    if not rows:
        raise FileNotFoundError(f"No metrics found for {method_key} under {results_root}")
    df = pd.DataFrame(rows).sort_values("total_size")
    return df

def plot_coverage_density(sjp_df, reta_df, outfile="coverage_density.pdf"):
    # Use the actual column values from the dataframe to determine the scale factors
    scale_avg_to_total = sjp_df["total_size"].iloc[-1] / sjp_df["avg_size_head"].iloc[-1]
    scale_norm_to_total = sjp_df["total_size"].iloc[-1] / sjp_df["norm_size_triple"].iloc[-1]

    # Setup figure
    WIDTH = 418.25555 / 72.27
    fig, ax_cov = plt.subplots(figsize=(WIDTH, WIDTH * 1.0))  # Made plot higher

    # Base x-axis will be total_size
    # Coverage (left axis)
    ax_cov.plot(sjp_df["total_size"], sjp_df["coverage_macro"],
                marker="o", markersize=4, linestyle="-", label="SJP Coverage", color='seagreen')
    ax_cov.plot(reta_df["total_size"], reta_df["coverage_macro"],
                marker="s", markersize=4, linestyle="--", label="RETA Coverage", color='seagreen')
    ax_cov.set_xlabel("Total Candidate Size")
    ax_cov.set_ylabel("Coverage (Macro)", color='seagreen')
    ax_cov.tick_params(axis='y', labelcolor='seagreen')
    ax_cov.set_ylim(0, 1)
    ax_cov.set_xlim(left=0)  # Prevent negative x-axis
    ax_cov.grid(True, alpha=0.3)

    # Density (right axis)
    ax_den = ax_cov.twinx()
    ax_den.plot(sjp_df["total_size"], sjp_df["density_macro"],
                marker="^", markersize=4, linestyle="-", label="SJP Density", color='dodgerblue')
    ax_den.plot(reta_df["total_size"], reta_df["density_macro"],
                marker="v", markersize=4, linestyle="--", label="RETA Density", color='dodgerblue')
    ax_den.set_ylabel("Density (Macro)", color='dodgerblue')
    ax_den.tick_params(axis='y', labelcolor='dodgerblue')

    # Bottom axes adjustments
    
    # 1st bottom axis: Total (Primary)
    ax_cov.xaxis.set_label_position("bottom")
    ax_cov.xaxis.set_ticks_position("bottom")
    ax_cov.spines['bottom'].set_position(('outward', 0))

    # 2nd bottom axis: Average Size per Head
    ax_avg = ax_cov.twiny()
    ax_avg.set_xlabel("Average Candidate Size (Head)")
    ax_avg.set_xlim(ax_cov.get_xlim())
    ticks = ax_cov.get_xticks()
    ax_avg.set_xticks(ticks)
    # Calculate avg size from total size
    ax_avg.set_xticklabels([f"{t / scale_avg_to_total:.1f}" for t in ticks])
    ax_avg.xaxis.set_ticks_position('bottom')
    ax_avg.xaxis.set_label_position('bottom')
    ax_avg.spines['bottom'].set_position(('outward', 40))
    ax_avg.spines['top'].set_visible(False)

    # 3rd bottom axis: Normalized Size per Triple
    ax_norm = ax_cov.twiny()
    ax_norm.set_xlabel("Normalized Candidate Size (Triple)")
    ax_norm.set_xlim(ax_cov.get_xlim())
    ax_norm.set_xticks(ticks)
    # Calculate norm size from total size
    ax_norm.set_xticklabels([f"{t / scale_norm_to_total:.1f}" for t in ticks])
    ax_norm.xaxis.set_ticks_position('bottom')
    ax_norm.xaxis.set_label_position('bottom')
    ax_norm.spines['bottom'].set_position(('outward', 80))
    ax_norm.spines['top'].set_visible(False)

    # Combine legends into one box
    lines_cov, labels_cov = ax_cov.get_legend_handles_labels()
    lines_den, labels_den = ax_den.get_legend_handles_labels()
    ax_cov.legend(lines_cov + lines_den, labels_cov + labels_den, loc='center right')

    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight", pad_inches=0.01)
    plt.close()
    print(f"Saved {outfile}")

if __name__ == "__main__":
    results_root = "../results"  # Adjust based on where you run this from
    try:
        sjp_df = collect_results(results_root, "sjp")
        reta_df = collect_results(results_root, "reta")
        
        out_dir = "./pdf_plots"
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "coverage_density.pdf")
        
        plot_coverage_density(sjp_df, reta_df, outfile=out_file)
    except Exception as e:
        print(f"Error: {e}")
