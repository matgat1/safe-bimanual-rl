import os
import math
import matplotlib.pyplot as plt


def save_plots(metrics: dict, save_dir: str, run_name: str):
    """


    Args:
        metrics (dict): _description_
        save_dir (str): _description_
        run_name (str): _description_
    """
    n = len(metrics)
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    epochs = None

    fig, axes = plt.subplots(rows, cols)
    axes = [axes] if n == 1 else list(axes.flat)

    for ax, (label, values) in zip(axes, metrics.items()):
        if epochs is None:
            epochs = list(range(len(values)))
        ax.plot(epochs, values)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(True)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(run_name)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{run_name}_plots.png"))
    plt.close(fig)
