import os
import matplotlib.pyplot as plt


def save_plots(J_values: list, R_values: list, save_dir: str, run_name: str) -> None:
    epochs = list(range(len(J_values)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, J_values)
    axes[0].set_title("Discounted Return (J)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("J")
    axes[0].grid(True)

    axes[1].plot(epochs, R_values)
    axes[1].set_title("Undiscounted Return (R)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("R")
    axes[1].grid(True)

    fig.suptitle(run_name)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{run_name}_plots.png"))
    plt.close(fig)
