# src/utils.py
import matplotlib.pyplot as plt
import os


def plot_training_history(history, fine_history=None, save_path=None):
    """Plot training history"""
    if fine_history:
        # Combine histories
        combined_history = {}
        for key in history.history.keys():
            if key in fine_history.history:
                combined_history[key] = history.history[key] + \
                    fine_history.history[key]
            else:
                combined_history[key] = history.history[key]
    else:
        combined_history = history.history

    # Determine what metrics we have
    available_metrics = list(combined_history.keys())

    # Create appropriate subplot grid
    num_metrics = len(
        [m for m in available_metrics if not m.startswith('val_')])
    num_cols = 2
    num_rows = (num_metrics + 1) // 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    if num_rows == 1:
        axes = [axes] if num_cols == 1 else axes

    metric_idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if metric_idx < len(available_metrics):
                metric = available_metrics[metric_idx]
                if not metric.startswith('val_'):
                    ax = axes[i][j] if num_rows > 1 else axes[j]

                    # Plot training metric
                    ax.plot(combined_history[metric],
                            label=f'Training {metric}')

                    # Plot validation metric if available
                    val_metric = f'val_{metric}'
                    if val_metric in combined_history:
                        ax.plot(combined_history[val_metric],
                                label=f'Validation {metric}')

                    ax.set_title(f'Model {metric.title()}')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel(metric.title())
                    ax.legend()

                    metric_idx += 1

    # Hide empty subplots
    for i in range(metric_idx, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        if num_rows > 1:
            axes[row][col].set_visible(False)
        else:
            axes[col].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
