import matplotlib.pyplot as plt
import seaborn as sns

def grain_area_histogram(data, filename, output_dir):
    with plt.ioff():
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data, bins='auto', kde=True, log_scale=True, color='skyblue', edgecolor='black', ax=ax)
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        ax.set_title('Grain areas nm²')
        plt.tight_layout()
        fig.savefig(output_dir / f"{filename}_grain_areas_hist.png", dpi=300)
        plt.close(fig)


def grain_circularity_histogram(data, filename, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data, bins='auto', kde=True, color='skyblue', edgecolor='black', ax=ax)
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_title('Grain circularities (0-1)')
    plt.tight_layout()
    fig.savefig(output_dir / f"{filename}_grain_circularity_hist.png", dpi=300)
    plt.close(fig)
