import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wandb
from manylatents.callbacks.embedding.plot_embeddings import PlotEmbeddings
from manylatents.popgen.data.manifold_genetics_dataset import ManifoldGeneticsDataset

logger = logging.getLogger(__name__)


class PlotAdmixture(PlotEmbeddings):
    """
    Callback to plot embeddings colored by admixture proportions.

    Extends PlotEmbeddings to create a subplot grid showing each admixture 
    component colored by its proportion. Only works with PlinkDataset objects 
    that have admixture data loaded.
    """

    def __init__(
        self,
        save_dir: str = "outputs",
        experiment_name: str = "experiment",
        admixture_K: int = 5,
        figsize_per_plot: tuple = (4, 4),
        point_size: float = 4,
        alpha: float = 0.4,
        cmap: str = "seismic",
        legend: bool = False,
    ):
        """
        Args:
            save_dir (str): Directory where the plot will be saved.
            experiment_name (str): Name of the experiment for the filename.
            admixture_K (int): Number of admixture components to plot (max 10).
            figsize_per_plot (tuple): Size of each individual subplot (width, height).
            point_size (float): Size of scatter plot points.
            alpha (float): Transparency of points (0 = transparent, 1 = opaque).
            cmap (str): Matplotlib colormap to use for coloring proportions.
            legend (bool): Whether to show legend (inherited from PlotEmbeddings).
        """
        if admixture_K > 10:
            raise ValueError("admixture_K must be <= 10")

        # Initialize parent with basic settings
        super().__init__(
            save_dir=save_dir,
            experiment_name=experiment_name,
            figsize=figsize_per_plot,
            legend=legend,
            alpha=alpha,
        )
        
        self.admixture_K = admixture_K
        self.figsize_per_plot = figsize_per_plot
        self.point_size = point_size
        self.cmap = cmap

        logger.info(
            f"PlotAdmixture initialized with directory: {self.save_dir}, "
            f"experiment name: {self.experiment_name}, K={admixture_K}"
        )

    def _get_subplot_layout(self, K: int) -> tuple:
        """Determine subplot layout based on number of components."""
        if K <= 5:
            return (1, K)
        else:
            ncols = (K + 1) // 2
            return (2, ncols)

    def _check_admixture_available(self, dataset: any) -> tuple:
        """Check if dataset has admixture data. Returns (bool, DataFrame or None)."""
        if not isinstance(dataset, ManifoldGeneticsDataset):
            logger.warning(
                f"PlotAdmixture skipped: dataset is {type(dataset).__name__}, "
                "not ManifoldGeneticsDataset"
            )
            return False, None

        if not hasattr(dataset, 'admixture_ratios') or dataset.admixture_ratios is None:
            logger.warning("PlotAdmixture skipped: dataset.admixture_ratios is None")
            return False, None

        # Handle both int and string keys
        K_key = self.admixture_K
        if K_key not in dataset.admixture_ratios:
            K_key = str(self.admixture_K)
        if K_key not in dataset.admixture_ratios:
            logger.warning(
                f"PlotAdmixture skipped: K={self.admixture_K} not found. "
                f"Available: {list(dataset.admixture_ratios.keys())}"
            )
            return False, None

        return True, dataset.admixture_ratios[K_key]

    def _plot_embeddings(self, dataset: any, embeddings_to_plot: np.ndarray, color_array: np.ndarray) -> str:
        """Override parent to create admixture subplot grid."""
        has_admixture, admixture_df = self._check_admixture_available(dataset)
        
        if not has_admixture:
            logger.error("Cannot plot admixture: no valid data available")
            return super()._plot_embeddings(dataset, embeddings_to_plot, color_array)
        
        return self._plot_admixture_grid(embeddings_to_plot, admixture_df)

    def _plot_admixture_grid(self, embeddings_2d: np.ndarray, admixture_df: pd.DataFrame) -> str:
        """Create admixture proportion subplot grid."""
        admixture_numeric = admixture_df.iloc[:, 1:]

        K = min(self.admixture_K, admixture_numeric.shape[1])
        nrows, ncols = self._get_subplot_layout(K)

        fig_width = ncols * self.figsize_per_plot[0]
        fig_height = nrows * self.figsize_per_plot[1]
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))

        if K == 1:
            axes = np.array([axes])
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for idx in range(K):
            ax = axes[idx]
            anc_col = admixture_numeric.iloc[:, idx].values

            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=anc_col,
                s=self.point_size,
                alpha=self.alpha,
                cmap=self.cmap,
                vmin=0,
                vmax=1
            )

            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Component {idx+1} proportion', fontsize=13)
            plt.colorbar(scatter, ax=ax, label='Proportion')

        for idx in range(K, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(
            f'Embeddings: Admixture Proportions (K={K})',
            fontsize=18,
            fontweight='bold',
            y=1.02
        )
        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"admixture_plot_K{K}_{self.experiment_name}_{timestamp}.png"
        save_path = os.path.join(self.save_dir, filename)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved admixture plot to {save_path}")
        if wandb.run is not None:
            wandb.log({"admixture_plot": wandb.Image(save_path)})

        return save_path

    def on_latent_end(self, dataset: any, embeddings: dict) -> dict:
        """Main callback entry point using parent's helper methods."""
        has_admixture, _ = self._check_admixture_available(dataset)

        if not has_admixture:
            return self.callback_outputs

        embeddings_2d = self._get_embeddings(embeddings)
        color_array = self._get_color_array(dataset, embeddings)
        path = self._plot_embeddings(dataset, embeddings_2d, color_array)

        self.register_output("admixture_plot_path", path)
        return self.callback_outputs
