"""
Omics-specific extension of PlotEmbeddings callback.

Extends the core PlotEmbeddings to add support for genomics datasets
with colormaps from manifold-genetics outputs or legacy hardcoded mappings.
"""

from manylatents.callbacks.embedding.plot_embeddings import PlotEmbeddings as PlotEmbeddingsBase
from manylatents.popgen.data.manifold_genetics_dataset import ManifoldGeneticsDataset

# Legacy dataset imports for backward compatibility
from manylatents.popgen.data.hgdp_dataset import HGDPDataset
from manylatents.popgen.data.ukbb_dataset import UKBBDataset
from manylatents.popgen.data.mhi_dataset import MHIDataset
from manylatents.popgen.data.aou_dataset import AOUDataset
from manylatents.popgen.utils.mappings import cmap_pop as cmap_pop_HGDP
from manylatents.popgen.utils.mappings import cmap_ukbb_superpops as cmap_pop_UKBB
from manylatents.popgen.utils.mappings import cmap_mhi_superpops as cmap_pop_MHI
from manylatents.popgen.utils.mappings import race_ethnicity_only_pca_colors as cmap_pop_AOU


class PlotEmbeddings(PlotEmbeddingsBase):
    """
    Omics-specific version of PlotEmbeddings that adds support for
    genomics datasets with colormaps.

    Supports:
    - ManifoldGeneticsDataset: Uses colormap.json from manifold-genetics outputs
    - Legacy datasets (HGDP, UKBB, MHI, AOU): Uses hardcoded colormaps

    Inherits all functionality from core PlotEmbeddings and extends
    _get_colormap() to handle genomics datasets.
    """

    def _get_colormap(self, dataset: any) -> any:
        """
        Get colormap for plotting with omics dataset support.

        Priority:
        1. ManifoldGeneticsDataset with colormap.json (recommended)
        2. Legacy dataset-specific hardcoded colormaps
        3. Fall back to parent class for other datasets

        Args:
            dataset: Dataset object to extract colormap from

        Returns:
            Colormap dict or None
        """
        # Handle new ManifoldGeneticsDataset with colormap.json
        if isinstance(dataset, ManifoldGeneticsDataset):
            colormap = dataset.get_colormap()
            if colormap is not None:
                return colormap
            # If no colormap loaded, fall through to default handling
        
        # Handle legacy omics-specific datasets with hardcoded colormaps
        elif isinstance(dataset, HGDPDataset):
            return cmap_pop_HGDP
        elif isinstance(dataset, UKBBDataset):
            return cmap_pop_UKBB
        elif isinstance(dataset, MHIDataset):
            return cmap_pop_MHI
        elif isinstance(dataset, AOUDataset):
            return cmap_pop_AOU
        
        # Fall back to parent class for DLAtree and generic datasets
        return super()._get_colormap(dataset)
