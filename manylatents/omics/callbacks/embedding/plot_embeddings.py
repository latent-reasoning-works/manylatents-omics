"""
Omics-specific extension of PlotEmbeddings callback.

Extends the core PlotEmbeddings to add support for genomics datasets
(HGDP, UKBB, MHI, AOU) with their specific colormaps.
"""

from manylatents.callbacks.embedding.plot_embeddings import PlotEmbeddings as PlotEmbeddingsBase
from manylatents.omics.data.hgdp_dataset import HGDPDataset
from manylatents.omics.data.ukbb_dataset import UKBBDataset
from manylatents.omics.data.mhi_dataset import MHIDataset
from manylatents.omics.data.aou_dataset import AOUDataset
from manylatents.omics.utils.mappings import cmap_pop as cmap_pop_HGDP
from manylatents.omics.utils.mappings import cmap_ukbb_superpops as cmap_pop_UKBB
from manylatents.omics.utils.mappings import cmap_mhi_superpops as cmap_pop_MHI
from manylatents.omics.utils.mappings import race_ethnicity_only_pca_colors as cmap_pop_AOU


class PlotEmbeddings(PlotEmbeddingsBase):
    """
    Omics-specific version of PlotEmbeddings that adds support for
    HGDP, UKBB, MHI, and AOU datasets with their custom colormaps.

    Inherits all functionality from core PlotEmbeddings and extends
    _get_colormap() to handle genomics datasets.
    """

    def _get_colormap(self, dataset: any) -> any:
        """
        Get colormap for plotting with omics dataset support.

        Falls back to parent class for non-omics datasets (DLAtree, generic).
        """
        # Handle omics-specific datasets first
        if isinstance(dataset, HGDPDataset):
            return cmap_pop_HGDP
        elif isinstance(dataset, UKBBDataset):
            return cmap_pop_UKBB
        elif isinstance(dataset, MHIDataset):
            return cmap_pop_MHI
        elif isinstance(dataset, AOUDataset):
            return cmap_pop_AOU
        else:
            # Fall back to parent class for DLAtree and generic datasets
            return super()._get_colormap(dataset)
