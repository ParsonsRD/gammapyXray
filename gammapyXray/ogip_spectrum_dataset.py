# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from gammapy.datasets import SpectrumDatasetOnOff
from gammapy.maps import MapAxis

__all__ = ["StandardOGIPDataset"]


class StandardOGIPDataset(SpectrumDatasetOnOff):
    """Dataset containing spectral data as defined by X-ray OGIP compliant files.

    A few elements are added that are not supported by the current SpectrumDataset in gammapy.

    grouping contains the information on the grouping scheme, namely the group number of each bin.

    Parameters
    ----------
    models : `~gammapy.modeling.models.Models`
        Source sky models.
    counts : `~gammapy.maps.WcsNDMap`
        Counts cube
    counts_off : `~gammapy.maps.WcsNDMap`
        Ring-convolved counts cube
    acceptance : `~gammapy.maps.WcsNDMap`
        Acceptance from the IRFs
    acceptance_off : `~gammapy.maps.WcsNDMap`
        Acceptance off
    exposure : `~gammapy.maps.WcsNDMap`
        Exposure cube
    mask_fit : `~gammapy.maps.WcsNDMap`
        Mask to apply to the likelihood for fitting.
    psf : `~gammapy.irf.PSFKernel`
        PSF kernel. Unused here.
    edisp : `~gammapy.irf.EDispKernel`
        Energy dispersion
    mask_safe : `~gammapy.maps.WcsNDMap`
        Mask defining the safe data range.
    grouping_axis : `~gammapy.maps.MapAxis`
        MapAxis defining the grouping scheme.
    gti : `~gammapy.data.GTI`
        GTI of the observation or union of GTI if it is a stacked observation
    meta_table : `~astropy.table.Table`
        Table listing information on observations used to create the dataset.
        One line per observation for stacked datasets.
    name : str
        Name of the dataset.

    """

    stat_type = "wstat"
    tag = "StandardOGIPDataset"

    def __init__(self, *args, **kwargs):
        self._grouped = None
        axis = kwargs.pop("grouping_axis", None)

        super().__init__(*args, **kwargs)

        self._models = None
        self._mask_fit = None
        self._mask = None
        
        self.grouping_axis = axis
        if axis is not None:
            self.apply_grouping(self.grouping_axis)
        else:
            self._grouped = self.to_spectrum_dataset_onoff(name=self.name)
            #print("gr", self._grouped)

    @property
    def grouped(self):
        """Return grouped SpectrumDatasetOnOff."""
        #print("gr", self._grouped)
        return self._grouped

    @property
    def _is_grouped(self):
        return self.grouped is not None

    def apply_grouping(self, axis=None):
        """Apply grouping."""
        if axis is None:
            raise ValueError("A grouping MapAxis must be provided.")
        else:
            dataset = self.to_spectrum_dataset_onoff(name=self.name)
            self._grouped = dataset.resample_energy_axis(
                axis, name=f"group_{self.name}"
            )
            self._grouped._mask_safe = self.mask_safe.resample_axis(
                axis=axis, ufunc=np.logical_or
            )
    def set_min_true_energy(self, energy):
        """Resamples the true energy axis of the grouped dataset, by eliminating all bins below a given energy.
        Parameters
        -------------
        energy: `~astropy.units.Quantity`
            Minimum energy (exclusive) for the resampled true energy axis
        """
        edges = self.grouped.geoms["geom_exposure"].axes["energy_true"].edges
        mask = np.where(edges > energy)
        edges_resampled = edges[mask]

        resampled_true_energy_axis = MapAxis.from_edges(
            edges_resampled, name="energy_true"
        )
        self.grouped.exposure = self.grouped.exposure.resample_axis(
            resampled_true_energy_axis
        )
        self.grouped.edisp.edisp_map = self.grouped.edisp.edisp_map.resample_axis(
            resampled_true_energy_axis
        )
        self.grouped.edisp.exposure_map = self.grouped.edisp.exposure_map.resample_axis(
            resampled_true_energy_axis
        )

    @property
    def models(self):
        """Models (`~gammapy.modeling.models.Models`)."""
        if self._is_grouped:
            return self.grouped.models
        return self._models
    
    @models.setter
    def models(self, models):
        """Models setter"""
        if self._is_grouped:
            self.grouped.models = models
        else:
            self._models = models


    @property
    def mask_fit(self):
        """RegionNDMap providing the fitting energy range."""
        if self._is_grouped:
            return self.grouped.mask_fit
        else:
            return self._mask_fit

    @mask_fit.setter
    def mask_fit(self, mask_fit):
        """RegionNDMap providing the fitting energy range."""
        if self._is_grouped:
            self.grouped.mask_fit = mask_fit.resample_axis(
                axis=self.grouping_axis, ufunc=np.logical_or
            )
        else:
            self._mask_fit = mask_fit

    @property
    def mask(self):
        """Combined fit and safe mask"""
        if self._is_grouped:
            return self.grouped.mask
        else:
            return self._mask
        
    # @property
    # def mask_safe(self):
    #     """Combined fit and safe mask"""
    #     if self._is_grouped:
    #         return self.grouped.mask_safe
    #     else:
    #         return self._mask_safe
    
    # @mask_safe.setter
    # def mask_safe(self, mask_safe):
    #     """Combined fit and safe mask"""
    #     #print(mask_safe.resample_axis(
    #     #        axis=self.grouping_axis, ufunc=np.logical_or
    #     #    ))
    #     if self._is_grouped:
    #         self.grouped.mask_safe = mask_safe.resample_axis(
    #             axis=self.grouping_axis, ufunc=np.logical_or
    #         )
    #     else:
    #         self._mask_safe = mask_safe

        
    def npred(self):
        """Predicted source and background counts
        Returns
        -------
        npred : `Map`
            Total predicted counts
        """
        return self.grouped.npred()

    def npred_signal(self, model_names=None):
        """ "Model predicted signal counts.
        If a model is passed, predicted counts from that component is returned.
        Else, the total signal counts are returned.
        Parameters
        -------------
        model_name: str
            Name of  SkyModel for which to compute the npred for.
            If none, the sum of all components (minus the background model)
            is returned
        Returns
        ----------
        npred_sig: `gammapy.maps.Map`
            Map of the predicted signal counts
        """
        return self.grouped.npred_signal()#model_name=model_name)

    def stat_sum(self):
        """Total statistic given the current model parameters."""
        return self.grouped.stat_sum()

    def _stat_sum_likelihood(self):
        """Total statistic given the current model parameters without the priors."""
        return self.grouped._fit_statistic.stat_sum_dataset(self.grouped)
    
    # @property
    # def counts(self):
    #     """Counts map"""
    #     if self._is_grouped:
    #         return self.grouped.counts
    #     else:
    #         return self._counts

    # @counts.setter
    # def counts(self, counts):
    #     """Counts map"""
    #     if self._is_grouped:
    #         self.grouped.counts = counts
    #     else:
    #         self._counts = counts
        
    # @property
    # def data(self):
    #     """Data vector"""
    #     if self._is_grouped:
    #         return self.grouped.data
    #     else:
    #         return self._data
    
    # @data.setter
    # def data(self, data):
    #     """Data vector"""
    #     if self._is_grouped:
    #         self.grouped.data = data
    #     else:
    #         self._data = data

    def plot_fit(
        self,
        ax_spectrum=None,
        ax_residuals=None,
        kwargs_spectrum=None,
        kwargs_residuals=None,
    ):
        ax_spectrum, ax_residuals = self.grouped.plot_fit(
            ax_spectrum, ax_residuals, kwargs_spectrum, kwargs_residuals
        )

        return ax_spectrum, ax_residuals

    def plot_residuals_spectral(self, ax=None, method="diff", region=None, **kwargs):
        """Plot spectral residuals.
        The residuals are extracted from the provided region, and the normalization
        used for its computation can be controlled using the method parameter.
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes to plot on.
        method : {"diff", "diff/sqrt(model)"}
            Normalization used to compute the residuals, see `SpectrumDataset.residuals`.
        region: `~regions.SkyRegion` (required)
            Target sky region.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.axes.Axes.errorbar`.
        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object.
        """
        return self.grouped.plot_residuals_spectral(
            ax=ax, method=method, region=region, **kwargs
        )

    @classmethod
    def read(cls, filename, name=None):
        """Read from file

        For now, filename is assumed to the name of a PHA file where BKG file, ARF, and RMF names
        must be set in the PHA header and be present in the same folder.

        For formats specs see `OGIPDatasetReader.read`

        Parameters
        ----------
        filename : `~pathlib.Path` or str
            OGIP PHA file to read
        """
        from .io_ogip import StandardOGIPDatasetReader

        reader = StandardOGIPDatasetReader(filename=filename)
        return reader.read(name=name)

    def write(self, filename, overwrite=False, format="ogip"):
        raise NotImplementedError("Standard OGIP writing is not supported.")

    def to_spectrum_dataset_onoff(self, name=None):
        """convert to spectrum dataset on off by dropping the grouping axis.
        Parameters
        ----------
        name : str
            Name of the new dataset.

        Returns
        -------
        dataset : `SpectrumDatasetOnOff`
            Spectrum dataset on off.
        """
        kwargs = {"name": name}

        kwargs["acceptance"] = self.acceptance
        kwargs["acceptance_off"] = self.acceptance_off
        kwargs["counts_off"] = self.counts_off
        dataset = self.to_spectrum_dataset()

        return SpectrumDatasetOnOff.from_spectrum_dataset(dataset=dataset, **kwargs)

    def info_dict(self):
        """HTML representation of the dataset.

        Returns
        -------
        html : str
            HTML code.
        """
        if self._is_grouped:
            return self.grouped.info_dict()
        else:
            return super().info_dict()