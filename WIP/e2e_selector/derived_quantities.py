import numpy as np


class DerivedQuantityRegistry:
    """Central registry for catalog derived quantities, dependency tracking, and inverse mappings."""

    def __init__(self):
        self._registry = {}

    def register(self, name, required_cols=None, inverse_func=None, description=""):
        """
        Decorator to register a derived quantity calculation and optional inverse mapping.

        Parameters
        ----------
        name : str
            Unique name/key for the derived quantity.
        required_cols : list of str, optional
            Native or antecedent columns required in the catalog to compute this quantity.
        inverse_func : callable, optional
            Function to map derived values or bounds back to native quantities.
        description : str, optional
            Brief summary of what this derived quantity represents.
        """
        def decorator(func):
            self._registry[name] = {
                "name": name,
                "func": func,
                "inverse_func": inverse_func,
                "required_cols": required_cols or [],
                "description": description,
            }
            return func
        return decorator

    def list_all(self):
        """Returns metadata for all registered derived quantities."""
        return {
            name: info["description"]
            for name, info in self._registry.items()
        }

    def get_info(self, name):
        """Retrieves registered entry metadata for a given derived quantity name."""
        if name not in self._registry:
            raise KeyError(f"Derived quantity '{name}' is not registered.")
        return self._registry[name]

    def get_computable(self, available_native_cols):
        """Checks dataset columns against prerequisites to return computable quantities."""
        native_set = set(available_native_cols)
        computable = {}
        for name, info in self._registry.items():
            if set(info["required_cols"]).issubset(native_set):
                computable[name] = info["description"]
        return computable

    def get_required_native_cols(self, derived_names):
        """
        Extracts all unique native columns required to compute a list of derived properties.
        Handles nested dependencies if a derived property depends on another derived property.
        """
        required_native = set()
        visited = set()

        def _resolve(col_name):
            if col_name in visited:
                return
            visited.add(col_name)

            if col_name in self._registry:
                for req in self._registry[col_name]["required_cols"]:
                    _resolve(req)
            else:
                required_native.add(col_name)

        for name in derived_names:
            _resolve(name)

        return sorted(list(required_native))

    def compute(self, name, df):
        """Executes the calculation function for a registered derived quantity."""
        if name not in self._registry:
            raise KeyError(f"Derived quantity '{name}' is not registered.")
        return self._registry[name]["func"](df)

    def invert(self, name, value_or_bounds):
        """Executes the inverse function for a registered quantity (e.g. mag bounds -> flux bounds)."""
        if name not in self._registry:
            raise KeyError(f"Derived quantity '{name}' is not registered.")
        inv_fn = self._registry[name]["inverse_func"]
        if inv_fn is None:
            raise NotImplementedError(f"No inverse function registered for '{name}'.")
        return inv_fn(value_or_bounds)


# Global singleton instance
derived_registry = DerivedQuantityRegistry()


# =========================================================================
# Photometric Band Registration Helpers (Flux <-> Magnitude)
# =========================================================================

def _mag_to_flux(mag_val):
    """
    Inverse helper: Converts AB magnitude back to flux in nJy.
    Formula: f = 10^((22.5 - m) / 2.5)
    """
    if isinstance(mag_val, (list, tuple)):
        return [10.0 ** ((22.5 - m) / 2.5) for m in mag_val]
    return 10.0 ** ((22.5 - mag_val) / 2.5)

def _mag_err_to_flux_err(mag_err, flux):
    """
    Inverse helper: Converts AB magnitude error back to flux error in nJy.

    Parameters
    ----------
    mag_err : float, list, or np.ndarray
        Magnitude error (sigma_m).
    flux : float, list, or np.ndarray
        Corresponding flux in nJy (f).

    Returns
    -------
    flux_err : float or np.ndarray
        Flux error in nJy (sigma_f).
    """
    mag_err = np.asarray(mag_err)
    flux = np.asarray(flux)
    return (np.log(10) / 2.5) * flux * mag_err


def register_photometric_bands(survey_name, suffix, bands, photometry_types=None):
    """Dynamically registers flux-to-magnitude and mag-error derived functions."""
    if photometry_types is None:
        photometry_types = ["gold"]

    for p in photometry_types:
        for band in bands:
            flux_col = f"flux_{p}{suffix}_{band}"
            flux_err_col = f"flux_err_{p}{suffix}_{band}"
            mag_col = f"mag_{p}{suffix}_{band}"
            mag_err_col = f"mag_err_{p}{suffix}_{band}"

            # Register AB Magnitude calculation
            @derived_registry.register(
                name=mag_col,
                required_cols=[flux_col],
                inverse_func=_mag_to_flux,
                description=f"Derived AB magnitude for {survey_name} {p} band {band}"
            )
            def _calc_mag(df, _col=flux_col):
                f = np.where(df[_col] <= 0, np.nan, df[_col])
                return 22.5 - 2.5 * np.log10(f)

            # Register Magnitude Error calculation
            @derived_registry.register(
                name=mag_err_col,
                required_cols=[flux_col, flux_err_col],
                inverse_func=_mag_err_to_flux_err,
                description=f"Derived AB magnitude error for {survey_name} {p} band {band}"
            )
            def _calc_mag_err(df, _fcol=flux_col, _fecol=flux_err_col):
                f = np.where(df[_fcol] <= 0, np.nan, df[_fcol])
                fe = df[_fecol]
                return (2.5 / np.log(10)) * (fe / f)


# Register default computable magnitudes and mag errors
register_photometric_bands("roman", "", "YJH", photometry_types=["gold", "pgauss"])
register_photometric_bands("lsst", "_LSST", "ugrizy", photometry_types=["gold", "pgauss"])

