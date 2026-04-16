"""Copyright 2026 DTCG Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


=====

Functionality for ensuring metadata is CF compliant: https://cfconventions.org/.
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from importlib import resources

import rioxarray  # noqa: F401
import xarray as xr
import yaml
from schema import Optional, Schema


class MetadataMapper:
    """Class for applying CF-compliant metadata to xarray Datasets.

    Attributes
    ----------
    METADATA_SCHEMA_DATA : schema.Schema
        Validation schema for data variable metadata.
    METADATA_SCHEMA_COORDS : schema.Schema
        Validation schema for coordinates metadata.
    metadata_mappings_data : dict
        Dictionary of metadata mappings for data variables loaded from a YAML
        file.
    metadata_mappings_coords : dict
        Dictionary of metadata mappings for coordinates loaded from a YAML file.
    """

    metadata_mappings_data: dict  # as this is not explicitly passed to __init__().
    metadata_mappings_coords: dict  # as this is not explicitly passed to __init__().

    def __init__(
        self: MetadataMapper,
        metadata_mapping_data: str = "",
        metadata_mapping_coords: str = "",
    ):
        """Initialise MetadataMapper with a given or default mapping file.

        Parameters
        ----------
        metadata_mapping_data : str, optional
            Path to the YAML file containing variable metadata mappings.
            If empty, defaults to 'metadata_mapping_data.yaml' provided
            by the ``dtcg`` package.
        metadata_mapping_coords : str, optional
            Path to the YAML file containing variable metadata mappings.
            If empty, defaults to 'metadata_mapping_coords.yaml'
            provided by the ``dtcg`` package.
        """

        if not metadata_mapping_data:
            metadata_mapping_data = resources.files("oggmzarr.datacube").joinpath(
                "metadata_mapping_data.yaml"
            )
        self.METADATA_SCHEMA_DATA = Schema(
            {
                "standard_name": str,
                "long_name": str,
                "units": str,
                Optional("author"): str,
                "institution": str,
                "source": str,
                "comment": str,
                "references": str,
            }
        )
        self.metadata_mappings_data = self.read_metadata_mappings(
            self.METADATA_SCHEMA_DATA, metadata_mapping_data
        )

        if not metadata_mapping_coords:
            metadata_mapping_coords = resources.files("oggmzarr.datacube").joinpath(
                "metadata_mapping_coords.yaml"
            )
        self.METADATA_SCHEMA_COORDS = Schema(
            {
                "standard_name": str,
                "long_name": str,
                "units": str,
            }
        )
        self.metadata_mappings_coords = self.read_metadata_mappings(
            self.METADATA_SCHEMA_COORDS, metadata_mapping_coords
        )

    def read_metadata_mappings(
        self: MetadataMapper, schema: Schema, map_file: str
    ) -> dict:
        """Load and validate metadata mappings from a YAML file.

        Parameters
        ----------
        schema : Schema
            The schema structure used for validation
        map_file : str
            Path to the YAML file containing metadata mappings.

        Return
        ------
        dict
            Metadata mappings loaded from YAML file.

        Raises
        ------
        schema.SchemaError
            If any of the metadata entries fail schema validation.
        """
        with open(map_file) as f:
            config_dict = yaml.safe_load(f)

        for _, metadata in config_dict.items():
            schema.validate(metadata)

        return config_dict

    @staticmethod
    def _update_shared_metadata(dataset: xr.Dataset, ds_name: str) -> None:
        """Add shared metadata attributes to the dataset and ensure CRS is set.

        Parameters
        ----------
        dataset : xarray.Dataset
            The dataset to which shared metadata and CRS should be
            applied.
        ds_name : str
            Name of dataset.

        Notes
        -----
        If a CRS is not present, it is set from the dataset's
        `pyproj_srs` attribute. Shared metadata includes CF conventions,
        title, and summary.
        """

        # update metadata shared across all variables
        shared_metadata = {
            "Conventions": "CF-1.12",
            "comment": (
                "The DTC Glaciers project is developed under the European Space "
                "Agency's Digital Twin Earth initiative, as part of the Digital Twin "
                "Components (DTC) Early Development Actions."
            ),
            "date_created": datetime.now().isoformat(),
            "RGI-ID": dataset.attrs["RGI-ID"],
            "glacier_attributes": dataset.attrs.get("glacier_attributes", {}),
        }
        if "L1" in ds_name:
            if not (
                "spatial_ref" in dataset.data_vars or "spatial_ref" in dataset.coords
            ):
                # create a spatial_ref layer in the dataset
                if not dataset.rio.crs and not {"x", "y"}.isdisjoint(dataset.dims):
                    dataset.rio.write_crs(dataset.pyproj_srs, inplace=True)
            shared_metadata.update(
                {
                    "title": "Datacube of glacier-domain variables.",
                    "summary": (
                        "Resampled glacier-domain variables from multiple sources "
                        f"for RGI6-ID '{dataset.attrs['RGI-ID']}'. "
                        "Generated for the DTC Glaciers project."
                    ),
                }
            )
        elif "L2" in ds_name:
            shared_metadata.update(
                {
                    "title": "Datacube of observation-informed modelled variables.",
                    "summary": (
                        "Observation-informed modelled variables for RGI6-ID "
                        f"'{dataset.attrs['RGI-ID']}'. "
                        "Generated for the DTC Glaciers project."
                    ),
                }
            )
            # L2 must contain a description of the applied calibration strategy
            if "calibration_strategy" not in dataset.attrs:
                pass
                # raise ValueError(
                #     "Missing required attribute 'calibration_strategy' in"
                #     "dataset.attrs. Add a description of the applied "
                #     "calibration strategy."
                # )
            else:
                shared_metadata["calibration_strategy"] = dataset.attrs[
                "calibration_strategy"
            ]

        dataset.attrs.clear()  # clear old metadata
        dataset.attrs.update(shared_metadata)

    def update_metadata(
        self: MetadataMapper, dataset: xr.Dataset, ds_name: str
    ) -> xr.Dataset:
        """Apply variable and shared metadata to an xarray Dataset.

        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset to which the metadata should be applied.
        ds_name : str
            Name of dataset.

        Returns
        -------
        xarray.Dataset
            The input dataset with updated metadata.

        Warns
        -----
        UserWarning
            If any dataset variables are missing in the metadata mapping.

        Notes
        -----
        This function adds both per-variable and global metadata attributes.
        Missing variable mappings are reported as warnings, not errors.
        """
        # check there are mappings for all variables in the dataset
        difference_data = set(dataset.data_vars) - set(
            self.metadata_mappings_data.keys()
        )
        difference_coords = set(dataset.coords) - set(
            self.metadata_mappings_coords.keys()
        )
        for difference in [difference_data, difference_coords]:
            # remove eolis check as they contain the metadata
            not_needed = [
                "eolis_elevation_change_sigma_timeseries",
                "eolis_elevation_change_timeseries",
                "eolis_gridded_elevation_change",
                "eolis_gridded_elevation_change_sigma",
                "spatial_ref",
            ]
            difference = [x for x in difference if x not in not_needed]
            if difference:
                warning_msg = (
                    "Metadata mapping is missing for the following variables: "
                    f"{sorted(difference)}. The metadata for these variables "
                    "might not be compliant with Climate and Forecast "
                    "conventions https://cfconventions.org/."
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.warn(warning_msg, UserWarning, stacklevel=2)

        # special treatment for model parameters, to convert some of their attrs
        model_variables = [
            "volume",
            "area",
            "length",
            "off_area",
            "on_area",
            "melt_off_glacier",
            "melt_on_glacier",
            "liq_prcp_off_glacier",
            "liq_prcp_on_glacier",
            "snowfall_off_glacier",
            "snowfall_on_glacier",
            "melt_off_glacier_monthly",
            "melt_on_glacier_monthly",
            "liq_prcp_off_glacier_monthly",
            "liq_prcp_on_glacier_monthly",
            "snowfall_off_glacier_monthly",
            "snowfall_on_glacier_monthly",
            "runoff_monthly",
            "runoff_monthly_cumulative",
            "runoff",
            "specific_mb",
            "specific_mb_calendar_cum",
            "snowline",
            "prcp",
            "temp",
            "temp_std",
        ]
        model_coordinates = [
            "member",
            "time",
            "rgi_id",
            "hydro_year",
            "hydro_month",
            "calendar_year",
            "calendar_month",
            "month_2d",
            "calendar_month_2d",
        ]

        # small helper function to rename some model output attributes
        def _rename_key(attrs, new_key, old_key):
            if new_key not in attrs:
                default = "N/A"
            else:
                default = attrs[new_key]
            attrs[new_key] = attrs.pop(old_key, default)

        # simple function to apply metadata to all layers in an xarray dataset
        for metadata_mappings in [
            self.metadata_mappings_data,
            self.metadata_mappings_coords,
        ]:
            for data_name, metadata in metadata_mappings.items():
                if data_name in dataset.data_vars or data_name in dataset.coords:
                    dataset[data_name].attrs.update(metadata)

                    # special treatment of model output attributes
                    if data_name in model_variables:
                        _rename_key(dataset[data_name].attrs, "units", "unit")
                    if data_name in model_coordinates + model_variables:
                        _rename_key(
                            dataset[data_name].attrs, "long_name", "description"
                        )

        self._update_shared_metadata(dataset, ds_name)

        return dataset
