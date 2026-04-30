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

Functionality for exporting a GeoZarr file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from numcodecs import Blosc
import geopandas as gpd

from oggmzarr.datacube.update_metadata import MetadataMapper


class GeoZarrHandler(MetadataMapper):
    def __init__(
        self: GeoZarrHandler,
        ds: xr.Dataset = None,
        ds_name: str = "L1",
        target_chunk_mb: float = 5.0,
        compressor: Optional[Blosc] = None,
        metadata_mapping_data: str = None,
        metadata_mapping_coords: str = None,
        zarr_format: int = 2,
    ):
        """Initialise a GeoZarrHandler object.

        Parameters
        ----------
        ds : xarray.DataTree | xarray.Dataset, default None
            Input dataset with dimensions ('x', 'y') or ('t', 'x', 'y').
            Must include coordinate variables. Accepts either a dataset
            or data tree.
        data_tree : xarray.DataTree, default None
            Input data_tree. Either ds or data_tree must be provided.
        ds_name : str, default 'L1'
            Name of datacube.
        target_chunk_mb : float, default 5.0
            Approximate chunk size in megabytes for efficient storage.
        compressor : Blosc, default None
            Compressor to apply on arrays. If None, the compression will
            be Blosc with zstd.
        metadata_mapping_data : str, default None
            Path to the YAML file containing variable metadata mappings.
            If None, defaults to 'metadata_mapping_data.yaml' in the current
            directory.
        metadata_mapping_coords : str, default None
            Path to the YAML file containing time coordinate metadata mappings.
            If None, defaults to 'metadata_mapping_data.yaml' in the current
            directory.
        zarr_format : int, default 2
            Zarr format version to use (2 or 3).
        """
        super().__init__(
            metadata_mapping_data=metadata_mapping_data,
            metadata_mapping_coords=metadata_mapping_coords,
        )

        self.target_chunk_mb = target_chunk_mb
        self.compressor = compressor or Blosc(
            cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE
        )
        self.zarr_format = zarr_format
        self.encoding = {}
        self._set_data(ds=ds, ds_name=ds_name)

    def _set_data(
        self, ds: xr.Dataset | xr.DataTree = None, ds_name: str = "L1"
    ) -> None:
        """Validate and set data.

        Parameters
        ----------
        ds : xarray.DataTree | xarray.Dataset, default None
            Input dataset with dimensions ('x', 'y') or ('t', 'x', 'y').
            Must include coordinate variables. Accepts either a dataset
            or data tree.
        ds_name : str, default "L1"
            Input data_tree. Either ds or data_tree must be provided.
        """

        if ds is None:
            raise ValueError("No dataset provided.")
        elif isinstance(ds, xr.Dataset):
            self.ds_name = ds_name
            ds = self._validate_dataset(ds)
            ds = self._update_metadata(ds, ds_name)
            self._define_encodings(ds, ds_name)

            # convert dataset to datatree
            self.data_tree = xr.DataTree.from_dict({ds_name: ds})
        elif isinstance(ds, xr.DataTree):
            # define encodings for potential exporting later on
            self.data_tree = ds
            for tree_level in self.data_tree:
                if tree_level in ["L1"]:
                    self._define_encodings(
                        ds=self.data_tree[tree_level].ds, ds_name=tree_level
                    )
                elif "L2" in tree_level or "L3" in tree_level:
                    for datacube_type in self.data_tree[tree_level]:
                        if datacube_type not in [
                            "monthly",
                            "annual_hydro",
                            "daily_smb",
                        ]:
                            raise ValueError(
                                "We currently only support model output datacubes of "
                                "the types 'monthly', 'annual_hydro' and 'daily_smb'."
                            )
                        self._define_encodings(
                            ds=self.data_tree[tree_level][datacube_type],
                            ds_name=tree_level,
                            ds_type=datacube_type,
                        )
        else:
            raise TypeError("Dataset should either be an xarray Dataset or DataTree.")

    def _validate_dataset(self: GeoZarrHandler, ds: xr.Dataset) -> xr.Dataset:
        """Validate the input dataset to ensure it includes required
        dimensions and associated coordinate variables.

        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset with dimensions ('x', 'y') or ('t', 'x', 'y').
            Must include coordinate variables.

        Raises
        ------
        ValueError
            - If 'x' or 'y' dimensions are missing.
            - If any dimension does not have an associated coordinate
              variable.
        """

        # TODO: get accepted dims from metadata mapping?
        # accepted_dims = {"x", "y", "t", "t_wgms", "t_sfc_type", "snowcover_frac"}
        # if not set(ds.dims).issubset(accepted_dims):
        #     raise ValueError(
        #         "Incorrect dataset dimensions."
        #         f" Accepted data dimensions are: {accepted_dims}"
        #     )
        for dim in ds.dims:
            if dim not in ds.coords:
                raise ValueError(
                    f"Coordinate variable for dimension '{dim}' is missing in "
                    "the dataset."
                )
        return ds

    def _calculate_chunk_sizes(
        self: GeoZarrHandler, var: xr.DataArray
    ) -> dict[str, int]:
        """Calculate chunk sizes for a given variable to match the
        target chunk size in megabytes.

        Parameters
        ----------
        var : xr.DataArray
            Data array whose dtype and dimensions are used to compute
            chunk sizes.

        Returns
        -------
        dict[str, int]
            A dictionary of chunk sizes for dimensions 'x', 'y', and
            optionally 't'.
        """
        target_bytes = self.target_chunk_mb * 1024 * 1024
        if "t_sfc_type" in var.dims:
            t_var = "t_sfc_type"
        elif "t_wgms" in var.dims:
            t_var = "t_wgms"
        else:
            t_var = "t"
        t_size = var.sizes.get(t_var, 1)  # Defaults to 1 if no 't' dimension
        chunk_sizes = {}

        if "x" in var.dims and "y" in var.dims:
            x_size = var.sizes["x"]
            y_size = var.sizes["y"]
            # Calculate the number of elements allowed per chunk
            # After accounting for a full 't' slice
            elements_per_t_slice = target_bytes // (var.dtype.itemsize * t_size)

            # Determine side length based on remaining budget
            side_length = int(np.sqrt(elements_per_t_slice))

            chunk_x = min(x_size, side_length)
            chunk_y = min(y_size, side_length)

            chunk_sizes["x"] = chunk_x
            chunk_sizes["y"] = chunk_y

        if t_var in var.dims:
            # Use the full length of 't' - this allows more efficient loading,
            # assuming the user is always interested in the full time series
            chunk_sizes[t_var] = t_size

        for dim in var.dims:
            if dim in ["member", "snowcover_frac"]:
                # use one to save each dimension separately
                chunk_sizes[dim] = 1
            elif dim not in [t_var, "x", "y"]:
                chunk_sizes[dim] = var.sizes[dim]

        return chunk_sizes

    def _define_encodings(
        self: GeoZarrHandler, ds: xr.Dataset, ds_name: str, ds_type: str = None
    ) -> None:
        """Define encoding settings for each data variable in the
        dataset, including chunking and compression.

        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset with dimensions ('x', 'y') or ('t', 'x', 'y').
            Must include coordinate variables.
        ds_name : str
            Dataset name to be used for this node of the tree.

        Notes
        -----
        Chunk sizes are computed using `_calculate_chunk_sizes`, and the
        compressor is set according to the class-level setting.
        """
        encoding_key = f"/{ds_name}"
        if ds_type is not None:
            encoding_key = f"{encoding_key}/{ds_type}"
        if encoding_key not in self.encoding:
            self.encoding[encoding_key] = {}

        for var in ds.data_vars:
            chunk_sizes = self._calculate_chunk_sizes(ds[var])
            chunks = tuple(chunk_sizes.get(dim) for dim in ds[var].dims)
            self.encoding[encoding_key][var] = {
                "chunks": chunks,
                "compressor": self.compressor,
            }

    def _update_metadata(self, ds: xr.Dataset, ds_name: str) -> xr.Dataset:
        """Update metadata to Climate and Forecast convention.

        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset with dimensions ('x', 'y') or ('t', 'x', 'y').
            Must include coordinate variables.
        ds_name : str
            Layer name for this node of the tree.

        Metadata is first updated using the ``update_metadata`` method.
        Each data variable is tagged with the ``grid_mapping`` attribute
        for spatial referencing.
        """
        ds = self.update_metadata(ds, ds_name)
        for var in ds.data_vars:
            var_dims = ds[var].dims
            if "x" in var_dims or "y" in var_dims:
                ds[var].attrs["grid_mapping"] = "spatial_ref"
        return ds

    def export(
        self: GeoZarrHandler, storage_directory: str, overwrite: bool = True
    ) -> None:
        """Write the dataset to GeoZarr format.

        Parameters
        ----------
        storage_directory : str
            Path to write the Zarr data.
        overwrite : bool, default True
            Whether to overwrite existing Zarr contents in the target
            location.
        """
        dir_path = Path(storage_directory).parent
        if not dir_path.exists():
            raise FileNotFoundError(
                f"Base directory of 'storage_directory' does not exist: {dir_path}"
            )
        self.data_tree.to_zarr(
            storage_directory,
            mode="w" if overwrite else "a",
            consolidated=True,
            zarr_format=self.zarr_format,
            encoding=self.encoding,
        )

    def add_layer(
        self: GeoZarrHandler, ds: xr.Dataset, ds_name: str, overwrite: bool = False
    ) -> None:
        """Add a new dataset as a child group of the DataTree at the root.
        Parameters
        ----------
        ds : xarray.Dataset
            New dataset layer to be added to the existing data tree.
        ds_name : str
            Layer name to be used for this node of the tree.
        overwrite : bool
            If True, allow a layer of the same name to be overwritten.
        """
        if ds_name in self.data_tree.children and not overwrite:
            raise ValueError(f"Group '{ds_name}' already exists.")

        # prepare new dataset
        ds = self._validate_dataset(ds)
        ds = self._update_metadata(ds, ds_name)

        # append additional encodings to the encodings class attribute
        self._define_encodings(ds, ds_name)

        # validate dataset attributes
        for var in ds.data_vars:
            attrs = ds[var].attrs.copy()
            attrs.pop("grid_mapping", None)
            self.METADATA_SCHEMA_DATA.validate(attrs)

        self.data_tree[ds_name] = xr.DataTree(dataset=ds)

    def add_shapefile(self: GeoZarrHandler, shapefile: gpd.GeoDataFrame, shapefile_name: str, overwrite: bool = False
    ) -> None:
        if shapefile_name in self.data_tree.children and not overwrite:
            raise ValueError(f"Shapefile '{shapefile_name}' already exists.")
        
        

    def add_datacube(
        self: GeoZarrHandler,
        datacubes: dict,
        datacube_name: str,
        overwrite: bool = False,
    ) -> None:
        """Add a new dataset as a child group of the DataTree at the root.

        Parameters
        ----------
        datacubes : dict
            A dictionary with keys one of the currently supported L2 datacubes
            ('monthly', 'annual_hydro', 'daily_smb') and values the
            corresponding xr.Dataset.
        datacube_name : str
            Layer name to be used for this node of the tree. It should either
            contain L2 or L3. If nothing from the both is included the name will
            get L2_ as suffix.
        overwrite : bool
            If True, allow a layer of the same name to be overwritten.
        """
        if "L2" not in datacube_name and "L3" not in datacube_name:
            # by default, we assume it is L2
            datacube_name = f"L2_{datacube_name}"

        if datacube_name in self.data_tree.children and not overwrite:
            raise ValueError(f"Group '{datacube_name}' already exists.")

        if not isinstance(datacubes, dict):
            raise ValueError(
                f"Datacubes need to be provided as dict with keys "
                f"one of 'monthly, 'annual_hydro' or 'daily_smb'."
            )

        # prepare leaves of new layer
        new_leaves = {}
        for datacube_type in datacubes:
            if datacube_type not in ["monthly", "annual_hydro", "daily_smb"]:
                raise ValueError(
                    "We currently only support model output "
                    "datacubes of the types 'monthly', "
                    "'annual_hydro' and 'daily_smb'."
                )
            datacube_tmp = datacubes[datacube_type]
            datacube_tmp = self._validate_dataset(datacube_tmp)
            datacube_tmp = self._update_metadata(datacube_tmp, datacube_name)

            # append additional encodings to the encodings class attribute
            self._define_encodings(
                ds=datacube_tmp, ds_name=datacube_name, ds_type=datacube_type
            )

            # validate dataset attributes
            for var in datacube_tmp.data_vars:
                attrs = datacube_tmp[var].attrs.copy()
                attrs.pop("grid_mapping", None)
                attrs.pop("inf_values", None)
                self.METADATA_SCHEMA_DATA.validate(attrs)

            for coord in datacube_tmp.coords:
                if coord not in ["spatial_ref"]:
                    attrs = datacube_tmp[coord].attrs.copy()
                    self.METADATA_SCHEMA_COORDS.validate(attrs)

            # after validation add to leaves
            new_leaves[datacube_type] = xr.DataTree(
                name=datacube_type, dataset=datacube_tmp
            )

        self.data_tree[datacube_name] = xr.DataTree(
            name=datacube_name, children=new_leaves
        )

    def get_layer(self: GeoZarrHandler, ds_name: str) -> xr.Dataset:
        """Get a dataset from a DataTree.

        Parameters
        ----------
        ds_name : str
            Layer name.

        Returns
        -------
        xr.Dataset
            Dataset layer in tree.

        Raises
        ------
        KeyError
            If the layer name is not present in the data tree.
        AttributeError
            If the layer does not contain a dataset.


        """
        try:
            layer = self.data_tree[ds_name].ds
        except KeyError:
            raise KeyError(f"{ds_name} layer not found in the data tree.")

        return layer

class OggmZarrHandler(GeoZarrHandler, MetadataMapper):
    def __init__(
        self: OggmZarrHandler,
        ds: xr.Dataset = None,
        ds_name: str = "L1",
        target_chunk_mb: float = 5.0,
        compressor: Optional[Blosc] = None,
        metadata_mapping_data: str = None,
        metadata_mapping_coords: str = None,
        zarr_format: int = 2,
    ):
        """Initialise an OggmZarrHandler object.

        Parameters
        ----------
        ds : xarray.DataTree | xarray.Dataset, default None
            Input dataset with dimensions ('x', 'y') or ('t', 'x', 'y').
            Must include coordinate variables. Accepts either a dataset
            or data tree.
        data_tree : xarray.DataTree, default None
            Input data_tree. Either ds or data_tree must be provided.
        ds_name : str, default 'L1'
            Name of datacube.
        target_chunk_mb : float, default 5.0
            Approximate chunk size in megabytes for efficient storage.
        compressor : Blosc, default None
            Compressor to apply on arrays. If None, the compression will
            be Blosc with zstd.
        metadata_mapping_data : str, default None
            Path to the YAML file containing variable metadata mappings.
            If None, defaults to 'metadata_mapping_data.yaml' in the current
            directory.
        metadata_mapping_coords : str, default None
            Path to the YAML file containing time coordinate metadata mappings.
            If None, defaults to 'metadata_mapping_data.yaml' in the current
            directory.
        zarr_format : int, default 2
            Zarr format version to use (2 or 3).
        """
        super().__init__(
            metadata_mapping_data=metadata_mapping_data,
            metadata_mapping_coords=metadata_mapping_coords,
        )

        self.target_chunk_mb = target_chunk_mb
        self.compressor = compressor or Blosc(
            cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE
        )
        self.zarr_format = zarr_format
        self.encoding = {}
        self._set_data(ds=ds, ds_name=ds_name)
