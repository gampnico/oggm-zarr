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

Tests for GeoZarr datacubes.
"""

import os

import numpy as np
import pytest
import xarray as xr
import yaml
import zarr

from oggmzarr.datacube.geozarr import GeoZarrHandler


class TestGeoZarrWriter:
    """Tests for GeoZarrWriter functionality and metadata compliance."""

    @pytest.fixture(name="test_dataset", scope="function")
    def fixture_test_dataset(self, tmp_path):
        """Generate a test xarray dataset and metadata mapping file."""
        ds = xr.Dataset(
            data_vars=dict(
                temperature=(("t", "y", "x"), np.random.rand(100, 100, 100)),
                precipitation=(("y", "x"), np.random.rand(100, 100)),
            ),
            coords=dict(
                x=("x", np.linspace(0, 1000, 100)),
                y=("y", np.linspace(0, 2000, 100)),
                t=("t", np.linspace(1e5, 1e6, 100)),
            ),
            attrs={
                "pyproj_srs": "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 "
                "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
                "RGI-ID": "RGI60-11-00001",
            },
        )

        metadata_mapping = {
            "temperature": {
                "standard_name": "bond",
                "long_name": "james bond",
                "units": "007",
                "institution": "mi6",
                "source": "wibble",
                "comment": "wobble",
                "references": "moneypenny",
            },
            "precipitation": {
                "standard_name": "bond",
                "long_name": "james bond",
                "units": "007",
                "institution": "mi6",
                "source": "wibble",
                "comment": "wobble",
                "references": "moneypenny",
            },
        }
        metadata_path = os.path.join(tmp_path, "meta.yaml")
        with open(metadata_path, "w") as f:
            yaml.dump(metadata_mapping, f)

        return ds, metadata_path

    @pytest.mark.filterwarnings("ignore:Metadata mapping is missing")
    def test_geozarr_attributes(self, tmp_path, test_dataset):
        """Test that spatial_ref, metadata, and dimensions are correct."""
        ds, metadata_path = test_dataset

        store_dir = tmp_path / "geozarr_store"

        writer = GeoZarrHandler(
            ds=ds,
            metadata_mapping_data=metadata_path,
        )

        writer.export(store_dir)

        root = zarr.open_group(store=store_dir, mode="r")
        root_group = root["L1"]

        assert "spatial_ref" in root_group
        assert "crs_wkt" in root_group["spatial_ref"].attrs

        for param in ["temperature", "precipitation"]:
            assert param in root_group
            assert "_ARRAY_DIMENSIONS" in root_group[param].attrs
            assert root_group[param].attrs["grid_mapping"] == "spatial_ref"
            assert root_group[param].attrs["standard_name"] == "bond"
            assert root_group[param].attrs["references"] == "moneypenny"

        for coord in ["x", "y", "t"]:
            assert coord in root_group
            assert root_group[coord].attrs["_ARRAY_DIMENSIONS"] == [coord]

    @pytest.mark.skip(reason="Requires rewrite for validation framework")
    def test_missing_required_dims_raises(self, test_dataset):
        """Test that missing required dimensions raises ValueError."""
        ds, _ = test_dataset
        ds = ds.rename({"x": "x_coordinate"})

        with pytest.raises(ValueError, match="Incorrect dataset dimensions"):
            GeoZarrHandler(ds=ds)

    @pytest.mark.filterwarnings("ignore:Metadata mapping is missing")
    def test_correct_chunking(self, test_dataset, tmp_path):
        """Test that chunking is computed as expected."""
        ds, _ = test_dataset
        writer = GeoZarrHandler(
            ds=ds,
            target_chunk_mb=1,
        )

        output_path = os.path.join(tmp_path, "test_zarr.zarr")
        writer.export(output_path)

        root = zarr.open_group(store=output_path, mode="r")
        root_group = root["L1"]

        temp_chunks = root_group["temperature"].chunks
        precip_chunks = root_group["precipitation"].chunks

        assert isinstance(temp_chunks, tuple)
        assert temp_chunks == (100, 36, 36)
        assert isinstance(precip_chunks, tuple)
        assert precip_chunks == (100, 100)

    def test_missing_dimension(self, test_dataset):
        ds, _ = test_dataset
        assert "t" in ds.dims
        ds = ds.drop_vars("t")
        assert "t" not in ds.coords
        with pytest.raises(
            ValueError, match="Coordinate variable for dimension 't' is missing"
        ):
            GeoZarrHandler(ds=ds)

    def test_add_layer(self, test_dataset):
        """Test that add layer functionality works correctly."""
        ds, metadata_path = test_dataset
        ds2 = ds.copy(deep=True)
        ds2.attrs["calibration_strategy"] = "test copy"

        handler = GeoZarrHandler(ds=ds, metadata_mapping_data=metadata_path)

        handler.add_layer(ds2, "L2")

        assert "L2" in handler.data_tree
        assert isinstance(handler.data_tree["L2"], xr.DataTree)
        assert isinstance(handler.data_tree["L2"].ds, xr.Dataset)
        for var in ["temperature", "precipitation", "x", "y", "t"]:
            xr.testing.assert_equal(ds2[var], handler.data_tree["L2"].ds[var])
        for var in ["temperature", "precipitation"]:
            assert "grid_mapping" in handler.data_tree["L2"].ds[var].attrs

    def test_add_datacube(self, test_dataset):
        """Test that add layer functionality works correctly."""
        ds, metadata_path = test_dataset
        ds2 = ds.copy(deep=True)
        ds2.attrs["calibration_strategy"] = "test copy"

        handler = GeoZarrHandler(ds=ds, metadata_mapping_data=metadata_path)

        handler.add_datacube({"monthly": ds2}, "L2")

        assert "L2" in handler.data_tree
        assert isinstance(handler.data_tree["L2"], xr.DataTree)
        assert isinstance(handler.data_tree["L2"]["monthly"].ds, xr.Dataset)
        for var in ["temperature", "precipitation", "x", "y", "t"]:
            xr.testing.assert_equal(
                ds2[var], handler.data_tree["L2"]["monthly"].ds[var]
            )
        for var in ["temperature", "precipitation"]:
            assert "grid_mapping" in handler.data_tree["L2"]["monthly"].ds[var].attrs

        # L2 datacubes are expected to be only OGGM model output at the moment
        # with no x, y coordinates and not "spatial_ref"
        # assert "spatial_ref" in handler.data_tree["L2"]["monthly"].ds.coords

    def test_get_layer(self, test_dataset):
        """Test getting a datatree layer."""
        ds, metadata_path = test_dataset

        handler = GeoZarrHandler(
            ds=ds,
            metadata_mapping_data=metadata_path,
        )

        datacube = handler.get_layer(ds_name="L1")
        assert isinstance(datacube, xr.Dataset)
        for var in ["temperature", "precipitation", "x", "y", "t"]:
            xr.testing.assert_equal(ds[var], datacube[var])
        for var in ["temperature", "precipitation"]:
            assert "grid_mapping" in datacube[var].attrs
        assert "spatial_ref" in datacube.coords

    def test_get_layer_errors(self, test_dataset):
        """Test errors are correctly raised when getting a layer."""

        ds, metadata_path = test_dataset

        handler = GeoZarrHandler(
            ds=ds,
            metadata_mapping_data=metadata_path,
        )

        # Search for a non-existent layer

        with pytest.raises(KeyError, match="L2 layer not found in the data tree."):
            handler.get_layer("L2")
