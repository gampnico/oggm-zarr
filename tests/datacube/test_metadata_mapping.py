"""Copyright 2025 DTCG Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import logging
import tempfile
import warnings

import numpy as np
import pytest
import xarray as xr
import yaml
from pyproj import CRS

from oggmzarr.datacube.update_metadata import MetadataMapper

logger = logging.getLogger(__name__)

pytest_plugins = "oggm.tests.conftest"


class TestMetadataMapper:

    @pytest.fixture(name="temp_metadata_file", scope="function")
    def fixture_temp_metadata_file(self):
        metadata = {
            "var1": {
                "standard_name": "air_temperature",
                "long_name": "Air Temperature",
                "units": "K",
                "institution": "Test Institute",
                "source": "Simulated",
                "comment": "Sample comment",
                "references": "http://example.com",
            }
        }
        with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w") as f:
            yaml.dump(metadata, f)
            return f.name

    @pytest.fixture(name="test_dataset", scope="function")
    def fixture_test_dataset(self):
        # Create a spatial dataset with CRS
        data = np.random.rand(3, 3)
        ds = xr.Dataset(
            {"var1": (["y", "x"], data)},
            coords={"x": np.arange(3), "y": np.arange(3)},
            attrs={
                "pyproj_srs": "+proj=longlat +datum=WGS84 +no_defs +type=crs",
                "RGI-ID": "RGI60-11-00001",
            },
        )
        return ds

    def test_load_metadata(self, temp_metadata_file):
        mapper = MetadataMapper(temp_metadata_file)
        assert "var1" in mapper.metadata_mappings_data
        assert isinstance(mapper.metadata_mappings_data["var1"], dict)

    def test_apply_metadata_to_variables(self, temp_metadata_file, test_dataset):
        mapper = MetadataMapper(temp_metadata_file)
        result = mapper.update_metadata(test_dataset.copy(), "L1")

        expected = mapper.metadata_mappings_data["var1"]
        for key, val in expected.items():
            assert result["var1"].attrs[key] == val

    def test_shared_metadata_attributes_and_crs(self, temp_metadata_file, test_dataset):
        # Ensure CRS is preserved or written correctly
        mapper = MetadataMapper(temp_metadata_file)
        result = mapper.update_metadata(test_dataset.copy(), "L1")

        for attr in [
            "Conventions",
            "title",
            "summary",
            "comment",
            "date_created",
            "glacier_attributes",
        ]:
            assert attr in result.attrs

        assert result.rio.crs is not None
        assert CRS.from_user_input(result.rio.crs).equals(CRS(test_dataset.pyproj_srs))

    def test_warns_on_unmapped_variables(self, temp_metadata_file):
        ds = xr.Dataset(
            {
                "var1": (["x", "y"], [[1.0, 2.0], [3.0, 4.0]]),
                "var2": (["x", "y"], [[4.0, 5.0], [6.0, 2.0]]),
                "var3": (["x", "y"], [[4.0, 5.0], [6.0, 3.0]]),
            },
            attrs={
                "pyproj_srs": CRS(3413).to_proj4(),
                "RGI-ID": "RGI60-11-00001",
            },
        )

        mapper = MetadataMapper(temp_metadata_file)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mapper.update_metadata(ds, "L1")
            assert (
                "Metadata mapping is missing for the following variables: "
                "['var2', 'var3']" in str(w[0].message)
            )
