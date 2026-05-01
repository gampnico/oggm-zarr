"""Copyright (c) 2026, Nicolas Gampierakis

"""
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import shapely
import xarray as xr

import oggmzarr.datacube.convert as convert


class TestConvert:
    def test_get_pickle_paths(self, tmp_path):
        (tmp_path / "foo.pkl").touch()
        (tmp_path / "bar.pkl").touch()
        (tmp_path / "ignored.txt").touch()
        gdir = MagicMock()
        gdir.dir = str(tmp_path)

        result = convert.get_pickle_paths(gdir)

        assert len(result) == 2
        assert all(p.suffix == ".pkl" for p in result)
        assert {p.stem for p in result} == {"foo", "bar"}

    @pytest.mark.parametrize("type_only", [False, True])
    def test_get_tranche(self, type_only):
        data = {"a": 1, "b": "hello", "c": np.array([1, 2, 3])}

        result = convert.get_tranche(data, type_only=type_only)
        assert isinstance(result, dict)

        

        if not type_only:
            for v in result.values():
                assert result["a"] == data["a"]
                assert result["b"] == data["b"]
                assert np.array_equal(result["c"], data["c"])
        else:
            assert result["a"] is int
            assert result["b"] is str
            assert result["c"] is np.ndarray

    def test_get_tranche_types(self):
        data = {"a": 1, "b": "hello"}

        result = convert.get_tranche(data, type_only=False)

        assert result["a"] == 1
        assert result["b"] == "hello"

    def test_filter_arrays_from_dict(self):
        d = {
            "arr": np.array([1, 2, 3]),
            "lst": [1, 2, 3],
            "num": 42,
            "arr2": np.array([4.0, 5.0]),
        }

        result = convert.filter_arrays_from_dict(d)

        assert set(result.keys()) == {"arr", "arr2"}
        assert all(isinstance(v, np.ndarray) for v in result.values())

    def test_filter_lists_from_dict(self):
        d = {
            "lst": [1, 2, 3],
            "arr": np.array([1, 2, 3]),
            "num": 42,
            "lst2": ["a", "b"],
        }

        result = convert.filter_lists_from_dict(d)

        assert set(result.keys()) == {"lst", "lst2"}
        assert all(isinstance(v, list) for v in result.values())

    def test_get_pickle_data_dict_pickle(self):
        gdir = MagicMock()
        gdir.read_pickle.return_value = {"key": np.array([1, 2, 3])}
        pickle_files = [Path("some_pickle.pkl")]

        result = convert.get_pickle_data(pickle_files, gdir)

        assert "some_pickle" in result
        np.testing.assert_array_equal(result["some_pickle"]["key"], np.array([1, 2, 3]))

    def test_get_pickle_data_from_list(self):
        gdir = MagicMock()
        gdir.read_pickle.return_value = [1, 2, 3]
        pickle_files = [Path("some_pickle.pkl")]

        result = convert.get_pickle_data(pickle_files, gdir)

        assert result["some_pickle"] == [int, int, int]

    def test_get_pickle_data_from_list_of_dicts(self):
        gdir = MagicMock()
        gdir.read_pickle.return_value = [{"key": 1.0}, {"key": 2.0}]
        pickle_files = [Path("some_pickle.pkl")]

        result = convert.get_pickle_data(pickle_files, gdir)

        assert "some_pickle" in result
        assert isinstance(result["some_pickle"], list)
        assert result["some_pickle"][0] == {"key": 1.0}
        assert result["some_pickle"][1] == {"key": 2.0}

    def test_get_pickle_data_type_only(self):
        gdir = MagicMock()
        gdir.read_pickle.return_value = {"key": np.array([1, 2, 3])}
        pickle_files = [Path("some_pickle.pkl")]

        result = convert.get_pickle_data(pickle_files, gdir, type_only=True)

        assert result["some_pickle"]["key"] is np.ndarray

    def test_get_pickle_data_non_parseable(self, capsys):
        gdir = MagicMock()
        gdir.read_pickle.return_value = 42
        pickle_files = [Path("unparseable.pkl")]

        result = convert.get_pickle_data(pickle_files, gdir)

        assert "unparseable" not in result
        assert "not parseable" in capsys.readouterr().out

    def test_get_pickle_data_exception(self, capsys):
        gdir = MagicMock()
        gdir.read_pickle.side_effect = Exception("read error")
        pickle_files = [Path("broken.pkl")]

        result = convert.get_pickle_data(pickle_files, gdir)

        assert result == {}
        assert "read error" in capsys.readouterr().out

    def test_get_pickle_data_multiple_files(self):
        gdir = MagicMock()
        gdir.read_pickle.side_effect = [
            {"a": np.array([1.0])},
            {"b": np.array([2.0])},
        ]
        pickle_files = [Path("first.pkl"), Path("second.pkl")]

        result = convert.get_pickle_data(pickle_files, gdir)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"first", "second"}

    def test_get_downstream_line_converts_linestring(self):
        line = shapely.LineString([(0, 0), (1, 1), (2, 2)])
        pickle = {"downstream_line": line}

        result = convert.get_downstream_line(pickle)

        assert isinstance(result["downstream_line"], xr.DataArray)
        assert result["downstream_line"].dims == ("x", "y")


    def test_convert_pickle_to_datatree_returns_datatree(self):
        pickle_data = {
            "inversion_input": {
                "/": xr.Dataset({"dx": xr.DataArray([10.0, 20.0])})
            },
        }

        result = convert.convert_pickle_to_datatree(pickle_data)

        assert isinstance(result, xr.DataTree)

    def test_convert_pickle_to_datatree_adds_dict_as_child(self):
        pickle_data = {
            "inversion_input": {
                "/": xr.Dataset({"dx": xr.DataArray([10.0, 20.0])})
            },
        }

        result = convert.convert_pickle_to_datatree(pickle_data)

        assert "inversion_input" in result.children

    def test_convert_pickle_to_datatree_list_uses_first_element(self):
        pickle_data = {
            "some_group": [
                {"/": xr.Dataset({"dx": xr.DataArray([1.0])})},
                {"/": xr.Dataset({"dx": xr.DataArray([2.0])})},
            ],
        }

        result = convert.convert_pickle_to_datatree(pickle_data)

        assert "some_group" in result.children

    def test_convert_pickle_to_datatree_downstream_line(self):
        line = shapely.LineString([(0, 0), (1, 1)])
        pickle_data = {
            "downstream_line": {"downstream_line": line},
        }

        result = convert.convert_pickle_to_datatree(pickle_data)

        assert isinstance(result, xr.DataTree)
        assert "downstream_line" in result.children

    def test_convert_pickle_to_datatree_skips_non_parseable(self, capsys):
        pickle_data = {"unparseable": 42}

        result = convert.convert_pickle_to_datatree(pickle_data)

        assert isinstance(result, xr.DataTree)
        assert "unparseable" not in result.children
        assert "Not parseable" in capsys.readouterr().out

    def test_write_zarr_creates_file(self, tmp_path):
        ds = xr.Dataset({"var": xr.DataArray([1, 2, 3])})
        data_tree = xr.DataTree(dataset=ds)
        zarr_path = str(tmp_path / "test.zarr")

        convert.write_zarr(data_tree, zarr_path)

        assert (tmp_path / "test.zarr").exists()

    def test_write_zarr_overwrite_does_not_raise(self, tmp_path):
        ds = xr.Dataset({"var": xr.DataArray([1, 2, 3])})
        data_tree = xr.DataTree(dataset=ds)
        zarr_path = str(tmp_path / "test.zarr")

        convert.write_zarr(data_tree, zarr_path)
        convert.write_zarr(data_tree, zarr_path, overwrite=True)

    def test_write_zarr_missing_base_dir_raises(self, tmp_path):
        data_tree = xr.DataTree()
        missing_path = str(tmp_path / "nonexistent" / "test.zarr")

        with pytest.raises(FileNotFoundError):
            convert.write_zarr(data_tree, missing_path)

    def test_add_datacube_adds_child_node(self):
        data_tree = xr.DataTree()
        datacubes = {"/": xr.Dataset({"var": xr.DataArray([1.0, 2.0])})}

        result = convert.add_datacube(data_tree, datacubes, "new_group")

        assert "new_group" in result.children

    def test_add_datacube_returns_data_tree(self):
        data_tree = xr.DataTree()
        datacubes = {"/": xr.Dataset({"var": xr.DataArray([1.0])})}

        result = convert.add_datacube(data_tree, datacubes, "group")

        assert isinstance(result, xr.DataTree)

    def test_add_datacube_overwrite_false_raises_on_duplicate(self):
        data_tree = xr.DataTree()
        datacubes = {"/": xr.Dataset({"var": xr.DataArray([1.0])})}
        data_tree = convert.add_datacube(data_tree, datacubes, "group")

        with pytest.raises(ValueError, match="already exists"):
            convert.add_datacube(data_tree, datacubes, "group", overwrite=False)

    def test_add_datacube_overwrite_true_replaces_existing(self):
        data_tree = xr.DataTree()
        datacubes = {"/": xr.Dataset({"var": xr.DataArray([1.0])})}
        data_tree = convert.add_datacube(data_tree, datacubes, "group")

        result = convert.add_datacube(data_tree, datacubes, "group", overwrite=True)

        assert "group" in result.children

    def test_add_datacube_non_dict_raises(self):
        data_tree = xr.DataTree()

        with pytest.raises(ValueError, match="dict"):
            convert.add_datacube(data_tree, "not_a_dict", "group")
