import xarray as xr
from pathlib import Path
import numpy as np
import os
import shapely


def get_pickle_paths(gdir) -> list[Path]:
    """Get all available pickles in the glacier directory."""
    return [Path(f) for f in os.listdir(gdir.dir) if f[-4:] == ".pkl"]


def get_tranche(data: dict, type_only=False) -> dict:
    """Extract a tranche of data from a dictionary.

    Parameters
    ----------
    data : dict
        The input dictionary to extract the tranche from.
    type_only : bool, default True
        If True, only the types of the data will be extracted. If False, the actual data will be extracted.
    """
    tranche = {}
    for k, v in data.items():
        if not type_only:
            tranche[k] = v
        else:
            tranche[k] = type(v)
    return tranche


def filter_arrays_from_dict(d: dict) -> dict:
    return {k: v for k, v in d.items() if isinstance(v, np.ndarray)}


def filter_lists_from_dict(d: dict) -> dict:
    return {k: v for k, v in d.items() if isinstance(v, list)}


def get_pickle_data(pickle_files: list[Path], gdir, type_only=False):
    """Read pickle files and extract their data into a dictionary.

    Parameters
    ----------
    pickle_files : list[Path]
        List of paths to pickle files.
    gdir : oggm.GlacierDirectory
        GlacierDirectory object to read the pickles from.
    type_only : bool, default False
        If True, only the types of the data will be extracted. If False, the actual data will be extracted.

    Returns
    -------
    dict
        A dictionary with the pickle base names as keys and the extracted data as values, or their types if `type_only` is True.
    """
    pickle_data = {}
    for pickle in pickle_files:
        try:
            stem = gdir.read_pickle(pickle.stem)
            if isinstance(stem, list):
                slices = []
                for i in stem:
                    if isinstance(i, dict):
                        slices.append(get_tranche(i, type_only=type_only))
                    else:
                        slices.append(type(i))
                pickle_data[pickle.stem] = slices
            elif isinstance(stem, dict):
                pickle_data[pickle.stem] = get_tranche(stem, type_only=type_only)
            else:
                print(f"Pickle {pickle.stem} not parseable.")
        except Exception as e:
            print(e)
            print(f"Pickle {pickle.stem} of type {type(pickle.stem)} not parseable.")

    return pickle_data


def get_downstream_line(pickle: dict) -> dict:
    """Convert downstream_line pickle data into zarr-compatible structure."""
    downstream_line = pickle["downstream_line"]
    if isinstance(downstream_line, shapely.LineString):
        coordinates = xr.DataArray(
            np.array(shapely.geometry.mapping(downstream_line)["coordinates"]),
            dims=["x", "y"],
        )
        pickle["downstream_line"] = coordinates
    return pickle


def convert_pickle_to_datatree(pickle_data: dict) -> xr.DataTree:
    """Convert a dictionary of pickles into an xarray DataTree."""
    data_tree = xr.DataTree()
    for name, pickle in pickle_data.items():
        try:
            if name == "downstream_line":
                data = get_downstream_line(pickle)

            elif isinstance(pickle, list):
                data = pickle[0]
            elif isinstance(pickle, dict):
                data = pickle
            else:
                raise TypeError("Not parseable")
            if isinstance(data, dict):
                data_tree = add_datacube(
                    data_tree=data_tree,
                    datacubes=data,
                    datacube_name=name,
                    overwrite=True,
                )
        except TypeError as e:
            print(e)
    return data_tree


def write_zarr(
    data_tree,
    storage_directory: str,
    overwrite: bool = True,
    zarr_format: int = 2,
    encoding: dict = None,
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
    data_tree.to_zarr(
        storage_directory,
        mode="w" if overwrite else "a",
        consolidated=True,
        zarr_format=zarr_format,
        encoding=encoding,
    )


def add_datacube(
    data_tree,
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

    if datacube_name in data_tree.children and not overwrite:
        raise ValueError(f"Group '{datacube_name}' already exists.")

    if not isinstance(datacubes, dict):
        raise ValueError(f"Datacubes need to be provided as dict")

    data_tree[datacube_name] = xr.DataTree.from_dict(name=datacube_name, data=datacubes)
    return data_tree
