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


=====

Provides shared fixtures for tests.

Use these to replace duplicated code.

For generating objects within a test function's scope, call a fixture
directly:

    .. code-block:: python

        def test_foobar(self, conftest_mock_grid):
            grid_object = conftest_mock_grid
            grid_object.set_foo(foo=bar)
            ...
"""

from types import ModuleType
from typing import Any
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest


# Function patches
@pytest.fixture(scope="function", autouse=False)
def conftest_mock_check_file_exists():
    """Override checks when mocking files."""

    patcher = patch("os.path.exists")
    mock_exists = patcher.start()
    mock_exists.return_value = True


@pytest.fixture(scope="function", autouse=False)
def conftest_mock_check_directory_exists():
    """Override checks when mocking directories."""

    patcher = patch("pathlib.Path.is_dir")
    mock_exists = patcher.start()
    mock_exists.return_value = True


@pytest.fixture(scope="function", autouse=False)
def conftest_hide_plot():
    """Suppress plt.show(). Does not close plots."""

    patcher = patch("matplotlib.pyplot.show")
    mock_exists = patcher.start()
    mock_exists.return_value = True


@pytest.fixture(name="conftest_rng_seed", scope="function", autouse=False)
def fixture_conftest_rng_seed():
    """Set seed for random number generator to 444.

    Returns:
        np.random.Generator: Random number generator with seed=444.
    """

    random_generator = np.random.default_rng(seed=444)
    assert isinstance(random_generator, np.random.Generator)

    yield random_generator


class TestBoilerplate:
    """Provides boilerplate methods for serialising tests.

    The class is instantiated via the `conftest_boilerplate` fixture.
    The fixture is autoused, and can be called directly within a test::

    ..code-block:: python

        def test_foo(self, conftest_boilerplate)"

            foobar = [...]
            conftest_boilerplate.bar(foobar)

    Methods are arranged with their appropriate test::

    .. code-block:: python

        @pytest.mark.dependency(name="TestBoilerplate::foo")
        def foo(self, ...):
            pass

        @pytest.mark.dependency(
            name="TestBoilerplate::test_foo", depends=["TestBoilerplate::foo"]
            )
        def test_foo(self ...):
            pass

    """

    def check_output(self, variable: Any, x_type: Any, x_value: Any) -> bool:
        """Check a variable matches an expected type and value.

        Args:
            variable: Variable to check.
            x_type: Expected variable type.
            x_value: Expected variable value.

        Returns:
            True when all assertions pass.
        """

        assert isinstance(variable, x_type)
        if np.issubdtype(type(variable), np.number):
            assert np.isclose(variable, x_value)
        else:
            assert variable == x_value

        return True

    def test_check_output(self):
        variable_list = [[1.0, float], ["test", str], [1, int], [True, bool]]

        for pair in variable_list:
            assert self.check_output(variable=pair[0], x_type=pair[1], x_value=pair[0])
        test_array = [0.0, 0.5, 0.6]
        test_value = max(test_array)
        assert test_value == 0.6
        assert isinstance(test_value, float)
        assert self.check_output(
            variable=max(test_array), x_type=float, x_value=test_value
        )

    def check_gpd_dataframe(self, gpd_dataframe: gpd.GeoDataFrame) -> bool:
        assert isinstance(gpd_dataframe, gpd.GeoDataFrame)
        assert not gpd_dataframe.empty
        return True

    def test_check_gpd_dataframe(self):
        pass

    def patch_variable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: ModuleType,
        new_params: dict,
    ):
        """Patch any variable in a module.

        Patch the module where the variable is used, not where it's
        defined. The patched variable only exists within the test
        function's scope, so test parametrisation is still supported.

        Example:
            To patch constants used by `cpkernel.node.Node`:

                .. code-block:: python

                    patches = {"dt": 7200, "air_density": 1.0}
                    conftest.boilerplate.patch_variable(
                        monkeypatch,
                        cosipy.cpkernel.node.constants,
                        patches,
                        )

        Args:
            monkeypatch: Monkeypatch instance.
            module: Target module for patching.
            new_params: Variable names as keys, desired patched values as values:

                .. code-block:: python

                    new_params = {"foo": 1, "bar": 2.0}
        """

        if not isinstance(new_params, dict):
            note = "Pass dict with variable names and patched values as items."
            raise TypeError(note)
        for key in new_params:
            monkeypatch.setattr(module, key, new_params[key])

    def test_boilerplate_integration(self):
        """Integration test for boilerplate methods."""

        self.test_check_output()


@pytest.fixture(name="conftest_boilerplate", scope="function", autouse=False)
def conftest_boilerplate():
    """Yield class containing methods for common tests."""

    test_boilerplate = TestBoilerplate()
    test_boilerplate.test_boilerplate_integration()

    yield TestBoilerplate()
