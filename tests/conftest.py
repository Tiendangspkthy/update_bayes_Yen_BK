import shutil
from pathlib import Path

import pytest


class BaseTest:
    """Base test class with common fixtures for all tests."""

    @pytest.fixture(autouse=True)
    def copy_config_folder(self):
        config_folder = Path("pedago_ml_configs")
        if not config_folder.exists():
            shutil.copytree(
                "../pedago_ml_configs",
                config_folder,
            )

    @pytest.fixture(autouse=True)
    def change_test_dir(self, request, monkeypatch):
        test_dir = request.fspath.dirname
        monkeypatch.chdir(request.fspath.dirname)
        return test_dir


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup a testing directory once we are finished."""

    def remove_test_dir():
        paths = [
            "pedago_ml_configs/",
        ]
        import os

        for f in paths:
            if "/" not in f and os.path.exists(f):
                os.remove(f)
            else:
                shutil.rmtree(
                    f"{os.path.dirname(os.path.realpath(__file__))}/{f}",
                    ignore_errors=True,
                )

    request.addfinalizer(remove_test_dir)
