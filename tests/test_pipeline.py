import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from conftest import BaseTest


class TestPipeline(BaseTest):
    @pytest.mark.order("last")
    def test_pipeline(self):
        pass
