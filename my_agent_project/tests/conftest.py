import pytest
import sys
from unittest.mock import Mock
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
