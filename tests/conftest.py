import pytest
from PIL import Image


@pytest.fixture
def tiny_rgb():
    def make(w=32, h=32, colour=(0, 0, 0)):
        return Image.new("RGB", (w, h), colour)
    return make


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    return root
