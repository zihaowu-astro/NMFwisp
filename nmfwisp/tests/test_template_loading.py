import numpy as np
import pytest

from ..nmfwisp import fit_wisp, load_wisp_templates


def test_load_templates_case_insensitive():
    """Bundled template filenames are lowercase; lookup must succeed on
    case-sensitive filesystems regardless of the case of the inputs."""
    for filter_name in ("F200W", "f200w"):
        for detector_name in ("nrcb4", "NRCB4"):
            wisp_template, _, wmask, high_snr_region = load_wisp_templates(
                None, detector_name, filter_name)
            assert wisp_template is not None
            assert wmask is not None
            assert high_snr_region is not None


def test_fit_wisp_bundled_templates():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, (2048, 2048)).astype("f4")
    err = np.ones((2048, 2048), "f4")
    mask = np.zeros((2048, 2048), bool)
    wisp, wisp_e = fit_wisp(data, err, mask,
                            detector_name="nrcb4", filter_name="F200W")
    assert wisp.shape == data.shape
    assert wisp_e.shape == data.shape
    assert np.isfinite(wisp).all()


def test_missing_template_raises():
    data = np.zeros((2048, 2048), "f4")
    err = np.ones((2048, 2048), "f4")
    mask = np.zeros((2048, 2048), bool)
    with pytest.raises(FileNotFoundError, match="F162M"):
        fit_wisp(data, err, mask, detector_name="nrcb4", filter_name="F162M")
