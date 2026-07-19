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


def test_missing_template_returns_zeros_with_warning():
    """A whitelisted combination with no shipped template file (F162M) should
    behave like the other no-wisp cases: warn and return zero arrays."""
    data = np.ones((2048, 2048), "f4")
    err = np.ones((2048, 2048), "f4")
    mask = np.zeros((2048, 2048), bool)
    with pytest.warns(UserWarning, match="F162M"):
        wisp, wisp_e = fit_wisp(data, err, mask,
                                detector_name="nrcb4", filter_name="F162M")
    assert wisp.shape == data.shape
    assert not wisp.any()
    assert not wisp_e.any()


def test_no_wisp_cases_return_zeros_with_warning():
    """Filters/detectors without wisps warn and return (wisp, wisp_e) zero
    arrays, unpackable by the fit_wisp callers."""
    data = np.ones((2048, 2048), "f4")
    err = np.ones((2048, 2048), "f4")
    mask = np.zeros((2048, 2048), bool)
    for detector_name, filter_name in [("nrcb4", "F070W"),   # filter without wisps
                                       ("nrca1", "F200W"),   # detector without wisps
                                       ("nrcb4", "F480M")]:  # filter without templates
        with pytest.warns(UserWarning):
            wisp, wisp_e = fit_wisp(data, err, mask,
                                    detector_name=detector_name,
                                    filter_name=filter_name)
        assert wisp.shape == data.shape
        assert not wisp.any()
        assert not wisp_e.any()
