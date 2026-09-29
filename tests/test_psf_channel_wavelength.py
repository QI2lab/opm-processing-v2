"""Choose the requested PSF wavelength for simultaneous laser channels."""

import pytest

from opm_processing.process import _psf_wavelength_um


@pytest.mark.unit
@pytest.mark.parametrize(
    "channel, expected",
    [
        ("488nm", 0.488),
        ("561", 0.561),
        ("637nm", 0.637),
        ("488 + 637", 0.637),
        ("637 + 488", 0.637),
        ("488nm + 637nm", 0.637),
        (" 488 NM + 637 NM ", 0.637),
    ],
)
def test_psf_channel_wavelength(channel, expected):
    """Keep single channels unchanged and select 637 nm for the combined label."""
    assert _psf_wavelength_um(channel) == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.parametrize("channel", ["GFP", "488 +", "nan", "0", "-488"])
def test_invalid_psf_channel_wavelength(channel):
    """Require a usable wavelength rather than generating an invalid PSF."""
    with pytest.raises(ValueError, match="supply --decon-psf-paths"):
        _psf_wavelength_um(channel)
