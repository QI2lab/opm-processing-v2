"""Store relocation and split-root links resolve against prescribed provenance."""

import pytest

from opm_processing.dataio.processing_state import resolve_store_path

pytestmark = pytest.mark.unit


def test_store_link_resolution_respects_provenance_and_local_copy_policy(
    store_link_case,
):
    """Select existing original or copied stores without losing portable basenames."""
    case = store_link_case
    resolved = resolve_store_path(
        case.recorded, case.directory, prefer_local=case.prefer_local
    )
    assert resolved == case.expected
    if case.available:
        assert (resolved / "identity").read_text("utf-8") == case.identity
    else:
        assert not resolved.exists()
