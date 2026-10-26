"""Standalone targets must produce every derived input they consume."""

from csi_maize.cli import workflow
from csi_maize.engine import ordered_stages


def test_isolated_phenotype_controls_produces_model_fold_provenance():
    order = ordered_stages(workflow("learning"), ["phenotype-controls"])
    assert "phenotype-models" in order
    assert order.index("phenotype-models") < order.index("phenotype-controls")
