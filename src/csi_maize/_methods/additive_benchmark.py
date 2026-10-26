"""Evidence guards for source-linked challenge cases and annotation status."""

from bootstrap import ROOT
from experiment_lab import sha256_file
from pathlib import Path


def validate_case(case):
    if (
        case.get("is_ground_truth")
        and case.get("annotation_status") != "independent_manual_reference"
    ):
        raise ValueError("An automatic mask is not independent ground truth")
    if case.get("split") == "independent_test":
        raise ValueError("Reused development cases cannot claim an untouched test split")
    return case


def source_case(case_id, domain, path, root=ROOT):
    return {
        "case_id": case_id,
        "domain": domain,
        "source": Path(path).relative_to(root).as_posix(),
        "source_sha256": sha256_file(path),
        "annotation_status": "automatic_prediction",
        "is_ground_truth": False,
        "split": "reused_development_challenge",
    }
