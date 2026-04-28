"""Smoke tests for the RQ1 pipeline (no network)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq1.classify import heuristic_label, _text  # noqa: E402
from rq1.collect import _keyword_match, _split_url  # noqa: E402
from rq1.verify_fix import verdict_of, CLOSE_REF_RE  # noqa: E402


def _rubric() -> dict:
    return json.loads((ROOT / "configs" / "rubric.json").read_text())


def test_keyword_match_positive():
    assert _keyword_match("model returns NaN sometimes", ["nan"])
    assert _keyword_match("Output is INCORRECT after train()", ["incorrect"])


def test_keyword_match_negative():
    assert not _keyword_match("please add a feature for resnet", ["nan", "incorrect"])


def test_split_url():
    repo, n = _split_url("https://github.com/pytorch/pytorch/issues/12345")
    assert repo == "pytorch/pytorch"
    assert n == 12345


def test_close_ref_re():
    assert CLOSE_REF_RE.search("Fixes #4242").group(2) == "4242"
    assert CLOSE_REF_RE.search("This is now resolved: #777").group(2) == "777"
    assert CLOSE_REF_RE.search("closes #1") is not None
    assert CLOSE_REF_RE.search("see also #1") is None


def test_verdict():
    assert verdict_of({"is_pull_request": True}) == "excluded_pull_request"
    assert verdict_of({"closed_with_commit": True}) == "fix_verified"
    assert verdict_of({"closed_with_merged_pr": True}) == "fix_verified"
    assert verdict_of({"body_refs_merged_pr": True}) == "fix_verified"
    assert verdict_of({}) == "unverified"


def test_heuristic_gradient_dominant():
    issue = {
        "title": "autograd computes wrong gradient when requires_grad=True after detach()",
        "body": "calling .backward() yields NaN; functorch.vmap also affected.",
        "labels": ["module: autograd"],
    }
    label, scores = heuristic_label(issue, _rubric())
    assert label == "A"
    assert scores["A"] >= 1


def test_heuristic_execution_mode():
    issue = {
        "title": "BatchNorm output incorrect after model.eval()",
        "body": "Wrong values vs manual normalization. Switching to torch.compile + autocast also broken.",
        "labels": ["module: nn"],
    }
    label, _ = heuristic_label(issue, _rubric())
    assert label == "B"


def test_heuristic_distribution():
    issue = {
        "title": "NCCL all_reduce ReduceOp.BXOR silently maps to Sum",
        "body": "torch.distributed: incorrect collective result across DistributedDataParallel workers.",
        "labels": ["module: ddp"],
    }
    label, _ = heuristic_label(issue, _rubric())
    assert label == "C"


def test_heuristic_other_state():
    issue = {
        "title": "manual_seed does not reset Generator state across DataLoader workers",
        "body": "running_mean and running_var diverge from numpy reference.",
        "labels": [],
    }
    label, _ = heuristic_label(issue, _rubric())
    assert label == "D"


def test_heuristic_non_state():
    issue = {
        "title": "torch.bincount returns wrong dtype on empty input",
        "body": "Pure shape/dtype issue at the kernel boundary; no other state involved.",
        "labels": [],
    }
    label, _ = heuristic_label(issue, _rubric())
    assert label == "E"
