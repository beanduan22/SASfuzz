# State-Aware Library Bug Reproducers

This directory contains anonymized state-related PyTorch and TensorFlow bug reproducers. Each bug has a separate entry file named `state_bug_###.py`; original issue IDs and source URLs are intentionally omitted.

```bash
python bugs/reproducers/state/state_bug_001.py
python bugs/reproducers/state/_state_common.py --list
python bugs/reproducers/state/_state_common.py --all
```

## Counts

| Dimension | Bug files |
| --- | ---: |
| Gradient tracking | 18 |
| Execution mode | 14 |
| Distribution strategy | 16 |
| Total | 48 |

## Status Counts

| Status | Bug files |
| --- | ---: |
| Fixed | 16 |
| Confirmed | 32 |

`_state_common.py` contains only shared execution and output-packing logic. The 48 public repro entry points are `state_bug_001.py` through `state_bug_048.py`.
