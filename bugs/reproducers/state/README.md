# State-Aware Library Bug Reproducers

This directory contains 48 anonymized state-related PyTorch and TensorFlow bug reproducers. Each `state_bug_###.py` file is a complete standalone script and does not import any local helper file.

```bash
python bugs/reproducers/state/state_bug_001.py
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

Original issue IDs and source URLs are intentionally omitted.
