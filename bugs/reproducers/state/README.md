# State-Aware Library Bug Reproducers

This directory contains a single runner for PyTorch and TensorFlow state-related bug reproducers.

```bash
python bugs/reproducers/state/state_library_bug_repros.py --list
python bugs/reproducers/state/state_library_bug_repros.py --case tf-62553
python bugs/reproducers/state/state_library_bug_repros.py --all
```

## Counts

| Dimension | Bugs |
| --- | ---: |
| Gradient tracking | 18 |
| Execution mode | 14 |
| Distribution strategy | 15 |
| Total | 47 |

## Status Counts

| Status | Bugs |
| --- | ---: |
| Fixed | 16 |
| Confirmed | 31 |

The runner executes each case in an isolated child process when `--all` is used, so fatal reproducers are reported without stopping the whole batch.
