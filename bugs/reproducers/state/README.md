# State-Aware Library Bug Reproducers

This directory contains 47 issue-level PyTorch and TensorFlow state-related bug
reproducers from the RQ3 bug study. Each `bug_###.py` file is a complete
standalone script and does not import any local helper file. The mapping from
script number to GitHub issue is recorded in `issue_index.json`.

```bash
python bugs/reproducers/state/bug_001.py
```

## Counts

The artifact is counted at the issue/case level. Some cases intentionally share
the same core API/input reproducer because separate GitHub issues reported the
same underlying state-sensitive behavior under different status or state
contexts.

| Framework | Bug files |
| --- | ---: |
| PyTorch | 18 |
| TensorFlow | 29 |
| Total | 47 |

| Dimension | Bug files |
| --- | ---: |
| Gradient tracking | 18 |
| Execution mode | 14 |
| Distribution strategy | 15 |
| Total | 47 |
