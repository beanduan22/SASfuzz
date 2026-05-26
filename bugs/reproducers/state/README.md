# State-Aware Library Bug Reproducers

This directory contains the anonymized state-related PyTorch and TensorFlow bug
reproducers counted in the paper RQ3 bug study. Each
`state_bug_###.py` file is a complete standalone script and does not import any
local helper file.

```bash
python bugs/reproducers/state/state_bug_001.py
```

## Counts

The artifact intentionally records only the executable reproducer, target
framework, and runtime-state dimension. It omits external source metadata and
maintainer outcome metadata.

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
