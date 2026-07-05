# Empirical Study Numbers

This report is pinned to the paper audit dataset. The live GitHub collection scripts remain available, but this artifact reports the exact numbers used in the manuscript.

## Collection

| Metric | Count |
|---|---:|
| Raw issues returned by repository-label queries | 6,011 |
| Unique issues after URL deduplication | 5,897 |
| Correctness-relevant reports after keyword filtering | 1,122 |
| Fix-verified issues | 329 |
| Unverified issues | 793 |

## Table 1: Counts for the 329 Fix-Verified Issues

| Stratum | Framework | # Issues | State | Non (E) | U/DUP | %State |
|---|---|---:|---:|---:|---:|---:|
| Fix-verified | PyTorch | 194 | 125 | 58 | 11 | 64.4% |
| Fix-verified | TensorFlow | 135 | 81 | 47 | 7 | 60.0% |
|  | Total | 329 | 206 | 105 | 18 | 62.6% |

## Table 2: Distribution over the 206 State-Related Issues

| Primary dimension | PyTorch # | PyTorch % | TensorFlow # | TensorFlow % | Total # | Total % |
|---|---:|---:|---:|---:|---:|---:|
| A. Gradient tracking | 41 | 32.8% | 14 | 17.3% | 55 | 26.7% |
| B. Execution mode | 31 | 24.8% | 43 | 53.1% | 74 | 35.9% |
| C. Distribution strategy | 43 | 34.4% | 19 | 23.5% | 62 | 30.1% |
| D. Other state | 10 | 8.0% | 5 | 6.2% | 15 | 7.3% |
| A--C total | 115 | 92.0% | 76 | 93.8% | 191 | 92.7% |
