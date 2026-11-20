# P7-R Turn Surgery Outcome Contract

## Frozen score contrast

For every completed branch, the formal implementation computes:

`effect = surgery_score - original_SHARDED_score`

No epsilon or post-hoc threshold is used.

- **POSITIVE / improve:** `effect > 0`
- **NO_CHANGE:** `effect == 0`
- **NEGATIVE / worsen:** `effect < 0`

These definitions are recovered from the formal P5 analysis code and are unchanged in P7-R.

## Task-specific evaluator contract

| Task | Evaluator scale | Score type | Failure | Success | Fail → success |
|---|---:|---|---|---|---|
| Actions | 0 or 1 | Binary function-call validity | 0 | 1 | original 0 and surgery 1 |
| Code | 0 or 1 | Binary pass@1 over frozen tests | 0 | 1 | original 0 and surgery 1 |
| Database | 0 or 1 | Binary SQL execution match | 0 | 1 | original 0 and surgery 1 |
| Math | 0 or 1 | Binary exact normalized answer match | 0 | 1 | original 0 and surgery 1 |
| Data-to-Text | `[0,1]` | Continuous BLEU/100 | No formal binary state | No formal binary state | Not used |
| Summary | `[0,1]` | Continuous joint coverage–citation score | No formal binary state | No formal binary state | Not used |

The 144 reported fail-to-success branches therefore come only from the four binary tasks. A trajectory can contribute more than one branch, so the corresponding trajectory-level count is reported separately: 76 unique trajectories.

## Statistical unit

Branch outcomes are descriptive, but their uncertainty is computed by resampling whole trajectories and retaining all nested branches. The 1,440 completed branches are not treated as 1,440 independent experimental units.

## Scientific boundary

A positive branch identifies a conversational-history turn whose neutralization improves the final task score under frozen-user replay. It does not identify a hidden reasoning state or automatically coincide with a human semantic first error.
