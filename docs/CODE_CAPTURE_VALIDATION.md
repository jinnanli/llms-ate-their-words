# Code stdout capture replacement

The historical `Capturing` list/context-manager class was used only by
`grade_stdio`. It is removed, not renamed or copied into a helper/test fixture.
The caller now uses standard-library `StringIO`, `redirect_stdout` and
`patch.object`. It reads the full buffer before leaving the context; the outer
StringIO context closes it after stdout restoration. Scoring, decimal matching,
error codes, timeout handling and call-based evaluation are unchanged.

The program-facing close override retains an unusual legacy behavior: it accepts
one argument and returns 1, while a zero-argument close is a runtime error. This
avoids silently changing scores for programs that close stdout. The diagnostic
TypeError function name changes with the implementation; classification remains
Runtime Error / -4. Wall-clock execution times are naturally not identical.

## Regression verification

From the repository root, with the documented CPU dependencies:

```sh
python code/review/test_code_capture.py
python code/release_checks.py
```

The capture suite covers 13 synthetic program cases: exact output, multiple
test inputs, whitespace, blank lines, numeric equivalence, empty output,
successive writes, wrong value/length, runtime exception, SystemExit and both
legacy close call forms. Additional tests cover timeout classification and
KeyboardInterrupt propagation. Stdout restoration is checked after each case.

During this repair, `--compare-index` loaded the pre-change evaluator directly
from the local Git staging index. All 13 cases produced the same scores and
scoring metadata (excluding runtime duration and implementation-named exception
text). That one-time comparison completed before staging the replacement; it
does not constitute a repeatable historical baseline after the index is updated.
The normal tests have explicit expectations and need no Git history.

All nine existing offline release commands passed. The 433 frozen numerical
fields (208 L1 + 225 L3) matched using the existing checker tolerances
(`abs_tol=1e-8`, `rel_tol=1e-6`); Review reference comparisons also passed.
Frozen measurements were not edited. These checks establish regression behavior
and frozen-statistic consistency, not re-execution of every historical generated
Code solution: full benchmark programs/tests are external to the compact release.

No model calls, GPU experiments or benchmark sample selection were performed.
The author confirmed MIT publication rights for the newly written implementation
and tests on 2026-11-20, as recorded in the root notices.
