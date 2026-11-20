# Public release changes

Base: WSDM2027_supplementary_anonymous.zip (135 files). The original directory
layout, compact measurements, figures and implementation interfaces were kept.
This release does not include manuscript sources or a paper PDF.

- Replaced the anonymous landing page with a unified README covering usage;
  retained separate attribution and test records.
- Split/pinned CPU dependencies and retained separate API/GPU requirements.
- Added missing prompt templates needed by inherited evaluators.
- Added numeric-only review data, frozen references and a portable CPU analyzer.
- Retained the original replay constructors and added an opt-in external-history
  adapter; live generation is not claimed to have been release-tested.
- Isolated figure regeneration under results/reproduced/ rather than overwriting
  frozen supplied PDFs; removed private provider labels from logging defaults.
- Added ignore/attribute rules and synthetic replay tests.
- No raw history, request/response logs, account configuration, databases or
  weights were copied. No experiment or inference was run for this release.

Pandas 3 initially changed some exact-boundary classifications when reading
CSV floats. Pinning pandas 2.2.3 restored the frozen L1/L3 numeric results;
measurements and analysis thresholds were not edited. The recorded final test
environment is a release environment, not proof of historical API/GPU versions.

Author approval is still required for publication, citation metadata and the
third-party provenance items in THIRD_PARTY_NOTICES.md. This directory was
prepared locally; no GitHub push was performed.

Final third-party audit: compared all 20 task/evaluator Python files and 27 prompt
templates against pinned public sources. Added seven upstream license copies,
file-level provenance, six comment-only evaluator notices, and Actions/Summary prompt
attribution sidecars. Experimental logic, prompt inputs, root LICENSE and frozen
measurements were unchanged at that audit stage.

Subsequent pre-publication repair: removed the Stack Overflow-derived Capturing
class and used standard-library stdout redirection directly in grade_stdio.
Added synthetic scoring/exception regression tests; all nine offline checks and
433 frozen numerical comparisons passed. See docs/CODE_CAPTURE_VALIDATION.md.
On 2026-11-20 the author explicitly confirmed MIT publication rights for project
additions and the redistribution conditions for third-party prompts, and
authorized the initial commit and public push. Inherited notices remain intact.
