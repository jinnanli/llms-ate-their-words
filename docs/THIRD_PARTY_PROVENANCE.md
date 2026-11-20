# File-level third-party provenance audit

Audit date: 2026-11-20. Scope: all 20 Python files in `code/evaluation/tasks/`
and all 27 text templates in `prompts/`, including embedded Code/Math prompts.
This is an evidence inventory, not a legal opinion or a blanket clearance.

## Method and status

Each release file was compared with the corresponding Microsoft file at
`c865793fe34a929d316119b0451d01bd9183bcfd`. All 47 have an upstream counterpart:
4 byte-identical, 39 differing only in line-ending serialization, and 4 modified
before this audit. All 27 prompt templates match Microsoft after LF normalization.
The four pre-existing modifications are `eval_code.py`, `eval_summhay.py`,
`task_summary.py`, and `tasks.py`. This audit adds attribution comments only to
six evaluators without changing their ASTs. A subsequent pre-publication repair
replaced Code stdout capture using standard-library context managers. Prompt
bytes remain unchanged. The comparison hashes below describe the initial audit;
see `CODE_CAPTURE_VALIDATION.md` for the replacement and its regression checks.

Pinned comparisons below identify public evidence, not necessarily the historical
import commit. BFCL and LiveCodeBench commits are explicitly cited by the source;
Spider, ToTTo and SummHay revisions are comparison snapshots resolved during this
audit. Distinctive functions and prompt text were inspected as well as diffs;
similarity alone is not treated as proof of authorship.

- **VERIFIED**: immediate public source and its license inspected. Redistribution
  is conditional on the listed notices and upstream terms.
- **NEEDS_AUTHOR_ACTION**: source is known but a licensing condition remains unresolved.
- **UNCERTAIN**: the exact historical revision/rights beyond the inspected source
  cannot be established. Such uncertainty is stated separately from source matching.

MIT components require preservation of the copyright and permission notice.
Apache-2.0 components require a license copy, retention of applicable notices,
and identification of modifications (section 4). No relevant standalone NOTICE
file was found in the checked upstream trees. Placeholder copyright lines in
Apache's appendix are not actual author attributions; none have been invented.

## File inventory

`Source` is the immediate Microsoft source. `Earlier source` traces additional
embedded/adapted material. Empty package initializers are listed for completeness.
Unchanged means unchanged *text* relative to Microsoft, ignoring CRLF/LF.

| File | Source | Earlier source | License | Modification / comparison | Redistribution / status | Required attribution |
|---|---|---|---|---|---|---|
| `code/evaluation/tasks/__init__.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/__init__.py) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `code/evaluation/tasks/actions/__init__.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/actions/__init__.py) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `code/evaluation/tasks/actions/eval_bfcl.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/actions/eval_bfcl.py) | [bfcl_utils](https://github.com/ShishirPatil/gorilla/blob/c5ffaf09fd3556e88aed947a3aefde173ca31611/berkeley-function-call-leaderboard/bfcl/model_handler/utils.py); [bfcl_ast](https://github.com/ShishirPatil/gorilla/blob/c5ffaf09fd3556e88aed947a3aefde173ca31611/berkeley-function-call-leaderboard/bfcl/eval_checker/ast_eval/ast_checker.py) | Apache-2.0 + MIT adapter | Modified: combined helpers/checker, inlined constants; source comments added in this release. | VERIFIED: redistribution subject to listed licenses and notices | BFCL/Gorilla contributors; Microsoft Corporation |
| `code/evaluation/tasks/actions/task_actions.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/actions/task_actions.py) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `code/evaluation/tasks/code/__init__.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/code/__init__.py) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `code/evaluation/tasks/code/eval_code.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/code/eval_code.py) | [lcb_testing](https://github.com/LiveCodeBench/LiveCodeBench/blob/998c52d394b836f15fff3b9a29866191108ff81b/lcb_runner/evaluation/testing_util.py); [HumanEval execution.py](https://github.com/openai/human-eval/blob/b9453e8f1865df0a0057a1383767caa400acfbcf/human_eval/execution.py) | MIT inherited portions and author-authorized project additions | Modified: evaluator adaptation, platform alarms, and HumanEval guard. Stack Overflow-derived Capturing removed; stdout uses standard-library context managers with regression-tested behavior. | VERIFIED inherited MIT sources; Capturing removed; author authorization confirmed 2026-11-20 | Copyright (c) 2024 LiveCodeBench; Copyright (c) OpenAI (https://openai.com); Microsoft Corporation |
| `code/evaluation/tasks/code/task_code.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/code/task_code.py) | [lcb_prompt](https://github.com/LiveCodeBench/LiveCodeBench/blob/998c52d394b836f15fff3b9a29866191108ff81b/lcb_runner/prompts/code_generation.py) | MIT | Embedded code-format prompt adapted from LiveCodeBench; adapter otherwise unchanged from Microsoft except EOL. | VERIFIED: redistribution subject to listed licenses and notices | Copyright (c) 2024 LiveCodeBench; Microsoft Corporation |
| `code/evaluation/tasks/data2text/__init__.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/data2text/__init__.py) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `code/evaluation/tasks/data2text/task_data2text.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/data2text/task_data2text.py) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `code/evaluation/tasks/data2text/utils_totto.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/data2text/utils_totto.py) | [totto_utils](https://github.com/google-research/language/blob/865fae65f63ef7e6b2989d4ff8b47f61750415a8/language/totto/table_to_text_html_utils.py) | Apache-2.0 | Modified: removed style/header helpers and simplified HTML output; original Google header retained. | VERIFIED: redistribution subject to listed licenses and notices | Copyright 2018 The Google AI Language Team Authors. |
| `code/evaluation/tasks/database/__init__.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/database/__init__.py) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `code/evaluation/tasks/database/eval_spider_exec.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/database/eval_spider_exec.py) | [spider_exec](https://github.com/taoyds/test-suite-sql-eval/blob/e97acc546ecbee8fa27fa8dbf025ef61493a876c/exec_eval.py) | Apache-2.0 + MIT adapter | Modified: imports, logging, and 100-candidate cap inherited from Microsoft. Attribution comments added here. | VERIFIED: redistribution subject to listed licenses and notices | test-suite-sql-eval contributors; Microsoft Corporation (adapter) |
| `code/evaluation/tasks/database/eval_spider_parse.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/database/eval_spider_parse.py) | [spider_parse](https://github.com/taoyds/test-suite-sql-eval/blob/e97acc546ecbee8fa27fa8dbf025ef61493a876c/parse.py) | Apache-2.0 | Exact LF-normalized implementation match before this audit; renamed and attribution comments added. | VERIFIED: redistribution subject to listed licenses and notices | test-suite-sql-eval contributors; no named copyright line in checked source/LICENSE |
| `code/evaluation/tasks/database/task_database.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/database/task_database.py) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `code/evaluation/tasks/math/__init__.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/math/__init__.py) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `code/evaluation/tasks/math/task_math.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/math/task_math.py) | [harness_metrics](https://github.com/EleutherAI/lm-evaluation-harness/blob/52df63b7b30da53c481ed9090598d9189fab1d91/lm_eval/api/metrics.py); [GSM8K YAML](https://github.com/EleutherAI/lm-evaluation-harness/blob/bb098f13b05e361f01a5afe7b612779ce362b3f2/lm_eval/tasks/gsm8k/gsm8k.yaml) | MIT | Adapted extraction/normalization; direct source comments also pin gsm8k.yaml at bb098f13b05e361f01a5afe7b612779ce362b3f2. | VERIFIED: redistribution subject to listed licenses and notices | Copyright (c) 2020 EleutherAI; Microsoft Corporation |
| `code/evaluation/tasks/summary/__init__.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/summary/__init__.py) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `code/evaluation/tasks/summary/eval_summhay.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/summary/eval_summhay.py) | [summhay_eval](https://github.com/salesforce/summary-of-a-haystack/blob/837ce0a96d05fd21ea3360c44d83d39fa743c70a/utils_eval.py) | Apache-2.0 + MIT adapter | Modified: scoring integration, trimming, evaluator calls; local call_role metadata added before audit. | VERIFIED: redistribution subject to listed licenses and notices | Salesforce/SummHay contributors; Microsoft Corporation (adapter) |
| `code/evaluation/tasks/summary/task_summary.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/summary/task_summary.py) | - | MIT (Microsoft distribution) | Modified from Microsoft; see notes below. | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `code/evaluation/tasks/tasks.py` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/tasks/tasks.py) | - | MIT (Microsoft distribution) | Modified from Microsoft; see notes below. | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/actions/actions_full_prompt.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/actions/actions_full_prompt.txt) | - | MIT (Microsoft distribution) | Unchanged bytes. | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/actions/actions_system_prompt.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/actions/actions_system_prompt.txt) | [bfcl_constant](https://github.com/ShishirPatil/gorilla/blob/c5ffaf09fd3556e88aed947a3aefde173ca31611/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py) | Apache-2.0 + MIT adaptation | Adapted DEFAULT_SYSTEM_PROMPT: function placeholder changed, multiturn completion paragraph omitted, spacing changed. | VERIFIED: redistribution subject to listed licenses and notices | BFCL/Gorilla contributors; Microsoft Corporation (adaptation) |
| `prompts/actions/apis_system_prompt_multiturn.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/actions/apis_system_prompt_multiturn.txt) | [bfcl_constant](https://github.com/ShishirPatil/gorilla/blob/c5ffaf09fd3556e88aed947a3aefde173ca31611/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py) | Apache-2.0 | Matches DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC wording; whitespace layout differs. | VERIFIED: redistribution subject to listed licenses and notices | BFCL/Gorilla contributors |
| `prompts/actions/bfcl_lazification_conversational.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/actions/bfcl_lazification_conversational.txt) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/actions/bfcl_lazification_segment.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/actions/bfcl_lazification_segment.txt) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/data2text/data2text_full_prompt.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/data2text/data2text_full_prompt.txt) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/data2text/data2text_system_prompt.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/data2text/data2text_system_prompt.txt) | - | MIT (Microsoft distribution) | Unchanged bytes. | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/database/database_full_prompt.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/database/database_full_prompt.txt) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/database/database_lazification_oneshot.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/database/database_lazification_oneshot.txt) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/database/database_system_prompt.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/database/database_system_prompt.txt) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/lazification_conversational.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/lazification_conversational.txt) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/lazification_gradual.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/lazification_gradual.txt) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/lazification_segment.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/lazification_segment.txt) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/lazification_verification.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/lazification_verification.txt) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/math/gsm8k_lazification_conversational.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/math/gsm8k_lazification_conversational.txt) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/math/gsm8k_lazification_segment.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/math/gsm8k_lazification_segment.txt) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/math/math_full_prompt.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/math/math_full_prompt.txt) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/math/math_system_prompt.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/math/math_system_prompt.txt) | - | MIT (Microsoft distribution) | Unchanged bytes. | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/summary/summary_full_prompt_conv.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/summary/summary_full_prompt_conv.txt) | [summhay_conv](https://github.com/salesforce/summary-of-a-haystack/blob/837ce0a96d05fd21ea3360c44d83d39fa743c70a/prompts/summarization_conv.txt) | Apache-2.0 + MIT adaptation | Modified upstream: template variable names and participant wording. No prompt text changed here; adjacent .license describes adaptation. | VERIFIED: redistribution subject to listed licenses and notices | Salesforce/SummHay contributors; Microsoft Corporation (adaptation) |
| `prompts/summary/summary_full_prompt_news.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/summary/summary_full_prompt_news.txt) | [summhay_news](https://github.com/salesforce/summary-of-a-haystack/blob/837ce0a96d05fd21ea3360c44d83d39fa743c70a/prompts/summarization_news.txt) | Apache-2.0 + MIT adaptation | Modified upstream: template variable names. No prompt text changed here; adjacent .license describes adaptation. | VERIFIED: redistribution subject to listed licenses and notices | Salesforce/SummHay contributors; Microsoft Corporation (adaptation) |
| `prompts/summary/summary_system_prompt.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/summary/summary_system_prompt.txt) | - | MIT (Microsoft distribution) | Unchanged bytes. | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/summary/summhay_evaluation.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/summary/summhay_evaluation.txt) | [summhay_prompt](https://github.com/salesforce/summary-of-a-haystack/blob/837ce0a96d05fd21ea3360c44d83d39fa743c70a/prompts/eval_summhay.txt) | Apache-2.0 | Same source lines as Salesforce; terminal newline differs. No prompt text changed here. | VERIFIED: redistribution subject to listed licenses and notices | Salesforce/SummHay contributors |
| `prompts/system_answer_extraction.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/system_answer_extraction.txt) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/system_answer_extraction_gen.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/system_answer_extraction_gen.txt) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/system_answer_extraction_prefix_suffix.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/system_answer_extraction_prefix_suffix.txt) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/system_turn_categorization.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/system_turn_categorization.txt) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |
| `prompts/user_agent.txt` | [Microsoft](https://github.com/microsoft/lost_in_conversation/blob/c865793fe34a929d316119b0451d01bd9183bcfd/prompts/user_agent.txt) | - | MIT (Microsoft distribution) | Unchanged text (EOL-only differences). | VERIFIED: permitted subject to MIT notice retention | Copyright (c) Microsoft Corporation.; retain root LICENSE |

## Modifications and unresolved items

1. **Code / Capturing (REMOVED).** The historical evaluator linked to
   [kindall's answer](https://stackoverflow.com/a/16571630). That class and its
   call site have been removed from the release. The replacement uses
   `redirect_stdout`, `StringIO` and `patch.object` directly, without copying the
   answer's context-manager implementation. No copy of the old class is bundled
   in regression tests. This resolves the identified snippet in the release
   source, rather than relicensing it. Historical provenance remains recorded;
   the author confirmed MIT publication rights for new project code on 2026-11-20.
2. **Code / reliability_guard.** Comparison with the pinned HumanEval implementation
   confirms closely corresponding body/docstring: the annotation was removed,
   formatting changed, and `builtins.exit = None` commented out. The OpenAI MIT
   notice is now supplied in addition to LiveCodeBench's MIT notice.
3. **BFCL and Summary prompts.** Two Actions prompts trace to BFCL constant.py:
   one retains the multiturn instructions; the other removes that paragraph and
   changes the functions placeholder. The Summary scoring prompt has identical source lines to SummHay
   (a terminal newline differs); the conversation/news prompts change template
   placeholders and participant wording. Adjacent `.license` files carry source,
   license and modification notices without changing runtime prompt input.
   The author confirmed the redistribution conditions and this sidecar packaging
   on 2026-11-20; adding text directly to model inputs would
   change the experiment and was deliberately avoided.
4. **Existing project changes.** `eval_code.py` adds platform-aware alarms;
   `eval_summhay.py` adds evaluator-call metadata; `task_summary.py` allows an
   evaluator environment setting; `tasks.py` adds entropy task dispatch. These
   changes predate this audit. The author confirmed authority to license
   their additions; Microsoft copyright is not assigned to new project work.
5. **Historical import revisions (UNCERTAIN).** Public matching sources establish
   redistribution evidence, not the exact date or chain of every historical
   import. The immediate Microsoft source is verified for all 47 files. No
   independent earlier origin was identified for the other 22 prompt templates;
   their verified distribution source is Microsoft, not an invented benchmark
   copyright attribution. This audit does not certify authorship of example
   text beyond the inspected upstream licensing evidence.
6. **External assets.** Evaluator code licenses do not automatically cover BFCL,
   LiveCodeBench/HumanEval problems, ToTTo/Wikipedia content, Spider databases,
   GSM8K questions, or SummHay documents. These corpora are not bundled. Users
   obtain the original distributions and applicable data/model terms separately.

## Retained license texts

The root Microsoft `LICENSE` is unchanged. The following upstream license files
are reproduced byte-for-byte; SHA256 identifies each inspected copy. BFCL's
unrelated tree-sitter license and Google's unrelated frost data license are not
attributed to these components.

| Local license | Pinned source | SHA256 |
|---|---|---|
| [docs/licenses/bfcl-LICENSE.txt](../docs/licenses/bfcl-LICENSE.txt) | [upstream](https://github.com/ShishirPatil/gorilla/blob/c5ffaf09fd3556e88aed947a3aefde173ca31611/LICENSE) | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| [docs/licenses/lcb-LICENSE.txt](../docs/licenses/lcb-LICENSE.txt) | [upstream](https://github.com/LiveCodeBench/LiveCodeBench/blob/998c52d394b836f15fff3b9a29866191108ff81b/LICENSE) | `104099948b78ebc36f2e951b3a7973b956adfa7d39aaec0d1512f97a2686d15d` |
| [docs/licenses/spider-LICENSE.txt](../docs/licenses/spider-LICENSE.txt) | [upstream](https://github.com/taoyds/test-suite-sql-eval/blob/e97acc546ecbee8fa27fa8dbf025ef61493a876c/LICENSE) | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| [docs/licenses/totto-LICENSE.txt](../docs/licenses/totto-LICENSE.txt) | [upstream](https://github.com/google-research/language/blob/865fae65f63ef7e6b2989d4ff8b47f61750415a8/LICENSE) | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| [docs/licenses/summhay-LICENSE.txt](../docs/licenses/summhay-LICENSE.txt) | [upstream](https://github.com/salesforce/summary-of-a-haystack/blob/837ce0a96d05fd21ea3360c44d83d39fa743c70a/LICENSE) | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| [docs/licenses/harness-LICENSE.txt](../docs/licenses/harness-LICENSE.txt) | [upstream](https://github.com/EleutherAI/lm-evaluation-harness/blob/52df63b7b30da53c481ed9090598d9189fab1d91/LICENSE.md) | `a806e42547620dffbc2ec0333322c8b3523b4af11e059be14ef81aa5b1ae021f` |
| [docs/licenses/humaneval-LICENSE.txt](../docs/licenses/humaneval-LICENSE.txt) | [upstream](https://github.com/openai/human-eval/blob/b9453e8f1865df0a0057a1383767caa400acfbcf/LICENSE) | `bcba3de214851cce46ed5af42d6698044616eeace887c3231bc7a20474ab639e` |
