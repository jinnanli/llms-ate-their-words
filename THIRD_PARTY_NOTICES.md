# Third-party notices

The evaluation framework and prompt templates are distributed through
[Microsoft Lost in Conversation](https://github.com/microsoft/lost_in_conversation/tree/c865793fe34a929d316119b0451d01bd9183bcfd).
The original [MIT license](LICENSE), including **Copyright (c) Microsoft
Corporation.**, is preserved byte-for-byte. It does not replace the terms of
components adapted from other projects.

The [file-level provenance audit](docs/THIRD_PARTY_PROVENANCE.md) covers all
20 task/evaluator Python files and 27 prompt templates. It records pinned source
URLs, modifications, redistribution conditions and unresolved questions.
The accompanying [audit record](docs/third_party_provenance_audit.json) preserves
the source comparisons and input hashes.

## Components and retained terms

| Component | Included material | License and attribution |
|---|---|---|
| BFCL / Gorilla | `eval_bfcl.py`, `actions_system_prompt.txt`, `apis_system_prompt_multiturn.txt` | [Apache-2.0](docs/licenses/bfcl-LICENSE.txt); BFCL/Gorilla contributors; adapted through Microsoft |
| LiveCodeBench | `eval_code.py`, code-format prompt in `task_code.py` | [MIT](docs/licenses/lcb-LICENSE.txt); Copyright (c) 2024 LiveCodeBench; modified evaluator |
| HumanEval | Adapted `reliability_guard` in `eval_code.py` | [MIT](docs/licenses/humaneval-LICENSE.txt); Copyright (c) OpenAI (https://openai.com) |
| Spider test-suite SQL evaluation | `eval_spider_exec.py`, `eval_spider_parse.py` | [Apache-2.0](docs/licenses/spider-LICENSE.txt); test-suite-sql-eval contributors; execution wrapper modified, parser implementation unchanged |
| Google Research ToTTo | `utils_totto.py` HTML helpers | [Apache-2.0](docs/licenses/totto-LICENSE.txt); Copyright 2018 The Google AI Language Team Authors.; modified HTML assembly |
| Salesforce SummHay | Adapted scoring helpers and three Summary prompts | [Apache-2.0](docs/licenses/summhay-LICENSE.txt); Salesforce/SummHay contributors; modifications recorded in source/adjacent prompt `.license` files |
| EleutherAI lm-evaluation-harness | GSM8K extraction/normalization in `task_math.py` | [MIT](docs/licenses/harness-LICENSE.txt); Copyright (c) 2020 EleutherAI |

Retain these license texts, upstream attribution and modification notices when
redistributing the corresponding material. The inspected Apache repositories
have no relevant standalone NOTICE file; the audit does not invent copyright
holders or substitute the license appendix's placeholder for an actual notice.
Other prompt templates have a verified Microsoft distribution source under MIT;
their individual paths are recorded in the audit.

## Publication actions

The historical Stack Overflow-derived `Capturing` class has been removed from
the release, including its call site. Output capture now uses Python's
`redirect_stdout`, `StringIO` and `patch.object` directly. Regression tests cover
the actual evaluator call pattern; they do not bundle the removed code. See
[capture validation](docs/CODE_CAPTURE_VALIDATION.md). This is a replacement,
not a claim that the historical snippet was MIT-licensed.

The author explicitly confirmed licensing authority for project-authored additions
and satisfaction of the redistribution conditions for adapted Actions/Summary
prompts on 2026-11-20. Their adjacent
`.license` files preserve attribution/modification notices without changing the
model input. Historical import dates for some components remain unverified;
the audit distinguishes those dates from the pinned public comparison sources.

Microsoft's MIT notice applies to its inherited framework, not authorship of
this project's additions. BFCL, Spider, ToTTo and SummHay portions retain their
Apache-2.0 terms; LiveCodeBench, HumanEval and EleutherAI portions retain their
MIT notices. Adapted prompts retain the adjacent `.license` notices. The
MIT licensing of project-added replay, analysis, release-check and regression
code, including the stdout replacement, is authorized by the author. The MIT
permission and warranty terms in the root LICENSE apply to these additions;
Microsoft's copyright attribution remains specific to its inherited work.
This records the author's confirmation, not an independent legal determination.

## External data and safe execution

Benchmark corpora, Spider SQLite databases, model weights, full generated
conversations and API request logs are not bundled. Compact tables contain
experimental identifiers and measurements. Obtain external assets from their
original distributions under their own licenses and model/API terms; evaluator
code licensing is not a grant to redistribute those assets.

The Code evaluator executes untrusted programs. Its reliability guard is not a
security sandbox; use an isolated disposable execution environment.
