# What if LLMs Ate Their Words: Causal History Effects in Multi-Turn Interaction

**Accepted at WSDM 2027** | [Paper (arXiv)](https://arxiv.org/abs/2609.05882)

## Overview

This work studies how an LLM's own previous replies affect its performance in
multi-turn interaction. Whole-History Neutralization measures the overall effect
of assistant history, while Turn Surgery locates influential individual turns.
Both interventions keep the recorded user-message sequence fixed. Across tasks,
models, and turns, assistant history can help, hurt, or leave performance unchanged.
The repository provides behavioral experiments, Qwen3-14B internal analyses,
and offline reproduction tools.

## Installation

Use **Python 3.11**. Run all commands below from the repository root.

```bash
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or
`.venv\Scripts\Activate.ps1` in PowerShell, then install the CPU dependencies:

```bash
python -m pip install -r requirements.txt
```

Dependencies are pinned in `requirements.txt`. Keep pandas 2.2.3:
CSV float parsing affects classifications at the exact epsilon=.01 boundary.
API and GPU dependencies are provided separately in `requirements-api.txt` and
`requirements-gpu.txt`.

## Quick Start

Validate the included tables and reproduce the Null-Replay statistics:

```bash
python code/validate_package.py
python code/review/reproduce_review.py
```

Expected messages are `PACKAGE_VALIDATION: PASS` and `REVIEW_REPRODUCTION: PASS`.
The second command writes CSVs and plots to `results/reproduced/review/`.
These commands run on CPU using the included data, without API credentials.

## Reproducing Results

### Behavioral analyses

```bash
python code/statistics/analyze_phase_l1.py
python code/statistics/analyze_phase_l3.py
```

| Analysis | Input | Output directory |
|---|---|---|
| Turn Surgery structure, position effects, and clustered statistics | `data/rq3/turn_surgery_branches_compact.csv` | `results/reproduced/phase_l1/` |
| Whole-history neutralization effects and outcome transitions | `data/rq2/rq2_paired_trajectory.csv` | `results/reproduced/phase_l3/` |

The inputs cover 237 trajectories / 1,440 Surgery branches and 2,973
SHARDED–SHORT pairs, respectively. Reference statistics are in
`results/rq2/` and `results/rq3/`. The original Surgery `delta` is
**Surgery − Original**; the Null-Replay analysis below estimates **Surgery − Null**.

RQ1 cell means and intervals are in `data/rq1/rq1_cell_summary.csv` and
`results/combined/P5_FINAL_STATISTICAL_RESULTS.json`. The Null-Replay reproduction
command cross-checks all 90 model/task/mode means and denominators and writes
`rq1_task_macro_point_estimates.csv`. Recomputing the original RQ1 bootstrap
intervals with `code/turn_surgery/p5_analyze.py` requires the source-trajectory
and stability manifests. The original `code/statistics/analyze_phase_l2.py`
likewise requires the FULL-trajectory join; cell-level correlation is available
through the compact-data workflow below.

### Null-Replay and supporting analyses

On 40 selected Qwen3-14B trajectories across five tasks, Turn Surgery improves
the average score over unedited replay by **0.0847 (95% CI: [0.0270, 0.1543])**.

```bash
python code/review/reproduce_review.py
```

| Input under `data/review/` | Analysis |
|---|---|
| `rq3_branches.csv`, `rq3_trajectories.csv` | Surgery − Null on 40 trajectories / 219 branches; trajectory-equal and branch-weighted estimates |
| `rq2_pairs.csv` | SHORT − Null, MATCHED − Null, and MATCHED − SHORT on 40 trajectories / 120 conditions |
| `stability_pairs.csv` | Three repeats for each of five preselected branches |
| `rq1_rq2_cells.csv` | RQ1–RQ2 association across 30 model/task cells |
| `token_by_turn.csv`, `token_by_trajectory.csv` | Actual token-count summaries |

Outputs under `results/reproduced/review/` include `rq3_statistics.csv`,
`rq2_statistics.csv`, leave-one-task-out and leave-one-trajectory-out tables,
stability summaries, exact/practical outcome classes, correlation summaries,
token summaries, and PNG plots. Recomputed tables are checked against
`results/review/`.

The analyses use 10,000 percentile bootstrap replicates, resampling whole
trajectories. RQ3 primary statistics use seed 20261118; RQ2 and sensitivity
analyses use 20261119. Exact signs and continuous-task epsilon=.01 classes are
reported separately. RQ2 has eight trajectories per task, so its point mean
also equals the five-task macro mean. Correlation uncertainty uses descriptive
crossed model/task resampling; the input table retains each experiment's cell
denominators.

### Figures and offline tests

```bash
python code/plotting/plot_figure1_framework.py
python code/plotting/plot_refined_figure2.py
python code/plotting/plot_refined_figures_3_5.py
```

The renderers use `data/figures/` and write Figures 1–5 as PDFs and PNGs to
`results/reproduced/figures/`. Supplied figures remain in `results/figures/`.

```bash
python code/review/test_replay.py
python code/release_checks.py
```

The first command tests replay construction and Code output capture. The second
runs all offline validation, statistics, figures, and numeric reference checks,
writing its report to `docs/release_validation.json`.

## Running Experiments

### Setup

```bash
python -m pip install -r requirements-api.txt
```

Export `OPENAI_API_KEY` and, for a compatible provider, `OPENAI_BASE_URL` in your
shell. [.env.example](.env.example) lists the configuration fields. Keep filled-in
credentials and live request logs private. Model configurations and evaluator
metadata are in `configs/`; prompt templates are in `prompts/`.

Prepare the benchmark at `data/sharded_instructions_600.json`. Each sample
contains `task_id`, `task`, `shards`, and the reference fields used by its adapter
in `code/evaluation/tasks/`. History interventions additionally require recorded
SHARDED conversations and the manifests described below.

### FULL, CONCAT, and SHARDED

```bash
python code/evaluation/run_simulations.py --dataset_file data/sharded_instructions_600.json --models Qwen/Qwen3-14B --tasks actions --N_full_runs 1 --N_concat_runs 1 --N_sharded_runs 1 --assistant_temperature 0 --user_temperature 0 --system_model gpt-5.6-luna --user_model gpt-5.6-luna --N_workers 1 --log_folder results/live/simulations
```

This runs one of each condition for the selected task and writes new trajectories
to `results/live/simulations/`. Pass temperatures explicitly, as library defaults
differ from the experimental settings. Configure model names to match your
provider and record any changes from `configs/p5_model_panel.yaml`.

### Whole-history neutralization and Turn Surgery

SHORT replaces non-final assistant replies with `Acknowledged.`. MATCHED repeats
that phrase to match each reply's whitespace-separated span count. Turn Surgery
replaces one non-final reply and regenerates every later assistant reply using
the stored user messages.

Provide CSV manifests with `trajectory_id`, `instruction_id`, `task`,
`tested_model`, `raw_sharded_path`, `SHARDED_score`, and, for Surgery,
`surgery_turn`; `repeat_index` is optional. Each `raw_sharded_path` points to
the original trace. The benchmark file is required for scoring and planning.

```bash
python code/interventions/p5_counterfactual_runner.py --mode neutral_ack --manifest external_raw/neutral_manifest.csv --output results/live/neutral.jsonl --dry-run
python code/interventions/p5_counterfactual_runner.py --mode length_matched_ack --manifest external_raw/matched_manifest.csv --output results/live/matched.jsonl --dry-run
python code/interventions/p5_counterfactual_runner.py --mode surgery --manifest external_raw/surgery_manifest.csv --output results/live/surgery.jsonl --dry-run
```

Check the planned population, then remove `--dry-run` to execute. This
runner uses the Summary evaluator alias `gpt-4o`. Use the following
adapter for new paired Null-Replay comparisons with fixed-version Summary scoring.

### Null-Replay controls

Turn Surgery controls regenerate all assistant replies after the target in both
the unedited and edited conditions (`--mode p0a`). Whole-history controls compare
Null/SHORT/MATCHED histories while regenerating only the final answer (`--mode p0b`).

Provide JSONL with one row per branch (`p0a`) or trajectory (`p0b`):
`pair_id`, `trajectory_id`, `task`, `model`, `source_instance_id`,
`original_messages`, `user_sequence_hash`, and `target_turn` (`p0a` only,
zero-based). Messages are ordered role/content objects with one system message
and matching user/assistant turns. Compute the user hash with
`code/review/p0_common.py`'s `user_sequence_hash`; `source_instance_id` joins
the benchmark's `task_id`.

```bash
python code/review/run_null_replay.py --mode p0a --manifest external_raw/null_replay_branches.jsonl --output-dir results/live/review-rq3
python code/review/run_null_replay.py --mode p0b --manifest external_raw/null_replay_trajectories.jsonl --output-dir results/live/review-rq2
```

These commands validate inputs. Add `--live` to generate and score. Generation
uses temperature 0 and max_tokens 1000. Summary scoring requires the actual
returned ID `gpt-4o-2024-08-06`; Math uses the `gpt-5.6-luna` answer extractor.
Check helper-model identities and configurations before analyzing a new run.

Each output directory contains condition/pair JSON files, response caches,
`requests.jsonl`, and `errors.jsonl`. Completed conditions are reused on resume.
An exclusive worker lock prevents overlapping runs; verify process state before
clearing a stale lock. Compute net effects from contemporaneous generated and
comparably scored conditions, keeping the original score as historical reference.

## Internal Analysis

The Qwen pipeline includes attention, counterfactual attention redistribution,
continuation margin, direct/full knockout, state propagation, and state restoration.
Use `Qwen/Qwen3-14B` at revision
`40c069824f4251a91eefaf281ebe4c544efd3e18`, with bfloat16, greedy decoding,
and thinking disabled.

Install a CUDA-compatible PyTorch build, then:

```bash
python -m pip install -r requirements-gpu.txt
```

Inputs include the reference dataset, exact user histories, native baseline and
surgery histories, and branch/pair manifests. L40-class hardware was used;
bf16 model weights occupy about 28 GB before KV-cache and probe activations.
Required memory grows with sequence length and the probe configuration.

```bash
python code/internal_analysis/run_qwen14b_formal_heldout_behavior.py --source-manifest external_raw/heldout.csv --source-payload external_raw/heldout.jsonl --dataset data/sharded_instructions_600.json --model Qwen/Qwen3-14B --revision 40c069824f4251a91eefaf281ebe4c544efd3e18 --thinking-mode disabled --output-dir results/live/heldout
python code/internal_analysis/run_qwen14b_state_restoration.py --pairs external_raw/restoration_pairs.csv --native-dir results/live/heldout --dataset data/sharded_instructions_600.json --model Qwen/Qwen3-14B --revision 40c069824f4251a91eefaf281ebe4c544efd3e18 --output-dir results/live/restoration --validate-inputs-only
```

The first command generates native behavior. The second validates restoration
inputs; remove `--validate-inputs-only` to run the probe. Its native directory
contains `server_native_baselines.jsonl` and `server_native_surgeries.jsonl`.
Restoration outputs record pair/layer/direction measurements and validation
records. Use `--help` for layer and sharding options. Keep discovery,
formal held-out, and prospective populations separate during aggregation.

## Data

| Directory | Contents |
|---|---|
| `data/rq1/`–`data/rq4/` | Compact behavioral and internal-analysis inputs |
| `data/review/` | Paired Null-Replay scores, repeat results, positions, and token counts |
| `data/figures/` | Frozen plotting tables |
| `results/` | Reference statistics and figures |
| `configs/`, `prompts/` | Model settings, schemas, and prompt templates |

Offline analyses use the included result tables. Running model experiments
requires the corresponding benchmark data, conversation histories, and model
access. Obtain BFCL, LiveCodeBench, ToTTo, Spider, GSM8K, and SummHay assets from
their original distributions under their respective terms. Compact measurements
are included here; benchmark corpora, full histories, databases, and weights
are external inputs. See the
[file-level source audit](docs/THIRD_PARTY_PROVENANCE.md) for evaluator and prompt origins.

- **Database:** prepare the compatible Spider **test-suite** files under
  `data/spider/databases/<db_id>/`. The evaluator executes reference and predicted
  SQL across that directory with `plug_value=True` and `keep_distinct=False`.
  Use the matching test-suite version, schemas, and file hashes rather than
  substituting ordinary Spider databases.
- **Code evaluation:** run generated programs in a disposable sandbox isolated
  from credentials and valuable data.
- **Null-Replay population:** the five-task Qwen subset contains selected
  degraded/replay-eligible trajectories: Actions, Code, D2T, Math, and Summary.
  Its paired estimates are separate from the original five-model results.
- **Length and model provenance:** MATCHED controls whitespace spans, not exact
  model tokens. Token summaries use the counts and tokenizer revision in
  `data/review/tokenizer_provenance.json`; re-tokenization requires the original
  text. Whole-prompt counts include the final question and exclude its answer.
  The historical remote chat template is unknown, and returned model aliases
  leave immutable service snapshots unverified.

## Citation & License

If you find this work useful, please cite our paper:

```bibtex
@article{li2026if,
  title={What if LLMs Ate Their Words: Causal History Effects in Multi-Turn Interaction},
  author={Li, Jinnan and Fu, Zheren and Wang, Yue and Li, Jinzhe and Wu, Yuan and Chang, Yi},
  journal={arXiv preprint arXiv:2609.05882},
  year={2026}
}
```

The paper is accepted at WSDM 2027; the citation above refers to the arXiv version.

Project additions are released under [MIT](LICENSE). Microsoft Corporation's
copyright notice is retained for the inherited framework. Third-party evaluators
and prompts retain their respective licenses and attribution, detailed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). External data and models follow
their own terms.
