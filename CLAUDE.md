# CLAUDE.md

## Environment
Use conda env `pred`. Prefix all Python with `conda run -n pred python`. When using SSH in EC2, use `python3.11` only.

## Architecture
MLB API → Parquet (S3/local) → feature stores → two modeling stacks:

- **`classical_learning/`** — classic ML on pre-engineered pregame features (win, YRFI, totals, run diff); pregame market making only; all 21 market families priced at inference via distribution integration
- **`deep_learning/`** — deep learning (GameTransformer) on live in-game state (pitch sequences, game context); consumes tensors from its own 20-artifact feature store. Training pipeline implemented (`mlb_dl/train_unified.py`). **Real-time repricing is the goal, NOT the current state: nothing serves this stack today.** `mlb_dl/inference_engine.py` builds `LiveGameModel` while training produces `GameTransformer` — the state_dict intersection is empty and it raises in `__init__`

Key invariants:
- **No leakage**: features use only data prior to `target_game_date`; standardizer fit on train only
- **Target status**: only `trainable` rows train; scratched players are `settles_last_fair`, never treated as zero outcomes
- **Game-index decay**: uses sequential game distance (λ_intra=0.015, λ_inter=0.30), not calendar days, to avoid penalizing offseason gaps
- **Batch contract**: all models are dict-in/dict-out with consistent tensor key names
- **Player identity**: hash-bucket embeddings (blake2b), not integer IDs — handles unseen players at inference
- **Update comments**: always update memory and codebase comments directly after new findings or modifications if the comment becomes stale
- **Give logging**: after launching a script, monitor it to make sure it bootsraps and starts, and give the bash command so that the user can ssh in and tail the log

## Scientific Rigor
Every constant, threshold, or architectural choice must be backed by one of:
1. Empirical validation on held-out metrics
2. Published research (cited)
3. First-principles derivation (shown)

Unvalidated values must be marked `# TODO: validate — placeholder`.

## Statistical Rigor
For every statistical method and test used, clearly state their assumptions and check if the data satisfies their assumptions.

## Code Philosophy
- **Comment WHY, not WHAT.** Name the hidden constraint, tradeoff, or non-obvious invariant.
- **Simplicity first.** No unrequested features, abstractions, or flexibility.
- **Surgical edits.** Touch only what the request requires. Don't improve adjacent code.
- **Think before coding.** State assumptions, surface ambiguity, ask before guessing.
- **Plan before implementing.** Understand the problem and expected behavior, consider high-level examples before coding, write adversial stress tests using edge cases.

## Bug-Fix & Testing Discipline
- **Failing test first.** Before fixing any bug, write a test that defines expected behavior and demonstrates the failure. No patch without a reproducing test.
- **Isolate before fixing.** Define expected behavior → write unit test → show it fails → identify root cause → then implement the fix. Do NOT force the test to pass by changing the test itself.
- **Adversarial stress tests after basics.** Once a fix passes its unit test, write adversarial edge cases (year boundaries, traded players, doubleheaders, cold-start, sparse data).
- **Prove issues before changing.** For design-level decisions (not clear-cut bugs), reproduce the issue with data and raise to the user before modifying code.

## Logging
Two handlers everywhere: file at `DEBUG` (granular), stdout at `INFO` (milestones). Never remove or weaken existing log statements.