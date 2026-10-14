# Proposed atomic preservation commit — not staged, committed or pushed

Branch: `breakdown-mechanisms`. Suggested message:
`Preserve verified parental-care and crowding experiments`.
One atomic commit keeps the runners, their result-dependent tests, checkpoints and
reproduction tools together. Do not cherry-pick incomplete subsets for a clean checkout.

Include:

- `.gitignore`, `requirements-reproduction.txt`.
- `experiments/final_rewards.py`, `controlled_care.py`, `verify_completed.py`,
  `analyze_controlled_care.py`, `report_completed.py`, `archive_controlled_care.py`.
- `tests/test_final_rewards.py`, `tests/test_controlled_care.py`.
- All 62 files in `results/final_reward_v1`: twenty trained checkpoints, twenty training
  histories, twenty evaluation reports, manifest and summary (12.14 MB total).
- `results/controlled_care_v1/manifest.json` and `summary.json`.
- All 21 files in `results/controlled_care_v1_archive`: twenty lossless compressed raw
  trial files and receipt manifest (about 7.04 MB plus receipt metadata).
- `results/controlled_care_v1_analysis/analysis.json` and `training_seed_rates.csv`.
- Pilot `manifest.json` and `summary.json`, documenting protocol development.
- All reviewed final figures, captions and verification in `results/report_figures_v1`.
- Verification, environment and hash receipts in `results/preservation_v1`.
- This plan, scientific review, reproduction commands, Phase 3 readiness and the complete
  per-file uncommitted inventory under `docs/`.

Exclude from the proposed commit, without deleting local files:

- Twenty large uncompressed Phase 2 trial JSON files: exact content is in the archive.
- Four pilot trial JSON files: all 96 trial records are identical to full-run records.
- Earlier Phase 2 plot PNG/PDF files and its presentation `REPORT.md`: final figures and
  scientific review supersede these duplicate presentation outputs. Retain the analysis
  JSON and seed-rate CSV. Do not add the whole old analysis directory indiscriminately.
- `/workspace/work` scripts, logs, montages, replay/reanalysis duplicates, virtual
  environments and caches. No required reproducer now depends on them.

All 34 tracked legacy result files remain unchanged and remain tracked. Nothing is
removed from Git history. Every proposed new file is below GitHub's 100 MB per-file
limit; no Git LFS or remote service is necessary for this approximately 20–21 MB addition.

After approval, use explicit paths for staging, review `git diff --cached --stat` and
`git diff --cached --name-status`, ensure no raw duplicate or legacy change is staged,
run the verifier with archive-only files available and run the full test suite. Confirm
the author before making the one commit. Verify its file list and push only to
`breakdown-mechanisms`, respecting branch protection. Do not force-push or touch `main`.

Configured local identity:
`Mariam Latsabidze <65814453+mariamlats@users.noreply.github.com>`.
This is the GitHub attribution address for `mariamlats`. It persists for future tasks
using this same checkout; a fresh clone does not inherit repository-local configuration.

The work is prepared for review, not yet durably preserved on GitHub. Approve the commit
and push (or make a separate verified backup) before Phase 3 execution. Future work
must use a new result directory and frozen, hash-verified Phase 1 checkpoints.
