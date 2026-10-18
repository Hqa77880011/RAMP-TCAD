# RAMP-TCAD

Implementation of **Risk-aware multi-fidelity optimization for technology computer-aided design of wide-bandgap and gate-all-around devices**.

## Installation

Use Python 3.11+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Create `campaign.json`, a JSON list with one entry per design/fidelity evaluation. Required fields are `id`, `technology`, `x`, `fidelity`, `completed`, `runtime`, `recovery`, and `initial`. Successful records additionally contain `objectives` and `constraints`; failed records need neither response. `runtime` is compute exposure until termination, `recovery` the additional measured recovery exposure, and optional `censored: true` denotes timeout/right censoring. Use one common high-fidelity-equivalent unit for all cost fields. Constraint values are feasible at or below zero, and all objectives use minimization, so negate breakdown voltage/on-current when maximizing them.

The initial set must include at least two completed target-technology high-fidelity simulations. To use an auxiliary fidelity, include at least two completed paired designs at that fidelity and high fidelity with identical `x`. Set `initial: true` only for the registered shared seed set. Held-out candidate outcomes remain hidden from surrogate fitting and acquisition until selected by replay.

Optional `descriptor` arrays contain physically meaningful dimensionless quantities with identical meaning across technologies. They are required for shared models. Shared objective/constraint coordinates must likewise have aligned physical meanings and units or a registered dimensionless response mapping. Do not simply concatenate unrelated raw SiC and GAAFET geometry parameters. Only successful high-fidelity observations enter the shared response model; completion and runtime models remain target-technology-specific. Auxiliary-technology records belong in the known initial set and are not launched as target-device candidates.

Edit `config.json` to register design bounds, objective normalization, target technology, ordered fidelities, hypervolume reference, safe-launch risk, boundary weight, and Monte Carlo count. The supplied unit bounds illustrate schema; replace them with actual device definitions before optimization. Fix objective normalization and hypervolume reference before comparing campaigns.

## Run and evaluate

```bash
ramp-tcad --data campaign.json --config config.json --budget 100 --output outputs/full --seed 0
```

## Ablations

Supported controls: `hf`, `mfbo`, `no_failure`, `no_boundary`, `no_reliability`, `no_transfer`, `raw_sharing`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
ramp-tcad --data campaign.json --config config.json --budget 100 --output outputs/hf --seed 0 --variant hf
```

## Outputs

`history.json`, `acquisition.json`, `observed.json`, `metrics.json` and `front_progress.png` retain revealed campaign records, allocation/cost history and safe-front summaries. The command replays supplied TCAD records; it does not launch a TCAD solver.

Keep `config.json` with the run; it contains model/data settings and the seed where the CLI does not expose one. Use separate output directories for different seeds/conditions.

Source inspection: Python 3.13.5, numpy 2.1.3, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
