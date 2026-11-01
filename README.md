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

`history.json`, `acquisition.json`, `observed.json`, `metrics.json` and `front_progress.png` retain revealed campaign records, allocation/cost history and safe-front summaries. The basic command replays supplied TCAD records. The explicit `advanced live` entry below launches a registered solver adapter.

Keep `config.json` with the run; it contains model/data settings and the seed where the CLI does not expose one. Use separate output directories for different seeds/conditions.

## Surrogates and metrics

The replay optimizer fits objective/constraint GPs, fidelity discrepancies and completion/runtime models. Acquisition combines safe-front improvement, completion, reliability and boundary uncertainty. Metrics include Pareto hypervolume, regret, compute exposure and disjoint-bin Brier/ECE. The basic entry uses sequential finite-candidate Monte Carlo EHVI with independent outputs; the advanced entry supports joint batches and live workers.

## Joint batches, fantasies, Sobol and live workers

```bash
python -m pip install -e ".[advanced]"
python -m ramp_tcad.advanced replay --data campaign.json --config config.json --method qehvi --batch-size 2 --budget 100 --output outputs/batch --seed 0
python -m ramp_tcad.advanced replay --data campaign.json --config config.json --method nsga2 --batch-size 2 --budget 100 --output outputs/nsga2 --seed 0
python -m ramp_tcad.advanced replay --data campaign.json --config config.json --method safeopt --batch-size 2 --budget 100 --output outputs/safeopt --seed 0
```

Register `seconds_per_hf` (measured seconds per common high-fidelity-equivalent unit) in the config for these entries. They charge measured surrogate/acquisition overhead as well as worker termination/recovery, never reveal outcomes until the **whole batch is selected**, and retain `batch_plans.json`. The allocator can consume the last remaining exposure; no new worker starts after it exhausts the budget. Recorded allocator time remains the actual time, including any terminal overshoot.

[BoTorch qExpectedHypervolumeImprovement](https://botorch.readthedocs.io/en/stable/acquisition.html#botorch.acquisition.multi_objective.monte_carlo.qExpectedHypervolumeImprovement) evaluates joint feasible hypervolume improvement with Sobol posterior sampling (`--mc-samples 64`). The corrected/shared GP marginal means and variances come from the existing surrogate. A positive semidefinite cross-candidate covariance uses the fitted independent GP correlation rescaled to those marginals; outputs remain independent. Outputs use independent GP covariance blocks. Candidate batches are enumerated up to `--max-batches 128`, or a seeded random subset of combinations is scored. Batch completion is the product of per-worker completion probabilities, reliability their mean, cost their sum, and boundary value the sum of marginal boundary uncertainties. These batch aggregation conventions are configurable through batch size and the existing boundary coefficient.

`--pending pending.json` accepts metadata for already launched, unresolved runs. Their joint posterior outcomes are sampled (`--fantasies 4`), each draw **conditions** the remaining Gaussian posterior and updates the feasible fantasy frontier, and joint qEHVI is averaged across those scenarios. It does not insert fake completed responses into the campaign. Hypothetical observations use `fantasy_noise=1e-6` by default in normalized output coordinates.

[NSGA-II from pymoo](https://pymoo.org/algorithms/moo/nsga2.html) uses its actual ask/tell evolutionary algorithm, initialized with the shared completed high-fidelity seeds. Offspring are projected to the nearest unseen high-fidelity candidate in a replay catalog; live catalogs use Sobol candidates. Only revealed evaluations enter tell. A noncompleted offspring gets an **optimizer-only** worst-objective penalty (`objective_upper + objective_span`) and positive constraint penalty 1; its physical record continues to have no objectives/constraints. [SafeOpt](https://proceedings.mlr.press/v37/sui15.html) is a finite-domain implementation with confidence-certified constraints, potential minimizers and Lipschitz safe-set expanders. It uses equal normalized objective weights, beta 2 and Lipschitz constant 1 by default (`safeopt_weights`, `safeopt_beta`, `safeopt_lipschitz` in config). Set these scalarization and domain parameters in the config before a campaign.

For a live campaign, add an explicit solver configuration, replacing paths and physical bounds with the registered simulator adapter:

```json
{
  "seconds_per_hf": 100.0,
  "constraint_scale": [1.0],
  "solver": {
    "argv": ["python", "/absolute/path/to/adapter.py", "--request", "{request}", "--response", "{response}"],
    "timeout_seconds": 3600
  }
}
```

```bash
python -m ramp_tcad.advanced live --data seed_campaign.json --config device_config.json --method qehvi --candidates 32 --batch-size 2 --budget 100 --output outputs/live --seed 0
```

Each iteration draws scrambled Sobol physical designs within registered bounds and enumerates fidelities. Optional `descriptor_indices`, `descriptor_offset`, `descriptor_scale` define a registered dimensionless descriptor mapping; no raw cross-device mapping is inferred. The launcher uses an argument list with `shell=False`, a separate numbered work directory, explicit request/result JSON and stdout/stderr. The adapter must return `{"completed":true,"objectives":[...],"constraints":[...]}` with registered finite response dimensions; failed runs return `completed:false` or a nonzero exit code. The timeout is the smaller of the registered worker limit and remaining campaign exposure. It terminates only the owned worker process tree, records measured right-censored runtime, and never invents a device response. All live high-fidelity methods enforce the existing posterior chance constraint before launch. Any actual recovery work must be recorded by the campaign operator; the generic adapter does not execute unspecified recovery commands.

`examples/toy_solver.py` is a clearly labeled analytic **schema fixture**, not a TCAD simulator. It enables a reproducible process-level check without a proprietary simulator:

```bash
python smoke.py
python smoke_advanced.py
```

`smoke.py` uses a synthetic campaign; `smoke_advanced.py` checks joint acquisition, posterior conditioning, Sobol candidates, comparison methods, exposure accounting and the analytic worker fixture. For device optimization, supply solver executables, calibrated decks, physical mappings and seed campaign records through the registered adapter.
