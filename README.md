# RAMP-TCAD

Implementation of **Risk-aware multi-fidelity optimization for technology computer-aided design of wide-bandgap and gate-all-around devices**.

The optimizer separates device responses, numerical completion, and censored runtime. Paired low/high evaluations learn source discrepancies. Corrected auxiliary responses feed high-fidelity Gaussian processes with discrepancy-dependent observation variance. Acquisition divides completion- and reliability-weighted safe hypervolume improvement plus boundary uncertainty by expected termination/recovery cost. An optional dimensionless descriptor model shares information through a held-out local risk gate.

## Install and supply simulation records

```bash
python -m venv .venv
python -m pip install -e .
```

Python 3.11+ is required. Device simulations are supplied by a licensed TCAD solver or an existing measured campaign. Mesh, transport/quantum models, tolerances, device geometry, and physical constraint definitions belong to that solver configuration. No random-noise substitute is presented as a different fidelity.

Create `campaign.json`, a JSON list with one entry per design/fidelity evaluation. Required fields are `id`, `technology`, `x`, `fidelity`, `completed`, `runtime`, `recovery`, and `initial`. Successful records additionally contain `objectives` and `constraints`; failed records need neither response. `runtime` is compute exposure until termination, `recovery` the additional measured recovery exposure, and optional `censored: true` denotes timeout/right censoring. Use one common high-fidelity-equivalent unit for all cost fields. Constraint values are feasible at or below zero, and all objectives use minimization, so negate breakdown voltage/on-current when maximizing them.

The initial set must include at least two completed target-technology high-fidelity simulations. To use an auxiliary fidelity, include at least two completed paired designs at that fidelity and high fidelity with identical `x`. Set `initial: true` only for the registered shared seed set. Held-out candidate outcomes remain hidden from surrogate fitting and acquisition until selected by replay.

Optional `descriptor` arrays contain physically meaningful dimensionless quantities with identical meaning across technologies. They are required for shared models. Shared objective/constraint coordinates must likewise have aligned physical meanings and units or a registered dimensionless response mapping. Do not simply concatenate unrelated raw SiC and GAAFET geometry parameters. Only successful high-fidelity observations enter the shared response model; completion and runtime models remain target-technology-specific. Auxiliary-technology records belong in the known initial set and are not launched as target-device candidates.

Edit `config.json` to register design bounds, objective normalization, target technology, ordered fidelities, hypervolume reference, safe-launch risk, boundary weight, and Monte Carlo count. The supplied unit bounds illustrate schema; replace them with actual device definitions before optimization. Fix objective normalization and hypervolume reference before comparing campaigns.

## Run replay and comparisons

```bash
ramp-tcad --data campaign.json --config config.json --budget 100 --output outputs/full --seed 0
ramp-tcad --data campaign.json --config config.json --budget 100 --output outputs/hf --variant hf --seed 0
ramp-tcad --data campaign.json --config config.json --budget 100 --output outputs/mfbo --variant mfbo --seed 0
ramp-tcad --data campaign.json --config config.json --budget 100 --output outputs/no_failure --variant no_failure
ramp-tcad --data campaign.json --config config.json --budget 100 --output outputs/no_boundary --variant no_boundary
ramp-tcad --data campaign.json --config config.json --budget 100 --output outputs/no_reliability --variant no_reliability
ramp-tcad --data campaign.json --config config.json --budget 100 --output outputs/no_transfer --variant no_transfer
ramp-tcad --data campaign.json --config config.json --budget 100 --output outputs/raw_sharing --variant raw_sharing
```

This command performs finite-candidate sequential replay of real precomputed TCAD records. Each selection sees only candidate coordinates, fidelity, technology, and descriptors. After selection, that record's actual response, completion, and cost are revealed. No simulator process is launched. The same `Surrogate.choose` method accepts a live candidate list; append the solver's completed or failed record before constructing the next surrogate to use it in an external TCAD campaign.

`hf` uses only high-fidelity response fitting and launches, `mfbo` removes completion and local reliability discounting, and other variants remove the named mechanism. `raw_sharing` sets the sharing gate to one. Comparisons use the same seed records and compute cap. This implementation evaluates one candidate per step: Monte Carlo independent-objective EHVI is the `q=1` case. No result is labeled NSGA-II or SafeOpt without those distinct algorithms.

The censored runtime model is a ridge-regularized lognormal accelerated-failure-time fit. Completion uses logistic regression when both outcomes are observed; otherwise it uses the observed constant rate. Source reliability decays with local discrepancy magnitude and uncertainty relative to posterior response uncertainty. Boundary utility is posterior constraint standard deviation times density near zero. High-fidelity launches require every modeled constraint's posterior violation probability to pass the configured limit; no unsafe boundary exception is enabled. Empty safe acquisition sets stop with an explicit error so the seed design or launch rule can be reconsidered.

## Results

`history.json` records measured cost and the safe-front trajectory. `acquisition.json` retains every evaluated candidate's allocation score, completion, reliability, expected cost, feasibility, and transfer gate. `observed.json` contains only revealed simulation records. `metrics.json` includes normalized safe hypervolume, safe hypervolume regret, physical high-fidelity violation rate, failed-run/recovery exposure, total compute, boundary Brier/ECE, and allocation overhead. `front_progress.png` plots safe hypervolume against compute.

Normalization and safe regret use the feasible high-fidelity front of the supplied finite campaign as an evaluation reference; this retrospective front is never supplied to acquisition. It is not the unknown continuous-domain optimum. Failed simulations have no fabricated objectives and update only operational models. Transfer benefit/loss is obtained by paired differences in final safe hypervolume between full/no-transfer/raw-sharing runs; a negative full-minus-no-transfer difference is negative transfer. Repeat both device campaigns and seeds before calculating that rate. Solver runtime and optimization overhead have separate units unless converted by a declared hardware accounting rule. No campaign results are bundled.

## Reproduction logic conventions

Boundary Brier/ECE compare target high-fidelity predictions only with completed target-technology high-fidelity responses. Auxiliary-fidelity outcomes are not high-fidelity calibration labels. Physical-violation accounting likewise excludes seed records from other technologies.
