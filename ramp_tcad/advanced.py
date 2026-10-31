"""Batch posterior-fantasy acquisition, comparison policies and a bounded live launcher."""
import argparse
import itertools
import json
import math
import os
import signal
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc

from .cli import Surrogate, calibration_errors, coordinates, hypervolume


def metadata(row):
    return {key: row[key] for key in ['id', 'x', 'fidelity', 'technology', 'descriptor'] if key in row}


def sobol_candidates(config, count, seed, observed):
    lower, upper = np.asarray(config['lower']), np.asarray(config['upper'])
    if not np.isfinite(np.r_[lower, upper]).all() or np.any(upper <= lower):
        raise ValueError('Finite, strictly ordered physical design bounds are required')
    points = qmc.Sobol(len(lower), scramble=True, seed=seed).random_base2(math.ceil(math.log2(count)))[:count]
    rows = []
    for index, point in enumerate(lower + points * (upper - lower)):
        for fidelity in config['fidelities']:
            if any(r['fidelity'] == fidelity and r['technology'] == config['technology'] and np.allclose(r['x'], point, rtol=0, atol=1e-12) for r in observed):
                continue
            row = dict(id=f'sobol_{seed}_{index}_{fidelity}', x=point.tolist(), fidelity=fidelity, technology=config['technology'])
            if 'descriptor_indices' in config:
                value = point[config['descriptor_indices']]
                row['descriptor'] = ((value - np.asarray(config['descriptor_offset'])) / np.asarray(config['descriptor_scale'])).tolist()
            rows.append(row)
    return rows


def gaussian_model(source, candidates):
    """Expose the existing corrected/shared GP marginals as a BoTorch posterior."""
    import torch
    from botorch.models.model import Model
    from botorch.posteriors.gpytorch import GPyTorchPosterior
    from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
    config = source.config
    m = len(config['objective_lower'])
    references = {tuple(np.round(coordinates(row, config), 12)): metadata(row) for row in candidates}

    class CampaignGaussian(Model):
        def __init__(self, pending_x=None, pending_y=None):
            super().__init__()
            self.register_buffer('anchor', torch.empty(0, dtype=torch.double))
            self.pending_x, self.pending_y = pending_x, pending_y

        @property
        def num_outputs(self):
            return len(source.models)

        @property
        def batch_shape(self):
            return torch.Size([])

        def raw(self, unit):
            rows = []
            for point in unit:
                key = tuple(np.round(point, 12))
                rows.append(references.get(key, dict(x=(np.asarray(config['lower']) + point * (np.asarray(config['upper']) - config['lower'])).tolist(),
                                                       technology=config['technology'], fidelity=source.high, id='query')))
            mean, std, _ = source.predict(rows)
            covariance = []
            for j, fitted in enumerate(source.models):
                _, cov = fitted.predict(unit, return_cov=True)
                scale = std[:, j] / np.sqrt(np.diag(cov).clip(1e-12))
                covariance.append(cov * scale[:, None] * scale[None])
            scaling = np.r_[1 / (np.asarray(config['objective_upper']) - config['objective_lower']),
                            1 / np.asarray(config.get('constraint_scale', [1.] * (len(source.models) - m)))]
            mean[:, :m] = -(mean[:, :m] - config['objective_lower'])
            mean *= scaling
            covariance = np.asarray(covariance) * scaling[:, None, None] ** 2
            return mean, covariance

        def condition(self, x, y):
            return CampaignGaussian(np.asarray(x), np.asarray(y))

        def posterior(self, X, output_indices=None, observation_noise=False, posterior_transform=None):
            unit = X.detach().cpu().numpy()
            batch_shape, q, dimension = unit.shape[:-2], unit.shape[-2], unit.shape[-1]
            means, covariances = [], []
            for point in unit.reshape(-1, q, dimension):
                joined = np.vstack([self.pending_x, point]) if self.pending_x is not None else point
                mean, covariance = self.raw(joined)
                if self.pending_x is not None:
                    n = len(self.pending_x)
                    new_mean, new_covariance = [], []
                    for j in range(self.num_outputs):
                        observed = covariance[j, :n, :n] + np.eye(n) * config.get('fantasy_noise', 1e-6)
                        cross = covariance[j, n:, :n]
                        gain = np.linalg.solve(observed, cross.T).T
                        new_mean.append(mean[n:, j] + gain @ (self.pending_y[:, j] - mean[:n, j]))
                        new_covariance.append(covariance[j, n:, n:] - gain @ cross.T)
                    mean, covariance = np.asarray(new_mean).T, np.asarray(new_covariance)
                means.append(mean)
                covariances.append(covariance)
            mean = torch.tensor(np.asarray(means).reshape(*batch_shape, q, self.num_outputs), dtype=X.dtype, device=X.device)
            cov = torch.tensor(np.asarray(covariances).reshape(*batch_shape, self.num_outputs, q, q), dtype=X.dtype, device=X.device)
            cov = (cov + cov.transpose(-1, -2)) / 2 + torch.eye(q, dtype=X.dtype, device=X.device) * 1e-8
            mvns = [MultivariateNormal(mean[..., j], cov[..., j, :, :]) for j in range(self.num_outputs)]
            posterior = GPyTorchPosterior(MultitaskMultivariateNormal.from_independent_mvns(mvns) if len(mvns) > 1 else mvns[0])
            return posterior_transform(posterior) if posterior_transform is not None else posterior

    return CampaignGaussian()


def qehvi_scores(source, candidates, batches, mc=64, fantasies=4, pending=None, seed=0):
    """Joint qEHVI averaged over actual Gaussian-conditioned pending outcomes."""
    import torch
    from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
    from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
    from botorch.exceptions.warnings import NumericsWarning
    torch.set_num_threads(1)
    config, m = source.config, len(source.config['objective_lower'])
    pending = pending or []
    model = gaussian_model(source, candidates + pending)
    lower, span = np.asarray(config['objective_lower']), np.asarray(config['objective_upper']) - config['objective_lower']
    safe = [r for r in source.completed if r['fidelity'] == source.high and max(r['constraints'], default=-1) <= 0]
    front = -np.asarray([(np.asarray(r['objectives']) - lower) / span for r in safe]).reshape(-1, m)
    reference = -torch.tensor(config.get('reference', [1.1] * m), dtype=torch.double)
    sample_x = torch.tensor(np.array([[coordinates(candidates[i], config) for i in batch] for batch in batches]), dtype=torch.double)
    scenarios = [(model, front)]
    if pending:
        pending_x = torch.tensor(np.array([coordinates(row, config) for row in pending]), dtype=torch.double)
        draws = SobolQMCNormalSampler(torch.Size([fantasies]), seed=seed)(model.posterior(pending_x)).detach().numpy()
        scenarios = []
        for draw in draws:
            feasible = (draw[:, m:] <= 0).all(1)
            scenarios.append((model.condition(pending_x.numpy(), draw), np.vstack([front, draw[feasible, :m]])))
    scores = np.zeros(len(batches))
    for index, (conditioned, current_front) in enumerate(scenarios):
        partitioning = NondominatedPartitioning(ref_point=reference, Y=torch.tensor(current_front, dtype=torch.double))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', NumericsWarning)
            acquisition = qExpectedHypervolumeImprovement(conditioned, reference.tolist(), partitioning,
                            sampler=SobolQMCNormalSampler(torch.Size([mc]), seed=seed + index),
                            objective=IdentityMCMultiOutputObjective(outcomes=list(range(m))),
                            constraints=[lambda values, j=j: values[..., m + j] for j in range(model.num_outputs - m)])
            with torch.no_grad():
                scores += acquisition(sample_x).numpy() / len(scenarios)
    return scores


def choose_batch(source, candidates, size, rng, args, pending=None):
    _, traces = source.choose(candidates, rng)
    admissible = [i for i, row in enumerate(traces) if row['score'] is not None]
    size = min(size, len(admissible))
    combinations = list(itertools.islice(itertools.combinations(admissible, size), args.max_batches + 1))
    if len(combinations) > args.max_batches:
        combinations = list(dict.fromkeys(tuple(sorted(rng.choice(admissible, size, replace=False))) for _ in range(args.max_batches)))
    utility = qehvi_scores(source, candidates, combinations, args.mc_samples, args.fantasies, pending, args.seed)
    for j, batch in enumerate(combinations):
        probability = np.prod([traces[i]['completion'] for i in batch])
        reliability = np.mean([traces[i]['reliability'] for i in batch])
        mean, std, _ = source.predict([candidates[i] for i in batch])
        m = len(source.config['objective_lower'])
        boundary = float(np.sum(std[:, m:] * np.exp(-.5 * (mean[:, m:] / std[:, m:]) ** 2)))
        utility[j] = probability * reliability * (utility[j] + source.config.get('boundary_weight', .05) * boundary) / max(1e-9, sum(traces[i]['cost'] for i in batch))
    best = int(np.argmax(utility))
    return list(combinations[best]), dict(batch=[candidates[i]['id'] for i in combinations[best]], utility=float(utility[best]), traces=traces)


def safeopt(source, candidates, size):
    """Finite-domain SafeOpt with registered scalarization and Lipschitz expanders."""
    mean, std, _ = source.predict(candidates)
    m, beta = len(source.config['objective_lower']), source.config.get('safeopt_beta', 2.)
    safe = (mean[:, m:] + beta * std[:, m:] <= 0).all(1)
    if not safe.any():
        raise ValueError('SafeOpt has no confidence-certified candidate')
    span = np.asarray(source.config['objective_upper']) - source.config['objective_lower']
    weights = np.asarray(source.config.get('safeopt_weights', [1 / m] * m)) / span
    objective, deviation = mean[:, :m] @ weights, np.sqrt((std[:, :m] ** 2) @ (weights ** 2))
    maximizer = objective - beta * deviation <= (objective + beta * deviation)[safe].min()
    expanders = np.zeros(len(candidates), bool)
    x = np.array([coordinates(row, source.config) for row in candidates])
    for i in np.flatnonzero(safe):
        distance = np.linalg.norm(x[~safe] - x[i], axis=1)
        lower = mean[i, m:] - beta * std[i, m:]
        expanders[i] = np.any((lower + source.config.get('safeopt_lipschitz', 1.) * distance[:, None] <= 0).all(1)) if len(distance) else False
    eligible = np.flatnonzero(safe & (maximizer | expanders))
    return eligible[np.argsort(-deviation[eligible])[:size]].tolist()


class NSGAControl:
    def __init__(self, observed, config, seed, size):
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.population import Population
        from pymoo.core.problem import Problem
        self.config = config
        high = [r for r in observed if r['completed'] and r['fidelity'] == config['fidelities'][-1] and r['technology'] == config['technology']]
        initial = Population.new(X=np.array([coordinates(r, config) for r in high]),
                                 F=np.array([r['objectives'] for r in high]), G=np.array([r['constraints'] for r in high]))
        problem = Problem(n_var=len(config['lower']), n_obj=len(config['objective_lower']), n_ieq_constr=len(high[0]['constraints']), xl=0., xu=1.)
        self.algorithm = NSGA2(pop_size=max(4, size), sampling=initial)
        self.algorithm.setup(problem, termination=('n_gen', 1000000), seed=seed, verbose=False)
        self.algorithm.tell(infills=self.algorithm.ask())

    def select(self, candidates, size):
        self.pending = self.algorithm.ask()
        proposed = self.pending.get('X')
        high = [i for i, r in enumerate(candidates) if r['fidelity'] == self.config['fidelities'][-1]]
        selected = []
        for point in proposed:
            remaining = [i for i in high if i not in selected]
            if not remaining or len(selected) == size:
                break
            selected.append(min(remaining, key=lambda i: np.linalg.norm(coordinates(candidates[i], self.config) - point)))
        return selected

    def update(self, rows):
        from pymoo.core.population import Population
        if rows:
            # Optimizer-only penalties for noncompletion; physical records retain no outcomes.
            span = np.asarray(self.config['objective_upper']) - self.config['objective_lower']
            penalty = np.asarray(self.config['objective_upper']) + span
            constraints = self.algorithm.problem.n_ieq_constr
            self.algorithm.tell(infills=Population.new(X=np.array([coordinates(r, self.config) for r in rows]),
                F=np.array([r['objectives'] if r['completed'] else penalty for r in rows]),
                G=np.array([r['constraints'] if r['completed'] else [1.] * constraints for r in rows])))


def launch(row, config, folder, exposure_limit):
    """Run only an explicit argv, with an isolated request/result directory and timeout."""
    solver = config['solver']
    folder.mkdir(parents=True, exist_ok=False)
    request, response = folder / 'request.json', folder / 'response.json'
    request.write_text(json.dumps(metadata(row), indent=2))
    if not isinstance(solver.get('argv'), list) or not solver['argv'] or any(not isinstance(part, str) for part in solver['argv']):
        raise ValueError('solver.argv must be a nonempty list of executable/arguments')
    command = [part.format(request=str(request.resolve()), response=str(response.resolve())) for part in solver['argv']]
    timeout = min(float(solver.get('timeout_seconds', 3600)), exposure_limit)
    begin = time.perf_counter()
    with (folder / 'stdout.log').open('w') as stdout, (folder / 'stderr.log').open('w') as stderr:
        options = dict(start_new_session=True) if os.name != 'nt' else dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        process = subprocess.Popen(command, cwd=folder, stdout=stdout, stderr=stderr,
                                   env={**os.environ, 'OMP_NUM_THREADS': '1'}, **options)
        censored = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], check=True, stdout=stderr, stderr=stderr)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code, censored = -1, True
    seconds = time.perf_counter() - begin
    result = dict(**metadata(row), completed=False, runtime=seconds / config['seconds_per_hf'], recovery=0., initial=False, censored=censored)
    if code == 0:
        value = json.loads(response.read_text())
        if value['completed']:
            objective, constraint = value['objectives'], value['constraints']
            expected = len(config.get('constraint_scale', constraint))
            if len(objective) != len(config['objective_lower']) or len(constraint) != expected or not np.isfinite(objective + constraint).all():
                raise ValueError('Solver response dimensions/finite values do not match the registered campaign')
            result.update(completed=True, objectives=objective, constraints=constraint)
    return result


def run(args):
    config = json.loads(Path(args.config).read_text())
    if config.get('seconds_per_hf', 0) <= 0:
        raise ValueError('Register positive seconds_per_hf for worker/optimizer exposure accounting')
    rows = json.loads(Path(args.data).read_text())
    observed = [r for r in rows if r.get('initial')]
    complete = [r for r in observed if r['completed']]
    if not complete:
        raise ValueError('Completed initial records are required')
    config.setdefault('constraint_scale', [1.] * len(complete[0]['constraints']))
    pending = [r for r in rows if not r.get('initial') and r['technology'] == config['technology']]
    external_pending = json.loads(Path(args.pending).read_text()) if args.pending else []
    rng = np.random.default_rng(args.seed)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cost = sum(r['runtime'] + r.get('recovery', 0.) for r in observed)
    nsga = NSGAControl(observed, config, args.seed, args.batch_size) if args.method == 'nsga2' else None
    history, plans, probabilities, labels = [], [], [], []
    launches, allocation_cost = 0, 0.
    while cost < args.budget and launches < args.max_launches:
        begin = time.perf_counter()
        if args.command == 'live':
            pending = sobol_candidates(config, args.candidates, args.seed + len(plans), observed)
        if not pending:
            break
        source = Surrogate(observed, config, 'full', args.seed)
        candidates = [metadata(row) for row in pending]
        if args.method == 'qehvi':
            selected, plan = choose_batch(source, candidates, min(args.batch_size, args.max_launches - launches), rng, args, external_pending)
        elif args.method == 'safeopt':
            selected = safeopt(source, candidates, args.batch_size)
            plan = dict(batch=[candidates[i]['id'] for i in selected])
        else:
            selected = nsga.select(candidates, args.batch_size)
            plan = dict(batch=[candidates[i]['id'] for i in selected])
        overhead = (time.perf_counter() - begin) / config['seconds_per_hf']
        cost += overhead
        allocation_cost += overhead
        plans.append(plan)
        completed_batch = []
        # Selection is frozen before any selected response is revealed.
        for i in selected:
            if cost >= args.budget or launches >= args.max_launches:
                break
            chosen = candidates[i]
            mean, std, _ = source.predict([chosen])
            if args.command == 'live':
                if chosen['fidelity'] == source.high and np.any(ndtr(mean[0, len(config['objective_lower']):] / std[0, len(config['objective_lower']):]) > config.get('risk', .05)):
                    raise ValueError('Live high-fidelity launch violates the registered posterior risk limit')
                item = launch(chosen, config, out / 'launches' / f'{launches:06d}', (args.budget - cost) * config['seconds_per_hf'])
            else:
                item = pending[i]
                expense = item['runtime'] + item.get('recovery', 0.)
                if cost + expense > args.budget:
                    item = dict(**chosen, completed=False, runtime=args.budget - cost, recovery=0., censored=True, initial=False)
            cost += item['runtime'] + item.get('recovery', 0.)
            observed.append(item)
            completed_batch.append(item)
            launches += 1
            if item['completed'] and item['fidelity'] == source.high:
                j = len(config['objective_lower'])
                probabilities.extend(ndtr(-mean[0, j:] / std[0, j:]).tolist())
                labels.extend((np.asarray(item['constraints']) <= 0).tolist())
            safe = [r for r in observed if r['completed'] and r['fidelity'] == source.high and r['technology'] == config['technology'] and max(r['constraints'], default=-1) <= 0]
            front = np.array([(np.asarray(r['objectives']) - config['objective_lower']) / (np.asarray(config['objective_upper']) - config['objective_lower']) for r in safe]).reshape(-1, len(config['objective_lower']))
            history.append(dict(cost=cost, selected=item['id'], completed=item['completed'], safe_hypervolume=hypervolume(front, config.get('reference', [1.1] * len(config['objective_lower'])))))
        if nsga:
            nsga.update(completed_batch)
        if args.command == 'replay':
            pending = [row for i, row in enumerate(pending) if i not in selected]
    # Oracle values are read only after allocation for replay scoring. Live has no hidden oracle.
    m = len(config['objective_lower'])
    oracle_hv = None
    if args.command == 'replay':
        oracle = [r for r in rows if r['completed'] and r['fidelity'] == config['fidelities'][-1] and r['technology'] == config['technology'] and max(r['constraints'], default=-1) <= 0]
        normalized = np.asarray([(np.asarray(r['objectives']) - config['objective_lower']) / (np.asarray(config['objective_upper']) - config['objective_lower']) for r in oracle]).reshape(-1, m)
        oracle_hv = hypervolume(normalized, config.get('reference', [1.1] * m))
    for item in history:
        item['normalized_safe_hypervolume'] = item['safe_hypervolume'] / oracle_hv if oracle_hv else None
        item['safe_regret'] = oracle_hv - item['safe_hypervolume'] if oracle_hv is not None else None
    brier, ece = calibration_errors(probabilities, labels)
    successful_hf = [r for r in observed if r['completed'] and r['fidelity'] == config['fidelities'][-1] and r['technology'] == config['technology']]
    metrics = dict(compute_cost=cost, allocation_cost=allocation_cost, launches=launches, boundary_brier=brier, boundary_ece=ece,
                   failure_exposure=sum(r['runtime'] + r.get('recovery', 0.) for r in observed if not r['completed']),
                   physical_violation_rate=float(np.mean([max(r['constraints'], default=-1) > 0 for r in successful_hf])) if successful_hf else None,
                   final_safe_hypervolume=history[-1]['safe_hypervolume'] if history else None)
    for name, value in [('metrics', metrics), ('history', history), ('batch_plans', plans), ('observed', observed)]:
        (out / (name + '.json')).write_text(json.dumps(value, indent=2))
    from matplotlib import pyplot as plt
    plt.plot([v['cost'] for v in history], [v['safe_hypervolume'] for v in history])
    plt.xlabel('Worker and allocator HF-equivalent exposure'); plt.ylabel('Safe hypervolume')
    plt.tight_layout(); plt.savefig(out / 'front_progress.png', dpi=160); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['replay', 'live'])
    for key in ['data', 'config', 'output']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--method', choices=['qehvi', 'nsga2', 'safeopt'], default='qehvi')
    parser.add_argument('--budget', type=float, required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--mc-samples', type=int, default=64)
    parser.add_argument('--fantasies', type=int, default=4)
    parser.add_argument('--max-batches', type=int, default=128)
    parser.add_argument('--candidates', type=int, default=32)
    parser.add_argument('--max-launches', type=int, default=1000)
    parser.add_argument('--pending', help='Launched but unresolved candidate metadata; no outcomes')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if min(args.batch_size, args.mc_samples, args.fantasies, args.max_batches, args.candidates, args.max_launches) < 1:
        parser.error('Batch/sample/fantasy/candidate/launch counts must be positive')
    run(args)


if __name__ == '__main__':
    main()
