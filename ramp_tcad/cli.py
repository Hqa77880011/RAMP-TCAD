import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, ndtr, log_ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LogisticRegression


def pareto(values):
    values = np.asarray(values)
    if len(values) == 0:
        return values
    keep = [not np.any(np.all(values <= row, axis=1) & np.any(values < row, axis=1)) for row in values]
    return values[keep]


def hypervolume(values, reference):
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if values.size == 0:
        return 0.
    values = pareto(values[np.all(values < reference, axis=1)])
    if len(values) == 0:
        return 0.
    if values.shape[1] == 1:
        return float(reference[0] - values[:, 0].min())
    slices = np.unique(np.r_[values[:, 0], reference[0]])
    return float(sum((upper - lower) * hypervolume(values[values[:, 0] <= lower, 1:], reference[1:])
                     for lower, upper in zip(slices[:-1], slices[1:])))


def gp(x, y, seed=0, noise=1e-5):
    kernel = ConstantKernel(1., (1e-3, 1e3)) * Matern(length_scale=np.ones(x.shape[1]), nu=2.5) + WhiteKernel(1e-5)
    return GaussianProcessRegressor(kernel, alpha=noise, normalize_y=True, random_state=seed, n_restarts_optimizer=0).fit(x, y)


def coordinates(record, config):
    lower, upper = np.asarray(config['lower']), np.asarray(config['upper'])
    return (np.asarray(record['x']) - lower) / (upper - lower)


class Surrogate:
    def __init__(self, records, config, variant, seed):
        self.records, self.config, self.variant = records, config, variant
        self.high = config['fidelities'][-1]
        self.completed = [r for r in records if r['completed'] and r['technology'] == config['technology']]
        high = [r for r in self.completed if r['fidelity'] == self.high]
        if len(high) < 2:
            raise ValueError('At least two completed high-fidelity seed evaluations are required')
        self.models, self.discrepancy, self.transfer = [], {}, []
        xh = np.array([coordinates(r, config) for r in high])
        responses = np.array([r['objectives'] + r['constraints'] for r in high])
        count = responses.shape[1]
        for output in range(count):
            xs, ys, noises = list(xh), list(responses[:, output]), [1e-5] * len(high)
            for fidelity in config['fidelities'][:-1]:
                low = [r for r in self.completed if r['fidelity'] == fidelity]
                pairs = [(a, b) for a in low for b in high if np.allclose(a['x'], b['x'])]
                if len(pairs) < 2:
                    continue
                xp = np.array([coordinates(a, config) for a, b in pairs])
                differences = np.array([(b['objectives'] + b['constraints'])[output] - (a['objectives'] + a['constraints'])[output] for a, b in pairs])
                discrepancy = gp(xp, differences, seed)
                self.discrepancy[fidelity, output] = discrepancy
                if variant == 'hf':
                    continue
                xl = np.array([coordinates(r, config) for r in low])
                correction, uncertainty = discrepancy.predict(xl, return_std=True)
                low_values = np.array([(r['objectives'] + r['constraints'])[output] for r in low])
                xs.extend(xl)
                ys.extend(low_values + correction)
                noises.extend(uncertainty ** 2 + 1e-3)
            independent = gp(np.asarray(xs), np.asarray(ys), seed, np.asarray(noises))
            shared_records = [r for r in records if r['completed'] and r['fidelity'] == self.high and 'descriptor' in r]
            shared = None
            if len(shared_records) >= 4 and all('descriptor' in r for r in high):
                xd = np.array([r['descriptor'] for r in shared_records])
                yd = np.array([(r['objectives'] + r['constraints'])[output] for r in shared_records])
                shared = gp(xd, yd, seed)
            self.models.append(independent)
            self.transfer.append(shared)
        operational = [r for r in records if r['technology'] == config['technology']]
        features = np.array([np.r_[coordinates(r, config), config['fidelities'].index(r['fidelity']) / max(1, len(config['fidelities']) - 1)] for r in operational])
        completion = np.array([int(r['completed']) for r in operational])
        self.failure = LogisticRegression(C=1.).fit(features, completion) if len(np.unique(completion)) > 1 else float(completion.mean())
        runtime = np.log(np.array([max(r['runtime'], 1e-6) for r in operational]))
        censored = np.array([r.get('censored', False) for r in operational])
        design = np.c_[np.ones(len(features)), features]
        def objective(parameters):
            residual = (runtime - design @ parameters[:-1]) / np.exp(parameters[-1])
            loglike = np.where(censored, log_ndtr(-residual), -.5 * residual ** 2 - parameters[-1] - .5 * np.log(2 * np.pi))
            return -loglike.sum() + .01 * np.square(parameters[1:-1]).sum()
        fit = minimize(objective, np.r_[np.linalg.lstsq(design, runtime, rcond=None)[0], 0.], method='L-BFGS-B', bounds=[(None, None)] * design.shape[1] + [(-5, 5)])
        if not fit.success:
            raise RuntimeError('Censored runtime fit failed: ' + fit.message)
        self.runtime = fit.x
        failed = [r.get('recovery', 0.) for r in operational if not r['completed']]
        self.recovery = float(np.mean(failed)) if failed else 0.

    def predict(self, candidates):
        x = np.array([coordinates(c, self.config) for c in candidates])
        outputs, deviations, gates = [], [], []
        high = [r for r in self.completed if r['fidelity'] == self.high]
        for output, model in enumerate(self.models):
            mean, std = model.predict(x, return_std=True)
            gate = np.zeros(len(candidates))
            shared = self.transfer[output]
            if shared is not None and self.variant not in ('no_transfer', 'hf') and all('descriptor' in c for c in candidates):
                shared_mean, shared_std = shared.predict(np.array([c['descriptor'] for c in candidates]), return_std=True)
                if self.variant == 'raw_sharing':
                    gate[:] = 1
                else:
                    risks = []
                    for held in range(len(high)):
                        fit_records = [r for j, r in enumerate(high) if j != held]
                        if len(fit_records) < 2:
                            continue
                        local = gp(np.array([coordinates(r, self.config) for r in fit_records]),
                                   np.array([(r['objectives'] + r['constraints'])[output] for r in fit_records]))
                        shared_training = [r for r in self.records if r['completed'] and r['fidelity'] == self.high and 'descriptor' in r and r is not high[held]]
                        pooled = gp(np.array([r['descriptor'] for r in shared_training]), np.array([(r['objectives'] + r['constraints'])[output] for r in shared_training]))
                        target = (high[held]['objectives'] + high[held]['constraints'])[output]
                        local_error = (local.predict(coordinates(high[held], self.config)[None])[0] - target) ** 2
                        shared_error = (pooled.predict(np.array(high[held]['descriptor'])[None])[0] - target) ** 2
                        risks.append((coordinates(high[held], self.config), shared_error - local_error))
                    if risks:
                        distance = np.sum((x[:, None] - np.array([r[0] for r in risks])[None]) ** 2, axis=-1)
                        w = np.exp(-distance / .2)
                        difference = w @ np.array([r[1] for r in risks]) / w.sum(1).clip(1e-12)
                        gate = expit(-(difference + shared_std) / self.config.get('transfer_temperature', 1.))
                mean = (1 - gate) * mean + gate * shared_mean
                std = np.sqrt((1 - gate) ** 2 * std ** 2 + gate ** 2 * shared_std ** 2)
            outputs.append(mean)
            deviations.append(std.clip(1e-6))
            gates.append(gate)
        return np.array(outputs).T, np.array(deviations).T, np.array(gates).T

    def choose(self, candidates, rng):
        mean, std, gates = self.predict(candidates)
        m = len(self.config['objective_lower'])
        lower, upper = np.array(self.config['objective_lower']), np.array(self.config['objective_upper'])
        reference = np.array(self.config.get('reference', [1.1] * m))
        high = [r for r in self.completed if r['fidelity'] == self.high and max(r['constraints'], default=-1) <= 0]
        front = np.array([(np.array(r['objectives']) - lower) / (upper - lower) for r in high]).reshape(-1, m)
        base_hv = hypervolume(front, reference)
        acquisition, traces = [], []
        for i, candidate in enumerate(candidates):
            feature = np.r_[coordinates(candidate, self.config), self.config['fidelities'].index(candidate['fidelity']) / max(1, len(self.config['fidelities']) - 1)]
            completion = self.failure if isinstance(self.failure, float) else self.failure.predict_proba(feature[None])[0, 1]
            expected_time = np.exp(np.r_[1, feature] @ self.runtime[:-1] + .5 * np.exp(2 * self.runtime[-1]))
            cost = expected_time + (1 - completion) * self.recovery
            feasibility = float(np.prod(ndtr(-mean[i, m:] / std[i, m:])))
            reliability = 1.
            if candidate['fidelity'] != self.high:
                estimates = []
                for output in range(mean.shape[1]):
                    discrepancy = self.discrepancy.get((candidate['fidelity'], output))
                    if discrepancy is None:
                        estimates.append(.05)
                    else:
                        bias, uncertainty = discrepancy.predict(coordinates(candidate, self.config)[None], return_std=True)
                        estimates.append(float(np.exp(-(abs(bias[0]) + uncertainty[0]) / max(std[i, output], 1e-6))))
                reliability = float(np.mean(estimates))
            samples = rng.normal(mean[i, :m], std[i, :m], (self.config.get('mc_samples', 64), m))
            improvement = np.mean([max(0, hypervolume(np.vstack([front, (sample - lower) / (upper - lower)]), reference) - base_hv) for sample in samples])
            boundary = float(np.sum(std[i, m:] * np.exp(-.5 * (mean[i, m:] / std[i, m:]) ** 2)))
            if self.variant == 'no_boundary':
                boundary = 0.
            if self.variant in ('no_failure', 'mfbo'):
                completion = 1.
            if self.variant in ('no_reliability', 'mfbo'):
                reliability = 1.
            score = completion * reliability * (feasibility * improvement + self.config.get('boundary_weight', .05) * boundary) / max(cost, 1e-9)
            if candidate['fidelity'] == self.high and np.any(ndtr(mean[i, m:] / std[i, m:]) > self.config.get('risk', .05)):
                score = -np.inf
            if self.variant == 'hf' and candidate['fidelity'] != self.high:
                score = -np.inf
            acquisition.append(score)
            traces.append({'id': candidate['id'], 'fidelity': candidate['fidelity'], 'score': float(score) if np.isfinite(score) else None,
                           'completion': float(completion), 'reliability': reliability, 'cost': float(cost),
                           'feasibility': feasibility, 'transfer_gate': gates[i].tolist()})
        if not np.isfinite(acquisition).any():
            raise ValueError('No candidate satisfies the registered high-fidelity launch rule')
        return int(np.argmax(acquisition)), traces


def run(args):
    config = json.loads(Path(args.config).read_text())
    rows = json.loads(Path(args.data).read_text())
    observed = [r for r in rows if r.get('initial', False)]
    pending = [r for r in rows if not r.get('initial', False) and r['technology'] == config['technology']]
    rng = np.random.default_rng(args.seed)
    traces, history = [], []
    cost = sum(r['runtime'] + r.get('recovery', 0.) for r in observed)
    high = config['fidelities'][-1]
    m = len(config['objective_lower'])
    lo, hi = np.array(config['objective_lower']), np.array(config['objective_upper'])
    reference = config.get('reference', [1.1] * m)
    true_safe = [r for r in rows if r['completed'] and r['fidelity'] == high and r['technology'] == config['technology'] and max(r['constraints'], default=-1) <= 0]
    oracle_hv = hypervolume(np.array([(np.array(r['objectives']) - lo) / (hi - lo) for r in true_safe]).reshape(-1, m), reference)
    begin = time.perf_counter()
    while pending and cost < args.budget:
        model = Surrogate(observed, config, args.variant, args.seed)
        candidates = [{k: r[k] for k in ['id', 'x', 'fidelity', 'technology', 'descriptor'] if k in r} for r in pending]
        index, acquisition = model.choose(candidates, rng)
        selected = pending.pop(index)
        exposed = selected['runtime'] + selected.get('recovery', 0.)
        if cost + exposed > args.budget:
            remaining = max(0., args.budget - cost)
            observed.append({**candidates[index], 'completed': False, 'runtime': remaining,
                             'recovery': 0., 'censored': True, 'termination': 'campaign_budget'})
            cost += remaining
            traces.append(acquisition)
            break
        mean, std, _ = model.predict([candidates[index]])
        cost += exposed
        observed.append(selected)
        safe = [r for r in observed if r['completed'] and r['fidelity'] == high and r['technology'] == config['technology'] and max(r['constraints'], default=-1) <= 0]
        hv = hypervolume(np.array([(np.array(r['objectives']) - lo) / (hi - lo) for r in safe]).reshape(-1, m), reference)
        record = {'cost': cost, 'safe_hypervolume': hv, 'normalized_safe_hypervolume': hv / oracle_hv if oracle_hv > 0 else None,
                  'safe_regret': oracle_hv - hv, 'selected': selected['id'], 'completed': selected['completed'], 'fidelity': selected['fidelity'],
                  'predicted_constraint_safety': ndtr(-mean[0, m:] / std[0, m:]).tolist(),
                  'constraint_safety': (np.array(selected['constraints']) <= 0).tolist() if selected['completed'] else None}
        history.append(record)
        traces.append(acquisition)
    completed = [r for r in observed if r['completed'] and r['fidelity'] == high and r['technology'] == config['technology']]
    probabilities, labels = [], []
    for item in history:
        if item['fidelity'] == high and item['constraint_safety'] is not None:
            probabilities.extend(item['predicted_constraint_safety'])
            labels.extend(item['constraint_safety'])
    ece = 0.
    if probabilities:
        probabilities, labels = np.array(probabilities), np.array(labels)
        for low in np.arange(0, 1, .1):
            selected = (probabilities >= low) & (probabilities < low + .1 + 1e-12)
            if selected.any():
                ece += selected.mean() * abs(probabilities[selected].mean() - labels[selected].mean())
    result = {'final': history[-1] if history else None, 'compute_cost': cost,
              'failure_exposure': sum(r['runtime'] + r.get('recovery', 0.) for r in observed if not r['completed']),
              'physical_violation_rate': float(np.mean([max(r['constraints'], default=-1) > 0 for r in completed])) if completed else None,
              'boundary_brier': float(np.mean((probabilities - labels) ** 2)) if len(probabilities) else None,
              'boundary_ece': float(ece), 'allocation_overhead_seconds': time.perf_counter() - begin}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('history', history), ('acquisition', traces), ('metrics', result), ('observed', observed)]:
        (out / f'{name}.json').write_text(json.dumps(value, indent=2))
    plt.plot([v['cost'] for v in history], [v['safe_hypervolume'] for v in history])
    plt.xlabel('Measured compute exposure')
    plt.ylabel('Safe hypervolume')
    plt.tight_layout()
    plt.savefig(out / 'front_progress.png', dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    for key in ['data', 'config', 'output']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--budget', type=float, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--variant', choices=['full', 'hf', 'mfbo', 'no_failure', 'no_boundary', 'no_reliability', 'no_transfer', 'raw_sharing'], default='full')
    run(parser.parse_args())
