"""Small analytic records/worker checks; never a TCAD performance claim."""
import json
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning

from ramp_tcad import advanced as a
from ramp_tcad.cli import Surrogate, coordinates


def main():
    torch.set_num_threads(1)
    config = dict(technology='target', fidelities=['coarse', 'fine'], lower=[0.], upper=[1.],
                  objective_lower=[0., 0.], objective_upper=[1., 1.], reference=[1.1, 1.1],
                  risk=.49, boundary_weight=.05, mc_samples=4, seconds_per_hf=100., constraint_scale=[1.])
    rows = []
    for i, x in enumerate([.2, .8]):
        for fidelity in ['coarse', 'fine']:
            rows.append(dict(id=f'seed{i}{fidelity}', technology='target', x=[x], fidelity=fidelity,
                completed=True, runtime=1. if fidelity == 'fine' else .2, recovery=0., initial=True,
                objectives=[x + (.03 if fidelity == 'coarse' else 0), 1 - x], constraints=[-1.]))
    rows.append(dict(id='failed', technology='target', x=[.95], fidelity='coarse', completed=False,
                     runtime=.3, recovery=.1, initial=True, censored=True))
    for i, x in enumerate([.4, .5, .6]):
        rows.append(dict(id=f'candidate{i}', technology='target', x=[x], fidelity='fine', completed=True,
                        runtime=.7, recovery=0., initial=False, objectives=[x, 1 - x], constraints=[-1.]))
    observed = [r for r in rows if r['initial']]
    candidates = [a.metadata(r) for r in rows if not r['initial']]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        source = Surrogate(observed, config, 'full', 0)
        posterior_model = a.gaussian_model(source, candidates)
        x = torch.tensor(np.array([coordinates(r, config) for r in candidates]), dtype=torch.double)
        posterior = posterior_model.posterior(x)
        mean, deviation, _ = source.predict(candidates)
        np.testing.assert_allclose(posterior.mean.numpy(), np.c_[-mean[:, :2], mean[:, 2:]], atol=1e-12)
        np.testing.assert_allclose(posterior.variance.numpy(), deviation ** 2 + 1e-8, atol=1e-9)
        pending_x = x[:1].numpy()
        pending_y = posterior.mean[:1].numpy().copy()
        pending_y[0, 0] += .1
        conditioned = posterior_model.condition(pending_x, pending_y).posterior(x[:1])
        assert conditioned.variance[0, 0] < posterior.variance[0, 0]
        assert conditioned.mean[0, 0] > posterior.mean[0, 0]
        score = a.qehvi_scores(source, candidates, [(0, 1), (1, 2)], mc=4, fantasies=2)
        fantasy = a.qehvi_scores(source, candidates, [(0, 1), (1, 2)], mc=4, fantasies=2, pending=[candidates[0]])
        assert np.isfinite(score).all() and np.isfinite(fantasy).all()
        assert not np.allclose(score, fantasy, atol=1e-10)
        sobol = a.sobol_candidates(config, 4, 0, observed)
        assert len(sobol) == 8 and all(0 <= r['x'][0] <= 1 for r in sobol)
        assert all('objectives' not in r for r in sobol)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'campaign.json').write_text(json.dumps(rows))
            (root / 'config.json').write_text(json.dumps(config))
            for method in ['qehvi', 'nsga2', 'safeopt']:
                args = SimpleNamespace(command='replay', config=str(root / 'config.json'), data=str(root / 'campaign.json'),
                    output=str(root / method), method=method, budget=5., batch_size=2, max_launches=2,
                    candidates=4, mc_samples=4, fantasies=2, max_batches=4, pending=None, seed=0)
                a.run(args)
                metrics = json.loads((root / method / 'metrics.json').read_text())
                visible = json.loads((root / method / 'observed.json').read_text())
                plan = json.loads((root / method / 'batch_plans.json').read_text())[0]
                assert metrics['launches'] == 2 and len(set(plan['batch'])) == 2
                np.testing.assert_allclose(metrics['compute_cost'], metrics['allocation_cost'] + sum(r['runtime'] + r.get('recovery', 0) for r in visible))
                assert metrics['compute_cost'] <= 5. and metrics['failure_exposure'] >= .4 - 1e-12
            control = a.NSGAControl(observed, config, 0, 2)
            control.select(candidates, 1)
            control.update([dict(**candidates[0], completed=False)])
            assert control.select(candidates[1:], 1)
            config['solver'] = dict(argv=[sys.executable, str(Path('examples/toy_solver.py').resolve()), '--request', '{request}', '--response', '{response}'], timeout_seconds=20)
            (root / 'live_config.json').write_text(json.dumps(config))
            args.command, args.method, args.config, args.output = 'live', 'qehvi', str(root / 'live_config.json'), str(root / 'live')
            a.run(args)
            live = json.loads((root / 'live/observed.json').read_text())[-2:]
            assert len(live) == 2 and all(r['completed'] and r['runtime'] > 0 for r in live)
            for row in live:
                np.testing.assert_allclose(row['objectives'], [row['x'][0] + (.03 if row['fidelity'] == 'coarse' else 0), 1-row['x'][0]])
            config['solver'] = dict(argv=[sys.executable, '-c', 'import time; time.sleep(5)'], timeout_seconds=.1)
            timeout = a.launch(candidates[0], config, root / 'timeout', 1.)
            assert not timeout['completed'] and timeout['censored'] and 'objectives' not in timeout
    print('PASS: BoTorch joint qEHVI, Gaussian posterior fantasies, Sobol, real NSGA-II/SafeOpt replay, exposure accounting, bounded live worker and timeout')


if __name__ == '__main__':
    main()
