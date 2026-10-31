"""Synthetic TCAD-record replay; not a physical TCAD experiment."""
import json
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from ramp_tcad import cli as c


def main():
    np.testing.assert_allclose(c.hypervolume(np.array([[1., 3.], [2., 2.], [3., 3.]]), [4., 4.]), 5.)
    np.testing.assert_allclose(c.calibration_errors([0., .1, .9, 1.], [0, 1, 0, 1]), [.405, .45])
    config = dict(technology='target', fidelities=['coarse', 'fine'], lower=[0.], upper=[1.],
                  objective_lower=[0., 0.], objective_upper=[1., 1.], reference=[1.1, 1.1],
                  risk=.49, boundary_weight=.05, mc_samples=4)
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
    observed = [row for row in rows if row['initial']]
    candidates = [{k: row[k] for k in ['id', 'technology', 'x', 'fidelity']} for row in rows if not row['initial']]
    # Tiny fixtures can place GP kernels at their bounds; this does not suppress runtime-fit errors.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        model = c.Surrogate(observed, config, 'full', 0)
        prediction = model.predict(candidates)
        restored = c.Surrogate(json.loads(json.dumps(observed)), config, 'full', 0)
        for a, b in zip(prediction, restored.predict(candidates)):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        assert all(np.isfinite(a).all() for a in prediction)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps(config))
            (root / 'campaign.json').write_text(json.dumps(rows))
            original = c.Surrogate.choose
            def checked(self, pending, rng):
                assert all('objectives' not in r and 'constraints' not in r and 'completed' not in r for r in pending)
                return original(self, pending, rng)
            for variant in ['full', 'hf', 'no_failure']:
                with patch.object(c.Surrogate, 'choose', checked):
                    c.run(SimpleNamespace(config=str(root / 'config.json'), data=str(root / 'campaign.json'),
                                          output=str(root / variant), seed=0, budget=5., variant=variant))
                history = json.loads((root / variant / 'history.json').read_text())
                metrics = json.loads((root / variant / 'metrics.json').read_text())
                visible = json.loads((root / variant / 'observed.json').read_text())
                assert history and metrics['compute_cost'] <= 5. + 1e-12
                assert metrics['compute_cost'] == sum(r['runtime'] + r.get('recovery', 0.) for r in visible)
                assert metrics['failure_exposure'] >= .4 - 1e-12
                assert (root / variant / 'front_progress.png').stat().st_size > 0
    print('PASS: manual Pareto hypervolume/ECE/Brier, GP fit and reconstruct equality, failed/censored exposure, hidden candidate outcomes, resource-limited replay and controls')


if __name__ == '__main__':
    main()
