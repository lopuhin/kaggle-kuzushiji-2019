from typing import Dict

from ignite.engine import Events
import tqdm


def run_with_pbar(engine, loader, desc=None):
    pbar = tqdm.trange(len(loader), desc=desc)
    engine.on(Events.ITERATION_COMPLETED)(lambda _: pbar.update(1))
    engine.run(loader)
    pbar.close()


def print_metrics(metrics: Dict):
    print(' '.join(f'{k}={format_value(v)}' for k, v in metrics.items()))


def format_value(v):
    if isinstance(v, float):
        return f'{v:.5f}'
    else:
        return str(v)
