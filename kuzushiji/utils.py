from pathlib import Path

import pandas as pd


DATA_ROOT = Path(__file__).parent.parent / 'data'

UNICODE_MAP = {codepoint: char for codepoint, char in
               pd.read_csv(DATA_ROOT / 'unicode_translation.csv').values}
