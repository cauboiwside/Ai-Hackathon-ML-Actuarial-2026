"""Checks on generated Part 1 frames; explicitly skipped when frames are absent."""
import os
from pathlib import Path
import unittest

try:
    import pandas as pd
except ImportError:
    pd = None

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('IAA_WORK_DIR', ROOT / 'work')).expanduser()
LINES = ('motor', 'buildings', 'contents')
REQUIRED = [WORK / f'mf_{split}_{line}.parquet' for line in LINES for split in ('train', 'test')]


class GeneratedFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [path.name for path in REQUIRED if not path.is_file()]
        if missing:
            raise unittest.SkipTest('Generated model frames absent; no training-data QA was run: ' + ', '.join(missing))
        if pd is None:
            raise unittest.SkipTest('Generated-frame QA requires pandas and a parquet engine.')

    def test_exposure_in_years(self):
        for line in LINES:
            for split in ('train', 'test'):
                with self.subTest(line=line, split=split):
                    exposure = pd.read_parquet(WORK / f'mf_{split}_{line}.parquet', columns=['exposure']).exposure
                    self.assertTrue(exposure.between(0.002, 1.005).all(), (line, split, exposure.max()))

    def test_categorical_levels_match(self):
        for line in LINES:
            train = pd.read_parquet(WORK / f'mf_train_{line}.parquet')
            test = pd.read_parquet(WORK / f'mf_test_{line}.parquet')
            for column in train.columns:
                if str(train[column].dtype) in ('object', 'str') and column in test and not column.endswith('_id'):
                    with self.subTest(line=line, column=column):
                        self.assertTrue(set(test[column].dropna()) <= set(train[column].dropna()))


if __name__ == '__main__':
    unittest.main()
