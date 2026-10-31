"""Portable file guardrails. Run: python -m unittest discover -s tests -v."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('validate_submissions', ROOT / 'scripts' / 'validate_submissions.py')
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class FinalSubmissionTests(unittest.TestCase):
    def test_canonical_finals_have_valid_schema_ids_and_values(self):
        for part, path in validator.FINAL_FILES.items():
            with self.subTest(part=part):
                report = validator.validate_submission(path, expected_rows=validator.EXPECTED_ROWS[part])
                self.assertTrue(report['passed'], report['errors'])
                self.assertEqual(set(report['line_counts']), {'M', 'B', 'C'})


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'predictions.csv'

    def write_csv(self, rows, header=('row_id', 'prediction')):
        with self.path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def test_valid_file_and_official_csv_sample(self):
        self.write_csv([('M1', '0'), ('B2', '12.5')])
        report = validator.validate_submission(self.path, self.path, expected_rows=2)
        self.assertTrue(report['passed'], report['errors'])
        self.assertTrue(report['official_sample_check']['same_order'])
        self.assertEqual(report['prediction_summary']['B']['sum'], 12.5)

    def test_invalid_prediction_values_fail(self):
        for value in ('-1', 'NaN', 'inf', '-inf', '', 'bad'):
            with self.subTest(value=value):
                self.write_csv([('M1', value)])
                self.assertFalse(validator.validate_submission(self.path)['passed'])

    def test_duplicate_and_blank_ids_fail(self):
        for rows in ([('M1', 1), ('M1', 2)], [('', 1)], [(' M1 ', 1)]):
            with self.subTest(rows=rows):
                self.write_csv(rows)
                self.assertFalse(validator.validate_submission(self.path)['passed'])

    def test_wrong_columns_and_empty_file_fail(self):
        self.write_csv([('M1', 1)], header=('prediction', 'row_id'))
        self.assertFalse(validator.validate_submission(self.path)['passed'])
        self.write_csv([])
        self.assertFalse(validator.validate_submission(self.path)['passed'])

    def test_incorrect_field_count_fails(self):
        self.write_csv([('M1', 1, 'extra')])
        self.assertFalse(validator.validate_submission(self.path)['passed'])

    def test_missing_file_and_wrong_expected_row_count_fail(self):
        self.assertFalse(validator.validate_submission(self.path)['passed'])
        self.write_csv([('M1', 1)])
        self.assertFalse(validator.validate_submission(self.path, expected_rows=2)['passed'])

    def test_official_id_mismatch_fails(self):
        sample = Path(self.folder.name) / 'sample.csv'
        sample.write_text('row_id,prediction\nB2,0\n', encoding='utf-8')
        self.write_csv([('M1', 1)])
        report = validator.validate_submission(self.path, sample)
        self.assertFalse(report['passed'])
        self.assertEqual(report['official_sample_check']['missing_ids'], 1)
        self.assertEqual(report['official_sample_check']['extra_ids'], 1)

    def test_matching_ids_in_different_order_are_disclosed(self):
        sample = Path(self.folder.name) / 'sample.csv'
        sample.write_text('row_id,prediction\nB2,0\nM1,0\n', encoding='utf-8')
        self.write_csv([('M1', 1), ('B2', 2)])
        report = validator.validate_submission(self.path, sample)
        self.assertTrue(report['passed'], report['errors'])
        self.assertFalse(report['official_sample_check']['same_order'])
        self.assertTrue(any('row order differs' in message for message in report['warnings']))


if __name__ == '__main__':
    unittest.main()
