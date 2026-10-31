#!/usr/bin/env python3
"""Validate final prediction CSVs without loading or fitting a model.

CSV-only validation uses the Python standard library. Official parquet samples
require pyarrow. Run with --help for paths and optional official-sample checks.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_FILES = {
    'part1': REPO_ROOT / 'final_submissions' / 'part1_prediction_file.csv',
    'part2': REPO_ROOT / 'final_submissions' / 'part2_prediction_file.csv',
}
EXPECTED_ROWS = {'part1': 626639, 'part2': 209063}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_sample_ids(path):
    """Read the official row IDs, preserving their supplied order."""
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or 'row_id' not in reader.fieldnames:
                raise ValueError('Official sample must contain a row_id column.')
            ids = [row['row_id'] for row in reader]
    elif path.suffix.lower() in ('.parquet', '.pq'):
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise ValueError('Reading parquet samples requires pyarrow; install requirements.txt.') from exc
        ids = parquet.read_table(path, columns=['row_id']).column('row_id').to_pylist()
    else:
        raise ValueError('Official sample must be CSV or parquet.')
    if any(value is None or not str(value).strip() for value in ids):
        raise ValueError('Official sample contains missing row IDs.')
    ids = [str(value) for value in ids]
    if len(ids) != len(set(ids)):
        raise ValueError('Official sample contains duplicate row IDs.')
    return ids


def validate_submission(path, sample_path=None, expected_rows=None):
    """Return a JSON-serializable report; validation errors set passed=False."""
    path = Path(path).expanduser().resolve()
    result = {'file': str(path), 'passed': False, 'errors': [], 'warnings': [],
              'official_sample': str(Path(sample_path).expanduser().resolve()) if sample_path else None}
    errors, warnings = result['errors'], result['warnings']
    if not path.is_file():
        errors.append('Prediction file does not exist.')
        return result
    result.update(bytes=path.stat().st_size, sha256=sha256_file(path))
    ids, predictions, line_values = [], [], {}
    invalid_rows, missing_ids, invalid_values = [], [], []
    try:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, [])
            result['columns'] = header
            if header != ['row_id', 'prediction']:
                errors.append('Columns must be exactly row_id,prediction in that order.')
                return result
            for row_number, row in enumerate(reader, start=2):
                if len(row) != 2:
                    invalid_rows.append(row_number)
                    continue
                row_id, raw_prediction = row
                ids.append(row_id)
                if not row_id.strip() or row_id != row_id.strip():
                    missing_ids.append(row_number)
                try:
                    prediction = float(raw_prediction)
                except ValueError:
                    invalid_values.append(row_number)
                    continue
                if not math.isfinite(prediction) or prediction < 0:
                    invalid_values.append(row_number)
                    continue
                predictions.append(prediction)
                line_values.setdefault(row_id[:1], []).append(prediction)
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(f'Cannot read prediction CSV: {exc}')
        return result
    result['rows'] = len(ids)
    result['expected_rows'] = expected_rows
    if not ids:
        errors.append('Prediction file contains no data rows.')
    if invalid_rows:
        errors.append(f'{len(invalid_rows)} rows have the wrong number of fields; first rows: {invalid_rows[:5]}.')
    if missing_ids:
        errors.append(f'{len(missing_ids)} row IDs are blank or contain surrounding whitespace; first rows: {missing_ids[:5]}.')
    counts = Counter(ids)
    duplicate_count = len(ids) - len(counts)
    result['duplicate_row_ids'] = duplicate_count
    if duplicate_count:
        examples = [row_id for row_id, count in counts.items() if count > 1][:5]
        errors.append(f'{duplicate_count} duplicate row IDs; examples: {examples}.')
    if invalid_values:
        errors.append(f'{len(invalid_values)} predictions are nonnumeric, nonfinite, or negative; first rows: {invalid_values[:5]}.')
    if expected_rows is not None and len(ids) != expected_rows:
        errors.append(f'Expected {expected_rows} rows, found {len(ids)}.')
    result['line_counts'] = dict(sorted(Counter(row_id[:1] for row_id in ids).items()))
    result['prediction_summary'] = {
        line: {'count': len(values), 'min': min(values), 'max': max(values),
               'mean': math.fsum(values) / len(values), 'sum': math.fsum(values)}
        for line, values in sorted(line_values.items())
    }
    if predictions and not any(predictions):
        warnings.append('All predictions are zero; confirm this is intended.')
    if sample_path is not None:
        try:
            sample_ids = read_sample_ids(Path(sample_path).expanduser())
            missing = set(sample_ids) - set(ids)
            extra = set(ids) - set(sample_ids)
            same_order = ids == sample_ids
            result['official_sample_check'] = {
                'rows': len(sample_ids), 'missing_ids': len(missing),
                'extra_ids': len(extra), 'same_order': same_order,
                'sha256': sha256_file(Path(sample_path).expanduser()),
            }
            if missing or extra:
                errors.append(f'Official sample ID mismatch: {len(missing)} missing, {len(extra)} extra.')
            if not missing and not extra and not same_order:
                warnings.append('IDs match the official sample but row order differs; confirm the competition accepts ID alignment.')
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f'Cannot validate official sample: {exc}')
    else:
        warnings.append('No official sample supplied: exact expected row-ID coverage and order were not verified.')
    result['passed'] = not errors
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for part in FINAL_FILES:
        parser.add_argument(f'--{part}', type=Path, default=FINAL_FILES[part], help=f'{part} prediction CSV')
        parser.add_argument(f'--{part}-sample', type=Path, help=f'Official {part} sample CSV/parquet')
    parser.add_argument('--report', type=Path, help='Write the full JSON report to this path')
    args = parser.parse_args(argv)
    reports = {
        part: validate_submission(getattr(args, part), getattr(args, f'{part}_sample'), EXPECTED_ROWS[part])
        for part in FINAL_FILES
    }
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'passed': all(item['passed'] for item in reports.values()),
              'scope': 'Prediction-file integrity only; model accuracy and competition-rule compliance are not established.',
              'submissions': reports}
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding='utf-8')
    sys.stdout.write(text)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
