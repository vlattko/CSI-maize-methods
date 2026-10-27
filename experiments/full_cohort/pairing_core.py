"""Full-cohort pairing bookkeeping. Scientific image operators are public imports."""
from pathlib import Path
import csv
import hashlib
import json
import os

QUARTERS = (0, 10, 20, 30)
IDENTITY = ('treatment', 'measurement', 'genotype', 'replicate')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def write_csv(path, rows):
    path = Path(path)
    temporary = path.with_suffix('.csv.tmp')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with temporary.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()})
    os.replace(temporary, path)


def true(value):
    return str(value).lower() in ('true', '1')


def frame_lookup(rows):
    result = {}
    for row in rows:
        key = (row['session'], row['camera'].lower(), int(row['view']))
        if key in result:
            raise ValueError('Duplicate frame key: ' + repr(key))
        result[key] = row
    return result


def wrong_plant(sessions, anchor):
    """Rotate replicates within treatment/visit/genotype when possible.

    Remaining choices match treatment/visit, then visit. Never use the same
    treatment/genotype/replicate candidate identity across visits as a control.
    Choices use labels and availability, never outcome or registration fit.
    """
    candidate_id = lambda r: (r['treatment'], r['genotype'], r['replicate'])
    pool = [s for s in sessions if candidate_id(s) != candidate_id(anchor)]
    for fields in [('treatment', 'measurement', 'genotype'),
                   ('treatment', 'measurement'), ('measurement',), ()]:
        choices = sorted([s for s in pool if all(s[k] == anchor[k] for k in fields)],
                         key=lambda s: s['session'])
        if choices:
            later = [s for s in choices if s['session'] > anchor['session']]
            return (later or choices)[0], '+'.join(fields) or 'none'
    return None, 'unavailable'


def moment_affine(a, b, np):
    """Exact initialization of unchanged public affine_align, with empty guard."""
    if a.sum() < 10 or b.sum() < 10:
        raise ValueError('empty_or_tiny_foreground')
    def moments(x):
        y, z = np.where(x)
        return np.array([z.mean(), y.mean()]), np.array([max(z.std(), 1), max(y.std(), 1)])
    ca, sa = moments(a)
    cb, sb = moments(b)
    scale = np.clip(sb / sa, 0.6, 1.6)
    return np.array([[scale[0], 0, cb[0] - ca[0] * scale[0]],
                     [0, scale[1], cb[1] - ca[1] * scale[1]]], np.float32)


def cache_read(path, signature):
    if not Path(path).exists():
        return None
    result = json.loads(Path(path).read_text(encoding='utf-8'))
    if result.get('signature') != signature:
        return None
    expected = result.pop('payload_sha256')
    if expected != digest(result):
        raise ValueError('Cached pairing payload changed: ' + str(path))
    result['payload_sha256'] = expected
    return result
