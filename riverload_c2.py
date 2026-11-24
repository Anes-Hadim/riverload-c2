# =====================================================================
# RIVERLOAD C2: ONE-CELL COMPLETE OFFLINE KAGGLE SOLUTION
# Forecasting: historical analogues, quantile LightGBM, optional
# trajectory ensembles + strict chronological backtesting.
# Planning: stochastic expected cost, 2 routes x 25 departure slots,
# multi-port mixed-integer selection, validated tiered stowage.
# No external hydrological data. Score cannot be guaranteed until Kaggle.
# =====================================================================
import os,sys,subprocess,time,importlib.util
from pathlib import Path
import pandas as pd

THREADS=4
USE_TRAJECTORY=True    # switch False if time constrained
TUNE_C2_POLICY=True    # optimizer fits on permitted past voyages only
FINALISTS=6
MILP_TIME_SECONDS=0.60
MAKE_SAFE_ALTERNATIVE=False  # optional second submission; extra CPU time

os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']=str(THREADS)
os.environ['PYTHONHASHSEED']='2026'

# Prefer the C2 mount automatically. C1 and C2 share data but have
# different sample_submission.csv schemas.
search_root=Path('/kaggle/input') if Path('/kaggle/input').exists() else Path.cwd()
found=[]
for p in sorted(search_root.rglob('sample_submission.csv')):
    try:
        cols=set(pd.read_csv(p,nrows=0).columns)
        if {'id','departure_slot','slot_row','slot_tier'}.issubset(cols) and (
                'route' in cols or 'route_id' in cols) and (
                p.parent/'history'/'levels_hourly.parquet').exists() and (
                p.parent/'reference'/'segments.csv').exists():
            found.append(p.parent)
    except Exception:
        continue
if not found:
    raise FileNotFoundError('Attach RIVERLOAD C2 Kaggle competition dataset. '
                            'The C1 sample submission is not compatible.')
DATA_DIR=found[0]
print('C2 dataset:',DATA_DIR,flush=True)
for pkg,mod in [('numpy','numpy'),('scipy','scipy'),('pandas','pandas'),
                ('scikit-learn','sklearn'),('pyarrow','pyarrow'),
                ('lightgbm','lightgbm'),('threadpoolctl','threadpoolctl')]:
    if importlib.util.find_spec(mod) is None:
        raise RuntimeError(pkg+' is missing. This solution does not fetch packages online.')
WORK=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
CODE=WORK/'riverload_c2_complete_code'
OUT=WORK/'riverload_c2_best'
CODE.mkdir(exist_ok=True,parents=True)
OUT.mkdir(exist_ok=True,parents=True)
SOURCES={}
# ======= river_model.py =======
SOURCES['river_model.py'] = r'''
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
QUANTILES = np.array([0.05, 0.1, 0.25, 0.5])
QCOLS = ['dmin_q05_cm', 'dmin_q10_cm', 'dmin_q25_cm', 'dmin_q50_cm']
KEYS = ['scenario_id', 'route_id', 'departure_slot', 'leg']
SEED = 2026

class Files:

    def __init__(self, root):
        self.root = Path(root).resolve()

    def get(self, section, name):
        canonical = self.root / section / name
        if canonical.is_file():
            return canonical
        if section == 'train' and name in {'scenarios.csv', 'manifests.csv', 'lock_queues.csv'}:
            duplicate = self.root / name.replace('.csv', ' (1).csv')
            if duplicate.is_file():
                return duplicate
        flat = self.root / name
        if flat.is_file():
            return flat
        raise FileNotFoundError(f'Missing competition file: {canonical}')

    def scenarios(self, split):
        df = pd.read_csv(self.get(split, 'scenarios.csv'), parse_dates=['issue_time', 't0'])
        prefix = 'TR-' if split == 'train' else 'TS-'
        if not df.scenario_id.str.startswith(prefix).all():
            raise ValueError(f'{split} scenarios were confused with the other split')
        return df

@dataclass
class Piece:
    gauge: str
    offset: float
    enter: float
    exit: float

class Geometry:

    def __init__(self, files):
        self.segments = pd.read_csv(files.get('reference', 'segments.csv')).set_index('segment_id')
        self.pieces = pd.read_csv(files.get('reference', 'route_pieces.csv'))
        self.groups = {}
        self.arrivals = {}
        for route, frame in self.pieces.groupby('route_id'):
            elapsed, leg, delay = (0.0, 1, False)
            for p in frame.sort_values('order').itertuples():
                if p.kind == 'PORT':
                    self.arrivals[route, leg] = elapsed
                    elapsed += float(p.dwell_h)
                    leg += 1
                    continue
                duration = p.length_km / p.speed_kmh
                if pd.notna(p.locks):
                    duration += len(p.locks.split(';'))
                    delay = True
                if p.kind == 'RIVER':
                    s = self.segments.loc[p.segment_id]
                    self.groups.setdefault((route, leg), []).append(Piece(s.gauge_id, s.guaranteed_depth_cm - s.ref_level_shifted_cm, elapsed, elapsed + duration))
                elapsed += duration

    def group(self, route, leg):
        return self.groups[route, int(leg)]

class Water:

    def __init__(self, files):
        self.levels = pd.read_parquet(files.get('history', 'levels_hourly.parquet')).sort_index()
        self.flows = pd.read_parquet(files.get('history', 'discharge_hourly.parquet')).sort_index()
        if self.levels.index.max() >= pd.Timestamp('2022-01-01') or self.flows.index.max() >= pd.Timestamp('2022-01-01'):
            raise ValueError('Historical training data must stop on 31 December 2021')
        for frame in [self.levels, self.flows]:
            if frame.index.has_duplicates or not (frame.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
                raise ValueError('Expected unique consecutive hourly history')
        self.history = self.levels.join(self.flows)
        self.test_levels = pd.read_parquet(files.get('test', 'window_levels_hourly.parquet'))
        self.test_flows = pd.read_parquet(files.get('test', 'window_discharge_hourly.parquet'))
        self.test = {}
        flows = {k: g.set_index('timestamp').drop(columns='issue_id') for k, g in self.test_flows.groupby('issue_id')}
        for key, g in self.test_levels.groupby('issue_id'):
            self.test[key] = g.set_index('timestamp').drop(columns='issue_id').join(flows[key]).sort_index()
        self.names = list(self.history.columns)
        self.gauges = list(self.levels.columns)
        self.col = {c: i for i, c in enumerate(self.names)}
        self.level_array = self.levels.to_numpy()
        self.level_col = {c: i for i, c in enumerate(self.gauges)}
        self.origin = self.levels.index[0]

    def window(self, issue, issue_id=None):
        source = self.history if issue_id is None else self.test[issue_id]
        index = pd.date_range(issue - pd.Timedelta(days=21), issue - pd.Timedelta(hours=1), freq='h')
        window = source.reindex(index)[self.names]
        return window.ffill().to_numpy(dtype=np.float32)

def summaries(window, names):
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        last = window[-1]
        smooth = np.nanmean(window[-6:], axis=0)
        for i, name in enumerate(names):
            out[f'{name}_last'] = last[i]
            out[f'{name}_smooth6'] = smooth[i]
        for hours in [6, 24, 48, 72, 168, 336, 480]:
            lag = np.nanmean(window[-hours - 6:-hours], axis=0)
            delta = smooth - lag
            for i, name in enumerate(names):
                out[f'{name}_change{hours}'] = delta[i]
                if name.startswith('Q') and hours in [24, 72, 168]:
                    out[f'{name}_ratio{hours}'] = np.log(np.maximum(smooth[i], 1) / np.maximum(lag[i], 1))
        for hours in [24, 72, 168, 504]:
            values = window[-hours:]
            means = np.nanmean(values, axis=0)
            low, high = (np.nanmin(values, axis=0), np.nanmax(values, axis=0))
            for i, name in enumerate(names):
                out[f'{name}_mean{hours}'] = means[i]
                out[f'{name}_range{hours}'] = high[i] - low[i]
                if hours in [168, 504]:
                    out[f'{name}_above_min{hours}'] = smooth[i] - low[i]
    return out

def scenario_keys(scenarios):
    rows = []
    for s in scenarios.itertuples():
        for route in s.routes.split(';'):
            for slot in range(25):
                for leg in range(1, 2 if s.family == 'A' else 4):
                    rows.append((s.scenario_id, route, slot, leg))
    return pd.DataFrame(rows, columns=KEYS)

def attach_scenarios(keys, scenarios, queues):
    out = keys.merge(scenarios[['scenario_id', 'issue_id', 'issue_time']], on='scenario_id', validate='many_to_one')
    sums = queues.groupby('scenario_id').queue_h.sum()
    out['queue_h'] = out.scenario_id.map(sums).fillna(0)
    return out

def synthetic_rows(water, geometry, stride=2):
    dates = pd.date_range('2000-01-23 06:00', '2021-12-20 06:00', freq=f'{stride}D')
    chunks = []
    slots = [0, 4, 8, 12, 16, 20, 24]
    for route in ['B1', 'B2']:
        for leg in [1, 2, 3]:
            gauges = {p.gauge for p in geometry.group(route, leg)}
            start = max((water.levels[g].first_valid_index() for g in gauges)) + pd.Timedelta(days=21)
            use = dates[dates >= start]
            frame = pd.DataFrame({'issue_time': np.repeat(use.to_numpy(), len(slots)), 'departure_slot': np.tile(slots, len(use))})
            frame['scenario_id'] = 'H-' + frame.issue_time.dt.strftime('%Y%m%d')
            frame['issue_id'] = frame.scenario_id
            frame['route_id'], frame['leg'] = (route, leg)
            ordinal = (frame.issue_time - pd.Timestamp('2000-01-01')).dt.days.to_numpy()
            frame['queue_h'] = 1.5 + ordinal % 4
            frame['dmin_cm'], frame['target_end'] = exact_targets(frame, water, geometry)
            chunks.append(frame.dropna(subset=['dmin_cm']))
    return pd.concat(chunks, ignore_index=True)

def exact_targets(rows, water, geometry):
    target = np.full(len(rows), np.nan)
    end = np.zeros(len(rows))
    issue_h = (rows.issue_time - water.origin).dt.total_seconds().to_numpy() / 3600
    for (route, leg), idx in rows.groupby(['route_id', 'leg']).indices.items():
        departure = 48 + 6 * rows.departure_slot.to_numpy()[idx]
        delay = rows.queue_h.to_numpy()[idx] if route.endswith('2') else 0.0
        current = np.full(len(idx), np.inf)
        for p in geometry.group(route, leg):
            a = np.floor(issue_h[idx] + departure + delay + p.enter).astype(int)
            b = np.floor(issue_h[idx] + departure + delay + p.exit).astype(int)
            end[idx] = np.maximum(end[idx], departure + delay + p.exit)
            invalid = (a < 0) | (b >= len(water.level_array))
            v = np.full(len(idx), np.inf)
            for shift in range(int(np.ceil(p.exit - p.enter)) + 2):
                sample = np.clip(np.minimum(a + shift, b), 0, len(water.level_array) - 1)
                v = np.minimum(v, water.level_array[sample, water.level_col[p.gauge]] + p.offset)
            v[invalid] = np.nan
            current = np.minimum(current, v)
        target[idx] = current
    return (target, rows.issue_time + pd.to_timedelta(end, unit='h'))

def feature_matrix(rows, water, geometry, test=False):
    timestamps = rows[['issue_time', 'issue_id']].drop_duplicates('issue_time').sort_values('issue_time')
    stats = []
    for s in timestamps.itertuples():
        stats.append(summaries(water.window(s.issue_time, s.issue_id if test else None), water.names))
    summary = pd.DataFrame(stats, index=timestamps.issue_time).astype(np.float32)
    index = summary.index.get_indexer(rows.issue_time)
    X = summary.iloc[index].reset_index(drop=True)
    month_angle = rows.issue_time.dt.dayofyear.to_numpy() * 2 * np.pi / 365.25
    X['season_sin'], X['season_cos'] = (np.sin(month_angle), np.cos(month_angle))
    X['leg'] = rows.leg.to_numpy()
    X['branch'] = (rows.route_id.str.endswith('2') & (rows.leg == 1)).astype(int).to_numpy()
    X['departure_hours'] = 48 + 6 * rows.departure_slot.to_numpy()
    derived_names = ['current_min', 'current_second', 'current_gap', 'bottleneck', 'horizon', 'horizon_start', 'horizon_end', 'bottleneck_change24', 'bottleneck_change72', 'bottleneck_change168', 'bottleneck_above_min504', 'trend24_linear', 'trend72_linear', 'trend168_linear', 'trend24_decay72', 'trend72_decay168', 'recession72', 'leg_mean_depth', 'leg_depth_range']
    features = {name: np.full(len(rows), np.nan, dtype=np.float32) for name in derived_names}
    for (route, leg), idx in rows.groupby(['route_id', 'leg']).indices.items():
        pieces = geometry.group(route, leg)
        depth = np.column_stack([X[f'{p.gauge}_last'].to_numpy()[idx] + p.offset for p in pieces])
        changes = {h: np.column_stack([X[f'{p.gauge}_change{h}'].to_numpy()[idx] for p in pieces]) for h in [24, 72, 168]}
        above_min = np.column_stack([X[f'{p.gauge}_above_min504'].to_numpy()[idx] for p in pieces])
        departure = X.departure_hours.to_numpy()[idx]
        delay = rows.queue_h.to_numpy()[idx] if route.endswith('2') else np.zeros(len(idx))
        horizon = departure[:, None] + delay[:, None] + np.array([(p.enter + p.exit) / 2 for p in pieces])[None, :]
        which = np.argmin(depth, axis=1)
        sorted_depth = np.sort(depth, axis=1)
        local = np.arange(len(idx))
        features['current_min'][idx] = sorted_depth[:, 0]
        features['current_second'][idx] = sorted_depth[:, 1]
        features['current_gap'][idx] = sorted_depth[:, 1] - sorted_depth[:, 0]
        features['bottleneck'][idx] = [int(pieces[w].gauge[1:]) for w in which]
        features['horizon'][idx] = horizon[local, which]
        features['horizon_start'][idx] = departure + delay + pieces[0].enter
        features['horizon_end'][idx] = departure + delay + pieces[-1].exit
        for h in [24, 72, 168]:
            features[f'bottleneck_change{h}'][idx] = changes[h][local, which]
            features[f'trend{h}_linear'][idx] = np.min(depth + changes[h] / h * horizon, axis=1)
        for h, decay in [(24, 72), (72, 168)]:
            effective = decay * -np.expm1(-horizon / decay)
            features[f'trend{h}_decay{decay}'][idx] = np.min(depth + changes[h] / h * effective, axis=1)
        features['recession72'][idx] = np.min(depth + np.minimum(changes[72], 0) / 72 * 168 * -np.expm1(-horizon / 168), axis=1)
        features['bottleneck_above_min504'][idx] = above_min[local, which]
        features['leg_mean_depth'][idx] = np.mean(depth, axis=1)
        features['leg_depth_range'][idx] = np.max(depth, axis=1) - np.min(depth, axis=1)
    X = pd.concat([X, pd.DataFrame(features)], axis=1).astype(np.float32)
    base = X.current_min.to_numpy().copy()
    return (X, base)

def build_dataset(data_dir, cache_dir=None, stride=2):
    files = Files(data_dir)
    geometry, water = (Geometry(files), Water(files))
    train = files.scenarios('train')
    test = files.scenarios('test')
    official = attach_scenarios(pd.read_csv(files.get('train', 'labels_dmin.csv')), train, pd.read_csv(files.get('train', 'lock_queues.csv')))
    reconstructed, end = exact_targets(official, water, geometry)
    error = np.abs(reconstructed - official.dmin_cm.to_numpy())
    if not np.isfinite(error).all() or np.max(error) > 0.06:
        raise ValueError(f'Timetable reconstruction disagrees with training truth: {np.nanmax(error)} cm')
    official['target_end'], official['official'] = (end, 1)
    synthetic = synthetic_rows(water, geometry, stride)
    synthetic['official'] = 0
    rows = pd.concat([official, synthetic], ignore_index=True)
    print(f'Verified {len(official):,} labels; added {len(synthetic):,} historical voyages', flush=True)
    X, base = feature_matrix(rows, water, geometry)
    if not np.isfinite(base).all():
        raise ValueError('Missing a required current gauge observation')
    test_rows = attach_scenarios(scenario_keys(test), test, pd.read_csv(files.get('test', 'lock_queues.csv')))
    Xt, bt = feature_matrix(test_rows, water, geometry, test=True)
    print(f'Feature tables: train {X.shape}, test {Xt.shape}', flush=True)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(cache_dir / 'rows.parquet', index=False)
        X.to_parquet(cache_dir / 'features.parquet', index=False)
        test_rows.to_parquet(cache_dir / 'test_rows.parquet', index=False)
        Xt.to_parquet(cache_dir / 'test_features.parquet', index=False)
    return (rows, X, base, test_rows, Xt, bt)

def load_dataset(cache_dir):
    root = Path(cache_dir)
    rows, X, test_rows, Xt = [pd.read_parquet(root / name) for name in ['rows.parquet', 'features.parquet', 'test_rows.parquet', 'test_features.parquet']]
    return (rows, X, X.current_min.to_numpy(), test_rows, Xt, Xt.current_min.to_numpy())

def pinball(y, pred):
    error = np.asarray(y)[:, None] - np.asarray(pred)
    return np.maximum(QUANTILES * error, (QUANTILES - 1) * error).mean()

def diagnostics(y, pred, rows, base):
    y = np.asarray(y)
    result = {'pinball_cm': float(pinball(y, pred)), 'n_rows': len(y), 'n_issue_times': int(rows.issue_time.nunique()), 'coverage': dict(zip(QCOLS, (y[:, None] < pred).mean(axis=0).tolist())), 'median_mae_cm': float(np.abs(y - pred[:, -1]).mean())}
    for label, mask in [('low_water_current_under_250', base < 250), ('true_depth_under_250', y < 250)]:
        if np.sum(mask):
            result[label] = {'n_rows': int(mask.sum()), 'pinball_cm': float(pinball(y[mask], pred[mask])), 'coverage': dict(zip(QCOLS, (y[mask, None] < pred[mask]).mean(axis=0).tolist()))}
    for leg in sorted(rows.leg.unique()):
        mask = rows.leg.to_numpy() == leg
        result[f'leg_{leg}'] = float(pinball(y[mask], pred[mask]))
    return result
CONFIGS = {'regularized': dict(num_leaves=15, min_child_samples=180, learning_rate=0.04, colsample_bytree=0.8, reg_lambda=8.0, n_estimators=650), 'flexible': dict(num_leaves=31, min_child_samples=120, learning_rate=0.035, colsample_bytree=0.85, reg_lambda=5.0, n_estimators=800), 'compact_trend': dict(num_leaves=15, min_child_samples=280, learning_rate=0.04, colsample_bytree=0.9, reg_lambda=12.0, n_estimators=650)}

def compact_features(X):
    keep = []
    gauge_suffixes = ['_last', '_smooth6', '_change6', '_change24', '_change48', '_change72', '_change168', '_change336', '_above_min504', '_range72']
    flow_suffixes = ['_smooth6', '_ratio24', '_ratio72', '_ratio168', '_change72']
    for name in X:
        if not name.startswith(('G', 'Q')) or (name.startswith('G') and any((name.endswith(s) for s in gauge_suffixes))) or (name.startswith('Q') and any((name.endswith(s) for s in flow_suffixes))):
            keep.append(name)
    out = X[keep].copy()
    for name in X:
        if name.startswith('trend') or name == 'recession72':
            out[name + '_delta'] = X[name] - X.current_min
    out['slope_acceleration'] = X.bottleneck_change24 / 24 - X.bottleneck_change72 / 72
    return out

def fit_predict(X, residual, Xt, config='regularized', threads=4, weights=None, iterations=None, model_dir=None):
    settings = dict(CONFIGS[config])
    if iterations is not None:
        settings['n_estimators'] = int(iterations)
    predictions = np.zeros((len(Xt), 4))
    if model_dir:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
    for j, q in enumerate(QUANTILES):
        start = time.monotonic()
        model = lgb.LGBMRegressor(objective='quantile', alpha=float(q), random_state=SEED + j, n_jobs=threads, deterministic=True, force_col_wise=True, verbosity=-1, max_bin=127, **settings)
        model.fit(X, residual, sample_weight=weights)
        predictions[:, j] = model.predict(Xt)
        if model_dir:
            model.booster_.save_model(str(Path(model_dir) / f'q{int(q * 100):02}.txt'))
        print(f'{config} q{int(q * 100):02}: {time.monotonic() - start:.1f}s', flush=True)
    return predictions

def ordered(pred):
    return np.sort(np.clip(pred, -500, 2000), axis=1)

def to_forecast(rows, prediction):
    out = rows[KEYS].reset_index(drop=True).copy()
    out[QCOLS] = np.round(ordered(prediction), 4)
    return out

def to_submission(forecast, sample=None):
    sub = forecast.copy()
    sub['id'] = sub.scenario_id + '_' + sub.route_id + '_' + sub.departure_slot.astype(int).astype(str) + '_' + sub.leg.astype(int).astype(str)
    sub = sub[['id', *QCOLS]]
    if sample is not None and 'dmin_q50_cm' in sample:
        if sub.id.duplicated().any() or set(sub.id) != set(sample.id):
            raise ValueError('Submission IDs do not exactly match the sample')
        sub = sample[['id']].merge(sub, on='id', validate='one_to_one')
        if list(sub.columns) != list(sample.columns):
            raise ValueError('Submission columns do not match the sample')
    if not np.isfinite(sub[QCOLS].to_numpy()).all() or (np.diff(sub[QCOLS].to_numpy(), axis=1) < 0).any():
        raise ValueError('Nonfinite or crossing submission quantiles')
    return sub


'''

# ======= analog_model.py =======
SOURCES['analog_model.py'] = r'''
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from river_model import QUANTILES
FEATURES = ['current_min', 'horizon', 'bottleneck_change24', 'bottleneck_change72', 'bottleneck_change168', 'bottleneck_above_min504', 'season_sin', 'season_cos', 'G09_change72', 'G20_change72', 'G10_change72', 'G02_change72', 'G05_change72']
SCALES = np.array([100, 72, 22, 40, 80, 80, 4, 4, 80, 70, 50, 50, 60])

def weighted_quantile(values, weights, quantiles=QUANTILES):
    order = np.argsort(values, kind='stable')
    v, w = (values[order], weights[order])
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(quantiles, cdf, v)

@threadpool_limits.wrap(limits=1)
def analog_predict(rows, X, query_rows, query_X, settings=None, variants=None):
    if settings is None:
        settings = dict(neighbors=50, ridge=None, trend_adjustment=0.0)
    choices = variants or {'prediction': settings}
    training_matrix = np.nan_to_num(X[FEATURES].to_numpy() / SCALES, nan=0)
    query_matrix = np.nan_to_num(query_X[FEATURES].to_numpy() / SCALES, nan=0)
    residual = rows.dmin_cm.to_numpy() - X.current_min.to_numpy()
    trends = X.trend72_decay168.to_numpy() - X.current_min.to_numpy()
    query_trends = query_X.trend72_decay168.to_numpy() - query_X.current_min.to_numpy()
    output = {name: np.zeros((len(query_rows), 4)) for name in choices}
    for leg in [1, 2, 3]:
        for branch in [0, 1] if leg == 1 else [0]:
            ti = np.flatnonzero((rows.official.to_numpy() == 0) & (rows.leg.to_numpy() == leg) & (X.branch.to_numpy() == branch if leg == 1 else True))
            vi = np.flatnonzero((query_rows.leg.to_numpy() == leg) & (query_X.branch.to_numpy() == branch if leg == 1 else True))
            if not len(vi):
                continue
            if not len(ti):
                raise ValueError('No historical analogues for this leg')
            model = NearestNeighbors(n_neighbors=min(1000, len(ti)), n_jobs=2).fit(training_matrix[ti])
            distance, positions = model.kneighbors(query_matrix[vi])
            issues = rows.issue_time.to_numpy()[ti]
            max_neighbors = max((c['neighbors'] for c in choices.values()))
            for local, (dist, pos) in enumerate(zip(distance, positions)):
                used, keep = (set(), [])
                for j, k in enumerate(pos):
                    if issues[k] not in used:
                        used.add(issues[k])
                        keep.append(j)
                    if len(keep) == max_neighbors:
                        break
                for name, config in choices.items():
                    chosen = np.asarray(keep[:config['neighbors']], dtype=int)
                    neighbors = ti[pos[chosen]]
                    bandwidth = max(dist[chosen[-1]] * 0.6, 0.01)
                    weights = np.exp(-0.5 * (dist[chosen] / bandwidth) ** 2)
                    values = residual[neighbors].copy()
                    if config.get('ridge') is not None:
                        delta = training_matrix[neighbors] - query_matrix[vi[local]]
                        A = np.column_stack([np.ones(len(delta)), delta])
                        penalty = np.eye(A.shape[1]) * config['ridge']
                        penalty[0, 0] = 1e-06
                        coefficients = np.linalg.solve(A.T @ (weights[:, None] * A) + penalty, A.T @ (weights * values))
                        values = values - delta @ coefficients[1:]
                    adjustment = config.get('trend_adjustment', 0)
                    values += adjustment * (query_trends[vi[local]] - trends[neighbors])
                    output[name][vi[local]] = query_X.current_min.iloc[vi[local]] + weighted_quantile(values, weights)
    return output if variants is not None else output['prediction']


'''

# ======= state_analog.py =======
SOURCES['state_analog.py'] = r'''
import numpy as np
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from analog_model import FEATURES, SCALES, weighted_quantile
from river_model import ordered
VARIANTS = {'seasonal80': dict(neighbors=80, ridge=5.0, season=1.0, horizon=48.0, level=100.0), 'seasonal160': dict(neighbors=160, ridge=10.0, season=1.0, horizon=48.0, level=100.0), 'seasonal160_log': dict(neighbors=160, ridge=10.0, season=1.0, horizon=48.0, level=150.0, relative=True), 'local100': dict(neighbors=100, ridge=8.0, season=1.0, horizon=48.0, level=100.0, local=True), 'local160_log': dict(neighbors=160, ridge=12.0, season=1.0, horizon=48.0, level=150.0, relative=True, local=True), 'weekly100': dict(neighbors=100, ridge=10.0, season=1.0, horizon=48.0, level=120.0, week=True)}

def state_matrix(X, leg, setting):
    if setting.get('local', False):
        upstream = {1: ['G02', 'G05', 'G09', 'G20'], 2: ['G09', 'G20', 'G10', 'G03'], 3: ['G09', 'G20', 'G10', 'G03']}[leg]
        names = FEATURES[:8] + [g + '_change72' for g in upstream] + [g + '_change24' for g in upstream]
        scales = np.r_[SCALES[:8], [60.0] * len(upstream), [30.0] * len(upstream)]
    else:
        names = FEATURES
        scales = SCALES.copy()
    scales[0] = setting['level']
    scales[1] = setting['horizon']
    scales[6:8] = setting['season']
    matrix = np.nan_to_num(X[names].to_numpy() / scales, nan=0.0)
    if setting.get('relative', False):
        norm = np.maximum(X.current_min.to_numpy() + 100, 80) / 400
        idx = [i for i, n in enumerate(names) if 'change' in n or 'above_min' in n]
        matrix[:, idx] /= norm[:, None]
    return matrix

@threadpool_limits.wrap(limits=1)
def state_predict(rows, X, query_rows, query_X, variants=None):
    variants = variants or VARIANTS
    output = {name: np.zeros((len(query_rows), 4)) for name in variants}
    base = X.current_min.to_numpy()
    bt = query_X.current_min.to_numpy()
    for leg in [1, 2, 3]:
        for branch in [0, 1] if leg == 1 else [0]:
            ti = np.flatnonzero((rows.official.to_numpy() == 0) & (rows.leg.to_numpy() == leg) & (X.branch.to_numpy() == branch if leg == 1 else True))
            vi = np.flatnonzero((query_rows.leg.to_numpy() == leg) & (query_X.branch.to_numpy() == branch if leg == 1 else True))
            if not len(vi):
                continue
            issues = rows.issue_time.to_numpy()[ti]
            for name, setting in variants.items():
                print(f'State analog {name}, leg {leg}, branch {branch}', flush=True)
                a = state_matrix(X.iloc[ti], leg, setting)
                b = state_matrix(query_X.iloc[vi], leg, setting)
                model = NearestNeighbors(n_neighbors=min(2500, len(ti)), n_jobs=1).fit(a)
                ds, positions = model.kneighbors(b)
                residual = rows.dmin_cm.to_numpy()[ti] - base[ti]
                if setting.get('relative', False):
                    residual /= np.maximum(base[ti] + 100, 80) / 400
                for i, (distance, pos) in enumerate(zip(ds, positions)):
                    used = set()
                    keep = []
                    for j, k in enumerate(pos):
                        stamp = issues[k]
                        if setting.get('week', False):
                            stamp = stamp.astype('datetime64[D]').astype(int) // 7
                        if stamp not in used:
                            used.add(stamp)
                            keep.append(j)
                        if len(keep) >= setting['neighbors']:
                            break
                    chosen = np.asarray(keep, dtype=int)
                    ns = pos[chosen]
                    bandwidth = max(distance[chosen[-1]] * 0.65, 0.01)
                    weight = np.exp(-0.5 * (distance[chosen] / bandwidth) ** 2)
                    values = residual[ns].copy()
                    delta = a[ns] - b[i]
                    A = np.column_stack([np.ones(len(ns)), delta])
                    penalty = np.eye(A.shape[1]) * setting['ridge']
                    penalty[0, 0] = 1e-06
                    coef = np.linalg.solve(A.T @ (weight[:, None] * A) + penalty, A.T @ (weight * values))
                    values -= delta @ coef[1:]
                    quantiles = weighted_quantile(values, weight)
                    if setting.get('relative', False):
                        quantiles *= max(bt[vi[i]] + 100, 80) / 400
                    output[name][vi[i]] = bt[vi[i]] + quantiles
    return {name: ordered(p) for name, p in output.items()}


'''

# ======= enhanced_model.py =======
SOURCES['enhanced_model.py'] = r'''
from pathlib import Path
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from river_model import QUANTILES, ordered, Files, Geometry

def enhanced_features(X, data_dir='.', mode='relative'):
    out = X[['current_min', 'current_second', 'current_gap', 'bottleneck', 'horizon', 'horizon_start', 'horizon_end', 'leg', 'branch', 'season_sin', 'season_cos', 'bottleneck_change24', 'bottleneck_change72', 'bottleneck_change168', 'bottleneck_above_min504', 'leg_depth_range']].to_dict(orient='series')
    base = X.current_min.to_numpy()
    for name in X:
        if name.startswith('trend') or name == 'recession72':
            out[name + '_delta'] = X[name] - base
    for g in [f'G{i:02}' for i in range(1, 22)]:
        for h in [6, 24, 48, 72, 168, 336, 480]:
            out[f'{g}_change{h}'] = X[f'{g}_change{h}']
        for h in [72, 168, 504]:
            out[f'{g}_range{h}'] = X[f'{g}_range{h}']
        out[f'{g}_above_min504'] = X[f'{g}_above_min504']
        out[f'{g}_acceleration'] = X[f'{g}_change24'] / 24 - X[f'{g}_change72'] / 72
        if mode == 'full':
            out[f'{g}_last'] = X[f'{g}_last']
    for q in [f'Q{i:02}' for i in range(1, 17)]:
        for h in [24, 72, 168]:
            out[f'{q}_ratio{h}'] = X[f'{q}_ratio{h}']
        out[f'{q}_log'] = np.log1p(np.maximum(X[f'{q}_smooth6'], 0))
    geometry = Geometry(Files(data_dir))
    offsets = geometry.segments.dropna(subset=['gauge_id']).groupby('gauge_id').apply(lambda g: (g.guaranteed_depth_cm - g.ref_level_shifted_cm).min(), include_groups=False)
    for gauge, offset in offsets.items():
        out[f'{gauge}_depth_gap'] = X[f'{gauge}_last'] + offset - base
    gauge_groups = {1: ['G02', 'G05', 'G11', 'G18', 'G07', 'G21', 'G01', 'G04', 'G08'], 2: ['G09', 'G20', 'G10', 'G03', 'G19', 'G14', 'G15', 'G02'], 3: ['G09', 'G20', 'G10', 'G03']}
    for h in [24, 72, 168]:
        upstream = np.zeros(len(X))
        local = np.zeros(len(X))
        for leg, gs in gauge_groups.items():
            mask = X.leg.to_numpy() == leg
            upstream[mask] = X.loc[mask, [f'{g}_change{h}' for g in gs[:3]]].mean(axis=1)
            local[mask] = X.loc[mask, [f'{g}_change{h}' for g in gs[-3:]]].mean(axis=1)
        out[f'upstream_change{h}'] = upstream
        out[f'local_change{h}'] = local
        out[f'upstream_local_change{h}'] = upstream - local
    out = pd.DataFrame(out, index=X.index)
    if mode == 'compact':
        keep = [c for c in out if not c.startswith('G') or c.endswith(('_change72', '_change168', '_change336', '_above_min504', '_depth_gap'))]
        out = out[keep]
    return out.astype(np.float32)

def boosted_predict(rows, X, query_rows, query_X, settings, data_dir='.', threads=4, model_dir=None):
    engine = settings.get('engine', 'lightgbm')
    xx = enhanced_features(X, data_dir, settings.get('features', 'relative'))
    xt = enhanced_features(query_X, data_dir, settings.get('features', 'relative'))
    pred = np.zeros((len(query_rows), 4))
    groups = [1, 2, 3] if settings.get('separate_legs', False) else [0]
    for leg in groups:
        train = rows.official.to_numpy() == 0
        if settings.get('use_official', False):
            train = np.ones(len(rows), dtype=bool)
        valid = np.ones(len(query_rows), dtype=bool)
        if leg:
            train &= rows.leg.to_numpy() == leg
            valid &= query_rows.leg.to_numpy() == leg
        a = xx.loc[train].copy()
        b = xt.loc[valid].copy()
        baseline = settings.get('baseline', 'current')
        anchor = X.current_min.to_numpy() if baseline in ['current', 'log'] else X.trend72_decay168.to_numpy()
        anchor_test = query_X.current_min.to_numpy() if baseline in ['current', 'log'] else query_X.trend72_decay168.to_numpy()
        y = rows.dmin_cm.to_numpy() - anchor
        weights = 1.0 + (rows.leg.to_numpy() == 1).astype(float)
        if settings.get('balanced_legs', False) and (not leg):
            for l, mass in [(1, 0.5), (2, 0.25), (3, 0.25)]:
                count = ((rows.leg.to_numpy() == l) & train).sum()
                weights[rows.leg.to_numpy() == l] = mass * train.sum() / max(count, 1)
        if settings.get('use_official', False):
            weights *= np.where(rows.official.to_numpy() == 1, 0.12, 1.0)
        if baseline == 'log':
            y = np.log(np.maximum(rows.dmin_cm.to_numpy() + 100, 10) / np.maximum(anchor + 100, 10))
            weights *= np.maximum(anchor + 100, 10) / 400
        if engine == 'catboost':
            from catboost import CatBoostRegressor
            a = a.fillna(0)
            b = b.fillna(0)
            model = CatBoostRegressor(loss_function='MultiQuantile:alpha=0.05,0.1,0.25,0.5', iterations=settings.get('iterations', 900), depth=settings.get('depth', 6), learning_rate=settings.get('learning_rate', 0.04), l2_leaf_reg=settings.get('l2', 12), random_seed=2026, thread_count=threads, verbose=False, allow_writing_files=False, random_strength=0.5)
            start = time.monotonic()
            model.fit(a, y[train], sample_weight=weights[train])
            delta = model.predict(b)
            if model_dir:
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                model.save_model(str(Path(model_dir) / f'leg{leg}.cbm'))
            print(f'CatBoost leg {leg}: {time.monotonic() - start:.1f}s', flush=True)
        else:
            delta = np.zeros((valid.sum(), 4))
            for j, q in enumerate(QUANTILES):
                start = time.monotonic()
                model = lgb.LGBMRegressor(objective='quantile', alpha=float(q), n_estimators=settings.get('iterations', 850), learning_rate=settings.get('learning_rate', 0.035), num_leaves=settings.get('leaves', 15), min_child_samples=settings.get('min_child', 180), reg_lambda=settings.get('l2', 15), colsample_bytree=0.85, max_bin=127, random_state=2026 + j, n_jobs=threads, force_col_wise=True, deterministic=True, verbosity=-1)
                model.fit(a, y[train], sample_weight=weights[train])
                delta[:, j] = model.predict(b)
                print(f'Enhanced LightGBM leg {leg} q{int(q * 100):02}: {time.monotonic() - start:.1f}s', flush=True)
                if model_dir:
                    Path(model_dir).mkdir(parents=True, exist_ok=True)
                    model.booster_.save_model(str(Path(model_dir) / f'leg{leg}_q{int(100 * q)}.txt'))
        if baseline == 'log':
            pred[valid] = np.maximum(anchor_test[valid] + 100, 10)[:, None] * np.exp(np.clip(delta, -4, 4)) - 100
        else:
            pred[valid] = anchor_test[valid, None] + delta
    return ordered(pred)


'''

# ======= history_augment.py =======
SOURCES['history_augment.py'] = r'''
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from river_model import Files, Water, Geometry, exact_targets, feature_matrix

def proxy_features(levels):
    columns = ['G09', 'G10', 'G19', 'G20', 'G14']
    source = levels[columns].ffill()
    result = {g: source[g] for g in columns}
    for h in [6, 24, 72]:
        for g in columns:
            result[f'{g}_change{h}'] = source[g] - source[g].shift(h)
    angle = levels.index.dayofyear.to_numpy() * 2 * np.pi / 365.25
    result['season_sin'], result['season_cos'] = (np.sin(angle), np.cos(angle))
    return pd.DataFrame(result, index=levels.index).astype(np.float32)

def augmented_rows(data_dir, cutoff, cache_dir='experiments/deep/augmentation', stride=2, dutch=False):
    dest = Path(cache_dir) / (('full_' if dutch else '') + str(cutoff))
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'features.parquet').exists():
        return (pd.read_parquet(dest / 'rows.parquet'), pd.read_parquet(dest / 'features.parquet'))
    water = Water(Files(data_dir))
    geometry = Geometry(Files(data_dir))
    features = proxy_features(water.levels)
    fit = water.levels.G03.notna() & (water.levels.index < pd.Timestamp(f'{cutoff}-01-01'))
    fit &= np.arange(len(water.levels)) % 6 == 0
    y = water.levels.G03 - water.levels.G19
    model = lgb.LGBMRegressor(objective='regression', n_estimators=350, num_leaves=15, min_child_samples=100, learning_rate=0.04, reg_lambda=12, max_bin=127, n_jobs=2, random_state=2026, force_col_wise=True, deterministic=True, verbosity=-1)
    model.fit(features.loc[fit], y.loc[fit])
    valid = water.levels.G03.notna() & (water.levels.index >= pd.Timestamp(f'{cutoff}-01-01'))
    error = model.predict(features.loc[valid]) + water.levels.G19.loc[valid].to_numpy() - water.levels.G03.loc[valid].to_numpy() if valid.any() else np.array([])
    report = {'cutoff': cutoff, 'mapping_train_hours': int(fit.sum()), 'mapping_valid_mae': float(np.abs(error).mean()) if len(error) else None, 'mapping_valid_rmse': float(np.sqrt(np.mean(error ** 2))) if len(error) else None}
    missing = water.levels.G03.isna() & (water.levels.index < pd.Timestamp('2011-11-01'))
    replacement = model.predict(features.loc[missing]) + water.levels.G19.loc[missing].to_numpy()
    water.levels.loc[missing, 'G03'] = replacement.astype(np.float32)
    if dutch:
        from proxy_probe import features as dutch_features
        inputs = dutch_features(water.levels, water.flows)
        report['dutch_mapping'] = {}
        for gauge in ['G01', 'G04', 'G08', 'G21']:
            known = water.levels[gauge].notna() & (water.levels.index < pd.Timestamp(f'{cutoff}-01-01'))
            known &= np.arange(len(water.levels)) % 6 == 0
            proxy = lgb.LGBMRegressor(n_estimators=350, num_leaves=15, min_child_samples=80, learning_rate=0.04, reg_lambda=12, n_jobs=2, max_bin=127, force_col_wise=True, deterministic=True, random_state=2026, verbosity=-1)
            proxy.fit(inputs.loc[known], (water.levels[gauge] - water.levels.G07).loc[known])
            validation = water.levels[gauge].notna() & (water.levels.index >= pd.Timestamp(f'{cutoff}-01-01'))
            discrepancy = proxy.predict(inputs.loc[validation]) + water.levels.G07.loc[validation].to_numpy() - water.levels.loc[validation, gauge].to_numpy() if validation.any() else np.array([])
            report['dutch_mapping'][gauge] = {'mae': float(np.abs(discrepancy).mean()) if len(discrepancy) else None}
            absent = water.levels[gauge].isna() & (water.levels.index < pd.Timestamp('2015-01-01'))
            values = proxy.predict(inputs.loc[absent]) + water.levels.G07.loc[absent].to_numpy()
            water.levels.loc[absent, gauge] = values.astype(np.float32)
    water.history = water.levels.join(water.flows)
    water.level_array = water.levels.to_numpy()
    dates = pd.date_range('2000-01-23 06:00', '2011-10-19 06:00', freq=f'{stride}D')
    chunks = []
    slots = [0, 4, 8, 12, 16, 20, 24]
    for route in ['B1', 'B2']:
        for leg in [1, 2, 3] if dutch else [2, 3]:
            use_dates = pd.date_range('2000-01-23 06:00', '2014-12-20 06:00', freq=f'{stride}D') if leg == 1 else dates
            frame = pd.DataFrame({'issue_time': np.repeat(use_dates.to_numpy(), len(slots)), 'departure_slot': np.tile(slots, len(use_dates))})
            frame['scenario_id'] = 'P-' + frame.issue_time.dt.strftime('%Y%m%d')
            frame['issue_id'] = frame.scenario_id
            frame['route_id'], frame['leg'] = (route, leg)
            frame['queue_h'] = 1.5 + (frame.issue_time - pd.Timestamp('2000-01-01')).dt.days.to_numpy() % 4
            frame['dmin_cm'], frame['target_end'] = exact_targets(frame, water, geometry)
            frame['official'] = 0
            frame['proxy'] = 1
            chunks.append(frame.dropna(subset=['dmin_cm']))
    rows = pd.concat(chunks, ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        X, _ = feature_matrix(rows, water, geometry)
    rows.to_parquet(dest / 'rows.parquet', index=False)
    X.to_parquet(dest / 'features.parquet', index=False)
    report['additional_rows'] = len(rows)
    (dest / 'report.json').write_text(json.dumps(report, indent=2))
    print('Early-history augmentation', json.dumps(report), flush=True)
    return (rows, X)
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='+', default=[2018, 2020])
    a = ap.parse_args()
    for y in a.years:
        augmented_rows('.', y)


'''

# ======= proxy_probe.py =======
SOURCES['proxy_probe.py'] = r'''
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from river_model import Files

def features(levels, flows):
    source = levels[['G02', 'G05', 'G07', 'G11', 'G12', 'G18']].ffill()
    result = {g: source[g] for g in source}
    for h in [6, 24, 72]:
        for g in source:
            result[f'{g}_change{h}'] = source[g] - source[g].shift(h)
    for q in flows:
        if q not in ['Q11', 'Q15']:
            result[q] = flows[q].ffill()
    angle = levels.index.dayofyear.to_numpy() * 2 * np.pi / 365.25
    result['season_sin'], result['season_cos'] = (np.sin(angle), np.cos(angle))
    return pd.DataFrame(result, index=levels.index).astype(np.float32)

def main():
    f = Files('.')
    lv = pd.read_parquet(f.get('history', 'levels_hourly.parquet'))
    fl = pd.read_parquet(f.get('history', 'discharge_hourly.parquet'))
    x = features(lv, fl)
    report = {}
    tr = (lv.index >= pd.Timestamp('2015-01-01')) & (lv.index < pd.Timestamp('2018-01-01')) & (np.arange(len(lv)) % 6 == 0)
    va = (lv.index >= pd.Timestamp('2018-01-01')) & (np.arange(len(lv)) % 6 == 0)
    for g in ['G01', 'G04', 'G08', 'G21']:
        baseline = lv.G07
        model = lgb.LGBMRegressor(n_estimators=350, num_leaves=15, min_child_samples=80, learning_rate=0.04, reg_lambda=12, n_jobs=2, max_bin=127, force_col_wise=True, deterministic=True, random_state=2026, verbosity=-1)
        model.fit(x.loc[tr], (lv[g] - baseline).loc[tr])
        err = model.predict(x.loc[va]) + baseline.loc[va].to_numpy() - lv.loc[va, g].to_numpy()
        report[g] = {'mae': float(np.abs(err).mean()), 'rmse': float(np.sqrt(np.mean(err ** 2))), 'bias': float(err.mean()), 'q05': float(np.quantile(err, 0.05)), 'q95': float(np.quantile(err, 0.95))}
        print(g, report[g], flush=True)
    Path('experiments/deep/dutch_proxy_report.json').write_text(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()


'''

# ======= predict_v2.py =======
SOURCES['predict_v2.py'] = r'''
from pathlib import Path
import numpy as np
import pandas as pd
from river_model import ordered, to_forecast
from state_analog import state_predict
from enhanced_model import boosted_predict
from history_augment import augmented_rows

def blend_forecasts(predictions, rows, config):
    output = np.zeros((len(rows), 4))
    for name, pred in predictions.items():
        per_leg = config.get('leg_weights', {}).get(name)
        weight = np.array([per_leg[str(int(leg))] for leg in rows.leg])[:, None] if per_leg else config['weights'][name]
        output += weight * pred
    return ordered(output)

def raw_predict_v2(rows, X, query_rows, query_X, config, data_dir='.', cache_dir=None, model_dir=None):
    cutoff = int(rows.target_end.max().year) + 1
    needs_history = any((spec.get('augmentation', False) for spec in config['components'].values()))
    rr, xx = (rows, X)
    if needs_history:
        cache_dir = cache_dir or Path('experiments/deep/augmentation')
        ar, ax = augmented_rows(data_dir, cutoff, cache_dir, stride=config['stride'], dutch=config.get('dutch_augmentation', False))
        rr = pd.concat([rows, ar], ignore_index=True)
        xx = pd.concat([X, ax], ignore_index=True)
    predictions = {}
    for name, weight in config['weights'].items():
        spec = config['components'][name]
        train_rows, train_X = (rr, xx) if spec.get('augmentation', False) else (rows, X)
        if spec['kind'] == 'analogue':
            pred = state_predict(train_rows.reset_index(drop=True), train_X.reset_index(drop=True), query_rows.reset_index(drop=True), query_X.reset_index(drop=True), {name: spec['settings']})[name]
        else:
            dest = Path(model_dir) / name if model_dir else None
            pred = boosted_predict(train_rows, train_X, query_rows, query_X, spec['settings'], data_dir, config['threads'], dest)
        predictions[name] = pred
        if model_dir:
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            to_forecast(query_rows, pred).to_parquet(Path(model_dir) / f'queries_{name}.parquet', index=False)
    return blend_forecasts(predictions, query_rows, config)


'''

# ======= calibration.py =======
SOURCES['calibration.py'] = r'''
import numpy as np
from river_model import QUANTILES, ordered

def groups(rows, base):
    return np.array([f'{int(leg)}_{int(depth < 250)}' for leg, depth in zip(rows.leg, base)])

def fit_calibration(y, pred, rows, base, strength=0.5, grouped=True):
    errors = np.asarray(y)[:, None] - pred
    global_offset = np.array([np.quantile(errors[:, j], q) for j, q in enumerate(QUANTILES)])
    result = {'strength': float(strength), 'global': global_offset.tolist(), 'groups': {}}
    if grouped:
        codes = groups(rows, base)
        for code in sorted(set(codes)):
            mask = codes == code
            count = rows.loc[mask, 'issue_time'].nunique()
            local = np.array([np.quantile(errors[mask, j], q) for j, q in enumerate(QUANTILES)])
            shrink = count / (count + 12.0)
            result['groups'][code] = (shrink * local + (1 - shrink) * global_offset).tolist()
    return result

def apply_calibration(pred, rows, base, calibration):
    codes = groups(rows, base)
    offsets = np.array([calibration['groups'].get(c, calibration['global']) for c in codes])
    return ordered(pred + calibration['strength'] * offsets)


'''

# ======= organizer.py =======
SOURCES['organizer.py'] = r'''
import importlib.util
from pathlib import Path
import shutil
import sys
from river_model import Files

def load_package(data_dir):
    root = Path(data_dir).resolve()
    candidates = [root / 'riverload' / '__init__.py', root / '__init__.py']
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        try:
            import riverload
            return riverload
        except ImportError as exc:
            raise FileNotFoundError('The supplied riverload package is required for C2 validation') from exc
    spec = importlib.util.spec_from_file_location('riverload', path, submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules['riverload'] = module
    spec.loader.exec_module(module)
    return module

def load_env(data_dir, split, stage_dir):
    package = load_package(data_dir)
    files = Files(data_dir)
    if (files.root / 'reference' / 'route_pieces.csv').is_file() and (files.root / split / 'scenarios.csv').is_file():
        return package.Env(files.root, split)
    stage = Path(stage_dir).resolve()
    for section, names in {'reference': ['route_pieces.csv', 'segments.csv'], split: ['scenarios.csv', 'manifests.csv', 'lock_queues.csv'] + (['labels_dmin.csv', 'labels_best_value.csv'] if split == 'train' else [])}.items():
        (stage / section).mkdir(parents=True, exist_ok=True)
        for name in names:
            src, dst = (files.get(section, name), stage / section / name)
            if not dst.is_file() or dst.stat().st_mtime_ns != src.stat().st_mtime_ns:
                shutil.copy2(src, dst)
    return package.Env(stage, split)


'''

# ======= trajectory_analog.py =======
SOURCES['trajectory_analog.py'] = r'''
"""Trajectory-preserving ensemble analogue forecaster.

Data provenance: Only official pre-2022 history, and provided scenario input windows.
No pretrained hydrology models or post-2021 external information are used.

Each neighbour supplies a COMPLETE historical future trajectory.  We anchor that
trajectory at the query's last observed level, replay it on the exact geometry,
and compute the minimum, rather than using independent per-segment quantiles.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from river_model import QUANTILES, Water, Geometry, Files, ordered

# scales in approximately cm; all feature names exist in source notebook.
STATE_COLS = [
    'current_min', 'current_gap', 'bottleneck_change24',
    'bottleneck_change72', 'bottleneck_change168',
    'bottleneck_above_min504', 'season_sin', 'season_cos',
    'G02_change72', 'G05_change72', 'G09_change72',
    'G20_change72', 'G10_change72', 'G03_change72',
    'G02_change24', 'G05_change24', 'G09_change24',
    'G20_change24', 'G10_change24',
    'Q01_ratio72', 'Q02_ratio72', 'Q03_ratio72',
    'Q04_ratio72', 'Q05_ratio72', 'Q06_ratio72',
    'Q07_ratio72', 'Q08_ratio72', 'Q09_ratio72',
    'Q10_ratio72', 'Q11_ratio72', 'Q12_ratio72',
    'Q13_ratio72', 'Q14_ratio72', 'Q15_ratio72', 'Q16_ratio72',
]
STATE_SCALES = np.array([
    125., 90., 28., 52., 95., 95.,
    .85, .85, 95.,95.,95.,95.,95.,95.,
    50.,50.,50.,50.,50.,
    .55,.55,.55,.55,.55,.55,.55,.55,.55,.55,.55,.55,.55,.55,.55,.55,
], dtype=np.float32)
assert len(STATE_COLS) == len(STATE_SCALES)


def state_array(X, train_medians=None):
    a = X[STATE_COLS].to_numpy(dtype=np.float32, copy=True)
    a /= STATE_SCALES
    if train_medians is None:
        means = np.nanmedian(a, axis=0)
        means[~np.isfinite(means)] = 0
    else:
        means = train_medians
    missing = ~np.isfinite(a)
    if missing.any():
        a[missing] = np.broadcast_to(means, a.shape)[missing]
    return np.ascontiguousarray(a), means


def weighted_quantiles(values, weights):
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if valid.sum() < 12:
        return None
    v = values[valid]
    w = weights[valid]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(QUANTILES, cdf, v)


@threadpool_limits.wrap(limits=1)
def predict_trajectories(rows, X, query_rows, query_X, data_dir,
                         neighbors=96, trend_weight=0.35, kernel=0.8,
                         cutoff='2022-01-01', verbose=True):
    """Return N*4 depth quantiles and row-level availability flags.

    `rows` is chronologically permitted training only. The function *also*
    explicitly restricts the historical donor ending hour before `cutoff`.
    The historical dataset itself must end no later than 2021-12-31.
    All query observations come from `query_X`, which was built strictly using
    the 21 days *before* each query's issue_time.
    """
    t0 = time.monotonic()
    water = Water(Files(data_dir))
    geom = Geometry(Files(data_dir))
    yout = np.full((len(query_rows), 4), np.nan)
    available = np.zeros(len(query_rows), dtype=bool)
    hist_array = water.level_array
    col = water.level_col
    limit = min(pd.Timestamp(cutoff), water.levels.index.max() + pd.Timedelta(hours=1))
    # More than enough room for slot 24 plus the 3 legs, queue, and dwell.
    # The precise final timestamp is checked again for every donor/gauge.
    safety = limit - pd.Timedelta(hours=450)
    train_route = rows.route_id.to_numpy()
    official = rows.official.to_numpy()
    slots = rows.departure_slot.to_numpy()
    legs = rows.leg.to_numpy()
    cutoff_date = rows.issue_time.to_numpy() < np.datetime64(safety)
    arr_query_leg = query_rows.leg.to_numpy()
    q_route = query_rows.route_id.to_numpy()
    q_scenario = query_rows.scenario_id.to_numpy()
    q_slots = query_rows.departure_slot.to_numpy()
    q_queues = query_rows.queue_h.to_numpy(dtype=float)
    # NumPy caches avoid repeatedly materializing very large pandas columns
    # inside the inner (scenario x slot x river-piece) simulation loop.
    hist_last = {g: X[f'{g}_last'].to_numpy(dtype=np.float64)
                 for g in water.gauges} if hasattr(water, 'gauges') else {
                     g: X[f'{g}_last'].to_numpy(dtype=np.float64)
                     for g in col}
    hist_trend = {g: X[f'{g}_change72'].to_numpy(dtype=np.float64)
                  for g in col}
    q_last = {g: query_X[f'{g}_last'].to_numpy(dtype=np.float64)
              for g in col}
    q_trend = {g: query_X[f'{g}_change72'].to_numpy(dtype=np.float64)
               for g in col}

    # For each route/leg, simulate shared donor trajectories for all 25 slots.
    for (route, leg), inds in query_rows.groupby(['route_id','leg'], sort=False).indices.items():
        leg = int(leg)
        branch = str(route)[-1]
        donor_route = 'B' + branch
        candidates = np.flatnonzero((official == 0) & (train_route == donor_route) &
                                    (slots == 12) & (legs == leg) & cutoff_date)
        if len(candidates) < 16:
            if verbose:
                print(f'Trajectory ensemble: too few donors for {route}/leg{leg}: {len(candidates)}', flush=True)
            continue
        a, med = state_array(X.iloc[candidates])
        b, _ = state_array(query_X.iloc[inds], med)
        # The 25 slots of a scenario share a donor set; this preserves
        # temporal coherence and avoids repeated neighbour searches.
        idx = np.asarray(inds)
        scenario_groups = {}
        for local_idx, absolute_idx in enumerate(idx):
            scenario_groups.setdefault(q_scenario[absolute_idx], []).append(local_idx)
        reps = np.asarray([v[0] for v in scenario_groups.values()])
        n_neigh = min(max(neighbors * 3, 180), len(candidates))
        nn = NearestNeighbors(n_neighbors=n_neigh, algorithm='auto', n_jobs=1).fit(a)
        ds, positions = nn.kneighbors(b[reps])
        candidate_times = rows.issue_time.to_numpy()[candidates]
        # Donors have distinct issue times because we select only slot 12,
        # one route, one leg, and purely synthetic historical rows.
        donor_idxs = ((candidate_times - np.datetime64(water.origin)) /
                      np.timedelta64(1, 'h')).astype(np.int64)
        parts = geom.group(route, leg)
        for group_i, (_, positions_local) in enumerate(scenario_groups.items()):
            sampled = positions[group_i, :neighbors]
            distances = ds[group_i, :neighbors]
            chosen_candidates = candidates[sampled]
            donor_origin = donor_idxs[sampled]
            if len(sampled) < 12:
                continue
            denom = max(float(distances[min(len(distances)-1, neighbors-1)]) * kernel, 0.075)
            kernel_weights = np.exp(-0.5 * (distances / denom)**2)
            # all source gauge offsets, timelines from ONLY historical values
            for local_i in positions_local:
                row_i = idx[local_i]
                dep_hours = 48.0 + 6.0 * int(q_slots[row_i])
                queuing = q_queues[row_i] if str(route).endswith('2') else 0.0
                sampled_depth = np.full(len(sampled), np.inf, dtype=np.float64)
                for piece in parts:
                    gauge = piece.gauge
                    if gauge not in col:
                        raise KeyError('Reference geometry includes missing gauge ' + gauge)
                    begin = int(np.floor(dep_hours + queuing + piece.enter))
                    end = int(np.floor(dep_hours + queuing + piece.exit))
                    hours = np.arange(begin, end + 1, dtype=np.int64)
                    positions_h = donor_origin[:, None] + hours[None, :]
                    in_bounds = ((positions_h >= 0) & (positions_h < len(hist_array))).all(axis=1)
                    positions_h = np.clip(positions_h, 0, len(hist_array) - 1)
                    future = hist_array[positions_h, col[gauge]].astype(np.float64)
                    donor_levels = hist_last[gauge][chosen_candidates]
                    current_level = q_last[gauge][row_i]
                    change_diff = (q_trend[gauge][row_i] -
                                   hist_trend[gauge][chosen_candidates])
                    future_delta = future - donor_levels[:, None]
                    if trend_weight:
                        # Decay limits the effect of extrapolating the
                        # difference between current and donor prior trends.
                        decay_h = 120.0 * -np.expm1(-hours / 120.0)
                        future_delta += (trend_weight * change_diff[:, None] /
                                         72.0 * decay_h[None, :])
                    part_depth = np.nanmin(current_level + future_delta + piece.offset,
                                           axis=1)
                    part_depth[~in_bounds] = np.nan
                    sampled_depth = np.minimum(sampled_depth, part_depth)
                quantiles = weighted_quantiles(sampled_depth, kernel_weights)
                if quantiles is not None:
                    yout[row_i] = quantiles
                    available[row_i] = True
        if verbose:
            print(f'Trajectory ensemble: {route}/leg{leg}, {len(candidates)} donors, '
                  f'{int(available[idx].sum())}/{len(idx)} predicted', flush=True)
    if verbose:
        print(f'Trajectory ensemble complete in {time.monotonic()-t0:.1f}s '
              f'({available.mean():.1%} of rows covered)', flush=True)
    return ordered(np.where(np.isfinite(yout), yout, 0)), available


'''

# ======= test_upgrade.py =======
SOURCES['test_upgrade.py'] = r'''
"""Dependency-light synthetic unit tests, NOT a claim of contest performance."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import trajectory_analog as ta
from run import combine, learn_weights, learn_calibration
from river_model import QUANTILES, QCOLS, pinball


def test_weighted_quantile():
    a = ta.weighted_quantiles(np.arange(100.0), np.ones(100))
    assert len(a) == 4 and np.all(np.diff(a) >= 0)
    assert a[0] < a[1] < a[2] < a[3]
    assert ta.weighted_quantiles(np.array([np.nan]*20), np.ones(20)) is None


def test_ensemble_calibration():
    rng = np.random.default_rng(4)
    n = 1500
    d = rng.normal(275, 50, n)
    years = np.repeat([2018,2020], n//2)
    legs = np.tile([1,2,3], n//3)
    base = np.column_stack([d-18,d-14,d-8,d+3])
    phys = np.column_stack([d-11,d-9,d-5,d+1])
    oof = pd.DataFrame({'dmin_cm': d, 'year':years, 'leg':legs,
                       'physical_available':True, 'current_min':d,
                       'issue_time':pd.date_range('2018-01-01', periods=n, freq='D'),
                       'scenario_id':['TR-'+str(k) for k in range(n)]})
    for j,c in enumerate(QCOLS):
        oof[f'baseline_{c}']=base[:,j]
        oof[f'trajectory_{c}']=phys[:,j]
    weights, report=learn_weights(oof)
    assert all(0 <= v <= .60 for v in weights.values())
    cal, diagnostics=learn_calibration(oof,weights)
    assert 'global' in cal and len(cal['global'])==4
    assert np.isfinite(diagnostics['raw_ensemble_loss'])
    p = combine(base,phys,np.ones(n, dtype=bool),legs,weights)
    assert (np.diff(p,axis=1)>=0).all()


def test_trajectory_analogue():
    origin=pd.Timestamp('2015-01-01 00:00:00')
    nh=24*365*6
    time_axis=np.arange(nh)
    observed=235 + 30*np.sin(time_axis/280) + .002 * time_axis
    water=SimpleNamespace(levels=pd.DataFrame(index=pd.date_range(origin,periods=nh,freq='h')),
                          origin=origin,
                          level_array=observed.reshape(-1,1).astype(np.float32),
                          level_col={'G01':0})
    class FakeGeometry:
        def __init__(self, x): pass
        def group(self,route,leg):
            return [SimpleNamespace(gauge='G01',enter=0.,exit=3.,offset=-8.),
                    SimpleNamespace(gauge='G01',enter=3.,exit=7.,offset=-12.)]
    ta.Water=lambda x:water
    ta.Geometry=FakeGeometry
    ta.Files=lambda x:x
    donor_h=np.arange(1500, nh-950, 800)
    issues=origin+pd.to_timedelta(donor_h,unit='h')
    nr=len(issues)
    assert nr>=16
    rr=pd.DataFrame({'scenario_id':['H-'+str(i) for i in range(nr)],
                     'issue_time':issues,'route_id':'B1','leg':1,'departure_slot':12,
                     'queue_h':0.,'official':0})
    qtimes=[origin+pd.Timedelta(hours=nh-600)]
    qry=pd.DataFrame({'scenario_id':['TS-1']*3,'issue_time':qtimes*3,
                      'route_id':'B1','leg':1,'departure_slot':[0,12,24],
                      'queue_h':0.})
    xx=pd.DataFrame(np.zeros((nr,len(ta.STATE_COLS))),columns=ta.STATE_COLS)
    vx=pd.DataFrame(np.zeros((3,len(ta.STATE_COLS))),columns=ta.STATE_COLS)
    xx['current_min']=observed[donor_h-1]-12
    vx['current_min']=observed[nh-601]-12
    for col in ['G01_last','G01_change72']:
        xx[col]=observed[donor_h-1] if col=='G01_last' else 0.
        vx[col]=observed[nh-601] if col=='G01_last' else 0.
    result,good=ta.predict_trajectories(rr,xx,qry,vx,'unused',neighbors=24,cutoff='2021-01-01',verbose=False)
    assert good.tolist() == [True]*3, good
    assert result.shape==(3,4)
    assert np.isfinite(result).all()
    assert (np.diff(result,axis=1)>=0).all()
    assert np.all(result > 0)
    # Exact oracle: a linear river stage has the SAME future delta for each
    # historical donor. Every quantile should match the route minimum.
    straight = 100.0 + 0.5 * np.arange(nh)
    water.level_array = straight.reshape(-1,1).astype(np.float32)
    xx['G01_last'] = straight[donor_h - 1]
    vx['G01_last'] = 300.0
    linear, flags = ta.predict_trajectories(rr,xx,qry,vx,'unused',neighbors=24,
                                             cutoff='2021-01-01',trend_weight=0,
                                             verbose=False)
    # Segment 2 is four cm shallower but reached three hours later.
    expected = 300.0 + 0.5 * (48.0 + 6.0 * np.array([0,12,24]) + 4.0) - 12.0
    assert flags.all()
    assert np.allclose(linear, expected[:,None],atol=.06), (linear,expected)

if __name__ == '__main__':
    for t in [test_weighted_quantile,test_ensemble_calibration,test_trajectory_analogue]:
        t()
        print('PASS',t.__name__)


'''

# ======= model_config.json =======
SOURCES['model_config.json'] = r'''
{
  "version": 2,
  "weights": {
    "aug_local160_log": 0.5,
    "relative_lgb_smooth": 0.5
  },
  "components": {
    "aug_local160_log": {
      "kind": "analogue",
      "augmentation": true,
      "settings": {
        "neighbors": 160,
        "ridge": 12.0,
        "season": 1.0,
        "horizon": 48.0,
        "level": 150.0,
        "relative": true,
        "local": true
      }
    },
    "relative_lgb_smooth": {
      "kind": "boosted",
      "augmentation": false,
      "settings": {
        "baseline": "log",
        "features": "compact",
        "min_child": 800,
        "leaves": 9,
        "iterations": 550,
        "l2": 25,
        "balanced_legs": true
      }
    }
  },
  "dutch_augmentation": false,
  "calibration": {
    "years": [
      2018,
      2020
    ],
    "strength": 1.0,
    "grouped": false
  },
  "cargo": {
    "risk": 0.1,
    "buffer_cm": 5.0,
    "finalists": 2
  },
  "stride": 2,
  "threads": 4,
  "selection_years": [
    2018,
    2020
  ],
  "repeat_validation_year": 2021,
  "leg_weights": {
    "aug_local160_log": {
      "1": 0.75,
      "2": 0.25,
      "3": 0.25
    },
    "relative_lgb_smooth": {
      "1": 0.25,
      "2": 0.75,
      "3": 0.75
    }
  }
}

'''

# ======= forecast_model.py =======
SOURCES['forecast_model.py'] = r'''
"""RIVERLOAD C1 — reproducible CPU-only ensemble with honest chronological CV.

Usage: python run.py <data_dir> <out_dir> [--no-trajectory] [--holdout-2021]

All historical water used is contained in the supplied competition data and ends
at 2021-12-31. No network operations. Output: submission_c1.csv.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from calibration import apply_calibration, fit_calibration
from river_model import (Files, build_dataset, load_dataset, ordered, QUANTILES,
                         QCOLS, to_forecast, to_submission, diagnostics, pinball)
from predict_v2 import raw_predict_v2
from trajectory_analog import predict_trajectories
from organizer import load_env

SEED = 2026


def evaluate_cv_components(rows, X, config, data_dir, out, years=(2018, 2020),
                           use_trajectory=True, threads=4):
    outputs = []
    for year in years:
        cutoff = pd.Timestamp(f'{year}-01-01')
        train = rows.target_end < cutoff
        valid = (rows.official == 1) & (rows.issue_time.dt.year == year)
        tr = rows.loc[train].reset_index(drop=True)
        xx = X.loc[train].reset_index(drop=True)
        val = rows.loc[valid].reset_index(drop=True)
        vx = X.loc[valid].reset_index(drop=True)
        if len(val) == 0 or len(tr) == 0:
            raise ValueError(f'Fold {year} has no data: {len(tr)} train, {len(val)} validation')
        print(f'FOLD {year}: historical={len(tr)}; official holdout={len(val)}; '
              f'unseen issues={val.issue_time.nunique()}', flush=True)
        fold_out = out / f'fold_{year}'
        fold_out.mkdir(exist_ok=True, parents=True)
        # Exactly the already-supplied 9.063 candidate is the safe anchor.
        base = raw_predict_v2(tr, xx, val, vx, config, data_dir=data_dir,
                              cache_dir=out / 'augmentation_cache')
        physical = base.copy()
        good = np.zeros(len(val), dtype=bool)
        if use_trajectory:
            physical, good = predict_trajectories(tr, xx, val, vx, data_dir=data_dir,
                                                   cutoff=f'{year}-01-01',
                                                   verbose=True)
        result = val[['scenario_id', 'issue_time', 'route_id',
                      'departure_slot', 'leg', 'dmin_cm']].copy()
        result['current_min'] = vx.current_min.to_numpy()
        result['physical_available'] = good
        result['year'] = year
        for i, col in enumerate(QCOLS):
            result[f'baseline_{col}'] = base[:, i]
            result[f'trajectory_{col}'] = physical[:, i]
        result.to_parquet(fold_out / 'predictions.parquet', index=False)
        outputs.append(result)
        print(f'FOLD {year}: original-ensemble raw loss={pinball(val.dmin_cm, base):.5f} '
              f'physical available={good.mean():.1%}', flush=True)
    return pd.concat(outputs, ignore_index=True)


def get_matrix(frame, prefix):
    return frame[[f'{prefix}_{col}' for col in QCOLS]].to_numpy(dtype=np.float64)


def combine(base, physical, covered, legs, weights):
    w = np.array([float(weights[str(int(leg))]) for leg in legs])[:, None]
    w = w * covered[:, None]
    return ordered((1.0 - w) * base + w * physical)


def learn_weights(oof):
    y = oof.dmin_cm.to_numpy(dtype=float)
    base, physical = get_matrix(oof, 'baseline'), get_matrix(oof, 'trajectory')
    years = oof.year.to_numpy()
    covered = oof.physical_available.to_numpy(dtype=bool)
    legs = oof.leg.to_numpy()
    weights = {}
    audit = {}
    for leg in np.unique(legs):
        mask = (legs == leg)
        orig_by_year = {int(year): pinball(y[mask & (years == year)],
                                         base[mask & (years == year)])
                        for year in np.unique(years)}
        baseline_mean = np.mean(list(orig_by_year.values()))
        scored = []
        for w in np.linspace(0, 0.60, 13):
            candidate = combine(base[mask], physical[mask], covered[mask],
                                legs[mask], {str(int(leg)): w})
            year_losses = [pinball(y[mask & (years == year)],
                                   candidate[years[mask] == year])
                           for year in np.unique(years)]
            # Two separate years are given equal importance so a single
            # extreme river event / an overrepresented issue cannot dominate.
            # Penalize unstable, excessive physical-analogue reliance.
            objective = float(np.mean(year_losses) + .15 * w * w +
                              .025 * np.std(year_losses))
            scored.append((objective, w, year_losses))
        objective, winner, losses = min(scored)
        # Demand real improvement rather than fitting selection noise.
        # A learned 0.0 weight returns to the previously submitted model.
        if baseline_mean - np.mean(losses) < .08:
            winner = 0.0
        weights[str(int(leg))] = round(float(winner), 4)
        audit[str(int(leg))] = {
            'baseline_loss_by_year': orig_by_year,
            'baseline_mean': baseline_mean,
            'weight': float(winner),
            'improvement_raw_cm': float(baseline_mean - np.mean(losses)),
            'selected_objective': float(objective),
        }
    return weights, audit


def learn_calibration(oof, weights):
    base = get_matrix(oof, 'baseline')
    physical = get_matrix(oof, 'trajectory')
    cover = oof.physical_available.to_numpy(dtype=bool)
    legs = oof.leg.to_numpy()
    y = oof.dmin_cm.to_numpy(dtype=np.float64)
    raw = combine(base, physical, cover, legs, weights)
    years = oof.year.to_numpy()
    # Choose global vs regime-and-leg calibration via cross-year evaluation,
    # never in-sample. Results are robust if a year is drought-heavy.
    choices = []
    for grouped in [False, True]:
        for strength in [0.0, 0.5, 0.75, 1.0]:
            year_losses = []
            for year in sorted(np.unique(years)):
                tr = years != year
                va = ~tr
                if tr.sum() == 0:
                    continue
                cal = fit_calibration(y[tr], raw[tr], oof.loc[tr].reset_index(drop=True),
                                      oof.current_min.to_numpy()[tr],
                                      strength=float(strength), grouped=grouped)
                pred = apply_calibration(raw[va],
                                         oof.loc[va].reset_index(drop=True),
                                         oof.current_min.to_numpy()[va], cal)
                year_losses.append(float(pinball(y[va], pred)))
            if year_losses:
                objective = float(np.mean(year_losses) + (.04 if grouped else 0))
                choices.append((objective, grouped, strength, year_losses))
    best_score, best_grouped, best_strength, best_year_losses = min(choices)
    cal = fit_calibration(y, raw, oof,
                          oof.current_min.to_numpy(),
                          strength=best_strength, grouped=best_grouped)
    report = {
        'raw_baseline_loss': float(pinball(y, base)),
        'raw_ensemble_loss': float(pinball(y, raw)),
        'raw_baseline_by_year': {str(k): float(pinball(y[years == k], base[years == k]))
                                 for k in sorted(np.unique(years))},
        'raw_blend_by_year': {str(k): float(pinball(y[years == k], raw[years == k]))
                              for k in sorted(np.unique(years))},
        'calibration_cross_year_loss': best_year_losses,
        'calibration_cross_year_objective': best_score,
        'calibration_strength': best_strength,
        'calibration_grouped': best_grouped,
        'in_sample_calibrated_loss_NOT_an_unbiased_estimate': float(pinball(
            y, apply_calibration(raw, oof, oof.current_min.to_numpy(), cal))),
    }
    return cal, report


def validate_holdout_2021(rows, X, config, data_dir, out, weights, cal):
    """Optional genuinely later temporal holdout; adds a third slow training fold.

    This is a diagnostic only. It never auto-tunes the weights on 2021.
    """
    cv = evaluate_cv_components(rows, X, config, data_dir, out,
                                 years=(2021,), use_trajectory=True)
    pred = combine(get_matrix(cv, 'baseline'), get_matrix(cv, 'trajectory'),
                   cv.physical_available.to_numpy(dtype=bool),
                   cv.leg.to_numpy(), weights)
    pred_cal = apply_calibration(pred, cv, cv.current_min.to_numpy(), cal)
    score = {
        'baseline_raw': float(pinball(cv.dmin_cm, get_matrix(cv, 'baseline'))),
        'new_raw': float(pinball(cv.dmin_cm, pred)),
        'new_calibrated': float(pinball(cv.dmin_cm, pred_cal)),
        'held_out_year': 2021,
        'n_scenarios': int(cv.scenario_id.nunique()),
        'tail_coverage': {qcol: float((cv.dmin_cm.to_numpy() < pred_cal[:, j]).mean())
                          for j, qcol in enumerate(QCOLS)},
    }
    (out / 'holdout_2021.json').write_text(json.dumps(score, indent=2))
    print('UNTOUCHED 2021 HOLDOUT:', json.dumps(score), flush=True)
    return score


def validate_final(forecast, data_dir, out):
    env = load_env(data_dir, 'test', out / 'validation_stage')
    errors = env.validate(forecast=forecast)
    if errors:
        raise ValueError('Official forecast validator failed: ' + str(errors[:5]))
    return True


def run(args):
    start = time.monotonic()
    data_dir = str(Path(args.data_dir).resolve())
    out = Path(args.out_dir).resolve()
    out.mkdir(exist_ok=True, parents=True)
    np.random.seed(SEED)
    config = json.loads(Path(__file__).with_name('model_config.json').read_text())
    config['threads'] = int(args.threads)
    cache = out / 'features_cache'
    if all((cache / f).is_file() for f in
           ['rows.parquet','features.parquet','test_rows.parquet','test_features.parquet']):
        rows, X, _, test_rows, Xt, bt = load_dataset(cache)
    else:
        rows, X, _, test_rows, Xt, bt = build_dataset(data_dir, cache_dir=cache,
                                                        stride=config['stride'])
    oof = evaluate_cv_components(rows, X, config, data_dir, out,
                                 years=(2018, 2020),
                                 use_trajectory=not args.no_trajectory)
    weights, weight_report = learn_weights(oof)
    if args.no_trajectory:
        weights = {'1':0., '2':0., '3':0.}
    calibration, calibration_report = learn_calibration(oof, weights)
    baseline_weights = {'1':0., '2':0., '3':0.}
    baseline_cal, baseline_report = learn_calibration(oof, baseline_weights)
    if (calibration_report['calibration_cross_year_objective'] >
            baseline_report['calibration_cross_year_objective'] - .075):
        print('Physical ensemble rejected: it failed to beat the original '
              'after cross-year calibration by at least 0.075 cm.', flush=True)
        weights, calibration, calibration_report = (
            baseline_weights, baseline_cal, baseline_report)
    (out / 'blend_report.json').write_text(json.dumps({
        'weights': weights, 'leg_diagnostics': weight_report,
        'calibration': calibration_report,
        'baseline_calibration_cross_year_loss': baseline_report[
            'calibration_cross_year_objective'],
    }, indent=2))
    (out / 'calibration.json').write_text(json.dumps(calibration, indent=2))
    print('CHRONOLOGICAL CV AND SELECTED BLEND:',
          json.dumps({'weights':weights, 'calibration':calibration_report}, indent=2),
          flush=True)
    if args.holdout_2021:
        validate_holdout_2021(rows, X, config, data_dir, out, weights, calibration)

    print('TRAINING FINAL BASELINE / CHOSEN EXTENSION', flush=True)
    baseline = raw_predict_v2(rows, X, test_rows, Xt, config,
                              data_dir=data_dir, cache_dir=out/'augmentation_cache')
    physical = baseline.copy()
    available = np.zeros(len(test_rows), dtype=bool)
    if not args.no_trajectory and max(weights.values()) > 0:
        physical, available = predict_trajectories(rows, X, test_rows, Xt,
                                                   data_dir=data_dir,
                                                   cutoff='2022-01-01')
    raw = combine(baseline, physical, available, test_rows.leg.to_numpy(), weights)
    pred = apply_calibration(raw, test_rows, bt, calibration)
    forecast = to_forecast(test_rows, pred)
    forecast.to_csv(out / 'forecast.csv', index=False)
    sample = pd.read_csv(Files(data_dir).get('', 'sample_submission.csv'))
    submission = to_submission(forecast, sample)
    submission.to_csv(out / 'submission_c1.csv', index=False)
    validate_final(forecast, data_dir, out)

    meta = {
        'seed': SEED,
        'runtime_seconds': time.monotonic()-start,
        'training_end': '2021-12-31',
        'network_used': False,
        'models': ['historical analogue', 'quantile LightGBM'] +
                  ([] if args.no_trajectory else ['conditional trajectory simulation']),
        'weights': weights,
        'chronological_validation': calibration_report,
        'n_submission_rows': len(submission),
        'official_validation': 'passed',
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ['numpy','pandas','lightgbm','scikit-learn','scipy','threadpoolctl','pyarrow']},
    }
    (out / 'run_metadata.json').write_text(json.dumps(meta, indent=2))
    print('VALID SUBMISSION:', out/'submission_c1.csv',
          'rows=',len(submission),'time_min=',round(meta['runtime_seconds']/60, 2),
          flush=True)
    return submission


if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS']='1'
    os.environ['MKL_NUM_THREADS']='1'
    os.environ['PYTHONHASHSEED']=str(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Path to supplied RIVERLOAD competition riverload-data')
    parser.add_argument('out_dir', help='Where to write submission_c1.csv')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--no-trajectory', action='store_true', help='Regression baseline')
    parser.add_argument('--holdout-2021', action='store_true', help='Extra temporal validation fold')
    run(parser.parse_args())


'''

# ======= cargo_model.py =======
SOURCES['cargo_model.py'] = r'''
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from river_model import KEYS, QCOLS, QUANTILES
from organizer import load_env

def select_greedy(mass, rank, profit, caps, exponent=0.85, nslots=188):
    order = np.argsort(-(profit / mass ** exponent), kind='stable')
    selected, load = ([], np.zeros(len(caps)))
    for i in order:
        if len(selected) >= nslots:
            break
        if profit[i] > 0 and np.all(load[:rank[i] + 1] + mass[i] <= caps[:rank[i] + 1] + 1e-08):
            selected.append(int(i))
            load[:rank[i] + 1] += mass[i]
    return np.array(selected, dtype=int)

def select_milp(mass, rank, profit, caps, nslots=188):
    matrix = np.vstack([np.ones(len(mass))] + [mass * (rank >= leg) for leg in range(len(caps))])
    positive = (profit > 0).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = milp(-profit, integrality=np.ones(len(mass)), bounds=Bounds(0, positive), constraints=LinearConstraint(matrix, np.zeros(len(caps) + 1), np.r_[nslots, caps]), options={'mip_rel_gap': 0.003, 'node_limit': 1200, 'threads': 1, 'random_seed': 2026, 'parallel': False})
    if result.x is None:
        return select_greedy(mass, rank, profit, caps, nslots=nslots)
    return np.flatnonzero(result.x > 0.5)

def stow(selected, mass, rank, nports, ship):
    stack_boxes, stack_weights, stack_ports = ([], [], [])
    for port in reversed(range(nports)):
        boxes = selected[rank[selected] == port]
        ordered = sorted(boxes, key=lambda i: (-mass[i], int(i)))
        stacks = [[] for _ in range(int(np.ceil(len(boxes) / 4)))]
        weights = [0.0] * len(stacks)
        for i in ordered:
            candidates = [j for j, s in enumerate(stacks) if len(s) < 4 and weights[j] + mass[i] <= ship['stack_limit_t'] + 1e-09]
            if not candidates:
                stacks.append([])
                weights.append(0.0)
                candidates = [len(stacks) - 1]
            j = min(candidates, key=lambda j: (len(stacks[j]), weights[j], j))
            stacks[j].append(int(i))
            weights[j] += mass[i]
        for j in sorted(range(len(stacks)), key=lambda j: (-weights[j], j)):
            if stacks[j]:
                stack_boxes.append(stacks[j])
                stack_weights.append(weights[j])
                stack_ports.append(port)
    if len(stack_boxes) > 48:
        return None
    count = len(stack_boxes)
    x = (np.arange(48) // 4 - 5.5) * ship['bay_pitch_m']
    moment = np.zeros(nports)
    free, assigned = (list(range(48)), [])
    for weight, port in zip(stack_weights, stack_ports):
        r = min(free, key=lambda r: (np.max(np.abs(moment[:port + 1] + weight * x[r])), np.sum(np.abs(moment[:port + 1] + weight * x[r])), abs(x[r]), r))
        assigned.append(r)
        moment[:port + 1] += weight * x[r]
        free.remove(r)
    weights = np.r_[stack_weights, np.zeros(48 - count)]
    ports = np.r_[stack_ports, np.full(48 - count, -1)].astype(int)
    placement = np.array(assigned + free, dtype=int)
    leg_weights = np.array([weights * (ports >= leg) for leg in range(nports)])
    for _ in range(16):
        current = np.sum(np.abs(moment)) + np.max(np.abs(moment))
        best = None
        for a in range(count):
            for b in range(a + 1, 48):
                delta = (leg_weights[:, a] - leg_weights[:, b]) * (x[placement[b]] - x[placement[a]])
                trial = moment + delta
                score = np.sum(np.abs(trial)) + np.max(np.abs(trial))
                if score < current - 1e-07:
                    current, best = (score, (a, b, trial))
        if best is None:
            break
        a, b, moment = best
        placement[a], placement[b] = (placement[b], placement[a])
    output = []
    for j, boxes in enumerate(stack_boxes):
        for tier, i in enumerate(boxes, 1):
            output.append((i, int(placement[j] + 1), tier))
    trims = moment / ship['mctc_tm_per_cm']
    loads = np.array([mass[selected[rank[selected] >= leg]].sum() for leg in range(nports)])
    draughts = ship['light_draught_cm'] + loads / ship['tpc_t_per_cm'] + np.abs(trims) / 2
    return (output, draughts, trims)

def depth_at_risk(qvalues, risk):
    return np.array([np.interp(risk, QUANTILES, row) for row in qvalues])

def plan_cargo(env, forecast, risk=0.1, buffer_cm=5.0, finalists=2, use_milp=True, planner='classic', **settings):
    if planner == 'robust':
        from cargo_robust import plan_cargo_robust
        return plan_cargo_robust(env, forecast, risk=risk, buffer_cm=buffer_cm, finalists=finalists, **settings)
    if planner != 'classic' or settings:
        raise ValueError(f'Unknown cargo planner or settings: {planner}, {settings}')
    from riverload.constants import SHIP, COST
    from riverload import stowage
    f = forecast.set_index(['scenario_id', 'route_id', 'departure_slot', 'leg']).sort_index()
    decisions, plans = ([], [])
    for num, sc in enumerate(env.scenarios.itertuples(), 1):
        sid, ports = (sc.scenario_id, sc.ports.split(';'))
        man = env.manifest(sid).reset_index(drop=True)
        mass, value = (man.mass_t.to_numpy(), man.value_eur.to_numpy())
        ranks = man.port.map({p: i for i, p in enumerate(ports)}).to_numpy()
        deadline = pd.to_datetime(man.deadline).to_numpy().astype('datetime64[ns]').astype(np.int64)
        late_rate = man.late_penalty_eur_h.to_numpy()
        candidates = []
        for route in sc.routes.split(';'):
            lc = env.leg_constants(route)
            base_it = env.itinerary(sid, route, 12)
            base_arrival = dict(zip(base_it[base_it.kind == 'PORT'].port, base_it[base_it.kind == 'PORT'].t_enter))
            voyage_h = (base_arrival[ports[-1]] - sc.t0).total_seconds() / 3600
            for slot in range(25):
                depth = depth_at_risk(f.loc[(sid, route, slot), QCOLS].to_numpy(), risk) - buffer_cm
                depth = np.minimum(depth, [lc[l + 1] for l in range(len(ports))])
                usable = np.minimum(depth - SHIP['safety_margin_cm'], SHIP['max_draught_cm']) - SHIP['light_draught_cm']
                caps = np.maximum(0, (usable - 2.0) * SHIP['tpc_t_per_cm'])
                arrivals = np.array([(base_arrival[p] + pd.Timedelta(hours=6 * (slot - 12))).value for p in ports])
                late_h = np.maximum(0, (arrivals[ranks] - deadline) / 3600000000000.0)
                profit = value - late_rate * late_h
                pick = select_greedy(mass, ranks, profit, caps, nslots=192 if len(ports) == 1 else 188)
                cost = COST['c_t_eur_per_h'] * voyage_h + COST['c_b_eur_per_h'] * 6 * abs(slot - 12)
                objective = cost - profit[pick].sum()
                candidates.append((objective, route, slot, depth, caps, profit, cost, pick))
        candidates.sort(key=lambda c: (c[0], c[1], c[2]))
        best = None
        for candidate in candidates[:finalists]:
            _, route, slot, depth, caps, profit, cost, greedy = candidate
            pick = select_milp(mass, ranks, profit, caps, nslots=192 if len(ports) == 1 else 188) if use_milp else greedy
            for _ in range(len(pick) + 1):
                result = stow(pick, mass, ranks, len(ports), SHIP)
                if result is not None:
                    positions, draughts, trims = result
                    if np.all(draughts <= SHIP['max_draught_cm'] + 1e-09) and np.all(np.abs(trims) <= SHIP['trim_limit_cm'] + 1e-09) and np.all(draughts + SHIP['safety_margin_cm'] <= depth + 1e-09):
                        break
                if not len(pick):
                    break
                drop = min(pick, key=lambda i: (profit[i] / mass[i], int(i)))
                pick = pick[pick != drop]
            positions, draughts, trims = result
            objective = cost - profit[pick].sum()
            if best is None or objective < best[0]:
                best = (objective, route, slot, positions)
        _, route, slot, positions = best
        plan = pd.DataFrame([(man.container_id.iloc[i], r, tier) for i, r, tier in positions], columns=['container_id', 'slot_row', 'slot_tier'])
        plan = plan.astype({'slot_row': int, 'slot_tier': int})
        errors, state = stowage.check(plan, man, ports)
        if errors:
            raise ValueError(f'Invalid load plan for {sid}: {errors}')
        decisions.append({'scenario_id': sid, 'route_id': route, 'departure_slot': slot})
        plans.append(plan.assign(scenario_id=sid))
        if num % 20 == 0 or num == len(env.scenarios):
            print(f'C2: planned {num}/{len(env.scenarios)} scenarios', flush=True)
    decision = pd.DataFrame(decisions)
    loadplan = pd.concat(plans, ignore_index=True)[['scenario_id', 'container_id', 'slot_row', 'slot_tier']]
    errors = env.validate(decision=decision, loadplan=loadplan)
    if errors:
        raise ValueError(f'Official C2 validator: {errors[:5]}')
    return (decision, loadplan)

def c2_submission(decision, loadplan, manifests, route_column='route'):
    if route_column not in {'route', 'route_id'}:
        raise ValueError('C2 route column must be route or route_id')
    out = manifests[['scenario_id', 'container_id']].merge(decision, on='scenario_id', validate='many_to_one')
    out = out.merge(loadplan, on=['scenario_id', 'container_id'], how='left', validate='one_to_one')
    out[['slot_row', 'slot_tier']] = out[['slot_row', 'slot_tier']].fillna(0).astype(int)
    out['id'] = out.scenario_id + '_' + out.container_id
    out = out[['id', 'route_id', 'departure_slot', 'slot_row', 'slot_tier']]
    return out.rename(columns={'route_id': route_column})

def main():
    p = argparse.ArgumentParser()
    p.add_argument('data_dir')
    p.add_argument('forecast')
    p.add_argument('--risk', type=float, default=0.1)
    p.add_argument('--buffer', type=float, default=5.0)
    p.add_argument('--out', default='experiments/cargo')
    p.add_argument('--no-milp', action='store_true')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    forecast = pd.read_parquet(a.forecast) if a.forecast.endswith('.parquet') else pd.read_csv(a.forecast)
    split = 'train' if forecast.scenario_id.str.startswith('TR-').all() else 'test'
    env = load_env(a.data_dir, split, out / 'organizer_data')
    env = env.subset(forecast.scenario_id.unique())
    start = time.monotonic()
    decision, plan = plan_cargo(env, forecast, a.risk, a.buffer, use_milp=not a.no_milp)
    decision.to_csv(out / 'decision.csv', index=False)
    plan.to_csv(out / 'loadplan.csv', index=False)
    c2_submission(decision, plan, env.manifests[env.manifests.scenario_id.isin(env.scenarios.scenario_id)]).to_csv(out / 'submission_c2.csv', index=False)
    if split == 'train':
        score = env.evaluate(decision=decision, loadplan=plan, forecast=forecast[KEYS + QCOLS])
        score['c2_per_scenario'].to_csv(out / 'per_scenario.csv', index=False)
        summary = {k: v for k, v in score.items() if k != 'c2_per_scenario'}
        (out / 'score.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    print(f'C2 completed in {time.monotonic() - start:.1f}s', flush=True)
if __name__ == '__main__':
    main()

'''

# ======= cargo_v3.py =======
SOURCES['cargo_v3.py'] = r'''
"""RIVERLOAD C2: risk-aware route, departure and multi-port cargo optimization.
All features and probabilities come exclusively from competition data.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from cargo_model import stow
from river_model import QCOLS, KEYS

PROBS = np.array([0.05, 0.10, 0.25, 0.50], dtype=float)


def route_number(route):
    """Private organizer IDs A1/A2/B1/B2 -> Kaggle's choice 1 or 2."""
    s = str(route).strip()
    if not s or s[-1] not in '12':
        raise ValueError(f'Invalid internal route identifier {route!r}')
    return int(s[-1])


def make_submission(decisions, loadplan, manifests, sample):
    """Produces the exact actual C2 sample schema; uses 1/2, not A1/A2/B1/B2."""
    if not {'id','departure_slot','slot_row','slot_tier'}.issubset(sample.columns):
        raise ValueError('Expected a C2 sample_submission.csv, not a C1 sample file')
    route_cols = [c for c in ('route', 'route_id') if c in sample.columns]
    if len(route_cols) != 1:
        raise ValueError(f'Unexpected C2 sample columns: {sample.columns.tolist()}')
    route_col = route_cols[0]
    if decisions.scenario_id.duplicated().any() or loadplan[['scenario_id','container_id']].duplicated().any():
        raise ValueError('Duplicate scenario decisions or loadplan boxes')
    required = manifests[['scenario_id','container_id']].drop_duplicates()
    frame = required.merge(decisions, on='scenario_id', how='left', validate='many_to_one')
    frame = frame.merge(loadplan, on=['scenario_id','container_id'], how='left', validate='one_to_one')
    frame[['slot_row','slot_tier']] = frame[['slot_row','slot_tier']].fillna(0).astype(np.int64)
    frame['id'] = frame.scenario_id.astype(str) + '_' + frame.container_id.astype(str)
    frame[route_col] = frame.route_id.map(route_number).astype(np.int64)
    frame['departure_slot'] = frame.departure_slot.astype(np.int64)
    submission = sample[['id']].merge(frame[['id',route_col,'departure_slot','slot_row','slot_tier']],
                                     on='id',how='left',validate='one_to_one')
    submission = submission[sample.columns.tolist()].copy()
    if len(submission) != len(sample) or submission.isna().any().any():
        raise ValueError('C2 Kaggle IDs failed exact reconciliation with sample')
    if not submission.id.is_unique or not set(submission.id)==set(sample.id):
        raise ValueError('Duplicate or missing C2 IDs')
    if not submission[route_col].isin([1,2]).all():
        raise ValueError('Kaggle routes must be 1 or 2, not A1/B1 etc')
    if not submission.departure_slot.between(0,24).all():
        raise ValueError('Invalid departure slot')
    if not submission.slot_row.between(0,48).all() or not submission.slot_tier.between(0,4).all():
        raise ValueError('Invalid stowage coordinates')
    if not ((submission.slot_row==0) == (submission.slot_tier==0)).all():
        raise ValueError('The 0/0 pair must be used for unloaded containers')
    sid = submission.id.str.rsplit('_',n=1).str[0]
    if (submission.assign(sid=sid).groupby('sid')[route_col].nunique()!=1).any():
        raise ValueError('Inconsistent route within scenario')
    if (submission.assign(sid=sid).groupby('sid').departure_slot.nunique()!=1).any():
        raise ValueError('Inconsistent departure slot within scenario')
    return submission


def cdf_at(q, threshold):
    """Smooth interpolated marginal failure probability estimated from four C1 quantiles."""
    v = np.asarray(q,dtype=float)
    if not np.all(np.isfinite(v)):
        raise ValueError('Nonfinite C1 quantiles')
    v = np.maximum.accumulate(v)
    v = v + np.arange(4)*1e-6
    x = float(threshold)
    if x <= v[0]:
        scale = max(float(v[1]-v[0]) / np.log(2), 7.0)
        return float(max(0.00003, .05*np.exp(np.clip((x-v[0])/scale,-16,0))))
    if x >= v[-1]:
        scale = max(float(v[3]-v[2]), 18.)
        return float(0.5 + 0.49*(1.0-np.exp(np.clip(-(x-v[3])/scale,-16,0))))
    return float(np.interp(x, v, PROBS))


def joint_probability(p, correlation=0.75):
    """Conservative correlation mixture: comonotone bound vs independent hazards."""
    if not len(p):
        return 0.
    p = np.clip(np.asarray(p,dtype=float),0.,1.)
    maxp = p.max()
    indep = 1-np.prod(1-p)
    return float(np.clip(correlation*maxp+(1-correlation)*indep,0,1))


def expected_cost(profit, ranks, selected, actual_draught, q, fixed, cost, best_value_proxy,
                  penalty_weight=1.0, error_scale=1.08, corr=0.75):
    """E[cost] under marginal CDFs, failure can stop cargo at intermediate ports."""
    from riverload.constants import SHIP
    nlegs = len(q)
    p=[]
    for leg in range(nlegs):
        threshold = float(actual_draught[leg]+SHIP['safety_margin_cm'])
        if threshold >= float(fixed[leg])-1e-6:
            # Equality is permitted. The fixed pool is deterministic, not a random failure.
            if threshold > float(fixed[leg])+1e-7:
                p.append(1.0)
            else:
                p.append(cdf_at(q[leg], threshold))
        else:
            p.append(cdf_at(q[leg], threshold))
    p=np.minimum(0.999,np.maximum(0,np.asarray(p)*error_scale))
    if np.any(actual_draught + SHIP['safety_margin_cm'] > fixed+1e-7):
        return float('inf')
    total_p=joint_probability(p,corr)
    delivered=0.
    for leg in range(nlegs):
        success=1-joint_probability(p[:leg+1],corr)
        ix=selected[ranks[selected]==leg]
        delivered+=float(np.sum(profit[ix]))*success
    return float(cost-delivered+penalty_weight*best_value_proxy*total_p)


def greedy_select(mass, rank, profit, caps, exponent=0.75, nslots=188):
    """Constant-time capacity checking per leg for fast first-stage search."""
    util = np.asarray(profit) / np.maximum(mass,0.2)**float(exponent)
    order = np.argsort(-util,kind='stable')
    load0=load1=load2=0.0
    cap0 = float(caps[0]); cap1=float(caps[1]) if len(caps)>1 else float('inf')
    cap2 = float(caps[2]) if len(caps)>2 else float('inf')
    chosen=[]
    for index in order:
        i=int(index); m=float(mass[i]); port=int(rank[i])
        if len(chosen)>=nslots: break
        if profit[i]<=0: break
        if load0+m > cap0+1e-8: continue
        if port>=1 and load1+m > cap1+1e-8: continue
        if port>=2 and load2+m > cap2+1e-8: continue
        chosen.append(i); load0+=m
        if port>=1:load1+=m
        if port>=2:load2+=m
    return np.asarray(chosen,dtype=np.int32)


def mixed_integer_select(mass,rank,profit,caps,ship,seconds=0.65):
    """Deterministic node-limited MILP with per-port stack/count constraints."""
    n=len(mass); nports=len(caps)
    A=[];ub=[]
    for leg in range(nports):
        A.append(np.r_[mass*(rank>=leg),np.zeros(nports)])
        ub.append(float(caps[leg]))
    A.append(np.r_[np.ones(n),np.zeros(nports)])
    ub.append(192.0)
    A.append(np.r_[np.zeros(n),np.ones(nports)])
    ub.append(48.0)
    for port in range(nports):
        sc=np.zeros(nports);sc[port]=-4
        A.append(np.r_[(rank==port).astype(float),sc]);ub.append(0)
        sw=np.zeros(nports);sw[port]=-float(ship['stack_limit_t'])
        A.append(np.r_[mass*(rank==port),sw]);ub.append(0)
    arr=np.asarray(A,dtype=float); cap=np.asarray(ub,dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result=milp(c=np.r_[-profit,np.zeros(nports)],
                    integrality=np.ones(n+nports,dtype=np.int32),
                    bounds=Bounds(np.zeros(n+nports),np.r_[(profit>0).astype(float),np.full(nports,48)]),
                    constraints=LinearConstraint(arr,-np.inf,cap),
                    options={'node_limit':max(30,int(300*seconds)),'mip_rel_gap':0.008,'presolve':True})
    if result.x is not None:
        x=np.r_[np.round(result.x[:n]),np.ceil(result.x[n:])]
        if np.all(arr@x<=cap+1e-5):
            return np.flatnonzero(x[:n]>0.5).astype(np.int32)
    possibilities=[greedy_select(mass,rank,profit,caps,p,188) for p in [0.0,0.6,1.0]]
    return max(possibilities,key=lambda z:np.sum(profit[z]))


def smooth_forecast(forecast):
    """Reduce exploitation of isolated forecast wiggles when choosing departure slot."""
    out=forecast.copy()
    for (_,_,_), idx in out.groupby(['scenario_id','route_id','leg'],sort=False).indices.items():
        sub=out.iloc[idx]
        order=np.argsort(sub.departure_slot.to_numpy())
        ix=np.asarray(idx)[order]
        val=out.iloc[ix][QCOLS].to_numpy(float)
        if len(val)>2:
            sm=val.copy()
            sm[1:-1]=0.20*val[:-2]+.60*val[1:-1]+0.20*val[2:]
            out.iloc[ix,out.columns.get_indexer(QCOLS)]=np.sort(sm,axis=1)
    return out


@dataclass
class Candidate:
    route:str
    slot:int
    risk:float
    forecast:np.ndarray
    depth:np.ndarray
    fixed:np.ndarray
    caps:np.ndarray
    profit:np.ndarray
    cost:float
    selected:np.ndarray
    score:float


def stow_and_repair(pick,mass,rank,profit,depth,ship,guard=1.5):
    """Port-homogeneous stowage, local trim optimization, and feasibility repair."""
    pick=np.asarray(pick,dtype=np.int32)
    for k in range(len(pick)+1):
        sol=stow(pick,mass,rank,len(depth),ship)
        if sol is not None:
            positions,draught,trim=sol
            if (np.all(np.isfinite(draught))
                    and np.all(np.abs(trim)<=ship['trim_limit_cm']+1e-7)
                    and np.all(draught<=ship['max_draught_cm']+1e-7)
                    and np.all(draught+ship['safety_margin_cm']<=depth-guard+1e-7)):
                return pick,positions,draught,trim
        if not len(pick): break
        # Drop least profitable mass, prefer drops affecting the tightest leg.
        if sol is not None:
            _,d,_=sol
            tight=int(np.argmax(d+ship['safety_margin_cm']-depth))
            candidates=pick[rank[pick]>=tight]
            if not len(candidates):candidates=pick
        else:
            candidates=pick
        drop=min(candidates,key=lambda i:(profit[i]/max(mass[i],0.2),int(i)))
        pick=pick[pick!=drop]
    raise RuntimeError('Unable to produce a legal stowage even after unloading everything')


def _fixed_depth(env,route,nports):
    lc=env.leg_constants(route)
    return np.array([float(lc[l+1]) for l in range(nports)],dtype=float)


def _itinerary_times(env,sid,route,slot,ports):
    times=env.itinerary(sid,route,int(slot))
    pp=times.loc[times.kind=='PORT',['port','t_enter']].drop_duplicates('port',keep='last')
    arrivals=dict(zip(pp.port, pd.to_datetime(pp.t_enter)))
    if not all(p in arrivals for p in ports):
        raise ValueError(f'{sid}, {route}: missing destination ports in itinerary: {arrivals}')
    return arrivals


def plan_cargo_v3(env,forecast,
                  sample_for_reporting=None,
                  buffer_cm=2.5,
                  trim_reserve_cm=1.8,
                  error_scale=1.08,
                  penalty_weight=1.0,
                  corr=0.75,
                  finalists=6,
                  milp_seconds=0.60,
                  greedy_only=False,
                  smooth=True,
                  verbose=True,
                  **_unused):
    from riverload.constants import SHIP,COST
    from riverload import stowage
    forecast=smooth_forecast(forecast) if smooth else forecast
    fx=forecast.set_index(KEYS)[QCOLS].sort_index()
    decisions=[];plans=[];debug=[]
    tt=time.monotonic()
    for seq,scenario in enumerate(env.scenarios.itertuples(),1):
        sid=scenario.scenario_id
        ports=scenario.ports.split(';')
        m=env.manifest(sid).reset_index(drop=True)
        mass=m.mass_t.to_numpy(dtype=float)
        vals=m.value_eur.to_numpy(dtype=float)
        rank=m.port.map({p:i for i,p in enumerate(ports)}).to_numpy(dtype=int)
        deadlines=pd.to_datetime(m.deadline).to_numpy().astype('datetime64[ns]').astype(np.int64)
        rates=m.late_penalty_eur_h.to_numpy(dtype=float)
        if np.any(~np.isfinite(rank)):
            raise ValueError(f'{sid}: unknown destination port')
        unlimited=np.full(len(ports),(SHIP['max_draught_cm']-SHIP['light_draught_cm']-3)*SHIP['tpc_t_per_cm'])
        max_pick=max([greedy_select(mass,rank,vals,unlimited,x,nslots=192) for x in [0,.5,1]],
                     key=lambda z:np.sum(vals[z]))
        V_physical=float(np.sum(vals[max_pick]))
        # A perfect clairvoyant operator may exploit a favorable departure slot,
        # but cannot load at the maximum draught when the river is shallow.
        # Estimate available cargo value at a median river state, then shrink
        # toward the physical maximum to avoid systematically underpricing risk.
        V_median=0.0
        for route in str(scenario.routes).split(';'):
            limits=_fixed_depth(env,route,len(ports))
            for future_slot in range(25):
                forecast_median=fx.loc[(sid,route,future_slot)][QCOLS[-1]].to_numpy(dtype=float)
                d=np.minimum(forecast_median,limits)
                capacity=np.maximum(0.0,(np.minimum(d-SHIP['safety_margin_cm'],SHIP['max_draught_cm'])
                                           -SHIP['light_draught_cm']-1.5)*SHIP['tpc_t_per_cm'])
                estimate=greedy_select(mass,rank,vals,capacity,0.6,nslots=192)
                V_median=max(V_median,float(np.sum(vals[estimate])))
        V=max(1.0,0.35*V_physical + 0.65*V_median)
        shortlist=[]
        for route in str(scenario.routes).split(';'):
            fixed=_fixed_depth(env,route,len(ports))
            for slot in range(25):
                arrival=_itinerary_times(env,sid,route,slot,ports)
                departure=pd.Timestamp(scenario.issue_time)+pd.Timedelta(hours=48+6*slot)
                voyage_h=(arrival[ports[-1]]-departure).total_seconds()/3600
                cost=COST['c_t_eur_per_h']*voyage_h + COST['c_b_eur_per_h']*6*abs(slot-12)
                late=np.maximum(0,(np.array([arrival[p].value for p in ports])[rank]-deadlines)/3.6e12)
                profit=vals-rates*late
                q=fx.loc[(sid,route,slot)].to_numpy(dtype=float)
                if q.shape!=(len(ports),4):
                    raise ValueError(f'Unexpected C1 predictions for {sid}/{route}/{slot}: {q.shape}')
                if not np.all(np.isfinite(q)):
                    raise ValueError(f'Missing forecast for {sid}/{route}/{slot}')
                # Candidate likelihood thresholds: aggressive and very cautious.
                risks=[.015,.035,.055,.085,.13,.20]
                for risk in risks:
                    quant=np.array([np.interp(risk,PROBS,v) if risk>=.05
                                    else v[0]-(.05-risk)/.05*max(v[1]-v[0],5.0)*.65
                                    for v in q])
                    depth=np.minimum(quant-buffer_cm,fixed)
                    usable=np.minimum(depth-SHIP['safety_margin_cm'],SHIP['max_draught_cm'])
                    caps=np.maximum(0,(usable-SHIP['light_draught_cm']-trim_reserve_cm)*SHIP['tpc_t_per_cm'])
                    # Baseline demands every downstream leg to remain physically feasible.
                    if np.any(depth<SHIP['light_draught_cm']+SHIP['safety_margin_cm']):
                        continue
                    approx=[greedy_select(mass,rank,profit,caps,x,nslots=192-2*(len(ports)-1))
                            for x in (.4,1.)]
                    pick=max(approx,key=lambda z:float(np.sum(profit[z])))
                    loads=np.array([mass[pick[rank[pick]>=leg]].sum() for leg in range(len(ports))])
                    draught=SHIP['light_draught_cm']+loads/SHIP['tpc_t_per_cm']+trim_reserve_cm/2
                    score=expected_cost(profit,rank,pick,draught,q,fixed,cost,V,
                                        penalty_weight=penalty_weight,error_scale=error_scale,corr=corr)
                    shortlist.append(Candidate(route,slot,risk,q,depth,fixed,caps,profit,float(cost),pick,float(score)))
        if not shortlist:
            raise RuntimeError(f'No feasible departure/route for {sid}; even empty ship needs 130.6 cm')
        shortlist.sort(key=lambda c:(c.score,c.cost,c.route,c.slot,c.risk))
        best_candidates=[];seen=set();seen_routes=set()
        for c in shortlist:
            signature=(c.route,c.slot,tuple(np.round(c.caps,1)))
            if signature in seen:continue
            seen.add(signature)
            best_candidates.append(c)
            seen_routes.add(c.route)
            if len(best_candidates)>=finalists:break
        # Keep best backup for each route in the shortlist.
        for route in str(scenario.routes).split(';'):
            if route not in seen_routes:
                b=next((c for c in shortlist if c.route==route),None)
                if b is not None:best_candidates.append(b)
        best=None
        for c in best_candidates:
            choices=[]
            if not greedy_only:
                choices.append(mixed_integer_select(mass,rank,c.profit,c.caps,SHIP,seconds=milp_seconds))
            choices.append(c.selected)
            for pick in choices:
                try:
                    pick,positions,draught,trim=stow_and_repair(pick,mass,rank,c.profit,c.depth,SHIP)
                except RuntimeError:
                    continue
                score=expected_cost(c.profit,rank,pick,draught,c.forecast,c.fixed,c.cost,V,
                                    penalty_weight=penalty_weight,error_scale=error_scale,corr=corr)
                if best is None or score<best[0]:
                    best=(score,c,pick,positions,draught,trim)
        if best is None:
            raise RuntimeError(f'{sid}: no feasible stowage after solving candidate programs')
        expected,c,pick,positions,draught,trim=best
        plan=pd.DataFrame([(m.container_id.iloc[i],r,t) for i,r,t in positions],
                          columns=['container_id','slot_row','slot_tier']).astype({'slot_row':int,'slot_tier':int})
        errors,_=stowage.check(plan,m,ports)
        if errors:
            raise ValueError(f'{sid}: official stowage check failed: {errors[:5]}')
        decisions.append(dict(scenario_id=sid,route_id=c.route,departure_slot=int(c.slot)))
        plans.append(plan.assign(scenario_id=sid))
        debug.append(dict(scenario_id=sid,route=c.route,slot=c.slot,risk=c.risk,estimated_best_value=V,
                          loaded=len(pick),expected_objective=expected,
                          max_trim=float(np.max(np.abs(trim))),
                          min_depth_headroom=float(np.min(c.depth-draught-SHIP['safety_margin_cm']))))
        if verbose and (seq%10==0 or seq==len(env.scenarios)):
            print(f'C2 optimized {seq}/{len(env.scenarios)} | minutes {((time.monotonic()-tt)/60):.1f}',flush=True)
    decision=pd.DataFrame(decisions)
    plan=pd.concat(plans,ignore_index=True) if plans else pd.DataFrame(columns=['scenario_id','container_id','slot_row','slot_tier'])
    plan=plan[['scenario_id','container_id','slot_row','slot_tier']]
    errors=env.validate(decision=decision,loadplan=plan)
    if errors:
        raise ValueError(f'Official internal C2 validator: {errors[:5]}')
    return decision,plan,pd.DataFrame(debug)

'''

# ======= run.py =======
SOURCES['run.py'] = r'''
"""RIVERLOAD C2, deterministic offline CPU solver.
Usage: python run.py <competition_data_dir> <output_dir> [--threads 4] [--no-trajectory] [--skip-policy-tune]
Writes submission_c2.csv; copies submission.csv; no outside hydrology data.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import warnings
import numpy as np
import pandas as pd
from calibration import apply_calibration, fit_calibration
from river_model import (Files, build_dataset,load_dataset,ordered, QCOLS, KEYS,
                         to_forecast,pinball)
from predict_v2 import raw_predict_v2
from trajectory_analog import predict_trajectories
from organizer import load_env
from cargo_v3 import make_submission,plan_cargo_v3

# Forecast training + CV implementation is shared with the previous, audited C1 baseline.
from forecast_model import (evaluate_cv_components, learn_weights,learn_calibration,
                            combine, get_matrix)

SEED=2026


def chosen_forecast(oof, weights, cal):
    """Cross-fit calibration by year before testing policies; no fold's
    own water labels are used to calibrate predictions for that fold.
    """
    base=get_matrix(oof,'baseline');physical=get_matrix(oof,'trajectory')
    covers=oof.physical_available.to_numpy(dtype=bool)
    raw=combine(base,physical,covers,oof.leg.to_numpy(),weights)
    pred=np.full_like(raw,np.nan,dtype=float)
    year_values=oof.year.to_numpy(dtype=int)
    years=np.unique(year_values)
    if len(years)>1:
        for year in years:
            tr=year_values!=year;va=~tr
            fold_cal=fit_calibration(oof.dmin_cm.to_numpy(dtype=float)[tr],raw[tr],
                                     oof.loc[tr].reset_index(drop=True),base[tr],
                                     strength=float(cal['strength']),
                                     grouped=bool(cal.get('groups',{})))
            pred[va]=apply_calibration(raw[va],oof.loc[va].reset_index(drop=True),
                                       base[va],fold_cal)
    else:
        pred=apply_calibration(raw,oof,base,cal)
    assert np.isfinite(pred).all()
    out=oof[KEYS].copy()
    out[QCOLS]=pred
    return out


def parse_c2_score(evaluation):
    if isinstance(evaluation,(float,int,np.floating)):
        return float(evaluation)
    if isinstance(evaluation,dict):
        for k in ['c2','c2_score','score_c2','C2','c2_mean_score','normalized_cost']:
            v=evaluation.get(k)
            if isinstance(v,(int,float,np.number)) and np.isfinite(v):
                return float(v)
        table=evaluation.get('c2_per_scenario')
        if isinstance(table,pd.DataFrame):
            for name in ['c2_score','score','normalized_cost','scenario_score','c2']:
                if name in table.columns:
                    return float(table[name].mean())
        if isinstance(table,pd.Series) and pd.api.types.is_numeric_dtype(table):
            return float(table.mean())
    raise ValueError(f'Unknown organizer score structure: {str(type(evaluation))} keys={list(evaluation) if isinstance(evaluation,dict) else str(evaluation)[:100]}')


def subset_scenarios(oof,per_year=8):
    """Diverse, deterministic dry / middle / wet holdout scenarios; past data only."""
    rng=np.random.default_rng(SEED)
    rows=[]
    for year,yearframe in oof.groupby('year'):
        for family,group in yearframe.assign(family=yearframe.route_id.str[0]).groupby('family'):
            groups=group.groupby('scenario_id',sort=True).current_min.median().sort_values()
            number=min(per_year//2,len(groups))
            if not number:continue
            positions=np.unique(np.r_[np.linspace(0,len(groups)-1,number, dtype=int)])
            rows.extend(groups.index[positions].tolist())
    return sorted(set(rows))


def policy_tuning(oof,weights,cal,data_dir,out,enabled=True):
    default=dict(buffer_cm=2.75,trim_reserve_cm=1.6,penalty_weight=1.1,
                 error_scale=1.08,corr=0.75)
    if not enabled:
        return default,{'enabled':False,'default':default}
    select=subset_scenarios(oof,per_year=8)
    cv=chosen_forecast(oof,weights,cal)
    cv=cv[cv.scenario_id.isin(select)].reset_index(drop=True)
    # A small, interpretable search; avoids overfitting many hyperparameters
    # on just two historical years.
    candidates=[
        dict(buffer_cm=1.0,trim_reserve_cm=1.3,penalty_weight=0.75,error_scale=1.00,corr=0.8),
        default,
        dict(buffer_cm=4.0,trim_reserve_cm=1.6,penalty_weight=1.30,error_scale=1.10,corr=0.75),
        dict(buffer_cm=6.0,trim_reserve_cm=2.2,penalty_weight=1.5,error_scale=1.15,corr=0.70),
    ]
    env=load_env(data_dir,'train',out/'validation_package')
    if not hasattr(env,'subset'):
        print('Official Env.subset missing; skip policy tuning',flush=True)
        return default,{'enabled':False,'reason':'no Env.subset','default':default}
    env=env.subset(select)
    records=[]
    # We evaluate using organizer's actual historical groundings, not using
    # true depths as inference inputs or inspecting post-2021 river observations.
    for i,policy in enumerate(candidates):
        try:
            t=time.monotonic()
            d,lp,dbg=plan_cargo_v3(env,cv,greedy_only=True,finalists=2,
                                   smooth=True,verbose=False,**policy)
            eval_result=env.evaluate(decision=d,loadplan=lp,forecast=cv[KEYS+QCOLS])
            score=parse_c2_score(eval_result)
            records.append(dict(policy=i,score=score,seconds=time.monotonic()-t,parameters=policy))
            print(f'Honest C2 policy CV {i}: official score={score:.5f} '
                  f'({len(d)} pre-2022 scenarios, {records[-1]["seconds"]:.1f}s)',flush=True)
        except Exception as exc:
            print(f'Policy candidate {i} could not be scored: {type(exc).__name__}: {exc}',flush=True)
            records.append(dict(policy=i,score=None,error=str(exc),parameters=policy))
    ranked=[r for r in records if r['score'] is not None and np.isfinite(r['score'])]
    if not ranked:
        return default,{'enabled':True,'reported':records,'selected':'default due to scorer API mismatch'}
    winner=min(ranked,key=lambda r:r['score']+.004*abs(r['policy']-1))
    report={'enabled':True,'n_validation_scenarios':len(select),'candidates':records,
            'chosen_index':winner['policy'],'chosen_score':winner['score'],
            'WARNING':'C2 policy CV uses the same pre-2022 years as forecast calibration; not an untouched generalization estimate.'}
    return dict(winner['parameters']),report


def run(args):
    start=time.monotonic()
    root=Path(args.data_dir).resolve();out=Path(args.out_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    np.random.seed(SEED)
    sample=pd.read_csv(Files(root).get('','sample_submission.csv'))
    if not {'id','slot_row','slot_tier','departure_slot'}.issubset(sample.columns):
        raise ValueError('Attach the C2 competition; C1 sample_submission.csv is not accepted')
    config=json.loads(Path(__file__).with_name('model_config.json').read_text())
    config['threads']=int(args.threads)
    print('DATA:',root,'| sample rows:',len(sample),'| forecast trajectory:',not args.no_trajectory,flush=True)
    cache=out/'features_cache'
    if all((cache/p).exists() for p in ['rows.parquet','features.parquet','test_rows.parquet','test_features.parquet']):
        rows,X,_,test_rows,Xt,bt=load_dataset(cache)
    else:
        rows,X,_,test_rows,Xt,bt=build_dataset(root,cache_dir=cache,stride=config['stride'])
    # Fit/validate forecasting only from chronologically permitted training history.
    oof=evaluate_cv_components(rows,X,config,str(root),out,
                               years=(2018,2020),use_trajectory=not args.no_trajectory,
                               threads=args.threads)
    weights,weight_report=learn_weights(oof)
    if args.no_trajectory:weights={'1':0.0,'2':0.0,'3':0.0}
    cal,cal_report=learn_calibration(oof,weights)
    basew={'1':0.0,'2':0.0,'3':0.0}
    bascal,basreport=learn_calibration(oof,basew)
    if cal_report['calibration_cross_year_objective'] > basreport['calibration_cross_year_objective']-.075:
        weights,cal,cal_report=basew,bascal,basreport
        print('Trajectory model rejected after cross-year calibration check',flush=True)
    print('C1 CV:',json.dumps({'weights':weights,'cross_year':cal_report['calibration_cross_year_objective']},indent=2),flush=True)
    policy,polreport=policy_tuning(oof,weights,cal,str(root),out,enabled=not args.skip_policy_tune)
    (out/'c2_policy_validation.json').write_text(json.dumps(polreport,indent=2))
    (out/'c1_blend_report.json').write_text(json.dumps({'weights':weights,'weights_report':weight_report,'calibration':cal_report},indent=2))
    (out/'c1_calibration.json').write_text(json.dumps(cal,indent=2))
    print('C2 POLICY SELECTED:',json.dumps(policy),flush=True)
    if args.forecast_only:
        print('Forecast-only debug mode requested. Skipping final cargo plan.',flush=True)
        return
    # Refit one final forecasting model on ALL provided pre-2022 history.
    baseline=raw_predict_v2(rows,X,test_rows,Xt,config,data_dir=str(root),
                            cache_dir=out/'augmentation_cache')
    physical=baseline.copy();valid=np.zeros(len(test_rows),dtype=bool)
    if max(weights.values())>0 and not args.no_trajectory:
        physical,valid=predict_trajectories(rows,X,test_rows,Xt,data_dir=str(root),
                                            cutoff='2022-01-01',verbose=True)
    raw=combine(baseline,physical,valid,test_rows.leg.to_numpy(),weights)
    calibrated=apply_calibration(raw,test_rows,bt,cal)
    forecast=to_forecast(test_rows,calibrated)
    forecast.to_csv(out/'forecast_c1_for_c2.csv',index=False)
    env=load_env(str(root),'test',out/'validator_package')
    errors=env.validate(forecast=forecast)
    if errors:raise ValueError(f'Official C1 forecast validation failed: {errors[:5]}')
    print('Optimizing 94 C2 scenarios with voyage-cost-aware stochastic planner...',flush=True)
    decision,plan,diag=plan_cargo_v3(env,forecast,greedy_only=False,finalists=args.finalists,
                                    milp_seconds=args.milp_seconds,verbose=True,**policy)
    decision.to_csv(out/'decision.csv',index=False)
    plan.to_csv(out/'loadplan.csv',index=False)
    diag.to_csv(out/'cargo_diagnostics.csv',index=False)
    sub=make_submission(decision,plan,env.manifests,sample)
    sub.to_csv(out/'submission_c2.csv',index=False)
    sub.to_csv(Path('/kaggle/working/submission.csv') if Path('/kaggle/working').exists()
               else out/'submission.csv',index=False)
    if args.safe_alternative:
        # One independent safer hedge, in case the 2022-26 private set
        # has unusually frequent severe low-water episodes. It is NOT
        # automatically preferred over the validation-selected submission.
        safer=dict(policy)
        safer['buffer_cm']=policy['buffer_cm']+3.0
        safer['penalty_weight']=policy['penalty_weight']*1.3
        safer['error_scale']=policy['error_scale']*1.05
        print('Preparing optional safer C2 alternative:',safer,flush=True)
        ds,ps,_=plan_cargo_v3(env,forecast,greedy_only=False,
                             finalists=max(3,args.finalists-2),
                             milp_seconds=args.milp_seconds,
                             verbose=True,**safer)
        alternate=make_submission(ds,ps,env.manifests,sample)
        alternate.to_csv(out/'submission_c2_safe.csv',index=False)
        print('Optional second submission:',out/'submission_c2_safe.csv',flush=True)
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','lightgbm','scikit-learn','pyarrow','threadpoolctl']}
    metadata={'seconds':time.monotonic()-start,'seed':SEED,'packages':versions,
              'no_external_data':True,'train_end':'2021-12-31','c2_policy':policy,
              'forecast_weights':weights,'c2_rows':len(sub),'c2_scenarios':int(decision.shape[0]),
              'official_validation':'passed','python':platform.python_version()}
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    (out/'requirements_used.txt').write_text('\n'.join(f'{n}=={v}' for n,v in versions.items())+'\n')
    print('VALID C2 submission:',out/'submission_c2.csv','\nResult sample:',sub.head().to_string(index=False),flush=True)
    print('Total elapsed minutes:',round(metadata['seconds']/60,2),flush=True)
    if metadata['seconds']>3600:
        print('WARNING: exceeds organizer 60 minute CPU runtime limit on THIS machine; use the faster flags.',flush=True)
    return sub


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('data_dir');p.add_argument('out_dir')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--no-trajectory',action='store_true')
    p.add_argument('--skip-policy-tune',action='store_true')
    p.add_argument('--forecast-only',action='store_true')
    p.add_argument('--safe-alternative',action='store_true')
    p.add_argument('--finalists',type=int,default=6)
    p.add_argument('--milp-seconds',type=float,default=0.6)
    run(p.parse_args())

'''

# ======= test_c2.py =======
SOURCES['test_c2.py'] = r'''
"""Minimal route encoding / quantile / MILP tests. No hydrological test data."""
import numpy as np
import pandas as pd
from cargo_v3 import route_number,cdf_at,joint_probability,make_submission,greedy_select,mixed_integer_select
from riverload.constants import SHIP
assert route_number('A1')==1 and route_number('B2')==2
assert cdf_at([150,160,185,220],150)<=.051
assert cdf_at([150,160,185,220],200)>.25
assert joint_probability(np.array([.10,.12]))>=.12
mass=np.array([10.,20.,30.,40.])
rank=np.array([0,1,1,0])
val=np.array([1.,15.,12.,8.]);caps=np.array([90.,45.])
p=greedy_select(mass,rank,val,caps)
assert mass[p].sum()<=caps[0]+1e-6
assert mass[p[rank[p]>=1]].sum()<=caps[1]+1e-6
sel=mixed_integer_select(mass,rank,val,caps,SHIP,seconds=.2)
assert mass[sel].sum()<=caps[0]+1e-6
assert mass[sel[rank[sel]>=1]].sum()<=caps[1]+1e-6
sample=pd.DataFrame({'id':['TS-A_C001','TS-A_C002','TS-B_C001'],
                     'route':[1,1,1],'departure_slot':[12,12,12],
                     'slot_row':[0,0,0],'slot_tier':[0,0,0]})
m=pd.DataFrame({'scenario_id':['TS-A','TS-A','TS-B'],'container_id':['C001','C002','C001']})
d=pd.DataFrame({'scenario_id':['TS-A','TS-B'],'route_id':['A1','B2'],'departure_slot':[9,13]})
lp=pd.DataFrame({'scenario_id':['TS-A','TS-B'],'container_id':['C001','C001'],
                 'slot_row':[12,1],'slot_tier':[1,1]})
r=make_submission(d,lp,m,sample)
assert r['route'].tolist()==[1,1,2],r
assert r['slot_row'].tolist()==[12,0,1]
assert (r['departure_slot']==np.array([9,9,13])).all()
sample2=sample.rename(columns={'route':'route_id'})
r2=make_submission(d,lp,m,sample2)
assert r2['route_id'].tolist()==[1,1,2]
print('PASS: numerical route, schema variants, all IDs, 0/0, probability, MILP constraints')

'''

# ======= requirements.txt =======
SOURCES['requirements.txt'] = r'''
numpy
pandas
scipy
scikit-learn
lightgbm
pyarrow
threadpoolctl

'''

# Check syntax before training; write all modules for offline reproduction.
for filename,src in SOURCES.items():
    if filename.endswith('.py'):
        compile(src,filename,'exec')
    (CODE/filename).write_text(src,encoding='utf-8')
print('Prepared',len(SOURCES),'complete offline source files',flush=True)
sys.path.insert(0,str(CODE))
cmd=[sys.executable,'-u',str(CODE/'run.py'),str(DATA_DIR),str(OUT),
     '--threads',str(THREADS),'--finalists',str(FINALISTS),
     '--milp-seconds',str(MILP_TIME_SECONDS)]
if not USE_TRAJECTORY:cmd.append('--no-trajectory')
if not TUNE_C2_POLICY:cmd.append('--skip-policy-tune')
if MAKE_SAFE_ALTERNATIVE:cmd.append('--safe-alternative')
print('Running reproducible model:',cmd,flush=True)
start=time.monotonic()
result=subprocess.run(cmd,cwd=str(CODE),check=False,env=os.environ.copy())
if result.returncode:
    raise RuntimeError(f'RIVERLOAD C2 failed with exit code {result.returncode}. '
                       'Scroll up for its traceback.')
sub=pd.read_csv(WORK/'submission.csv')
sample=pd.read_csv(DATA_DIR/'sample_submission.csv')
assert list(sub.columns)==list(sample.columns)
assert sub.id.is_unique and sample.id.is_unique and set(sub.id)==set(sample.id)
route_col='route' if 'route' in sample.columns else 'route_id'
assert set(sub[route_col].astype(int).unique()).issubset({1,2})
assert ((sub.slot_row==0)==(sub.slot_tier==0)).all()
print('SUCCESS: submission.csv validated, rows:',len(sub),'minutes:',
      round((time.monotonic()-start)/60,2),flush=True)
print(sub.head(8).to_string(index=False))
from IPython.display import display,FileLink
display(FileLink(str(WORK/'submission.csv')))
print('Reproducible source:',CODE/'run.py',flush=True)
print('C2 policy CV report:',OUT/'c2_policy_validation.json',flush=True)
