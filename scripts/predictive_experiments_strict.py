# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_pipeline import normalize_missing, parse_docx_mappings, prepare_features

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
REPORT_DIR = ROOT / 'reports'
RANDOM_SEED = 42

# Strict functional leakage removal list (broad)
FUNC_PATTERNS = [
    'ADL', 'IADL', '运动', '步态', '步行', '握力', '衰弱', '跌倒', 'Fried', '肌少', '生活质量',
    '认知', '心理', '感知', '活力', '营养', '小腿围', 'BMI', '视力', '听力', 'SPPB',
]


def load_raw():
    xlsx = next(p for p in DATA_DIR.glob('*.xlsx') if not p.name.startswith('~$'))
    docx = next(DATA_DIR.glob('*.docx'), None)
    df = pd.read_excel(xlsx)
    df = normalize_missing(df)
    mappings = parse_docx_mappings(docx, df.columns.tolist())
    return df, mappings


def drop_functional_cols(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = []
    for c in df.columns:
        for p in FUNC_PATTERNS:
            if p in str(c):
                drop_cols.append(c)
                break
    return df.drop(columns=list(set(drop_cols)), errors='ignore')


def get_models():
    models = {
        'LR': LogisticRegression(max_iter=2000, n_jobs=-1),
        'RF': RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1),
    }
    if HAS_XGB:
        models['XGB'] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=RANDOM_SEED,
        )
    return models


def eval_binary(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_proba),
    }


def run_adl_iadl_strict(df: pd.DataFrame, mappings: dict[str, dict[str, str]]):
    adl_col = 'ADL量表-总分'
    iadl_col = 'IADL量表-总分'

    adl = pd.to_numeric(df[adl_col], errors='coerce')
    iadl = pd.to_numeric(df[iadl_col], errors='coerce')

    adl_thr = adl.quantile(0.25)
    iadl_thr = iadl.quantile(0.25)

    labels = {
        'ADL_low': (adl <= adl_thr).astype(int),
        'IADL_low': (iadl <= iadl_thr).astype(int),
    }

    outputs = {}
    for label_name, y in labels.items():
        # drop ADL/IADL and *all* functional patterns
        X_df = df.drop(columns=[adl_col, iadl_col], errors='ignore')
        X_df = drop_functional_cols(X_df)
        X, _, _, _ = prepare_features(X_df, mappings)

        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )

        results = []
        for name, model in get_models().items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics = eval_binary(y_test, y_pred, y_proba)
            metrics['model'] = name
            results.append(metrics)

        res_df = pd.DataFrame(results)
        res_df.to_csv(OUTPUT_DIR / f'PRED_{label_name}_metrics_strict.csv', index=False)
        outputs[label_name] = {'threshold': float(adl_thr if label_name=='ADL_low' else iadl_thr), 'metrics': res_df}

    return outputs


def write_report(outputs: dict):
    lines = []
    lines.append('# 严格去功能变量后的 ADL/IADL 预测结果')
    lines.append('')
    lines.append('## 严格剔除规则')
    lines.append('- 移除所有包含以下关键词的变量：')
    lines.append('  ADL, IADL, 运动, 步态, 步行, 握力, 衰弱, 跌倒, Fried, 肌少, 生活质量, 认知, 心理, 感知, 活力, 营养, 小腿围, BMI, 视力, 听力, SPPB')
    lines.append('')

    for name, info in outputs.items():
        lines.append(f'## {name} (阈值=第25分位: {info["threshold"]:.3f})')
        lines.append(info['metrics'].to_markdown(index=False))
        lines.append('')

    (REPORT_DIR / 'PREDICTION_EXPERIMENTS_STRICT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    df, mappings = load_raw()
    outputs = run_adl_iadl_strict(df, mappings)
    write_report(outputs)


if __name__ == '__main__':
    main()
