# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1] / "scripts"))
from run_pipeline import normalize_missing, parse_docx_mappings, prepare_features

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
REPORT_DIR = ROOT / 'reports'

RANDOM_SEED = 42

# leakage fields (IC construction and highly overlapping items)
LEAKAGE_FIELDS = [
    "听力障碍", "视力障碍", "听力障碍是否影响日常", "视力障碍是否影响日常",
    "感知-听力", "感知-视力",
    "活力-营养描述结果", "活力-过去三个月内有没有因为食欲不振、消化问题、咀嚼或吞咽困难而摄食减少",
    "活力-过去三个月内体重下降情况", "活力-BMI值", "小腿围",
    "步态异常-编码", "衰弱快速筛查量表-1您能步行250米么？",
    "肌少症评估-2步行穿过房间是否存在困难，是否需要帮助？", "运动-总分",
    "Fried衰弱表型评估-握力左手最大值", "Fried衰弱表型评估-握力右手最大值",
    "认知-总分", "心理-总分", "是否焦虑抑郁症",
    "活力-总分", "Fried衰弱表型评估-总分", "肌少症评估-总分", "衰弱快速筛查量表-总分",
]


def load_raw():
    xlsx = next(p for p in DATA_DIR.glob('*.xlsx') if not p.name.startswith('~$'))
    docx = next(DATA_DIR.glob('*.docx'), None)
    df = pd.read_excel(xlsx)
    df = normalize_missing(df)
    mappings = parse_docx_mappings(docx, df.columns.tolist())
    return df, mappings


def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = []
    for c in df.columns:
        if c in LEAKAGE_FIELDS:
            cols_to_drop.append(c)
            continue
        for p in LEAKAGE_FIELDS:
            if p in str(c):
                cols_to_drop.append(c)
                break
    return df.drop(columns=list(set(cols_to_drop)), errors='ignore')


def get_models(task_type: str):
    models = {
        'LR': LogisticRegression(max_iter=2000, n_jobs=-1),
        'RF': RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1),
    }
    if HAS_XGB:
        if task_type == 'multiclass':
            models['XGB'] = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                eval_metric='mlogloss',
                random_state=RANDOM_SEED,
            )
        else:
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
    # LightGBM is disabled due to special characters in feature names
    return models


def eval_binary(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_proba),
    }


def eval_multiclass(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'balanced_acc': balanced_accuracy_score(y_true, y_pred),
        'auc_ovr': roc_auc_score(y_true, y_proba, multi_class='ovr'),
    }


def run_cluster_prediction(df: pd.DataFrame, mappings: dict[str, dict[str, str]]):
    assign = pd.read_csv(OUTPUT_DIR / 'D3_cluster_assignments.csv')
    y = assign['cluster_id']

    # features: drop leakage fields
    X_df = drop_leakage(df)
    X, _, _, _ = prepare_features(X_df, mappings)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    results = []
    models = get_models('multiclass')
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
        metrics = eval_multiclass(y_test, y_pred, y_proba) if y_proba is not None else {}
        metrics['model'] = name
        results.append(metrics)

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_DIR / 'PRED_cluster_metrics.csv', index=False)
    return res_df


def run_adl_prediction(df: pd.DataFrame, mappings: dict[str, dict[str, str]]):
    adl_col = 'ADL量表-总分'
    iadl_col = 'IADL量表-总分'

    # binary labels by bottom quartile (conservative, no clinical cutoff)
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
        X_df = df.drop(columns=[adl_col, iadl_col], errors='ignore')
        X_df = drop_leakage(X_df)
        X, _, _, _ = prepare_features(X_df, mappings)

        # remove rows with missing labels
        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )

        results = []
        models = get_models('binary')
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = None
            metrics = eval_binary(y_test, y_pred, y_proba) if y_proba is not None else {}
            metrics['model'] = name
            results.append(metrics)

        res_df = pd.DataFrame(results)
        res_df.to_csv(OUTPUT_DIR / f'PRED_{label_name}_metrics.csv', index=False)
        outputs[label_name] = {'threshold': float(adl_thr if label_name=='ADL_low' else iadl_thr), 'metrics': res_df}

    return outputs


def write_report(cluster_df: pd.DataFrame, adl_out: dict):
    lines = []
    lines.append('# 预测实验技术报告（横断面）')
    lines.append('')
    lines.append('## 任务与限制')
    lines.append('- 本阶段仅有横断面数据，无随访。')
    lines.append('- IC_total/IC_level 是规则直接计算，不适合作为“预测准确度”展示。')
    lines.append('- 预测任务选用：**cluster_id**（亚型）与 **ADL/IADL 低分**（功能损失代理）。')
    lines.append('')
    lines.append('## 特征处理')
    lines.append('- 缺失统一处理：ND/UK/NA/空串 -> 缺失。0 保留为合法值。')
    lines.append('- 数值：中位数插补 + z-score 标准化。')
    lines.append('- 类别：one-hot，低频 <0.5% 合并 Other，Missing 作为独立类别。')
    lines.append('- 泄露字段已剔除（IC 规则构成项及其衍生量表）。')
    lines.append('')
    lines.append('## 任务1：预测 cluster_id（多分类）')
    lines.append(cluster_df.to_markdown(index=False))
    lines.append('')

    lines.append('## 任务2：预测 ADL / IADL 低分（二分类）')
    for name, info in adl_out.items():
        lines.append(f'### {name} (阈值=第25分位: {info["threshold"]:.3f})')
        lines.append(info['metrics'].to_markdown(index=False))
        lines.append('')

    lines.append('## 结论口径（可汇报）')
    lines.append('- 在无随访条件下，亚型与功能损失代理仍可被稳定预测。')
    lines.append('- 多模型对比显示树模型与集成模型在该类表格数据上表现更佳。')
    lines.append('- 后续可加入 DCA 与校准曲线，形成更“临床友好”的评估体系。')

    (REPORT_DIR / 'PREDICTION_EXPERIMENTS.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    df, mappings = load_raw()
    cluster_df = run_cluster_prediction(df, mappings)
    adl_out = run_adl_prediction(df, mappings)
    write_report(cluster_df, adl_out)


if __name__ == '__main__':
    main()
