"""
Treinamento do modelo clássico de ML para previsão do IDEB.

Usa RandomForestRegressor (modelo clássico, não-generativo) com
pré-processamento via ColumnTransformer. Salva:
  - ml/ideb_model.joblib    (pipeline completo: preprocessador + modelo)
  - ml/feature_info.json    (metadados de features para uso pelo agente)
  - ml/metrics.json         (métricas em treino/teste)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ideb_schools.csv"
MODEL_PATH = ROOT / "ml" / "ideb_model.joblib"
FEATURE_INFO_PATH = ROOT / "ml" / "feature_info.json"
METRICS_PATH = ROOT / "ml" / "metrics.json"

TARGET = "ideb"

CATEGORICAL_FEATURES = [
    "regiao",
    "uf",
    "dependencia_administrativa",
    "localizacao",
    "etapa_ensino",
]

NUMERIC_FEATURES = [
    "numero_alunos",
    "numero_professores",
    "pct_prof_ensino_superior",
    "tem_biblioteca",
    "tem_lab_informatica",
    "tem_lab_ciencias",
    "tem_quadra_esportes",
    "tem_internet",
    "tem_agua_potavel",
    "tem_energia_eletrica",
    "tem_jornada_estendida",
    "taxa_aprovacao",
    "nivel_socioeconomico",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", model)])


def main() -> None:
    print(f"Carregando dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  linhas: {len(df)}  colunas: {list(df.columns)}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Treinando RandomForestRegressor...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    metrics = {
        "r2_train": float(r2_score(y_train, y_train_pred)),
        "r2_test": float(r2_score(y_test, y_test_pred)),
        "mae_train": float(mean_absolute_error(y_train, y_train_pred)),
        "mae_test": float(mean_absolute_error(y_test, y_test_pred)),
        "rmse_test": float(np.sqrt(np.mean((y_test - y_test_pred) ** 2))),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    print("\nMétricas:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Feature importance (no nível das features originais, não do OHE)
    rf: RandomForestRegressor = pipeline.named_steps["regressor"]
    feature_names_out = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = dict(zip(feature_names_out, rf.feature_importances_.tolist()))
    top_features = sorted(importances.items(), key=lambda kv: -kv[1])[:10]
    print("\nTop 10 features mais importantes:")
    for name, imp in top_features:
        print(f"  {name}: {imp:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModelo salvo em: {MODEL_PATH}")

    # Estatísticas do dataset para o agente usar como contexto / ranges válidos
    feature_info = {
        "target": TARGET,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_values": {
            col: sorted(df[col].unique().tolist()) for col in CATEGORICAL_FEATURES
        },
        "numeric_ranges": {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
            }
            for col in NUMERIC_FEATURES
        },
        "top_feature_importances": dict(top_features),
    }
    with open(FEATURE_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_info, f, indent=2, ensure_ascii=False)
    print(f"Metadados salvos em: {FEATURE_INFO_PATH}")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Métricas salvas em: {METRICS_PATH}")


if __name__ == "__main__":
    main()
