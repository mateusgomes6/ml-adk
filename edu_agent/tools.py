"""
Ferramentas (tools) do agente ADK para análise educacional.

Cada função pública com type hints e docstring é registrada como
"tool" na definição do agente. O ADK usa a docstring e os nomes dos
parâmetros para decidir quando e como chamar cada ferramenta.

Ferramentas disponíveis:
  - prever_ideb              : inferência com o modelo de ML treinado
  - explicar_fatores_ideb    : top features do modelo (interpretabilidade)
  - buscar_municipio_ibge    : consulta API pública do IBGE
  - estatisticas_regiao      : estatísticas agregadas do dataset
  - listar_campos_esperados  : esquema de entrada para prever_ideb
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "ml" / "ideb_model.joblib"
FEATURE_INFO_PATH = ROOT / "ml" / "feature_info.json"
DATA_PATH = ROOT / "data" / "ideb_schools.csv"

IBGE_API_BASE = "https://servicodados.ibge.gov.br/api/v1"


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_PATH}. "
            "Rode primeiro: python ml/train.py"
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_feature_info() -> dict:
    with open(FEATURE_INFO_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

def prever_ideb(
    regiao: str,
    uf: str,
    dependencia_administrativa: str,
    localizacao: str,
    etapa_ensino: str,
    numero_alunos: int,
    numero_professores: int,
    pct_prof_ensino_superior: float,
    tem_biblioteca: int,
    tem_lab_informatica: int,
    tem_lab_ciencias: int,
    tem_quadra_esportes: int,
    tem_internet: int,
    tem_agua_potavel: int,
    tem_energia_eletrica: int,
    tem_jornada_estendida: int,
    taxa_aprovacao: float,
    nivel_socioeconomico: float,
) -> dict[str, Any]:
    """Prevê o IDEB (Índice de Desenvolvimento da Educação Básica) de uma
    escola a partir de suas características.

    Use esta ferramenta sempre que o usuário quiser estimar, prever ou
    simular o IDEB de uma escola (real ou hipotética).

    Args:
        regiao: Região do Brasil. Uma de: "Norte", "Nordeste", "Centro-Oeste",
            "Sudeste", "Sul".
        uf: Sigla do estado, ex: "SP", "SE", "BA".
        dependencia_administrativa: "Federal", "Estadual", "Municipal" ou "Privada".
        localizacao: "Urbana" ou "Rural".
        etapa_ensino: "Anos Iniciais", "Anos Finais" ou "Ensino Médio".
        numero_alunos: Número total de alunos matriculados (inteiro).
        numero_professores: Número total de professores (inteiro).
        pct_prof_ensino_superior: Proporção de professores com ensino superior
            completo, entre 0.0 e 1.0.
        tem_biblioteca: 1 se a escola tem biblioteca, 0 caso contrário.
        tem_lab_informatica: 1 se tem laboratório de informática, 0 caso contrário.
        tem_lab_ciencias: 1 se tem laboratório de ciências, 0 caso contrário.
        tem_quadra_esportes: 1 se tem quadra de esportes, 0 caso contrário.
        tem_internet: 1 se tem acesso à internet, 0 caso contrário.
        tem_agua_potavel: 1 se tem água potável, 0 caso contrário.
        tem_energia_eletrica: 1 se tem energia elétrica, 0 caso contrário.
        tem_jornada_estendida: 1 se oferece jornada estendida, 0 caso contrário.
        taxa_aprovacao: Taxa de aprovação dos alunos, entre 0.0 e 1.0.
        nivel_socioeconomico: Nível socioeconômico médio do alunado, na escala
            do INEP (entre 1.0 = muito baixo e 7.0 = muito alto).

    Returns:
        Dicionário com:
          - status: "sucesso" ou "erro"
          - ideb_previsto: float com a previsão do IDEB (0-10)
          - interpretacao: texto curto classificando o resultado
          - features_utilizadas: dict com os valores usados na inferência
    """
    try:
        model = _load_model()
        features = {
            "regiao": regiao,
            "uf": uf,
            "dependencia_administrativa": dependencia_administrativa,
            "localizacao": localizacao,
            "etapa_ensino": etapa_ensino,
            "numero_alunos": int(numero_alunos),
            "numero_professores": int(numero_professores),
            "pct_prof_ensino_superior": float(pct_prof_ensino_superior),
            "tem_biblioteca": int(tem_biblioteca),
            "tem_lab_informatica": int(tem_lab_informatica),
            "tem_lab_ciencias": int(tem_lab_ciencias),
            "tem_quadra_esportes": int(tem_quadra_esportes),
            "tem_internet": int(tem_internet),
            "tem_agua_potavel": int(tem_agua_potavel),
            "tem_energia_eletrica": int(tem_energia_eletrica),
            "tem_jornada_estendida": int(tem_jornada_estendida),
            "taxa_aprovacao": float(taxa_aprovacao),
            "nivel_socioeconomico": float(nivel_socioeconomico),
        }
        df = pd.DataFrame([features])
        pred = float(model.predict(df)[0])
        pred_clip = max(0.0, min(10.0, pred))

        if pred_clip >= 6.0:
            interp = f"IDEB {pred_clip:.2f} — acima da meta nacional (6.0). Desempenho considerado adequado."
        elif pred_clip >= 5.0:
            interp = f"IDEB {pred_clip:.2f} — abaixo da meta nacional (6.0), mas próximo. Desempenho intermediário."
        elif pred_clip >= 4.0:
            interp = f"IDEB {pred_clip:.2f} — significativamente abaixo da meta. Desempenho preocupante."
        else:
            interp = f"IDEB {pred_clip:.2f} — crítico, muito abaixo da meta nacional."

        return {
            "status": "sucesso",
            "ideb_previsto": round(pred_clip, 2),
            "interpretacao": interp,
            "features_utilizadas": features,
        }
    except Exception as exc:
        return {"status": "erro", "mensagem": str(exc)}

def explicar_fatores_ideb(top_n: int = 10) -> dict[str, Any]:
    """Retorna as features mais importantes do modelo de previsão do IDEB.

    Útil quando o usuário perguntar "quais fatores mais influenciam o IDEB?"
    ou "o que mais pesa na previsão?".

    Args:
        top_n: Quantas features retornar (padrão 10).

    Returns:
        Dicionário com lista das features mais importantes e suas
        importâncias relativas (0-1, somam ~1 entre todas as features do modelo).
    """
    try:
        info = _load_feature_info()
        items = list(info["top_feature_importances"].items())[:top_n]
        return {
            "status": "sucesso",
            "fatores_mais_importantes": [
                {"feature": name, "importancia": round(imp, 4)}
                for name, imp in items
            ],
            "observacao": (
                "Importâncias calculadas pelo RandomForestRegressor. "
                "Valores altos indicam maior peso na previsão do IDEB."
            ),
        }
    except Exception as exc:
        return {"status": "erro", "mensagem": str(exc)}

def buscar_municipio_ibge(nome_municipio: str, uf: str = "") -> dict[str, Any]:
    """Consulta a API pública do IBGE para obter dados oficiais de um município
    brasileiro: código IBGE, microrregião, mesorregião e estado.

    Esta é uma fonte de dados pública oficial (servicodados.ibge.gov.br),
    útil para enriquecer análises educacionais com contexto geográfico.

    Args:
        nome_municipio: Nome do município (ex: "Aracaju", "São Paulo").
        uf: Sigla do estado para desambiguar municípios homônimos (opcional).

    Returns:
        Dicionário com dados do município, ou erro se não encontrado.
    """
    try:
        url = f"{IBGE_API_BASE}/localidades/municipios"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        municipios = resp.json()

        alvo = nome_municipio.strip().lower()
        matches = [m for m in municipios if m["nome"].lower() == alvo]
        if uf:
            matches = [
                m for m in matches
                if m["microrregiao"]["mesorregiao"]["UF"]["sigla"].upper() == uf.upper()
            ]

        if not matches:
            return {
                "status": "nao_encontrado",
                "mensagem": f"Município '{nome_municipio}' não encontrado"
                + (f" em {uf}" if uf else ""),
            }

        m = matches[0]
        return {
            "status": "sucesso",
            "codigo_ibge": m["id"],
            "nome": m["nome"],
            "microrregiao": m["microrregiao"]["nome"],
            "mesorregiao": m["microrregiao"]["mesorregiao"]["nome"],
            "uf": m["microrregiao"]["mesorregiao"]["UF"]["sigla"],
            "uf_nome": m["microrregiao"]["mesorregiao"]["UF"]["nome"],
            "regiao": m["microrregiao"]["mesorregiao"]["UF"]["regiao"]["nome"],
            "total_homonimos_no_brasil": len(
                [x for x in municipios if x["nome"].lower() == alvo]
            ),
        }
    except requests.RequestException as exc:
        return {"status": "erro", "mensagem": f"Falha ao consultar IBGE: {exc}"}

def estatisticas_regiao(
    regiao: str = "",
    uf: str = "",
    dependencia_administrativa: str = "",
) -> dict[str, Any]:
    """Retorna estatísticas agregadas do IDEB a partir do dataset de treino,
    filtradas por região, UF e/ou dependência administrativa.

    Útil para benchmarks — por exemplo, comparar a previsão de uma escola
    com a média da sua região.

    Args:
        regiao: Filtro opcional por região ("Norte", "Nordeste", "Centro-Oeste",
            "Sudeste", "Sul"). Vazio para não filtrar.
        uf: Filtro opcional por UF (ex: "SE"). Vazio para não filtrar.
        dependencia_administrativa: Filtro opcional ("Federal", "Estadual",
            "Municipal", "Privada"). Vazio para não filtrar.

    Returns:
        Dicionário com n_escolas, média, mediana, mínimo, máximo e desvio
        padrão do IDEB no recorte solicitado.
    """
    try:
        df = _load_dataset().copy()
        if regiao:
            df = df[df["regiao"] == regiao]
        if uf:
            df = df[df["uf"] == uf.upper()]
        if dependencia_administrativa:
            df = df[df["dependencia_administrativa"] == dependencia_administrativa]

        if len(df) == 0:
            return {
                "status": "sem_dados",
                "mensagem": "Nenhuma escola encontrada com esses filtros.",
            }

        return {
            "status": "sucesso",
            "filtros": {
                "regiao": regiao or "todas",
                "uf": uf or "todas",
                "dependencia_administrativa": dependencia_administrativa or "todas",
            },
            "n_escolas": int(len(df)),
            "ideb_medio": round(float(df["ideb"].mean()), 2),
            "ideb_mediano": round(float(df["ideb"].median()), 2),
            "ideb_minimo": round(float(df["ideb"].min()), 2),
            "ideb_maximo": round(float(df["ideb"].max()), 2),
            "ideb_desvio_padrao": round(float(df["ideb"].std()), 2),
            "pct_acima_da_meta_6": round(float((df["ideb"] >= 6.0).mean() * 100), 1),
        }
    except Exception as exc:
        return {"status": "erro", "mensagem": str(exc)}

def listar_campos_esperados() -> dict[str, Any]:
    """Retorna o esquema de entrada esperado pela ferramenta prever_ideb,
    incluindo valores válidos para campos categóricos e faixas para
    campos numéricos.

    Use esta ferramenta quando o usuário perguntar "quais dados você
    precisa?" ou quando precisar validar se um valor é aceitável.

    Returns:
        Dicionário com campos categóricos (valores válidos) e numéricos
        (min/max/média observados no dataset de treino).
    """
    info = _load_feature_info()
    return {
        "status": "sucesso",
        "campos_categoricos": info["categorical_values"],
        "campos_numericos": info["numeric_ranges"],
        "observacao": (
            "Campos booleanos (tem_*) devem ser 0 ou 1. "
            "Taxa de aprovação e pct_prof_ensino_superior entre 0.0 e 1.0. "
            "Nível socioeconômico entre 1.0 e 7.0 (escala INEP)."
        ),
    }
