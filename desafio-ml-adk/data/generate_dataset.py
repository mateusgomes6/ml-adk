"""
Gerador de dataset sintético baseado na estrutura real dos microdados
do Censo Escolar (INEP) e do IDEB.

As features foram escolhidas espelhando variáveis que o INEP e o MEC
publicam nos portais oficiais (dadosabertos.mec.gov.br, inep.gov.br).

Observação: em produção, este script seria substituído por um pipeline
que baixa os microdados oficiais. Usamos dados sintéticos aqui para que
o projeto seja executável sem depender de downloads pesados (>1 GB).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)

REGIOES = {
    "Norte": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
    "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Centro-Oeste": ["DF", "GO", "MT", "MS"],
    "Sudeste": ["ES", "MG", "RJ", "SP"],
    "Sul": ["PR", "RS", "SC"],
}

DEPENDENCIAS = ["Federal", "Estadual", "Municipal", "Privada"]
LOCALIZACOES = ["Urbana", "Rural"]
ETAPAS = ["Anos Iniciais", "Anos Finais", "Ensino Médio"]


def gerar_escolas(n: int = 4000) -> pd.DataFrame:
    """Gera um DataFrame sintético com n escolas."""

    registros = []
    for i in range(n):
        regiao = RNG.choice(list(REGIOES.keys()), p=[0.10, 0.27, 0.08, 0.42, 0.13])
        uf = RNG.choice(REGIOES[regiao])

        # Dependência administrativa - distribuição aproximada do Censo Escolar
        dep = RNG.choice(DEPENDENCIAS, p=[0.005, 0.30, 0.55, 0.145])

        # Localização urbana é mais comum em escolas privadas/estaduais
        prob_urbana = {"Federal": 0.9, "Estadual": 0.85, "Municipal": 0.55, "Privada": 0.95}[dep]
        localizacao = "Urbana" if RNG.random() < prob_urbana else "Rural"

        etapa = RNG.choice(ETAPAS, p=[0.5, 0.3, 0.2])

        # Número de alunos (log-normal, varia muito)
        n_alunos = int(np.clip(RNG.lognormal(mean=5.5, sigma=0.8), 20, 3000))

        # Número de professores (proporcional)
        razao_aluno_prof = RNG.normal(loc=18, scale=4)
        n_professores = max(3, int(n_alunos / max(razao_aluno_prof, 8)))

        # % de professores com ensino superior (maior em privadas/federais)
        base_formacao = {"Federal": 0.95, "Estadual": 0.82, "Municipal": 0.72, "Privada": 0.88}[dep]
        pct_prof_superior = np.clip(RNG.normal(base_formacao, 0.08), 0.3, 1.0)

        # Infraestrutura (booleans)
        base_infra = {"Federal": 0.9, "Estadual": 0.65, "Municipal": 0.45, "Privada": 0.80}[dep]
        if localizacao == "Rural":
            base_infra *= 0.55

        tem_biblioteca = RNG.random() < base_infra
        tem_lab_informatica = RNG.random() < (base_infra * 0.85)
        tem_lab_ciencias = RNG.random() < (base_infra * 0.5)
        tem_quadra_esportes = RNG.random() < (base_infra * 0.9)
        tem_internet = RNG.random() < min(1.0, base_infra + 0.15)
        tem_agua_potavel = RNG.random() < min(1.0, base_infra + 0.25)
        tem_energia_eletrica = RNG.random() < min(1.0, base_infra + 0.35)

        # Taxa de aprovação (0-1)
        taxa_aprovacao = np.clip(
            RNG.normal(loc=0.85 + (0.05 if dep == "Privada" else 0), scale=0.08),
            0.4,
            1.0,
        )

        # Nível socioeconômico (1-7, escala do INEP)
        nse_base = {"Federal": 5.5, "Estadual": 4.2, "Municipal": 3.8, "Privada": 6.0}[dep]
        if localizacao == "Rural":
            nse_base -= 0.8
        nivel_socioeconomico = np.clip(RNG.normal(nse_base, 0.7), 1.0, 7.0)

        # Jornada estendida (escolas de tempo integral)
        tem_jornada_estendida = RNG.random() < (0.15 if dep != "Privada" else 0.25)

        # IDEB (0-10) — variável alvo. Fórmula realista que combina as features
        # com ruído. Valores típicos no Brasil: 3.5 a 7.5.
        ideb = (
            2.0
            + 0.35 * nivel_socioeconomico
            + 2.5 * taxa_aprovacao
            + 0.8 * pct_prof_superior
            + 0.15 * tem_biblioteca
            + 0.15 * tem_lab_informatica
            + 0.10 * tem_lab_ciencias
            + 0.10 * tem_quadra_esportes
            + 0.20 * tem_internet
            + 0.25 * tem_jornada_estendida
            + (0.3 if dep == "Federal" else 0.0)
            + (-0.25 if localizacao == "Rural" else 0.0)
            + (-0.15 if etapa == "Ensino Médio" else 0.0)
            + RNG.normal(0, 0.35)
        )
        ideb = float(np.clip(ideb, 0.0, 10.0))

        registros.append(
            {
                "codigo_escola": f"{100000 + i}",
                "regiao": regiao,
                "uf": uf,
                "dependencia_administrativa": dep,
                "localizacao": localizacao,
                "etapa_ensino": etapa,
                "numero_alunos": n_alunos,
                "numero_professores": n_professores,
                "pct_prof_ensino_superior": round(pct_prof_superior, 3),
                "tem_biblioteca": int(tem_biblioteca),
                "tem_lab_informatica": int(tem_lab_informatica),
                "tem_lab_ciencias": int(tem_lab_ciencias),
                "tem_quadra_esportes": int(tem_quadra_esportes),
                "tem_internet": int(tem_internet),
                "tem_agua_potavel": int(tem_agua_potavel),
                "tem_energia_eletrica": int(tem_energia_eletrica),
                "tem_jornada_estendida": int(tem_jornada_estendida),
                "taxa_aprovacao": round(taxa_aprovacao, 3),
                "nivel_socioeconomico": round(nivel_socioeconomico, 2),
                "ideb": round(ideb, 2),
            }
        )

    return pd.DataFrame(registros)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    df = gerar_escolas(n=4000)
    out_path = out_dir / "ideb_schools.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset gerado: {out_path}  ({len(df)} linhas)")
    print(df.describe().round(2))


if __name__ == "__main__":
    main()
