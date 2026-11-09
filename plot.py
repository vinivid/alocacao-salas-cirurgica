"""Gera gráficos e tabela-resumo a partir de resultados.csv.

Uso:  python plots.py [--csv resultados.csv] [--outdir graficos]
Métricas: status, valor da função objetivo, tempo e gap de otimalidade.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"Gurobi": "#c0392b", "SCIP": "#2471a3", "Simulated Annealing": "#27ae60", "Guloso (BFD)": "#95a5a6"}
SOLVERS = ["Gurobi", "SCIP"]
HEUR = ["Guloso (BFD)", "Simulated Annealing"]
STATUS_COLORS = {"Ótima": "#27ae60", "Factível (limite de tempo)": "#f39c12", "Infactível": "#c0392b", "Sem solução": "#7f8c8d"}


def normalize_status(row):
    s = str(row["status"]).lower()
    if s == "optimal":
        return "Ótima"
    if s == "infeasible":
        return "Infactível"
    return "Factível (limite de tempo)" if pd.notna(row["obj"]) else "Sem solução"


def load(path):
    df = pd.read_csv(path)
    mip = df.metodo.isin(SOLVERS)
    df["status_norm"] = None
    df.loc[mip, "status_norm"] = df[mip].apply(normalize_status, axis=1)
    best = df[df.viavel == True].groupby("instancia").obj.min().rename("best")
    df = df.join(best, on="instancia")
    df["desvio_pct"] = 100 * (df.obj - df.best) / df.best
    # gap recalculado a partir do bound final e da melhor solução inteira do próprio solver
    df["gap_calc"] = np.where(df.lower_bound.notna() & (df.obj > 0), 100 * (df.obj - df.lower_bound) / df.obj, np.nan)
    return df


def plot_status(df, ax):
    d = df[df.metodo.isin(SOLVERS)]
    ct = d.groupby(["metodo", "status_norm"]).size().unstack(fill_value=0).reindex(SOLVERS).fillna(0)
    bottom = np.zeros(len(ct))
    for st in ct.columns:
        ax.bar(ct.index, ct[st], bottom=bottom, label=st, color=STATUS_COLORS.get(st, "gray"))
        for k, v in enumerate(ct[st]):
            if v > 0:
                ax.text(k, bottom[k] + v / 2, int(v), ha="center", va="center", color="white", fontweight="bold")
        bottom += ct[st].values
    ax.set_title("Status da solução (nº de instâncias)")
    ax.set_ylabel("Instâncias")
    ax.legend(fontsize=8)


def plot_objective(df, ax):
    """Desvio percentual do objetivo em relação à melhor solução conhecida, por tamanho."""
    order = SOLVERS + HEUR
    sizes = sorted(df.n.unique())
    width = 0.2
    for k, m in enumerate(order):
        vals = [df[(df.metodo == m) & (df.n == n)].desvio_pct.mean() for n in sizes]
        ax.bar(np.arange(len(sizes)) + (k - 1.5) * width, vals, width, label=m, color=COLORS[m])
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Nº de cirurgias (n)")
    ax.set_ylabel("Desvio médio da melhor solução (%)")
    ax.set_title("Valor da função objetivo")
    ax.legend(fontsize=8)


def plot_objective_abs(df, ax):
    sizes = sorted(df.n.unique())
    for m in SOLVERS + HEUR:
        vals = [df[(df.metodo == m) & (df.n == n)].obj.mean() for n in sizes]
        ax.plot(sizes, vals, marker="o", label=m, color=COLORS[m], ls="--" if m in HEUR else "-")
    ax.set_xlabel("Nº de cirurgias (n)")
    ax.set_ylabel("Objetivo médio (custo total)")
    ax.set_title("Custo total médio por tamanho")
    ax.legend(fontsize=8)


def plot_time(df, ax):
    for m in SOLVERS:
        d = df[df.metodo == m]
        ax.scatter(d.n, d.tempo_s, color=COLORS[m], alpha=0.35, s=18)
        g = d.groupby("n").tempo_s.mean()
        ax.plot(g.index, g.values, marker="o", color=COLORS[m], label=f"{m} (média)")
    ax.set_yscale("log")
    ax.set_xlabel("Nº de cirurgias (n)")
    ax.set_ylabel("Tempo (s, escala log)")
    ax.set_title("Tempo de processamento")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)


def plot_gap(df, ax):
    d = df[df.metodo.isin(SOLVERS) & df.gap_calc.notna()]
    sizes = sorted(d.n.unique())
    width = 0.35
    for k, m in enumerate(SOLVERS):
        vals = [d[(d.metodo == m) & (d.n == n)].gap_calc.mean() for n in sizes]
        ax.bar(np.arange(len(sizes)) + (k - 0.5) * width, vals, width, label=m, color=COLORS[m])
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Nº de cirurgias (n)")
    ax.set_ylabel("Gap médio (%) = (UB − LB)/UB")
    ax.set_title("Gap de otimalidade ao final do limite de tempo")
    ax.legend(fontsize=8)


def plot_per_instance(df, ax):
    """Tempo por instância: Gurobi x SCIP (pontos acima da diagonal = SCIP mais lento)."""
    g = df[df.metodo == "Gurobi"].set_index("instancia").tempo_s
    s = df[df.metodo == "SCIP"].set_index("instancia").tempo_s
    both = pd.concat([g, s], axis=1, keys=["g", "s"]).dropna()
    ax.scatter(both.g, both.s, color="#8e44ad")
    lim = [min(both.min()) * 0.7, max(both.max()) * 1.4]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Gurobi (s)"); ax.set_ylabel("SCIP (s)")
    ax.set_title("Tempo por instância: SCIP × Gurobi")
    ax.grid(True, which="both", alpha=0.3)


def summary_table(df):
    rows = []
    for m in SOLVERS + HEUR:
        d = df[df.metodo == m]
        rows.append({
            "Método": m,
            "Instâncias": len(d),
            "Ótimas": int((d.status_norm == "Ótima").sum()) if m in SOLVERS else "-",
            "Obj. médio": round(d.obj.mean(), 1),
            "Desvio médio (%)": round(d.desvio_pct.mean(), 3),
            "Melhor/empate (nº)": int((d.desvio_pct.abs() < 1e-9).sum()),
            "Tempo médio (s)": round(d.tempo_s.mean(), 2),
            "Tempo máx (s)": round(d.tempo_s.max(), 2),
            "Gap médio (%)": round(d.gap_calc.mean(), 3) if m in SOLVERS else "-",
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="resultados.csv")
    ap.add_argument("--outdir", default="graficos")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = load(a.csv)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plot_status(df, axes[0, 0]); plot_objective(df, axes[0, 1])
    plot_time(df, axes[1, 0]); plot_gap(df, axes[1, 1])
    fig.suptitle("Alocação de cirurgias em salas: Gurobi × SCIP × heurísticas", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{a.outdir}/painel_geral.png", dpi=150)
    plt.close(fig)

    for name, fn in [("status", plot_status), ("objetivo", plot_objective), ("objetivo_absoluto", plot_objective_abs),
                     ("tempo", plot_time), ("gap", plot_gap), ("tempo_por_instancia", plot_per_instance)]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fn(df, ax)
        fig.tight_layout()
        fig.savefig(f"{a.outdir}/{name}.png", dpi=150)
        plt.close(fig)

    tab = summary_table(df)
    tab.to_csv(f"{a.outdir}/resumo.csv", index=False)
    print(tab.to_string(index=False))
    print(f"\nGráficos salvos em {a.outdir}/")


if __name__ == "__main__":
    main()
