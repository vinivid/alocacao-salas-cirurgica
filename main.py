"""Executa o experimento: Gurobi x SCIP x Heurística (BFD + Simulated Annealing).

Uso:  python main.py --sizes 15 25 40 --time 60 --sa-time 10 --seed 1
"""
import argparse
import csv
import time

from instance import generate
from heuristics import greedy_bfd, simulated_annealing
from solution import evaluate, print_schedule


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[15, 25, 40])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--time", type=float, default=60, help="limite de tempo dos solvers (s)")
    ap.add_argument("--sa-time", type=float, default=10, help="tempo do Simulated Annealing (s)")
    ap.add_argument("--solvers", nargs="+", default=["gurobi", "scip"])
    ap.add_argument("--show", action="store_true", help="imprime a grade da melhor solução")
    ap.add_argument("--out", default="resultados.csv")
    a = ap.parse_args()

    solvers = {}
    if "gurobi" in a.solvers:
        from model import solve_gurobi; solvers["Gurobi"] = solve_gurobi
    if "scip" in a.solvers:
        from model import solve_scip; solvers["SCIP"] = solve_scip

    rows = []
    for n in a.sizes:
        for seed in a.seeds:
            inst = generate(n, seed)
            print(f"\n=== {inst.name}: {inst.n} cirurgias, {inst.J} salas-dia ({inst.D} dias) ===")
            runs = []

            t0 = time.time(); g = greedy_bfd(inst); tg = time.time() - t0
            runs.append({"solver": "Guloso (BFD)", "status": "heuristic", "time": tg, "assign": g,
                         "bound": None, "gap": None})
            t0 = time.time(); sa, iters = simulated_annealing(inst, time_limit=a.sa_time, seed=seed, init=g)
            runs.append({"solver": "Simulated Annealing", "status": f"{iters} iter", "time": time.time() - t0,
                         "assign": sa, "bound": None, "gap": None})
            for name, fn in solvers.items():
                try:
                    runs.append(fn(inst, time_limit=a.time))
                except Exception as e:  # licença limitada do Gurobi, etc.
                    print(f"  [{name}] erro: {e}")

            best_obj, best_assign = None, None
            print(f"  {'Método':<22}{'Status':<12}{'Obj':>9}{'Salas':>7}{'OT(min)':>9}{'LB':>9}{'Gap%':>7}{'Tempo(s)':>10}  Viável")
            for r in runs:
                ev = evaluate(inst, r["assign"]) if r.get("assign") else None
                obj = ev["obj"] if ev else None
                if ev and ev["feasible"] and (best_obj is None or obj < best_obj):
                    best_obj, best_assign = obj, r["assign"]
                gap = None if r.get("gap") is None else 100 * r["gap"]
                print(f"  {r['solver']:<22}{r['status']:<12}{fmt(obj):>9}{(ev['rooms'] if ev else '-'):>7}"
                      f"{(ev['overtime'] if ev else '-'):>9}{fmt(r.get('bound')):>9}{fmt(gap):>7}{r['time']:>10.2f}"
                      f"  {'sim' if ev and ev['feasible'] else 'NÃO' if ev else '-'}")
                rows.append({"instancia": inst.name, "n": n, "seed": seed, "metodo": r["solver"], "status": r["status"],
                             "obj": obj, "salas": ev["rooms"] if ev else None, "overtime": ev["overtime"] if ev else None,
                             "lower_bound": r.get("bound"), "gap_pct": gap, "tempo_s": round(r["time"], 3),
                             "viavel": ev["feasible"] if ev else None})
            if a.show and best_assign:
                print("  Melhor grade:"); print_schedule(inst, best_assign)

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nResultados salvos em {a.out}")


if __name__ == "__main__":
    main()
