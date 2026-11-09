"""Modelo MILP (uma única especificação, resolvida por Gurobi e SCIP).

Conjuntos : I = cirurgias, J = salas-dia, S = cirurgiões, T = dias
Parâmetros: w_i = d_i + t_i, C, O, H, F_j, alpha, a_ij (compatibilidade + prazo)
Variáveis : x_ij in {0,1}  cirurgia i alocada na sala-dia j (só criada se a_ij = 1)
            y_j  in {0,1}  sala-dia j aberta
            s_j  in [0,O]  overtime da sala-dia j (min)

min  sum_j F_j y_j + alpha * sum_j s_j
s.a. (1) sum_j x_ij = 1                                   para todo i
     (2) sum_i w_i x_ij <= C y_j + s_j                    para todo j
     (3) s_j <= O y_j                                     para todo j
     (4) x_ij <= y_j                                      (reforço, opcional)
     (5) sum_{i in I_s} sum_{j in J_t} d_i x_ij <= H      para todo s, t
     (6) y_j >= y_{j+1} para salas-dia consecutivas do mesmo dia e tipo (quebra de simetria)
"""
from __future__ import annotations

import time
from collections import defaultdict


def build_spec(inst, strong_linking=True, symmetry=True):
    """Retorna (variáveis, restrições) em forma neutra, independente de solver."""
    vars_ = {}   # chave -> (vtype, lb, ub, custo)
    cons = []    # (nome, {chave: coef}, sentido, rhs)

    for i in range(inst.n):
        for j in range(inst.J):
            if inst.allowed[i][j]:
                vars_[("x", i, j)] = ("B", 0, 1, 0.0)
    for j in range(inst.J):
        vars_[("y", j)] = ("B", 0, 1, inst.Fj[j])
        vars_[("s", j)] = ("C", 0, inst.O, inst.alpha)

    # (1) cada cirurgia em exatamente uma sala-dia
    for i in range(inst.n):
        cons.append((f"assign_{i}", {("x", i, j): 1 for j in range(inst.J) if inst.allowed[i][j]}, "==", 1))

    for j in range(inst.J):
        # (2) capacidade com overtime
        co = {("x", i, j): inst.w[i] for i in range(inst.n) if inst.allowed[i][j]}
        co[("y", j)] = -inst.C
        co[("s", j)] = -1
        cons.append((f"cap_{j}", co, "<=", 0))
        # (3) overtime só se a sala estiver aberta
        cons.append((f"ot_{j}", {("s", j): 1, ("y", j): -inst.O}, "<=", 0))
        # (4) reforço x_ij <= y_j
        if strong_linking:
            for i in range(inst.n):
                if inst.allowed[i][j]:
                    cons.append((f"link_{i}_{j}", {("x", i, j): 1, ("y", j): -1}, "<=", 0))

    # (5) jornada máxima do cirurgião por dia
    by_surgeon = defaultdict(list)
    for i, s in enumerate(inst.surgeon):
        by_surgeon[s].append(i)
    for s, items in by_surgeon.items():
        for day in range(inst.D):
            co = {("x", i, j): inst.d[i] for i in items for j in range(inst.J)
                  if inst.room_day[j] == day and inst.allowed[i][j]}
            if co and sum(co.values()) > inst.H:  # só vale a pena se puder ser violada
                cons.append((f"surg_{s}_{day}", co, "<=", inst.H))

    # (6) quebra de simetria entre salas idênticas (mesmo dia e tipo)
    if symmetry:
        for j in range(inst.J - 1):
            if inst.room_day[j] == inst.room_day[j + 1] and inst.room_type[j] == inst.room_type[j + 1]:
                cons.append((f"sym_{j}", {("y", j): 1, ("y", j + 1): -1}, ">=", 0))
    return vars_, cons


def _extract(inst, xvals):
    assign = [-1] * inst.n
    for (_, i, j), v in xvals.items():
        if v > 0.5:
            assign[i] = j
    return assign


def solve_gurobi(inst, time_limit=60, **kw):
    import gurobipy as gp
    from gurobipy import GRB
    vars_, cons = build_spec(inst, **kw)
    m = gp.Model("cirurgias")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    m.Params.Threads = 1
    v = {k: m.addVar(lb=lb, ub=ub, obj=c, vtype=GRB.BINARY if t == "B" else GRB.CONTINUOUS,
                     name="_".join(map(str, k))) for k, (t, lb, ub, c) in vars_.items()}
    m.ModelSense = GRB.MINIMIZE
    for name, co, sense, rhs in cons:
        expr = gp.quicksum(c * v[k] for k, c in co.items())
        m.addConstr(expr <= rhs if sense == "<=" else expr >= rhs if sense == ">=" else expr == rhs, name=name)
    t0 = time.time()
    m.optimize()
    elapsed = time.time() - t0
    status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit", GRB.INFEASIBLE: "infeasible"}.get(m.Status, str(m.Status))
    res = {"solver": "Gurobi", "status": status, "time": elapsed, "obj": None, "bound": None, "gap": None, "assign": None}
    if m.SolCount > 0:
        res.update(obj=m.ObjVal, bound=m.ObjBound, gap=m.MIPGap,
                   assign=_extract(inst, {k: x.X for k, x in v.items() if k[0] == "x"}))
    return res


def solve_scip(inst, time_limit=60, **kw):
    from pyscipopt import Model, quicksum
    vars_, cons = build_spec(inst, **kw)
    m = Model("cirurgias")
    m.hideOutput()
    m.setParam("limits/time", time_limit)
    v = {k: m.addVar(name="_".join(map(str, k)), vtype=t, lb=lb, ub=ub, obj=c)
         for k, (t, lb, ub, c) in vars_.items()}
    m.setMinimize()
    for name, co, sense, rhs in cons:
        expr = quicksum(c * v[k] for k, c in co.items())
        m.addCons(expr <= rhs if sense == "<=" else expr >= rhs if sense == ">=" else expr == rhs, name=name)
    t0 = time.time()
    m.optimize()
    elapsed = time.time() - t0
    status = m.getStatus()
    res = {"solver": "SCIP", "status": status, "time": elapsed, "obj": None, "bound": None, "gap": None, "assign": None}
    if m.getNSols() > 0:
        gap = m.getGap()
        res.update(obj=m.getObjVal(), bound=m.getDualbound(), gap=None if gap == float("inf") else gap,
                   assign=_extract(inst, {k: m.getVal(x) for k, x in v.items() if k[0] == "x"}))
    return res
