"""Heurística construtiva (Best-Fit Decreasing adaptada) + Simulated Annealing.

Função de custo (alinhada ao MILP):  sum_{j usada} F_j + alpha * overtime_j
O SA usa uma versão suavizada:  custo - gamma * sum_j (carga_j / (C+O))^2,
que favorece concentrar carga em poucas salas e facilita esvaziar salas.
"""
from __future__ import annotations

import math
import random
import time
from collections import defaultdict


def _surgeon_load(inst, assign):
    sl = defaultdict(int)
    for i, j in enumerate(assign):
        sl[(inst.surgeon[i], inst.room_day[j])] += inst.d[i]
    return sl


def greedy_bfd(inst):
    """Ordena por (prazo, peso decrescente) e insere onde o custo incremental é mínimo,
    desempatando pela menor folga restante (best-fit)."""
    cmax = inst.C + inst.O
    load = [0] * inst.J
    cnt = [0] * inst.J
    sl = defaultdict(int)
    assign = [-1] * inst.n

    def tc(j, l, c):
        return 0.0 if c == 0 else inst.Fj[j] + inst.alpha * max(0, l - inst.C)

    for i in sorted(range(inst.n), key=lambda i: (inst.deadline[i], -inst.w[i])):
        best, best_key = None, None
        for j in range(inst.J):
            if not inst.allowed[i][j] or load[j] + inst.w[i] > cmax:
                continue
            if sl[(inst.surgeon[i], inst.room_day[j])] + inst.d[i] > inst.H:
                continue
            nl = load[j] + inst.w[i]
            key = (tc(j, nl, cnt[j] + 1) - tc(j, load[j], cnt[j]), abs(inst.C - nl), j)
            if best_key is None or key < best_key:
                best, best_key = j, key
        if best is None:
            raise RuntimeError(f"Heurística gulosa não encontrou sala para a cirurgia {i}")
        assign[i] = best
        load[best] += inst.w[i]
        cnt[best] += 1
        sl[(inst.surgeon[i], inst.room_day[best])] += inst.d[i]
    return assign


def simulated_annealing(inst, time_limit=10.0, seed=0, T0=300.0, Tf=1.0, gamma=100.0, init=None):
    """Meta-heurística: movimentos relocate/swap sempre mantendo a solução viável."""
    rnd = random.Random(seed)
    n, J, C, O, H = inst.n, inst.J, inst.C, inst.O, inst.H
    cmax = C + O
    assign = list(init if init is not None else greedy_bfd(inst))
    load, cnt = [0] * J, [0] * J
    for i, j in enumerate(assign):
        load[j] += inst.w[i]
        cnt[j] += 1
    sl = _surgeon_load(inst, assign)

    def rc(j, l, c):   # custo suavizado da sala
        return 0.0 if c == 0 else inst.Fj[j] + inst.alpha * max(0, l - C) - gamma * (l / cmax) ** 2

    def tc(j, l, c):   # custo real da sala
        return 0.0 if c == 0 else inst.Fj[j] + inst.alpha * max(0, l - C)

    true_cost = sum(tc(j, load[j], cnt[j]) for j in range(J))
    best_cost, best_assign = true_cost, assign[:]
    t_start = time.time()
    it, T = 0, T0
    while True:
        if it % 500 == 0:
            frac = (time.time() - t_start) / time_limit
            if frac >= 1:
                break
            T = T0 * (Tf / T0) ** frac
        it += 1
        if rnd.random() < 0.6:  # ---- relocate
            i = rnd.randrange(n)
            j1, j2 = assign[i], rnd.randrange(J)
            if j1 == j2 or not inst.allowed[i][j2] or load[j2] + inst.w[i] > cmax:
                continue
            d1, d2, s = inst.room_day[j1], inst.room_day[j2], inst.surgeon[i]
            if d1 != d2 and sl[(s, d2)] + inst.d[i] > H:
                continue
            delta = (rc(j1, load[j1] - inst.w[i], cnt[j1] - 1) - rc(j1, load[j1], cnt[j1])
                     + rc(j2, load[j2] + inst.w[i], cnt[j2] + 1) - rc(j2, load[j2], cnt[j2]))
            if delta <= 0 or rnd.random() < math.exp(-delta / T):
                dt = (tc(j1, load[j1] - inst.w[i], cnt[j1] - 1) - tc(j1, load[j1], cnt[j1])
                      + tc(j2, load[j2] + inst.w[i], cnt[j2] + 1) - tc(j2, load[j2], cnt[j2]))
                load[j1] -= inst.w[i]; cnt[j1] -= 1
                load[j2] += inst.w[i]; cnt[j2] += 1
                sl[(s, d1)] -= inst.d[i]; sl[(s, d2)] += inst.d[i]
                assign[i] = j2
                true_cost += dt
        else:                   # ---- swap
            a, b = rnd.randrange(n), rnd.randrange(n)
            j1, j2 = assign[a], assign[b]
            if j1 == j2 or not inst.allowed[a][j2] or not inst.allowed[b][j1]:
                continue
            na, nb = load[j1] - inst.w[a] + inst.w[b], load[j2] - inst.w[b] + inst.w[a]
            if na > cmax or nb > cmax:
                continue
            d1, d2 = inst.room_day[j1], inst.room_day[j2]
            if d1 != d2:
                chg = defaultdict(int)
                chg[(inst.surgeon[a], d1)] -= inst.d[a]; chg[(inst.surgeon[a], d2)] += inst.d[a]
                chg[(inst.surgeon[b], d2)] -= inst.d[b]; chg[(inst.surgeon[b], d1)] += inst.d[b]
                if any(sl[k] + v > H for k, v in chg.items() if v > 0):
                    continue
            delta = rc(j1, na, cnt[j1]) - rc(j1, load[j1], cnt[j1]) + rc(j2, nb, cnt[j2]) - rc(j2, load[j2], cnt[j2])
            if delta <= 0 or rnd.random() < math.exp(-delta / T):
                dt = tc(j1, na, cnt[j1]) - tc(j1, load[j1], cnt[j1]) + tc(j2, nb, cnt[j2]) - tc(j2, load[j2], cnt[j2])
                if d1 != d2:
                    for k, v in chg.items():
                        sl[k] += v
                load[j1], load[j2] = na, nb
                assign[a], assign[b] = j2, j1
                true_cost += dt
        if true_cost < best_cost - 1e-9:
            best_cost, best_assign = true_cost, assign[:]
    return best_assign, it
