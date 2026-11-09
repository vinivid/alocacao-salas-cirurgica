"""Validação independente de soluções (usada para MIP e heurísticas)."""
from collections import defaultdict


def evaluate(inst, assign):
    """assign[i] = índice da sala-dia da cirurgia i. Retorna dicionário com objetivo e violações."""
    viol = []
    load = [0] * inst.J
    sl = defaultdict(int)
    for i, j in enumerate(assign):
        if j is None or j < 0:
            viol.append(f"cirurgia {i} não alocada")
            continue
        if not inst.allowed[i][j]:
            viol.append(f"cirurgia {i} incompatível/fora do prazo na sala-dia {j}")
        load[j] += inst.w[i]
        sl[(inst.surgeon[i], inst.room_day[j])] += inst.d[i]
    for j in range(inst.J):
        if load[j] > inst.C + inst.O:
            viol.append(f"sala-dia {j} excede C+O ({load[j]})")
    for (s, day), t in sl.items():
        if t > inst.H:
            viol.append(f"cirurgião {s} excede jornada no dia {day} ({t})")
    used = [j for j in range(inst.J) if load[j] > 0]
    overtime = sum(max(0, load[j] - inst.C) for j in used)
    obj = sum(inst.Fj[j] for j in used) + inst.alpha * overtime
    return {"obj": obj, "rooms": len(used), "overtime": overtime,
            "feasible": not viol, "violations": viol, "load": load}


def print_schedule(inst, assign):
    by_room = defaultdict(list)
    for i, j in enumerate(assign):
        by_room[j].append(i)
    for j in sorted(by_room):
        items = by_room[j]
        tot = sum(inst.w[i] for i in items)
        print(f"  Dia {inst.room_day[j] + 1} | sala {j:>2} ({inst.room_type[j]:<8}) | "
              f"{tot:>3}/{inst.C} min | cirurgias {items}")
