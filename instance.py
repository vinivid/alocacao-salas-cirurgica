"""Instâncias do problema de alocação de cirurgias eletivas em salas cirúrgicas.

Itens   = cirurgias (peso = duração + tempo de limpeza/preparo)
Caixas  = pares (sala, dia) ou "sala-dia", com capacidade C minutos (+ overtime O)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Salas disponíveis por dia: 3 gerais + 1 com equipamento especial (ex.: robótica)
ROOM_TYPES = ["geral", "geral", "geral", "especial"]


@dataclass
class Instance:
    name: str
    d: list            # duração estimada da cirurgia (min)
    turnover: list     # tempo de limpeza/preparo após a cirurgia (min)
    needs_special: list  # exige sala com equipamento especial?
    deadline: list     # último dia (0-indexado) em que pode ser operada
    urgency: list      # rótulo: urgente / prioritaria / eletiva
    surgeon: list      # id do cirurgião responsável
    room_day: list     # dia de cada sala-dia
    room_type: list    # tipo de cada sala-dia
    C: int = 480       # tempo regular do bloco cirúrgico (min) = 8 h
    O: int = 60        # overtime máximo por sala-dia (min)
    H: int = 540       # jornada máxima de um cirurgião por dia (min)
    F: float = 1000.0  # custo fixo de abrir uma sala-dia
    alpha: float = 5.0  # custo por minuto de overtime
    day_eps: float = 1.0  # pequena preferência por dias mais cedo (quebra simetria)

    w: list = field(init=False)
    Fj: list = field(init=False)
    allowed: list = field(init=False)

    def __post_init__(self):
        self.w = [a + b for a, b in zip(self.d, self.turnover)]
        self.Fj = [self.F + self.day_eps * day for day in self.room_day]
        self.allowed = [[self.is_allowed(i, j) for j in range(self.J)] for i in range(self.n)]

    @property
    def n(self):
        return len(self.d)

    @property
    def J(self):
        return len(self.room_day)

    @property
    def D(self):
        return max(self.room_day) + 1

    def is_allowed(self, i, j):
        """a_ij: compatibilidade de equipamento e prazo (dia da sala <= deadline)."""
        if self.needs_special[i] and self.room_type[j] != "especial":
            return False
        return self.room_day[j] <= self.deadline[i]


def generate(n: int, seed: int = 0, C: int = 480, O: int = 60) -> Instance:
    """Gera instância sintética. Durações ~ lognormal (padrão na literatura de salas cirúrgicas)."""
    rng = np.random.default_rng(seed)
    d = np.clip(np.round(rng.lognormal(math.log(100), 0.5, n) / 5) * 5, 30, 300).astype(int)
    needs = rng.random(n) < 0.2
    turn = np.where(needs, 40, rng.choice([20, 30], n))
    w = d + turn

    R = len(ROOM_TYPES)
    n_spec = ROOM_TYPES.count("especial")
    # número de dias suficiente para ~65% de ocupação (folga para o empacotamento)
    D = max(3, math.ceil(w.sum() / (0.65 * R * C)),
            math.ceil(w[needs].sum() / (0.65 * n_spec * C)))

    urg = rng.choice(["urgente", "prioritaria", "eletiva"], n, p=[0.15, 0.25, 0.60])
    dl_map = {"urgente": 1, "prioritaria": D // 2, "eletiva": D - 1}
    deadline = [dl_map[u] for u in urg]
    surgeon = rng.integers(0, max(3, n // 5), n)

    room_day = [day for day in range(D) for _ in ROOM_TYPES]
    room_type = [t for _ in range(D) for t in ROOM_TYPES]
    return Instance(
        name=f"cirurgias_n{n}_s{seed}",
        d=d.tolist(), turnover=turn.tolist(), needs_special=needs.tolist(),
        deadline=deadline, urgency=urg.tolist(), surgeon=surgeon.tolist(),
        room_day=room_day, room_type=room_type, C=C, O=O,
    )
