# Alocação de Cirurgias Eletivas em Salas Cirúrgicas (Bin Packing com restrições)

Trabalho de Programação Matemática: Bin Packing e variantes, resolvido com **Gurobi**, **SCIP**
e uma **heurística/meta-heurística** (Best-Fit Decreasing + Simulated Annealing).

## Como executar
```bash
pip install -r requirements.txt
python main.py                                   # n = 15, 25, 40 com 3 sementes
python main.py --sizes 20 --seeds 1 --show       # imprime a grade de cirurgias
python main.py --time 120 --sa-time 20 --solvers gurobi scip
```
Saída: tabela no terminal e `resultados.csv`.
Obs.: o `gurobipy` instalado via pip traz licença restrita (2000 variáveis/restrições). Instâncias
maiores (n ≳ 50) exigem licença acadêmica gratuita (https://www.gurobi.com/academia/).
Alternativa: `generate()` aceita outros n, e `build_spec(..., strong_linking=False)` reduz o nº de restrições.

## Arquivos
| Arquivo | Conteúdo |
|---|---|
| `instance.py` | Dados, gerador de instâncias sintéticas (durações lognormais) |
| `model.py` | Modelo MILP único, resolvido via `solve_gurobi` e `solve_scip` |
| `heuristics.py` | Guloso BFD e Simulated Annealing |
| `solution.py` | Verificador independente de viabilidade e do objetivo |
| `main.py` | Experimentos e comparação |

## Modelagem matemática

**Conjuntos:** I = cirurgias; J = salas-dia (cada sala em cada dia do horizonte); S = cirurgiões; T = dias.
J_t ⊂ J são as salas-dia do dia t; I_s ⊂ I são as cirurgias do cirurgião s.

**Parâmetros:** d_i duração; t_i tempo de limpeza/preparo; w_i = d_i + t_i; C = 480 min (bloco regular);
O = 60 min (overtime máximo); H = 540 min (jornada máx. do cirurgião por dia); F_j = F + ε·dia(j) custo de abrir a
sala-dia; α custo por minuto de overtime; a_ij = 1 se a sala j tem o equipamento exigido por i **e** dia(j) ≤ prazo_i.

**Variáveis:** x_ij ∈ {0,1} (i operada em j; criada apenas se a_ij = 1); y_j ∈ {0,1} (sala-dia aberta); s_j ∈ [0, O] (overtime).

**Modelo:**

min Σ_j F_j y_j + α Σ_j s_j

| # | Restrição | Papel |
|---|---|---|
| 1 | Σ_j x_ij = 1, ∀i | toda cirurgia é agendada exatamente uma vez (integridade) |
| 2 | Σ_i w_i x_ij ≤ C y_j + s_j, ∀j | capacidade; o que passar de C vira overtime |
| 3 | s_j ≤ O y_j, ∀j | overtime limitado e só em sala aberta |
| 4 | x_ij ≤ y_j, ∀i,j | reforço de relaxação linear (opcional) |
| 5 | Σ_{i∈I_s} Σ_{j∈J_t} d_i x_ij ≤ H, ∀s,t | um cirurgião não opera além da jornada diária |
| 6 | y_j ≥ y_{j+1} (salas idênticas, mesmo dia) | quebra de simetria |

A compatibilidade e o prazo de urgência entram via a_ij (variáveis x_ij inexistentes quando a_ij = 0).
Com C fixo, F_j = 1 e α = 0, o modelo reduz-se ao **Bin Packing clássico**.
O ε·dia(j) (ε = 1 ≪ F) dá preferência a dias mais cedo e quebra simetria entre dias.

## Métodos de solução alternativos
- **Guloso BFD:** ordena por (prazo, duração decrescente) e insere cada cirurgia na sala-dia de menor custo incremental, desempatando pela menor folga.
- **Simulated Annealing:** parte da solução gulosa; movimentos *relocate* e *swap* que mantêm viabilidade
  (compatibilidade, prazo, capacidade, jornada do cirurgião); resfriamento geométrico por tempo; custo suavizado
  com termo −γΣ(carga/(C+O))² que incentiva esvaziar salas. Guarda a melhor solução pelo custo real.

## Roteiro do relatório (o que o professor pede)
1. **Introdução:** contexto (centro cirúrgico = recurso caro), revisão com referências (ABNT/IEEE).
2. **Modelagem:** tabela acima, com justificativas.
3. **Experimentos:** `resultados.csv` → tabelas de objetivo, gap, tempo; comparar Gurobi × SCIP × heurística.
4. **Discussão:** efeito do reforço (4) e da simetria (6) (`build_spec(inst, strong_linking=False, symmetry=False)`),
   escalabilidade, qualidade da heurística vs. ótimo.

## Referências (confirmar autores/ano/DOI antes de usar)
- Dexter, Macario, Traub (1999). *Which algorithm for scheduling add-on elective cases maximizes operating room utilization?* Anesthesiology.
- Cardoen, Demeulemeester, Beliën (2010). *Operating room planning and scheduling: A literature review.* EJOR.
- Guerriero, Guido (2011). *Operational research in the management of the operating theatre: a survey.* Health Care Manag. Sci.
- Denton, Viapiano, Vogl (2007). *Optimization of surgery sequencing and scheduling decisions under uncertainty.* Health Care Manag. Sci.
- Delorme, Iori, Martello (2016). *Bin packing and cutting stock problems: Mathematical models and exact algorithms.* EJOR.
