# Plano de ablation — v3 (avaliado contra o código atual)

> **Status:** auditoria interna, com resultados históricos preservados. A decisão
> de protocolo tomada em 6 de agosto de 2026 mantém o **pooled instance-level
> AUROC** como métrica oficial do leaderboard. Macro/per-task e a decomposição
> within/cross são diagnósticos obrigatórios. Quando as seções históricas abaixo
> propõem trocar a primária, prevalece o plano atual em
> [`PAPER_EXECUTION_PLAN.md`](PAPER_EXECUTION_PLAN.md).

Revisão do plano v2 contra `simplicial_sigreg_new` @ `f29b46e`. Três achados no
código mudam a ordem das fases; um deles muda o objeto do estudo.

---

## Parte I — O que o código diz antes de qualquer run

### 1. A métrica era descrita com o nome errado

O `README.md` na época desta auditoria definia o alvo como:

> Metric: **mean AUROC across `rel-avito` binary-classification tasks**, averaged over the val seeds.

`ValAUROCCallback` ([main.py:111-137](../rt/main.py#L111-L137)) acumula `preds`/`labels`
de **todos** os batches de **todas** as val tasks numa lista só e chama
`roc_auc_score` **uma vez**. Isso é AUROC **pooled**, não a média por tarefa.

Quais tarefas entram: `resolve_tasks` ([nanort_train.py:73-77](../scripts/nanort_train.py#L73-L77))
filtra por `f"{table}:Val" in table_info`. Em `~/scratch/pre/rel-avito/table_info.json`:

| task | chave | seeds de val |
|---|---|---|
| `user-visits` / `num_click` | `user-visits:Val` | 29 979 |
| `user-clicks` / `num_click` | `user-clicks:Val` | 21 183 |
| `SearchInfo` / `IsUserLoggedOn` | só `SearchInfo:Db` → **excluída** | — |

Então val = **2 tarefas com base rates diferentes**, num único AUROC pooled. E as
duas são distinguíveis na entrada da célula-alvo: `column_index.json` dá
`num_click of user-clicks` = 9 e `num_click of user-visits` = 3246415 — **índices de
texto distintos**, logo embeddings de `col_name` distintos, e `col_name` entra em `x`
incondicionalmente, inclusive na célula mascarada
([model.py:333-336](../rt/model.py#L333-L336)).

Consequência aritmética. Um score **constante dentro de cada tarefa**, ordenado pela
tarefa de maior base rate, carrega **zero** informação intra-tarefa e ainda assim
pontua, no pooled (n = 29 979 / 21 183):

```
AUROC_between = [n⁺_A·n⁻_B + ½(n⁺_A·n⁻_A + n⁺_B·n⁻_B)] / [(n⁺_A+n⁺_B)(n⁻_A+n⁻_B)]
```

| p(user-visits) ↓ / p(user-clicks) → | 0.05 | 0.10 | 0.20 | 0.30 |
|---|---|---|---|---|
| 0.60 | 0.786 | 0.754 | 0.697 | 0.646 |
| 0.70 | **0.822** | 0.794 | 0.743 | 0.695 |
| 0.80 | **0.864** | **0.840** | 0.794 | 0.751 |
| 0.90 | **0.916** | **0.896** | **0.857** | **0.821** |

**O alvo 0.81621 e o 0.87 reportado caem os dois dentro da faixa alcançável por pura
separação de base rate entre as duas tarefas.** Não é prova de que é isso que
acontece — é prova de que a métrica atual não consegue distinguir os dois casos.

Nota: o caminho de código que computa AUROC **por tarefa** já existe
([main.py:654-665](../rt/main.py#L654-L665)), mas `eval_tasks=[]`
([nanort_train.py:61](../scripts/nanort_train.py#L61)) e ele nunca roda.

Como a calibração e toda a linhagem usaram o mesmo pooled, os records permanecem
comparáveis. O README atual nomeia o estimando corretamente e exige a decomposição
para limitar sua interpretação.

### 2. O reenquadramento "aprendido vs. init" já está escrito no código

[main.py:96-99](../rt/main.py#L96-L99):

> Skip the step-1 eval when False: on the static-relation route the near-init auroc
> is high (~0.87) and would trigger `time_to_target` at step 1, which is not a
> trained-model crossing.

Ou seja: **AUROC ≈ 0.87 no step 1 já é fato conhecido e documentado**, e o flag
`eval_at_step1` existe para impedir que `time_to_target` dispare nele. A hipótese
central do plano v2 (§"o confound primário é aprendido vs. no init") não precisa ser
testada — está confirmada no repositório.

No commit auditado, o default do YAML era `eval_at_step1: "1"`. Como
`timer.start()` roda antes do callback, isso fazia `time_to_target` ficar próximo
da duração de um eval. O protocolo atual corrige o default para
`eval_at_step1: "0"`; o recorde #4 verificado (305,63 s) usa o primeiro eval elegível
no step 100. A run de 154 s abaixo é mantida apenas como diagnóstico de latência.

Consequência: `time_to_target` mistura treino e infraestrutura **dentro da
fronteira histórica do timer**, em especial avaliação e logging. O paper preserva
essa regra para comparabilidade, mas reporta os componentes e `AUROC@0`
separadamente.

Quanto os pesos andam até o step 25: Muon com warmup 3000 → LR(25) = 25/3000·5e-3 =
4.2e-5. E `enc_dict`/`dec_dict`/`mask_embs`/normas/LayerScale caem **todos** no grupo
AdamW ([optim/__init__.py](../rt/optim/__init__.py)), com OneCycle partindo de
`max_lr/25` = 4e-5. Nenhum dos dois moveu nada. Confirmado.

### 3. `relation_embs` não é parâmetro — a Fase de geometria é barata

[data.py:164-171](../rt/data.py#L164-L171): `relation_embs` é um tensor montado por DB
em `RelationalDataset.__init__` e entregue no dict do batch. Não é `nn.Parameter`;
"congelado" é por construção. Logo **todos** os arms `r-*` são uma transformação de
~20 linhas em `data.py`, aplicada por chunk de DB — o que automaticamente vale para
treino e para o holdout, porque são duas instâncias do mesmo `Dataset`. Sem tocar
modelo, sem tocar compile.

Detalhe: `R_train = Σ_DBs` mas `R_val = R_holdout` (offsets em
[data.py:105-113](../rt/data.py#L105-L113)), então `num_relations` **muda** entre treino
e eval. Já logado como `simplicial/r_table_size`.

Detalhe 2: `r_mask = torch.ones(...)` ([data.py:171](../rt/data.py#L171)) — o re-mask
defensivo de [model.py:357-358](../rt/model.py#L357-L358) é no-op hoje.

---

## Parte II — Correções técnicas ao plano v2

| item v2 | correção |
|---|---|
| 0.A.2 "o termo `k̄` é absorvível em `W_q`/`W_k`" | **Mais forte e exato**: `q_norm` é `nn.RMSNorm(d_head)` com `elementwise_affine=True`, logo `q̂ = w ⊙ (W_q x / rms)`. `diag(k̄)` funde-se **exatamente** em `q_norm.weight`. A componente-média do operador é reparametrização pura. |
| "…portanto redundante com o `feat`" | **Não** é redundante com `feat`: o suporte da softmax é outro (`feat` normaliza sobre a linha densa `same_node \| kv_in_f2p`; `nbr_attn` sobre o conjunto esparso de arestas FK por pai). A afirmação correta é "reduz a atenção bilinear sobre as arestas FK" — que é o **controle** certo, não uma redundância. |
| 0.A.3 SVD de `E ∈ R^{R×d_head}` | Fazer **por head**: `kr` é `(R,H,D)` = `(R,8,32)` pós-`kr_norm`. `rank(E_h) ≤ min(R,32)`. Com `R_avito` ≈ 20 (contagem de FK do schema), **não é low-rank por construção** — o argumento "R pequeno ⇒ rank baixo" só morde se `R ≲ 5`. Continua valendo como medida empírica. |
| 0.A.4 "`qk_norm` destrói a geometria global" | Provavelmente **falso no init**, e a previsão deve estar escrita antes: `W_kr` é 256→256 densa, então cada head vê uma projeção aleatória de 256→32 do vetor de relação; por JL a projeção **preserva cosseno** aproximadamente, e a RMSNorm remove só o raio. Previsão: correlação alta (≳0.8). Se der baixa, é achado; se der alta, o arm `r-gram` pode casar a Gram pré-norm. |
| ausente | **O gargalo de contraste está antes do `wkr`**: `enc_dict.relation` é `Linear(384,256, bias=True)`. Com texto unit-norm, ‖Wx‖ ≈ 0.47 e ‖b‖ ≈ 0.47 no init — **bias e sinal da mesma ordem**. Depois vem `norm_dict.relation` (RMSNorm global, mata o raio). Ou seja `r⁰ ≈ (viés comum) + (parte específica de relação de mesma magnitude)`, e as frases-FK (`"{col} between {child} and {parent}"`) já são mutuamente muito similares em MiniLM. **Previsão: `r-const ≈ r-text` e a Fase de geometria dá nulo.** Isso é mensurável a custo zero e é o primeiro gate. |
| D3 (`R` passes exatos de flex_attention) | **Morto por aritmética.** Custo `B·R·S²` = 256·20·1024² ≈ 5.4e9 vs `T` ≈ 1e5–4e5 arestas (`valid_fk_slots ≤ B·S·5` = 1.3M, com `parent_match_rate` < 1). Quatro ordens de grandeza pior. Remover do plano. |
| D2 marcado "✓ fusível" | **Não é fusível em `score_mod`.** `score_mod` recebe um escalar de score e índices — dá para somar um termo por `(relação, head)` (isso é D1), mas não para computar `q̂·(k̂ ⊙ e_r)`, que precisa dos vetores. D1 ✓, D2 ✗. |
| D1 | **Já existe**: branch `relation-score-mod` (bias de relação via `score_mod`). Reaproveitar em vez de reimplementar. A rota de fusão real é `flex_attention` com a máscara `kv_in_f2p` (já construída em [model.py:292](../rt/model.py#L292)) + `score_mod` indexando um `(R,H)` — o id de relação por par `(q,kv)` cabe em `int8` se `R<127`: 256·1024²·1B = 268 MB. |
| 0.C "`r-const` é o 'off' correto" | Concordo, e vale reforçar: desligar `simplicial_layers=none` remove módulos (muda contagem de params e grupos do Muon), `r-const` não. Manter `r-const` como o off canônico. |
| variância | Falta uma fonte, e ela é grande: **a rede toda é bf16** ([main.py:427](../rt/main.py#L427)), então os scores de val têm ~3 dígitos decimais e há muitos **empates**; `roc_auc_score` os resolve por média. Medir a fração de empates. Somar a isso: `index_add_` em [simplicial_attention.py:135,189](../rt/simplicial_attention.py#L135) é atômico ⇒ não-determinístico na GPU. Já o **subgrafo de val está pinado** (`seed=0` + `shuffle_py(0)`, `__getitem__` puro em `batch_idx`, `in_order=True`) — essa fonte pode ser descartada de antemão. |
| orçamento | Boa parte da Fase 0 é *código*, não GPU. `R` por DB já sai no log de todo run (`[artifact-check] … relation_types=`, [yaml:265-271](../plx_nanort_train.yaml#L265)), e `active_relation_types` já vai pro wandb. Nada de novo a rodar para 0.A.1. |

---

## Parte III — Plano revisado

Métrica oficial preservada para toda a série: `pooled_AUROC`. A análise deve sempre
reportar `mean_per_task_AUROC` e a decomposição within/cross ao lado, sem usá-las para
alterar retroativamente a elegibilidade do leaderboard.

### Fase 0 — Decomposição da métrica e do init (nenhum treino)

Gate para todo o resto. Custo: quase todo em código.

**0.1 — Decompor o AUROC pooled. ✅ IMPLEMENTADO** ([rt/val_metrics.py](../rt/val_metrics.py),
ligado em [main.py:161](../rt/main.py#L161); testes em
[tests/test_val_metrics.py](../tests/test_val_metrics.py), rodam sem torch/GPU).

O id de tarefa já estava no batch: `col_name_idxs[is_targets]` (9 = `num_click of
user-clicks`, 3246415 = `num_click of user-visits`), mapeado por `get_column_index`. Zero
plumbing novo. Logado por eval:

- `val/auroc_zero_shot` — pooled, **intocado**: é contra ele que os records e o
  `time_to_target` são medidos
- `val/auroc/<task>` e `val/auroc_mean_per_task` ← diagnóstico macro
- `val/base_rate/<task>`, `val/n_pos/<task>`, `val/n_neg/<task>`
- `val/auroc_between` — fórmula fechada acima, teto de base-rate puro
- `val/auroc_pooled_within`, `val/auroc_pooled_cross`, `val/pooled_cross_weight` —
  split **exato** do pooled: como AUROC é uma média de Mann-Whitney sobre pares
  (positivo, negativo), ela fatora sobre a tarefa de origem de cada elemento, e vale a
  identidade `pooled = (1−cw)·within + cw·cross` (verificada no teste a 1e-9).
  `cross_weight` é a fração da métrica reportada que repousa em pares **entre** tarefas,
  parte deliberada do estimando global. Nas contagens de val do avito, `cw ≈ 0.86`.
- `val/pred_pos_neg_tie_rate`, `val/pred_duplicate_fraction`, `val/pred_unique` —
  quantização/empates do bf16
- `val/n_tasks`, `val/n_unmapped_tasks` — se o mapa de id falhar, o split continua
  correto (agrupa pelo id observado), só sem nome resolvido

Descartado no caminho: a primeira versão logava "pooled após centrar os scores por
tarefa". Centrar pela média do grupo **não** remove o efeito de base rate — a média do
grupo depende dela, e o resultado ficou 0.70 contra 0.86 de `mean_per_task` num fixture
onde o sinal intra-tarefa era idêntico nos dois grupos. Sem interpretação; substituído
pelo split exato de Mann-Whitney.

Custo GPU: um eval. **Gate G0**: se `auroc_mean_per_task ≪ auroc_zero_shot ≈
auroc_between`, manter o recorde pooled, mas atribuir separadamente alinhamento
cross-task e discriminação within-task nas Fases 2–4.

Como obter a leitura sem gastar 45 min: habilitar explicitamente
`eval_at_step1=1` + `max_steps` pequeno entrega a decomposição no step 1, que é
próximo do init. Esse não é o default nem uma run elegível para o leaderboard.

#### Resultado — G0 FECHADO

Projeto wandb `nanort-ablation`, `holdout_db=rel-avito`, `sigreg_layer=0`,
`λ_col=1e-3`, `qk_norm=1`, **`update_relations=1`**, `muon_warmup_steps=200`, seed 0.
Runs `earthy-sea-1` / `golden-deluge-2` (step 1, réplicas exatas) e `deft-totem-3`
(51 steps, `val_eval_freq=5`).

**Base rates medidos — o cenário extremo:**

| task | n⁺ | n⁻ | base rate |
|---|---|---|---|
| `user-visits/num_click` | 27 086 | 2 893 | **0.9035** |
| `user-clicks/num_click` | 745 | 20 438 | **0.0352** |

Daí `auroc_between = 0.9246` e `pooled_cross_weight = 0.8559`: **86% da métrica
reportada repousa em pares entre tarefas** — comparar um positivo de `user-visits`
contra um negativo de `user-clicks`, que o benchmark nunca pede.

**A curva (`deft-totem-3`):**

| step | pooled | mean/task | visits | clicks | within | cross | duplicate frac. (log histórico) | mtp_loss |
|---|---|---|---|---|---|---|---|---|
| 1  | 0.3257 | 0.5264 | 0.5100 | 0.5427 | 0.5153 | 0.2938 | 0.966 | 0.7045 |
| 10 | 0.3330 | 0.5334 | 0.5197 | 0.5471 | 0.5242 | 0.3009 | 0.965 | 0.7035 |
| 25 | 0.4222 | 0.5595 | 0.5502 | 0.5688 | 0.5532 | 0.4002 | 0.953 | 0.7001 |
| 40 | 0.6980 | 0.5807 | 0.6038 | 0.5575 | 0.5963 | 0.7151 | 0.986 | 0.7127 |
| 50 | 0.7696 | 0.5733 | 0.6017 | 0.5448 | 0.5925 | 0.7994 | 0.989 | 0.7666 |

Decompondo o ganho de 50 steps pela identidade exata:

```
Δpooled  = 0.4439
  cross  = 0.856 · (0.7994 − 0.2938) = 0.4328   ← 97.5% do ganho
  within = 0.144 · (0.5925 − 0.5153) = 0.0111   ←  2.5% do ganho
```

**97.5% do ganho na métrica reportada vem da ordenação entre as duas tarefas.** Há
confirmação mecânica independente: a fração histórica de scores duplicados sobe de
0.966 para 0.989 e `mtp_loss` *sobe* de 0.700 (step 25) para 0.767 — o modelo está
colapsando as predições em dois níveis, um por coluna-alvo, que é exatamente a
solução que maximiza o pooled sem aprender nada relacional.

O sinal intra-tarefa existe mas é pequeno: `mean_per_task` vai de 0.526 a 0.581
(pico, step 40). Com n⁺=745 em `user-clicks`, o SE por eval é ≈0.018 nessa tarefa e
≈0.009 em `user-visits`, logo ≈0.010 na média — o 0.581 está ~8σ acima de 0.5, é
real, e é 0.58, não 0.87.

**Ruído de medida (0.3, de graça pelas réplicas):** deltas de **2–5·10⁻⁴** em todos
os AUROCs entre `earthy-sea-1` e `golden-deluge-2`. Contagens, `base_rate`,
`cross_weight`, a duplicação de scores e `pred_unique` bit-idênticos → o subgrafo de val
está pinado, confirmado; o resíduo é o `index_add_` atômico. Piso de ruído de kernel
muito melhor que o temido: ~5·10⁻⁴. **O SE amostral de `user-clicks` (0.018) é a
restrição real de poder, não o kernel.**

**`R_avito = 14`** (`r_table_size`), 13 tipos ativos por batch, `edge_count` ≈
1.16 M, `parent_match_rate` = 0.999, `edges_per_q` = 10.6. Responde 0.5, e enterra
`D3` definitivamente: `B·R·S²` = 3.8·10⁹ contra `T` = 1.2·10⁶, ~3000× pior.

**`time_to_target = −1`**: não cruzou 0.81621 em 50 steps. Extrapolando a curva de
`cross`, cruzaria por volta do step 60–70 — ou seja, o "record" é o tempo até o
modelo aprender os dois base rates.

**Custo: o eval domina o wall-clock.** Dos incrementos de `train/elapsed_seconds`
em `deft-totem-3`: um step de treino custa **1.6 s**, um eval custa **163 s** — um
eval vale ~100 steps. O `ValAUROCCallback` roda o val loader inteiro (~200 batches,
51 k seeds) sem o cap `max_eval_steps` que o `evaluate()` herdado usa, e chamava
`net(batch)` com o default `need_stats=True`, computando ~40 diagnósticos por batch
com `.unique()` e um sync GPU→CPU por stat.

Isso reenquadra o record: com `val_eval_freq=100` e `eval_at_step1=1`, chegar ao
eval do step 100 custa ≈ 163 (eval@1) + 160 (99 steps) + 163 (eval@100) ≈ **486 s**,
contra os **446 s** do record de 7m26s. Ou seja o record é, com boa aproximação,
**"o AUROC no step 100" — e mais da metade do wall-clock foi avaliação, não treino.**
Confirma a Fase 5: `t2t` precisa ser decomposto em startup / treino / eval antes de
qualquer comparação.

#### Resultado — o 0.87 é o init (`vague-eon-4`)

Config vencedora do doc: `update_relations=0`, `muon_warmup_steps=3000`,
`sigreg_layer=0`, `λ_col=1e-3`, com `eval_at_step0=1`.

| step | min | pooled | mean/task | within | cross | mtp_loss |
|---|---|---|---|---|---|---|
| **0** (zero gradientes) | 0.0 | **0.8651** | **0.5682** | 0.5707 | 0.9146 | 0.6898 |
| 1 | 0.0 | 0.8676 | 0.5686 | 0.5718 | 0.9174 | 0.6882 |
| 25 | 3.1 | **0.8753** ← pico | 0.5754 | 0.5802 | **0.9249** ← teto | 0.6880 |
| 50 | 6.2 | 0.8724 | 0.5781 | 0.5832 | 0.9211 | 0.6993 |
| 75 | 9.3 | 0.8726 | 0.5788 | 0.5843 | 0.9212 | 0.7058 |
| 100 | 12.3 | 0.8721 | 0.5789 | 0.5847 | 0.9204 | 0.7096 |
| 125 | 15.4 | 0.8699 | 0.5790 | 0.5849 | 0.9179 | 0.7108 |
| 150 | 18.5 | 0.8636 | 0.5790 | 0.5855 | 0.9104 | 0.7172 |
| 175 | 21.6 | 0.8585 | 0.5807 | 0.5878 | 0.9041 | 0.7172 |

**O platô é 0.58, e chega em 9 minutos.** `mean_per_task` vai de 0.5682 a ~0.579 no
step 75 e depois anda dentro do ruído (+0.0001 por janela de 25 steps até o 150).
O treino inteiro contribui **+0.012** sobre o init. Rodar os 45 min não muda nada:
o platô é atingido em 1/5 do budget — e desses 9 min, ~1.5 min são treino, o resto eval.

**O "decay do prior de texto" do §8 é o `cross` decaindo.** O pooled pica no step 25
e cai (0.8753 → 0.8585) enquanto o `within` *sobe* monotonicamente (0.5707 → 0.5878).
Não é o modelo esquecendo semântica: é o modelo saindo da saturação na ordenação
entre tarefas conforme passa a usar algum sinal real dentro de cada uma. Explicação
mecânica limpa do fenômeno que a receita do §7 tentava evitar com warmup longo — o
warmup preserva um artefato de init, não um prior útil.

**E a `mtp_loss` no holdout piora monotonicamente** (0.6880 → 0.7172 do step 25 ao
175). O objetivo de reconstrução mascarada não está transferindo para o `rel-avito`:
treinar melhora `within` em milésimos e piora a reconstrução zero-shot.

`val/target_crossed_at_init = 1`: **o alvo 0.81621 é cruzado com zero passos de
gradiente.** O 0.87 do doc está inteiro no init — 0.8651 antes de treinar, 0.8753 no
step 25. E `train/time_to_target` = **154 s**, disparado no step 1, que é exatamente
a duração de um eval (155 s medidos). Nesta config o "record" é um eval.

O **recorde oficial #4 é 305,63 s**, na run W&B `ltlneqhn`, com
`eval_at_step1=0` e primeiro eval
elegível no step 100. O número de 154 s acima pertence ao diagnóstico com step 1
ligado e não substitui o leaderboard.

Resposta à pergunta que estava aberta — o `within` do 0.87 é **0.568**, não 0.75.
O ganho de 25 steps no pooled é +0.0102, e 86% dele ainda é `cross`
(0.144·0.0095 + 0.856·0.0103 = 0.0102, identidade fechando). `cross` no step 25 é
0.9249 contra um teto de 0.9246: **saturado no teto de base rate.**

**O que `update_relations` realmente faz.** Comparando os inits (no step 0/1 o
warmup é irrelevante, então a comparação é limpa):

| | within@init | cross@init | pooled@init |
|---|---|---|---|
| `update_relations=1` | ~0.515 | **0.294** | 0.326 |
| `update_relations=0` | 0.571 | **0.915** | 0.865 |

O `within` difere pouco (0.52 → 0.57). O pooled difere **0.54**, e é quase todo a
*direção* do `cross`. Ou seja: "a rota estática é a vencedora" é, em grande parte,
sobre qual direção o init aponta no eixo que não interessa. O mecanismo é o `RelAttn`
(quase-uniforme, +56% na norma de `r`): com ele, a ordenação entre tarefas sai
invertida; sem ele, `r_update_frac = 1` (o prior de texto fica intacto) e sai certa.

**O achado positivo.** `mean_per_task = 0.568` no init, sem um único gradiente, é
~6.8σ acima de 0.5 (SE ≈ 0.010). Random features mais embeddings de texto de schema
dão AUROC por tarefa de 0.57 em zero-shot cross-DB **sem treino**. E 25 steps de
treino somam +0.007. Isso é o "paper 2" da lista: um baseline *training-free*.

**Custo do eval — correção.** A otimização de `need_stats` rendeu pouco: 163 s → 155 s
(~5%). O gargalo não eram os syncs de stats, é o forward sobre os ~200 batches do val
loader inteiro. Reduzir de verdade exige capar o número de batches, ao custo de
variância — e com n⁺ = 745 em `user-clicks` o SE já é 0.018, então não há espaço.

**0.2 — `AUROC@0` e sua distribuição sobre seeds.** O logging de step 0 já está
implementado e impedido por construção de disparar `time_to_target` ou selecionar
checkpoint. Resta executar a distribuição forward-only, ~1 min por seed,
**20 seeds**:

- distribuição de `AUROC@0` (pooled e per-task), com **sinal**: fração de seeds com AUROC < 0.5
- variante *head-only*: fixar a seed do backbone e variar só `dec_dict`/`mask_embs`

Este é o experimento que separa duas histórias muito diferentes. Se a distribuição está
centrada em ~0.87 com spread pequeno, a representação de entrada separa o rótulo quase
independentemente do readout → o achado é sobre dados+encoder, não sobre o ramo de
relação (e é um baseline *training-free*, potencialmente publicável). Se está centrada
em 0.5 com cauda larga e simétrica, o `best-by-val-AUROC` sobre uma curva decrescente
está apenas escolhendo o máximo do ruído — história oposta, e mais grave.

**0.3 — Reprodutibilidade.** Config champion, mesma seed, 3×. Comparar `AUROC@0` e a
curva. Fontes a atribuir: bf16/empates (0.1), `index_add_` atômico, e — descartada de
antemão — amostragem de val. Se o spread for da ordem de 0.01, nenhuma comparação de
arm com Δ < 0.02 é afirmável. Auditar em paralelo a config real do run 0.87 contra o
`.md` (`eval_at_step1`, `sigreg_layer`, `muon_warmup_steps`).

**0.4 — Geometria da relação no init.** Job de 1 min (precisa dos artefatos do cluster;
os locais em `~/scratch/pre/` são do pré-processamento antigo, sem
`relation_text_idxs.json`). Medir, nesta ordem, porque cada etapa pode zerar o
contraste:

1. cossenos par-a-par de `relation_embs` (texto puro)
2. após `enc_dict.relation` + `norm_dict.relation` → razão ‖b‖/‖Wx‖ e cossenos
3. após `wkr` + `kr_norm`, por head → cossenos, e correlação com (1)
4. `k̄` vs `e_r`: `‖e_r‖/‖k̄‖`
5. SVD **por head** de `E_h`, espectro e rank efetivo

**Gate G1**: se o contraste em (2) já for desprezível, `r-const ≈ r-text` é previsão, não
achado, e a Fase 3 só vale para confirmar o nulo com 1–2 arms.

**0.5 — `R` e cobertura.** Puro wandb/log: `relation_types` por DB do `[artifact-check]`,
`simplicial/r_table_size`, `active_relation_types`, `active_relation_type_frac`,
`parent_match_rate`, `edge_count`. Zero compute.

### Fase 1 — Protocolo

- **Oficial**: pooled AUROC e `time_to_target` histórico. Como diagnósticos de
  atribuição, reportar `mean_per_task_AUROC @ K` e within/cross em `K` fixo.
- Reportar sempre o quarteto: `@0` · `@K` · `@peak` (rotulado *oráculo*) · `t2t`
  (rotulado com sua fronteira temporal), mais a decomposição do pooled.
- **Pareamento**: mesma seed, mesma ordem de dados, mesmo stream de init entre arms;
  reportar diferenças pareadas. Com 3 seeds isso é a diferença entre ter e não ter poder.
- **Variância**: eval (SE analítico + empates do bf16), seed (spread), kernel
  (`index_add_`). O efeito mínimo detectável sai daí e vai no pré-registro.
- **`val_eval_freq`**: 100 é grosso demais para um pico no step ~25. Usar 5 nos arms curtos.
- **s/step por variante**: 50 steps, sem eval, sem checkpoint, mais `edges/s`.

### Fase 2 — Qual canal de texto carrega o sinal

Gate: G0. Grid em step-0 (**barato: ~1 min/célula, 10+ seeds**) e depois em treino curto.

`{r-text, r-perm, r-const} × {c-text, c-perm, c-const}`

- `r-*`: transformação em `data.py` sobre cada chunk de DB (ver Parte I §3).
- `c-*`: `col_name_idxs` são linhas da tabela de texto e `column_index.json` diz quais
  linhas são nomes de coluna → permutação/constante por DB, re-gather de
  `col_name_values` como transformação de batch em `main.py`.
- Permutação = bijeção fixa **dentro** de cada DB, mesmo procedimento em treino e holdout.

**Atenção**: `c-perm` sobre os nomes de coluna **destrói a distinguibilidade entre as duas
val tasks** se as duas linhas de `num_click` forem trocadas entre si — o que colapsa
`auroc_between`. Isso é feature, não bug: é exatamente o teste do confound do §1. Mas
tem de ser reportado com `auroc_between` ao lado, senão o Δ é ilegível.

Leitura: contribuição de cada canal + interação. Se `AUROC(r-const, c-text) ≈
AUROC(r-text, c-text)`, o ramo de relação não contribui no init.

### Fase 3 — Geometria ou identidade

Gate: `r-const < r-text` na Fase 2 (e G1 não ter previsto o nulo). Se o gate falhar,
rodar só `r-const` + `r-orth` para documentar o nulo e parar.

| arm | `relation_embs` | isola |
|---|---|---|
| `r-const` | vetor único compartilhado | piso |
| `r-text` | texto (champion) | referência |
| `r-perm` | bijeção fixa por DB | similaridade ↔ papel relacional |
| `r-orth` | ortogonalizado, pares equidistantes | toda a estrutura de similaridade, mantendo identidade |
| `r-gram` | aleatório com Gram casada **pós-`kr_norm`** | similaridade estatística sem semântica |
| `r-anon` | `"rel_7 between t3 and t9"` | léxico vs. formato de schema |
| `r-struct` | direção, cardinalidade, grau do pai, dtype → projeção linear | alternativa transferível, sem texto |

Restrições que a arquitetura impõe: `kr_norm` per-head remove o raio, então **arms que
diferem só em norma são no-op por construção**. `r-gram`/`r-orth` têm de gerar vetores
também para os tipos de FK do holdout, senão o teste é memorização e não transferência.
`r-anon` e `r-struct` exigem embedar texto novo → `rt/embed.py` (SentenceTransformer,
barato em CPU) num passo de pré-processamento. `r-struct` exige metadados de FK que
hoje não estão em `table_info.json` — **custo maior, agendar por último**.

Matriz de decisão: manter a do v2, com a ressalva de que todas as linhas passam a ser
lidas em `mean_per_task_AUROC`, não em pooled.

### Fase 4 — Decomposição do operador

Gate: `r-const ≪ r-text` **em `mean_per_task_AUROC`**.

Com `k̂^rel_r = k̄ + e_r`:

```
ℓ = (q̂ ⊙ k̄)·k̂_s /√d    ← funde exatamente em q_norm.weight → atenção bilinear sobre arestas FK
  + (q̂ ⊙ e_r)·k̂_s /√d   ← todo o sinal específico de relação
```

| arm | operador | fusível |
|---|---|---|
| `D0` | trilinear + scatter (champion) | — |
| `D1` | rank-1 `e_r ≈ c_r·u` → escalar por (relação, head) | ✓ `score_mod` — **já em `relation-score-mod`** |
| `D2` | rank-J via SVD por head (J de 0.4) | ✗ score_mod; só no scatter esparso |
| `D4` | gate só no value, score sem `r` | ✓ |
| `D5` | key aditiva `q̂·(k̂ + k̂_r)` | ✓ |
| `D6` | adaLN: pool de `r` nas arestas → scale+shift no bloco | ✓ |
| `D7` | cross-attn cells→relations (`S×R`) | ✓ |
| `D8` | champion com `simplicial_layerscale_init=0` | — |
| `Dbil` | **novo**: só o termo `k̄` (controle bilinear correto) | ✓ |

`D3` do v2 removido (aritmética, Parte II). `Dbil` é o controle que faltava: separa
"o operador trilinear ajuda" de "atenção sobre arestas FK ajuda".

`D8` testa o reenquadramento direto: com LayerScale init 0 o modelo no step 0 **é**
`r-const`, e o ramo tem de conquistar entrada. Se nunca recupera 0.87, confirma artefato
de init. Nota de implementação: `layerscale_init=0` mantém gradiente para os params do
`nbr_attn`? `dx = attn_out * nbr_scale` com `nbr_scale=0` → `∂L/∂W = 0` no primeiro step;
só `∂L/∂nbr_scale ≠ 0`. Sai do zero, mas devagar. Usar `1e-2` em vez de `0` como arm
adicional para não confundir "não ajuda" com "não treina" (é o mesmo modo de falha que o
commit `430b5d4` já corrigiu no `RelAttn`).

### Fase 5 — Custo e decomposição do `t2t`

- Fronteira `mean_per_task_AUROC @ K` × `s/step`.
- AUROC sob budget fixo de 45 min de wall-clock, fiel à competição.
- **Decompor, sem redefinir retroativamente**: reportar o `t2t` histórico junto de
  `startup`
  (download + build rustler + compile das flex masks, ~3 min pelo README) +
  `train` + `eval_overhead` (os evals contam no `TrainTimer`). E reportar
  `AUROC@0` ao lado, para expor quanto do `t2t` é a rede nem ter treinado.
- Se `AUROC@0 ≥ 0.81621` em `pooled`, dizer explicitamente que o `t2t` legado é
  degenerado neste regime.

### Fase 6 — Condicional, não é ablation

Só se a Fase 3 der positivo: relation dropout estilo CFG; testar se SIGReg **para** o
decay (agora mensurável como distância ao init, com `AUROC@0` na mão); softmax com
segmentação conjunta (endpoint livre) como direção de pesquisa fora do caminho crítico.

---

## Orçamento

| fase | o que é | GPU-h |
|---|---|---|
| 0.1 | código (~40 linhas) + 1 eval | ~0.05 |
| 0.2 | 20 seeds forward-only, step 0 | ~0.4 |
| 0.3 | 3 runs curtos (~10 min) | ~0.5 |
| 0.4 | 1 job de análise | ~0.05 |
| 0.5 | wandb/logs | 0 |
| 1 | reaproveita 0.2/0.3 + ~6 curtos | ~1.0 |
| 2 | 9 células × 10 seeds step-0, + 9 × 3 curtos | ~3.5 |
| 3 | ≤7 arms × 3 seeds × 10 min (gated) | ~3.5 |
| 4 | ≤8 arms × 3 seeds × 45 min (gated) | ~18 |
| 5 | reaproveita | ~1 |

**Caminho crítico: 0.1 → 0.4 → 0.2.** Decompor a métrica, medir o contraste da relação
no init, medir a distribuição de `AUROC@0`. É uma tarde, e as Fases 2–4 só têm objeto se
esses três não encerrarem a discussão.

---

## Disciplina estatística

Pré-registrar antes de rodar: `K`, lista de arms, métrica primária oficial
(`pooled_AUROC`), métricas diagnósticas de atribuição (`mean_per_task_AUROC` e
within/cross), efeito mínimo detectável vindo de 0.3, e os gates G0/G1. Com
~20 arms e picos de oráculo, vencedores espúrios são garantidos. Tudo fora do
pré-registro é exploratório e vai rotulado como tal.

## Os papers possíveis, reordenados

1. **A métrica de zero-shot no RelBench multi-task é dominada por base rate** — se G0
   confirmar, isto é um resultado metodológico sobre como esse tipo de avaliação é
   reportado, e vale independente de tudo o mais aqui. É o mais provável e o mais barato.
2. **Random features + texto de schema dão zero-shot forte sem treino** — se `AUROC@0`
   for alto e estável em `mean_per_task_AUROC` (não só em pooled). Baseline
   *training-free*, e reenquadra `t2t` como métrica de infraestrutura.
3. **Conditioning por tipo de aresta transfere zero-shot** (`r-orth ≈ r-text`) — claim
   mais fraco que o original mas limpo; `r-struct` dispensa o encoder de texto.
4. **Rank baixo recupera o trilinear a custo bilinear** (`D2` com `J` pequeno ≈ `D0`,
   contra `Dbil`) — contribuição de método.
5. **Semântica lexical genuína** (`r-perm` e `r-anon` caem, em `mean_per_task_AUROC`) —
   o claim original. Menos provável, dado o §1 e o gate G1.
