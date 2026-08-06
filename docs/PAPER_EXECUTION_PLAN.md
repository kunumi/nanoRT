# Plano de execução do paper nanoRT

Status: plano de trabalho para a submissão ao workshop
[CODEC-FM @ NeurIPS 2026](https://collaborative-open-decentralized-fomo.github.io/).
Data-alvo atual: **29 de agosto de 2026**.

Este documento é a referência operacional para o paper. Ele preserva a métrica
histórica do speedrun e substitui, para fins de decisão, a recomendação de trocar a
métrica primária em `docs/ABLATION_PLAN.md`. O plano antigo continua útil como auditoria
técnica e catálogo de controles.

## 1. Pergunta e narrativa central

Pergunta principal:

> Quão rapidamente um único H200 pode produzir um Relational Transformer que atinge
> um nível de qualidade zero-shot previamente fixado em um banco relacional excluído
> do treinamento, e quais mudanças de otimização, geometria e viés indutivo explicam
> essa redução de tempo?

A narrativa tem três intervenções cumulativas, cada uma associada a um gargalo:

1. **Muon:** melhora o caminho de otimização dos parâmetros matriciais.
2. **SIGReg:** controla a geometria das representações durante o treinamento, sem
   custo de inferência.
3. **Atenção ternária condicionada pela relação:** incorpora a identidade da relação
   no score e no valor de cada mensagem vizinha.

Resultado de leaderboard verificado para a configuração final: **305,63 s
(5m05,6s)**, contra 7m26s do record anterior. A run histórica é
[`ltlneqhn`](https://wandb.ai/franciscokunumi-universidade-federal-de-minas-gerais/nanort-train/runs/ltlneqhn),
e o checkpoint está arquivado em
[`nanort-simplicial-staticrel`](https://huggingface.co/GabrielVeloso/NanoRT/tree/main/checkpoints/rel-avito/nanort-simplicial-staticrel).
O manifesto técnico está em [`RECORD_04_MANIFEST.md`](RECORD_04_MANIFEST.md).
Antes da submissão, completar a verificação por seeds; não confundir esse tempo
oficial com os evals diagnósticos de step 0/1.

Usar “simplicial” como nome curto do módulo implementado, mas defini-lo tecnicamente
como **relation-conditioned trilinear neighbor attention** ou **schema-conditioned
ternary attention**. Não afirmar que o operador constrói um complexo simplicial ou
uma 2-simplex explícita: a implementação usa uma interação trilinear fatorada entre
query, célula vizinha e relação.

O paper deve separar três espécies de contribuição:

- **sistema/protocolo:** speedrun reprodutível, hardware e alvo fixos;
- **resultado empírico:** progressão AdamW → Muon → SIGReg → atenção relacional;
- **análise científica:** quais componentes mudam qualidade inicial, velocidade de
  aprendizado e custo por step.

### 1.1 Encaixe no CODEC-FM

O paper não deve fingir ser um método de treinamento descentralizado. O encaixe
honesto na chamada é:

- **open model development and community infrastructure:** leaderboard aberto,
  records cumulativos, configs e artefatos reproduzíveis;
- **benchmarks and evaluation:** protocolo time-to-quality e auditoria de métrica;
- **compute accessibility:** uma unidade experimental completa em um único
  acelerador, reduzindo a barreira para participantes independentes;
- **optimization:** evidência sobre Muon e regularização geométrica em foundation
  models relacionais.

A contribuição para colaboração é tornar melhorias pequenas acumuláveis e
verificáveis por diferentes participantes. Não é necessário alegar que um H200
isolado resolve os problemas de comunicação ou heterogeneidade do workshop.

## 2. Decisões congeladas

Não alterar estas decisões depois de observar os novos resultados:

- hardware oficial: `1 × NVIDIA H200`;
- conjunto de treinamento: os mesmos cinco bancos e a mesma mistura `alpha=0.3`;
- domínio-alvo: artefato congelado de `rel-avito`, nunca usado para gradientes;
- métrica oficial: **AUROC pooled por instância**, exatamente como calculada pela
  série histórica;
- alvo oficial: `0.81621`;
- orçamento máximo: 45 minutos de tempo ativo segundo a regra histórica;
- seeds, raio de três hops, embeddings e exemplos de validação permanecem congelados;
- diagnósticos adicionais nunca mudam a elegibilidade de um recorde retroativamente.

Antes da submissão, corrigir no README e no paper toda ocorrência de “mean AUROC
across tasks”. O nome honesto é **pooled instance-level zero-shot AUROC**. “Macro” ou
“mean per-task AUROC” designam outra quantidade.

## 3. Como definir e sustentar a métrica

Seja \(V=\bigcup_t V_t\) a união dos exemplos das tarefas de validação, mantendo as
frequências empíricas, e seja \(s(x)\) o score produzido pelo único modelo. A métrica
oficial é

\[
M_{pool}=\Pr[s(X^+)>s(X^-)]+\tfrac12\Pr[s(X^+)=s(X^-)],
\]

onde \(X^+\) e \(X^-\) são sorteados da união de positivos e negativos de `V`. Na
implementação, concatenam-se labels e scores de todas as tarefas elegíveis e chama-se
`roc_auc_score` uma vez.

### 3.1 Justificativas que devem aparecer no paper

1. **O estimando é por exemplo.** Cada decisão binária no conjunto congelado tem o
   mesmo peso; tarefas maiores recebem peso proporcional à população avaliada. Essa
   é uma escolha micro/instance-level, não uma estimativa da “tarefa média”.
2. **O modelo é único e schema-aware.** RT produz scores para todos os exemplos com
   um mesmo preditor e recebe o contexto de schema. Logo, alinhamento de scores entre
   contextos faz parte da capacidade avaliada, em vez de ser removido por uma
   normalização por tarefa.
3. **O alvo foi fixado antes das intervenções.** Muon, SIGReg e o bloco relacional
   foram comparados contra a mesma pipeline. Trocar agora a métrica quebraria a
   linhagem dos records e introduziria avaliação pós-hoc.
4. **O objetivo do speedrun é tempo até uma solução fixa.** A métrica é uma regra de
   término estável, barata e determinística sobre o artefato congelado. Ela não é
   apresentada como a única caracterização de transferência relacional.
5. **A não decomponibilidade é explícita.** `M_pool` recompensa ranking dentro das
   tarefas e ordenação entre tarefas. O paper deve declarar isso diretamente, não
   chamar o segundo efeito de calibração probabilística.

Formulação curta sugerida:

> We retain the challenge's historically fixed pooled AUROC because the speedrun targets
> a single schema-aware predictor over the empirical evaluation population. The
> metric weights examples rather than tasks and deliberately evaluates both
> within-context ranking and cross-context score alignment. To expose rather than
> obscure the latter contribution, we report an exact within/cross-task pair
> decomposition and per-task AUROC as diagnostics.

### 3.2 Limitação que não deve ser omitida

Diferenças de base rate e offsets de score entre tarefas podem elevar o pooled AUROC
mesmo quando o ranking dentro de uma tarefa é fraco. A defesa correta não é negar
isso; é declarar que pares entre tarefas pertencem ao estimando oficial e mostrar
quanto do número vem deles.

Para duas ou mais tarefas, decompor o numerador do AUROC em:

- pares **within-task**: positivo e negativo vêm da mesma tarefa;
- pares **cross-task**: positivo e negativo vêm de tarefas diferentes.

Logar, em todo eval:

- `val/auroc_pooled` — única métrica que aciona o alvo;
- `val/auroc/<task>` e `val/auroc_mean_per_task`;
- `val/auroc_pooled_within` — média dos AUROCs por tarefa ponderada por
  \(n_t^+n_t^-\);
- `val/auroc_pooled_cross` e `val/pooled_cross_weight`;
- `val/base_rate/<task>`, `n_pos`, `n_neg`;
- média, desvio e quantis dos scores por tarefa;
- fração de empates dos scores, relevante sob `bf16`;
- `val/auroc_task_prior_oracle`: score constante por tarefa ordenado pela prevalência
  empírica. Rotular como **oráculo analítico**, nunca como baseline treinável.

Esses números não substituem a métrica oficial. Eles sustentam sua interpretação e
permitem responder ao reviewer que perguntar se o resultado é apenas task identity.

## 4. Por que logar o step 0

`step 0` é a avaliação do modelo inicializado **antes de qualquer `optimizer.step()`**.
Ela deve ocorrer fora do cronômetro oficial e nunca pode:

- acionar `time_to_target`;
- criar um recorde;
- escolher checkpoint;
- influenciar early stopping ou hiperparâmetros durante a run.

Ela é necessária por seis razões:

1. **Controle de inicialização:** mede quanto do AUROC vem do encoder, do schema e do
   readout aleatório antes de aprender.
2. **Atribuição causal:** Muon não pode afetar `AUROC@0`; diferenças no step 0 entre
   arms revelam seeds ou inicializações não pareadas.
3. **Ganho aprendido:** permite reportar
   \(\Delta M(t)=M(t)-M(0)\), além do valor absoluto.
4. **Auditoria do alvo:** revela se `0.81621` já é satisfeito sem updates e impede que
   tempo de avaliação ou resolução do logging seja confundido com aprendizado.
5. **Análise do bloco relacional:** se ligar o módulo muda `AUROC@0`, o efeito inicial
   é arquitetural; se a diferença surge depois, é interação com treinamento.
6. **Transparência:** o próprio código registra que a rota estática pode ter AUROC
   alto próximo da inicialização. O paper fica mais forte ao quantificar isso.

Implementação desejada:

```text
build model
load frozen validation artifact
compile/warm up the train graphs without an optimizer update
evaluate_and_log(prefix="init", step=0, can_trigger_record=false)
for each real training step:
    update parameters
    after the first update, start the historical timer
    evaluate on the frozen schedule
    only post-update evaluations may trigger time_to_target
```

Logar separadamente `init_eval_seconds`, `compile_seconds`, `train_seconds`,
`eval_seconds` e `wall_clock_seconds`. A run oficial continua usando a definição
histórica, mas o paper deve mostrar o custo end-to-end ao lado.

### 4.1 Ramos narrativos pré-registrados

- **`AUROC@0` abaixo do alvo:** o speedrun mede de fato tempo até aprender o limiar;
  analisar slopes e crossings.
- **`AUROC@0` próximo do alvo:** reportar alta sensibilidade a seed e exigir
  crossings persistentes/repetidos para não confundir ruído com sucesso.
- **`AUROC@0` acima do alvo:** manter a métrica para continuidade, mas não chamar o
  crossing de “tempo de treinamento”. A narrativa vira **time-to-verified-solution**;
  Muon/SIGReg só recebem crédito por ganhos pós-init que apareçam em
  `Delta AUROC`, robustez, persistência ou custo, não pelo primeiro eval.

Não escolher o ramo depois por conveniência: publicar o que o controle mostrar.

## 5. Protocolo de holdout e o que é garantido

Usar a expressão **gradient holdout** ou **zero-shot target domain**, não “unseen test
set”. `rel-avito` é fisicamente excluído do mix de treinamento e não participa de
forward/backward para atualização, mas repetidas decisões de engenharia baseadas na
validação o transformam em benchmark de desenvolvimento.

Garantias a implementar e documentar:

1. manifest versionado com hashes SHA-256 dos artefatos de treino e validação;
2. lista explícita dos cinco DBs de treino e assertiva fatal se `rel-avito` aparecer;
3. caminhos físicos separados: `train/<other-dbs>` e `val/rel-avito`;
4. log de code commit, diff status, config resolvida, CUDA/PyTorch e GPU;
5. seeds de amostragem e exemplos de validação congelados;
6. máscara do target, corte temporal e remoção de colunas leaky testados por unidade;
7. nenhuma estatística de labels de validação usada em treino ou seleção dentro da
   run; o oráculo de task prior é somente análise posterior;
8. pacote de reprodução que recalcula a métrica a partir de logits salvos.

Se houver orçamento, reservar agora um segundo domínio nunca consultado para uma
checagem final de validade externa. Ele não muda o leaderboard de `rel-avito`.

## 6. Matriz de experimentos

### P0 — indispensável para o paper

Executar com batches, seeds, artefatos e schedule pareados:

| arm | optimizer | SIGReg | atenção relacional | pergunta |
|---|---|---:|---:|---|
| A | AdamW | off | off | baseline reproduzida |
| B | Muon | off | off | contribuição do optimizer |
| C | Muon | column | off | contribuição incremental de SIGReg |
| D | Muon | column | on | contribuição incremental do simplicial |
| E | Muon | off | on | simplicial depende de SIGReg? |
| F | AdamW | off | on | efeito do módulo sem Muon |

Para cada arm, reportar:

- `AUROC@0`, curva pooled e métricas diagnósticas;
- primeiro crossing, bracket temporal do crossing e sucesso até 45 min;
- `AUROC@K` em um step fixado antes de comparar arms;
- exemplos/s, arestas/s, ms/step, memória máxima e parâmetros;
- `Delta AUROC@K = AUROC@K - AUROC@0`;
- mediana e dispersão entre seeds.

### P0 — controles específicos do simplicial

Manter contagem de parâmetros comparável sempre que possível:

1. **relation text:** operador completo;
2. **relation constant:** a mesma arquitetura, mas toda aresta recebe a mesma relação;
3. **relation permutation:** mapeamento fixo e bijetivo dentro de cada DB;
4. **bilinear control:** mesma vizinhança e value path sem o fator relacional no score;
5. **score-only vs value-only:** relação modula apenas logits ou apenas mensagens.

Os controles 1–3 testam informação semântica; 4–5 testam o mecanismo. Não usar apenas
“módulo removido” se isso alterar parâmetros, grupos do Muon ou custo de compilação.

### P0 — SIGReg

- `lambda_col = 0` versus valor champion, com tudo mais idêntico;
- row-only, column-only e row+column como diagnóstico, sem forçar row-view na
  narrativa se ele for negativo;
- logar loss principal e termos SIGReg separadamente;
- medir espectro/collapse na camada regularizada em pontos fixos, não apenas AUROC.

### P1 — robustez e generalização

- pelo menos um segundo DB de holdout, escolhido antes de rodar;
- target um pouco abaixo e acima de `0.81621` para mostrar que o ranking de métodos
  não depende de um único limiar;
- curvas qualidade × tempo e qualidade × FLOPs, além do recorde pontual;
- sensibilidade à frequência de eval e custo de avaliação;
- long run do champion para detectar degradação tardia.

## 7. Estatística e regra de recorde

- Usar seeds pareadas entre arms; mesma seed deve controlar inicialização e stream de
  dados.
- Fazer 3 seeds para triagem, pelo menos 5 para tabelas principais e, se a regra
  `p<0.01` for mantida, usar amostra suficiente para o teste escolhido.
- Pré-definir o teste. Recomendação: diferenças pareadas de `time_to_target` ou
  `AUROC@K`, com teste unilateral de randomização/sign-flip e intervalo bootstrap.
  Com apenas 5 pares, um teste exato de sinais não consegue atingir `p<0.01`; não
  escrever essa regra sem resolver a aritmética.
- Runs que não cruzam até 45 min são censuradas, não removidas. Reportar taxa de
  sucesso; se forem frequentes, usar resumo de time-to-event.
- Como a avaliação é discreta, o crossing está no intervalo entre o último eval
  abaixo e o primeiro acima. Não inventar precisão sub-intervalo por interpolação.
- Guardar logits e labels para bootstrap e recomputação das decomposições.

## 8. Plano de escrita

### Independente dos novos resultados — escrever agora

- Introduction;
- Related Work;
- definição de RT e do problema relacional;
- protocolo do speedrun;
- holdout e prevenção de leakage;
- definição e decomposição da métrica;
- descrição de Muon, SIGReg e do operador relacional;
- detalhes de hardware, timer e reprodução.

### Dependente dos resultados — preencher depois

- abstract com números;
- tabela principal e record progression;
- ablations e curvas;
- análise de `AUROC@0`;
- conclusão e limitações específicas;
- escolha final entre “training speed” e “time-to-verified-solution”.

### Estrutura sugerida para 6 páginas

1. Introduction — 0.75 página
2. Background and Related Work — 0.75
3. nanoRT Speedrun Protocol — 1.0
4. Methods: Muon, SIGReg, Relational Attention — 1.25
5. Experiments and Ablations — 1.5
6. Analysis, Limitations, Conclusion — 0.75

Material suplementar: hashes, configs, derivação da métrica, tabela completa de
seeds, detalhes do sampler e controles adicionais.

## 9. Sequência de execução para outro Codex

Executar nesta ordem e não avançar pelos gates sem registrar o resultado:

1. **Auditar a linhagem:** recuperar config e logits dos records 1–3; confirmar que
   todos usaram o mesmo pooled AUROC, artefato e timer.
2. **Instrumentar sem mudar a régua:** adicionar `step 0`, AUROCs por tarefa,
   decomposição within/cross, base rates, ties e tempos separados.
3. **Testar a instrumentação:** casos sintéticos com duas tarefas, incluindo scores
   constantes por tarefa; provar que a soma within+cross reproduz o pooled.
4. **Criar manifest:** hashes de dados, commit, config e ambiente em cada run.
5. **Rodar controles baratos:** 20 inicializações forward-only para `AUROC@0` e 3
   repetições idênticas para variância de kernel.
6. **Congelar `K`, seeds e teste estatístico:** seguir o plano pré-registrado em
   [`POLYAXON_SWEEP_PLAN.md`](POLYAXON_SWEEP_PLAN.md).
7. **Executar arms A–F:** primeiro 3 seeds; promover somente a matriz já definida para
   5+ seeds, sem selecionar arms por uma seed favorável.
8. **Executar controles do simplicial e SIGReg.**
9. **Gerar automaticamente:** tabela principal, curvas, decomposição da métrica e
   tabela de custo.
10. **Preencher os blocos `TODO-RESULT` do draft**, atualizar abstract e limitações.
11. **Reproduzir do zero:** uma máquina/checkout limpo recalcula ao menos o champion e
    a métrica a partir dos logits.
12. **Checklist de submissão:** 6 páginas, anonimização se exigida, licença/código,
    links suplementares e conformidade com a chamada do workshop.

Critério de aceite: cada claim numérico do paper deve apontar para config, seed, log,
commit e script que o reproduz.

## 10. Cronograma até 29 de agosto

- **6–9 ago:** instrumentação, testes de métrica, manifest e step 0.
- **10–15 ago:** matriz P0 e replay dos records históricos.
- **16–19 ago:** controles simplicial/SIGReg e estatística final.
- **20–22 ago:** figuras, tabelas e preenchimento dos resultados.
- **23–25 ago:** primeira revisão integral e corte para 6 páginas.
- **26–27 ago:** reprodução limpa e revisão de claims.
- **28 ago:** PDF final, suplementar e checklist; não deixar upload para o deadline.

## 11. Referências de posicionamento

- [RelBench](https://arxiv.org/abs/2407.20060) e
  [Relational Transformer](https://arxiv.org/abs/2510.06377): problema e modelo-base.
- [PluRel](https://arxiv.org/abs/2602.04029): relational foundation models recentes.
- [Speedrunning Tabular Foundation Model Pretraining](https://arxiv.org/abs/2606.03681)
  e [nanoTabPFN](https://arxiv.org/abs/2511.03634): templates mais próximos.
- [Cramming](https://arxiv.org/abs/2212.14034),
  [MLPerf Training](https://arxiv.org/abs/1910.01500) e
  [DAWNBench analysis](https://arxiv.org/abs/1806.01427): tempo até qualidade sob
  orçamento fixo.
- [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt): cultura de
  speedrun e progressão aberta de records.
- [Muon](https://github.com/KellerJordan/Muon),
  [Muon is Scalable for LLM Training](https://openreview.net/pdf/1bff28e2c4dc62d54931563338d4dd25112a0203.pdf),
  [SIGReg](https://openreview.net/forum?id=IYnHchzvYB) e
  [LeJEPA](https://arxiv.org/abs/2511.08544): otimização e geometria.
- [2-simplicial attention](https://arxiv.org/abs/1909.00668) e
  [Fast and Simplex](https://arxiv.org/abs/2507.02754): contexto para interações
  ternárias, deixando explícita a diferença para o operador do nanoRT.
- [R-GCN](https://arxiv.org/abs/1703.06103) e
  [relation-aware self-attention](https://aclanthology.org/N18-2074/): antecedentes
  para condicionamento explícito por tipo de relação.

## 12. Prompt inicial para entregar a outro Codex

```text
Leia docs/PAPER_EXECUTION_PLAN.md e docs/PAPER_DRAFT.md integralmente antes de agir.
O objetivo é preparar o nanoRT para um paper de 6 páginas no CODEC-FM @ NeurIPS 2026.

Decisões não negociáveis:
- preserve val/auroc_pooled e o target 0.81621 como regra oficial histórica;
- não altere artefatos, seeds, mistura de treino ou timer legado;
- step 0 é diagnóstico fora do timer e nunca aciona recorde;
- toda nova métrica é diagnóstica e deve ser logada ao lado da oficial;
- não invente números nem complete TODO-RESULT sem log reproduzível.

Comece pelas etapas 1–5 da seção 9: audite records/configs, implemente a decomposição
within/cross-task e AUROC por tarefa, implemente AUROC@0, crie testes sintéticos e um
manifest de reprodução. Antes de rodar a matriz de ablations, apresente o diff, os
testes e uma tabela com as configs exatas propostas. Preserve mudanças não relacionadas
no worktree.
```
