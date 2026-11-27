# 🧬 COVID-EDA v2

Projeto de análise exploratória de variantes genéticas e classificação binária de COVID longa em uma base de participantes não vacinados. Os notebooks exploram a qualidade dos dados e a relação entre os loci genéticos e o desfecho `Long_COVID`. A linha de comando cobre a preparação dos dados, a busca de hiperparâmetros com Optuna, o treinamento, a avaliação e a interpretação dos modelos com SHAP.

## 📦 Instalação

O projeto usa Python 3.13. Com o Git e o `uv` instalados, clone o repositório e sincronize as dependências a partir do `uv.lock`. [Instalação do uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone https://github.com/FelipePassarela/covid-eda-v2.git
cd covid-eda-v2
uv sync --locked
```

Execute os comandos a seguir na raiz do repositório.

## 🗂️ Dados

Coloque uma cópia local de `longa_nao_vacinados.csv` em `data/raw/`. O esquema esperado inclui `id` para identificar cada amostra, `Long_COVID` para o desfecho binário (`0` ou `1`) e colunas com prefixo `chr` para os loci genéticos. Os CSVs em `data/` são ignorados pelo Git.

```bash
uv run split
```

Por padrão, `split` grava `data/interim/train.csv` e `data/interim/test.csv` com uma divisão estratificada de 80% para treino e 20% para teste, usando a semente 42. Consulte `uv run split --help` para informar outro arquivo de entrada ou ajustar a divisão.

## 📊 Treinamento, avaliação e interpretação

As etapas de experimentação registram resultados no projeto `covid` do Weights & Biases (W&B). Para manter os registros locais em `wandb/`, defina `WANDB_MODE=offline` no terminal antes de executar os exemplos.

```powershell
$env:WANDB_MODE = "offline"
```

Em Bash ou Zsh, use `export WANDB_MODE=offline`. Para acompanhar as execuções no W&B online, use `uv run wandb login` em uma sessão com o modo padrão.

```bash
uv run train pipeline=xgboost
uv run eval models/xgboost-tree.joblib
uv run explain models/xgboost-tree.joblib --data-path data/interim/test.csv
```

`train` usa a partição de treino e salva o pipeline ajustado em `models/`. `eval` usa a partição de teste por padrão e calcula métricas como acurácia balanceada, F1 e ROC AUC. `explain` registra um gráfico beeswarm e a importância média absoluta de cada variável segundo SHAP.

Os pipelines disponíveis incluem um baseline `dummy`, regressão logística, SVM, XGBoost, LightGBM e CatBoost. Para ajustar o limiar de decisão durante o treinamento, acrescente `tune_threshold=true` ao comando `train`.

## 🔍 Busca de hiperparâmetros

`tune` usa validação cruzada estratificada e repetida para buscar hiperparâmetros. O Optuna escolhe a melhor tentativa pela primeira métrica da configuração, `roc_auc` por padrão. `eval-tune` estima o desempenho da busca com validação cruzada aninhada. A busca ocorre nos folds internos estratificados e a avaliação usa os folds externos estratificados e repetidos.

Os exemplos usam `dataset=interim` para reservar `data/interim/test.csv` para a avaliação final. Cada busca é limitada a três tentativas para demonstrar o fluxo. As configurações padrão dos dois comandos usam `dataset=raw`.

```bash
uv run tune dataset=interim search=xgboost-selectk search.n_searches=3 search.n_fold_repeats=1 search.n_jobs=1
uv run eval-tune dataset=interim search=xgboost-selectk search.n_searches=3 outer_cv.n_repeats=1 outer_cv.n_jobs=1
```

O W&B registra os melhores parâmetros da busca e as métricas por fold externo da avaliação aninhada. Para aplicar os parâmetros escolhidos ao treinamento final, configure o pipeline usado por `train` em [packages/common/src/covid_common/conf/pipeline/](packages/common/src/covid_common/conf/pipeline/). As opções de busca estão em [packages/tuning/src/covid_tuning/conf/search/](packages/tuning/src/covid_tuning/conf/search/). Os comandos `tune`, `eval-tune` e `train` usam Hydra e aceitam `--cfg job` para inspecionar a configuração e `wandb.group=<nome>` para agrupar execuções.

Para comparar modelos, salve a tabela `result/outer_cv_report` de cada modelo como CSV, com uma linha por fold externo e uma coluna para a métrica desejada. No modo offline, os campos `columns` e `data` da tabela ficam nos arquivos `outer_cv_report_*.table.json` em `wandb/offline-run-*/files/media/table/result/`.

Passe os CSVs ao `boxplot` para gerar `reports/tuning-evaluation/boxplot.png`.

```bash
uv run boxplot reports/cv/xgboost.csv reports/cv/logistic-regression.csv --metric test_balanced_accuracy
```

## 📓 Notebooks

Instale as dependências dos notebooks com `uv sync --locked --group notebooks`. Abra os notebooks no Jupyter ou em um editor compatível e selecione o kernel Python da `.venv`. A maioria das análises usa a partição de treino gerada por `split`.

| Notebook | Conteúdo |
| --- | --- |
| [01_schema.ipynb](notebooks/01_schema.ipynb) | Colunas, tipos e estrutura do CSV |
| [02_quality.ipynb](notebooks/02_quality.ipynb) | Consistência, dados ausentes e variância dos loci |
| [03_univariate.ipynb](notebooks/03_univariate.ipynb) | Distribuição das variantes e equilíbrio entre classes |
| [04_bivariate.ipynb](notebooks/04_bivariate.ipynb) | Relação entre loci e o desfecho |
| [05_cleaning.ipynb](notebooks/05_cleaning.ipynb) | Exploração da limpeza dos dados |
| [06_embeddings.ipynb](notebooks/06_embeddings.ipynb) | Projeções dos dados em dimensões reduzidas |
| [07_training.ipynb](notebooks/07_training.ipynb) | Experimentos de treinamento |

## 🛠️ Desenvolvimento

Os membros do workspace compartilham o `uv.lock` da raiz.

| Pacote | Responsabilidade |
| --- | --- |
| [`covid-common`](packages/common/) | `split`, preparação de dados e pipelines compartilhados. |
| [`covid-eda`](packages/eda/) | Qualidade dos dados, análises univariada e bivariada e seleção por q-value. |
| [`covid-tuning`](packages/tuning/) | `tune`, `eval-tune` e `boxplot`. |
| [`covid-model`](packages/model/) | `train`, `eval` e `explain`. |

`covid-tuning` e `covid-model` dependem de `covid-common`. Para selecionar um pacote em particular, use `uv run --package covid-tuning tune --cfg job` ou `uv run --package covid-model train --cfg job`. A análise exploratória pode ser instalada separadamente com `uv sync --locked --package covid-eda`.

- Cada pacote mantém os próprios testes em `packages/<pacote>/tests/`. Use `uv run pytest packages/model/tests` para rodar só os testes de modelagem.
- [docs/](docs/) contém diagramas PlantUML da análise exploratória, da busca, da avaliação aninhada e da interpretação.

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run --locked lint-imports
```

Os únicos imports permitidos entre membros do workspace são de `covid-eda`, `covid-model` e `covid-tuning` para `covid-common`. O Import Linter verifica essa regra nos pull requests e nos pushes para a branch `main`.

Para habilitar o Ruff antes de cada commit, execute `uv run pre-commit install`.
