# METRICX: Quantitative Evaluation Framework for XAI Methods

**Dissertação de Mestrado – DCC/UFMG**

METRICX é um framework Python para avaliação quantitativa de métodos de interpretabilidade post-hoc (*Explainable Artificial Intelligence*, XAI) aplicados a modelos de aprendizado de máquina em dados tabulares.

O projeto investiga métricas de avaliação de XAI em experimentos conduzidos com os classificadores **Random Forest** e **XGBoost** sobre os datasets **FHS**, **SEPSIS**, **STROKE**, **UCI-THYROID-DXBIN** e **WDBC**.

---

## Estrutura do Repositório

```
.
├── src/
│   ├── data/
│   │   └── loaders.py          # Carregamento e pré-processamento dos datasets
│   ├── models/
│   │   └── classifiers.py      # Random Forest e XGBoost
│   ├── xai/
│   │   └── explainers.py       # SHAP e LIME
│   ├── metrics/
│   │   └── xai_metrics.py      # Métricas de avaliação de XAI
│   └── pipeline/
│       └── pipeline.py         # Orquestração do pipeline completo
├── experiments/
│   └── run_experiments.py      # Script de execução dos experimentos
├── tests/                      # Testes automatizados (pytest)
│   ├── test_loaders.py
│   ├── test_classifiers.py
│   ├── test_metrics.py
│   └── test_pipeline.py
├── data/                       # Diretório para os arquivos CSV dos datasets
├── results/                    # Resultados gerados (CSV)
├── requirements.txt
└── setup.py
```

---

## Datasets

| Identificador       | Descrição                                    | Arquivo CSV esperado        |
|---------------------|----------------------------------------------|-----------------------------|
| `WDBC`              | Wisconsin Diagnostic Breast Cancer (WDBC)    | `data/wdbc.csv`             |
| `FHS`               | Framingham Heart Study                       | `data/fhs.csv`              |
| `SEPSIS`            | Sepsis clinical dataset                      | `data/sepsis.csv`           |
| `STROKE`            | Stroke Prediction Dataset                    | `data/stroke.csv`           |
| `UCI-THYROID-DXBIN` | UCI Thyroid Disease (classificação binária)  | `data/uci_thyroid_dxbin.csv`|

> **Nota:** O dataset WDBC possui um *fallback* embutido via scikit-learn, permitindo execução sem arquivo CSV. Os demais datasets devem ser colocados no diretório `data/`.

---

## Instalação

```bash
# Criar e ativar ambiente virtual (opcional, mas recomendado)
python -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Instalar o pacote em modo editável
pip install -e .
```

---

## Uso

### Executar todos os experimentos

```bash
python experiments/run_experiments.py
```

### Opções disponíveis

```
usage: run_experiments.py [-h] [--datasets {FHS,SEPSIS,STROKE,UCI-THYROID-DXBIN,WDBC} [...]]
                           [--classifiers {RF,XGB} [...]] [--xai {SHAP,LIME} [...]]
                           [--data-dir DATA_DIR] [--output OUTPUT]
                           [--top-k TOP_K] [--n-explain N_EXPLAIN]
                           [--random-state RANDOM_STATE]
```

**Exemplos:**

```bash
# Apenas WDBC com Random Forest e SHAP
python experiments/run_experiments.py --datasets WDBC --classifiers RF --xai SHAP

# Todos os datasets com ambos os classificadores e métodos XAI
python experiments/run_experiments.py \
    --datasets FHS SEPSIS STROKE UCI-THYROID-DXBIN WDBC \
    --classifiers RF XGB \
    --xai SHAP LIME \
    --output results/metricx_all.csv

# Especificar diretório dos dados
python experiments/run_experiments.py --data-dir /path/to/data
```

---

## Métricas de Avaliação XAI

### Fidelidade (Faithfulness)

| Métrica                    | Descrição                                                                                     |
|----------------------------|-----------------------------------------------------------------------------------------------|
| `faithfulness_correlation` | Correlação de Pearson entre importâncias e variação da predição ao mascarar features          |
| `comprehensiveness`        | Queda média na probabilidade predita ao remover as top-k features (↑ = mais compreensivo)     |
| `sufficiency`              | Queda média na probabilidade predita ao manter apenas as top-k features (≈0 = suficiente)    |

### Estabilidade (Stability)

| Métrica           | Descrição                                                                               |
|-------------------|-----------------------------------------------------------------------------------------|
| `avg_sensitivity` | Sensibilidade média máxima da explicação a perturbações gaussianas na entrada (↓ = mais estável) |

### Complexidade (Complexity)

| Métrica               | Descrição                                                                              |
|-----------------------|----------------------------------------------------------------------------------------|
| `complexity`          | Entropia média da distribuição normalizada de importâncias (↑ = mais complexo)         |
| `effective_complexity`| Número médio de features com importância acima do limiar (↓ = mais simples)            |

---

## Testes

```bash
# Executar todos os testes
pytest

# Com detalhes
pytest -v

# Apenas um módulo específico
pytest tests/test_metrics.py -v
```

---

## Dependências Principais

- **scikit-learn** – Random Forest, pré-processamento e métricas
- **XGBoost** – Classificador XGBoost
- **SHAP** – Explicações SHAP (*SHapley Additive exPlanations*)
- **LIME** – Explicações LIME (*Local Interpretable Model-agnostic Explanations*)
- **pandas / numpy / scipy** – Manipulação de dados e cálculos numéricos
- **matplotlib / seaborn** – Visualizações

---

## Referências

- Samek, W. et al. (2017). *Evaluating the visualization of what a deep neural network has learned*. IEEE TNNLS.
- Yeh, C.-K. et al. (2019). *On the (in)fidelity and sensitivity of explanations*. NeurIPS.
- Doshi-Velez, F. & Kim, B. (2017). *Towards a rigorous science of interpretable machine learning*. arXiv:1702.08608.
- Lundberg, S. & Lee, S.-I. (2017). *A unified approach to interpreting model predictions*. NeurIPS.
- Ribeiro, M. T. et al. (2016). *"Why should I trust you?": Explaining the predictions of any classifier*. KDD.

---

## Licença

Projeto acadêmico desenvolvido no Departamento de Ciência da Computação (DCC) da Universidade Federal de Minas Gerais (UFMG).
