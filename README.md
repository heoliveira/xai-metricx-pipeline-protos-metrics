# xai-metricx-pipeline-protos-metrics
METRICX: Uma estratégia para seleção de métricas de interpretabilidade

Descrição
-----------
METRICX reúne notebooks e scripts para executar experimentos de classificação e avaliar métodos post-hoc de interpretabilidade (LIME, SHAP, Anchor, PFI, Surrogate Trees). O repositório contém pipelines para treinar classificadores, gerar previsões e calcular métricas quantitativas que suportam comparações entre métodos e configurações.

Este repositório corresponde a uma versão preliminar dos experimentos desenvolvidos no contexto da dissertação de mestrado em Ciência da Computação no DCC/UFMG. O objetivo é disponibilizar os pipelines experimentais, métodos de interpretabilidade e métricas utilizadas para avaliação quantitativa de explicações geradas por modelos de aprendizado de máquina.

Datasets incluídos
------------------
- FHS
- SEPSIS
- STROKE
- UCI-THYROID-DXBIN
- WDBC

Classificadores disponíveis
---------------------------
- Random Forest
- XGBoost

Métodos de interpretabilidade avaliados
---------------------------------------
Os experimentos utilizam diferentes técnicas post-hoc de interpretabilidade aplicadas aos modelos de classificação:

- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Anchor
- Permutation Feature Importance (PFI)
- Surrogate Decision Trees

Métricas de desempenho dos modelos
----------------------------------
As seguintes métricas são utilizadas para avaliar o desempenho dos classificadores utilizados nos experimentos:

1. Acurácia (Accuracy)
2. Acurácia Balanceada (Balanced Accuracy)
3. Precisão (Precision)
4. Revocação / Sensibilidade (Recall / Sensitivity)
5. F1-score
6. Especificidade (Specificity)
7. AUROC — ROC AUC (One-vs-Rest macro)
8. Área sob Precision-Recall (Average Precision / PR AUC)
9. Log Loss / Cross-Entropy (erro logarítmico)
10. Cross-Entropy por classe (quando aplicável)
11. Contagens da matriz de confusão: TP / FP / TN / FN
12. Taxas derivadas da matriz de confusão (TPR, TNR, FPR, FNR — em %)

Métricas de interpretabilidade avaliadas
----------------------------------------
A estratégia de avaliação utiliza, naturalmente, as métricas internas de desempenho dos modelos, sendo elas essenciais para nos calculos das métricas quantitativas de interpretabilidade . Entre as métricas avaliadas estão:

- Fidelity
- Faithfulness
- Infidelity
- Completeness
- Consistency
- Coverage
- Robustness
- Stability
- Simplicity
- Selectivity
- Sufficiency
- Directional Soundness

Essas permitem avaliar propriedades importantes das explicações geradas, como fidelidade ao modelo original, robustez a pequenas perturbações, estabilidade das explicações e capacidade de representar adequadamente o comportamento do modelo.

Análise estatística dos resultados
----------------------------------
Para comparar os métodos de interpretabilidade e analisar diferenças estatisticamente significativas entre os resultados obtidos nos diferentes datasets, foram utilizados procedimentos estatísticos não paramétricos amplamente empregados em avaliação comparativa de algoritmos de aprendizado de máquina.

As análises incluem:

- **Teste de Friedman**: utilizado para verificar se existem diferenças estatisticamente significativas entre múltiplos métodos avaliados em diferentes datasets ou configurações experimentais.

- **Teste pós-hoc de Nemenyi**: aplicado após o teste de Friedman quando diferenças significativas são detectadas, permitindo identificar quais pares de métodos apresentam diferenças estatisticamente relevantes.

- **Correlação de Spearman**: utilizada para analisar relações monotônicas entre métricas quantitativas de interpretabilidade e métricas de desempenho dos modelos, permitindo investigar possíveis associações entre qualidade explicativa e desempenho preditivo.

Essas análises permitem avaliar não apenas o desempenho individual dos métodos, mas também a consistência e a relação entre diferentes métricas utilizadas no estudo.

Essas análises são implementadas nos notebooks localizados no diretório `statistical_evaluation`.

Portabilidade
-------------
Para tornar o repositório portátil, todas as referências a caminhos absolutos foram substituídas por caminhos relativos baseados na estrutura das pastas do repositório (uso de `Path('../...')` nos notebooks). Também foram limpas execuções/outputs que continham caminhos do ambiente local. Nenhuma lógica experimental foi alterada — apenas caminhos e metadados de saída.

Como usar
--------
- Instale dependências listadas em `requirements.txt` nas pastas de cada dataset quando necessário.
- Abra os notebooks em `/*/notebooks/` (por exemplo, `WDBC/notebooks/`), ajuste kernels se necessário e execute as células para reproduzir os experimentos.

Contribuições
-------------
Pull requests são bem-vindos. Para alterações que afetem reprodutibilidade, favor manter caminhos relativos e não alterar a lógica de experimentos.