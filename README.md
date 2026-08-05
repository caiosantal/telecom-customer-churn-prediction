# 📶 Telecom Churn - Projeto de Ciência de Dados

Classificação e previsão de cancelamento de clientes (*Churn*) em telecomunicações — um projeto completo de machine learning focado na retenção proativa, cobrindo análise exploratória, tratamento do desbalanceamento de dados e comparação de modelos.

---

## 📖 Sobre o Projeto

O *Churn* (cancelamento de serviço) é um dos maiores desafios estratégicos em empresas de telecomunicações. Adquirir um novo cliente pode custar até 5 vezes mais do que reter um já existente. Este projeto utiliza dados reais de clientes para construir um pipeline de Ciência de Dados capaz de identificar antecipadamente perfis de alto risco.

O objetivo é transformar análises preditivas em ações operacionais de retenção, fornecendo diagnósticos claros sobre os fatores que impulsionam o cancelamento.

---

## 🎯 Objetivo

* **Prever o risco de Churn:** Construir um classificador para prever se um cliente irá cancelar o serviço no próximo trimestre.
* **Maximizar o Recall:** Identificar a maioria dos clientes em risco de cancelamento (Classe 1), reduzindo falsos negativos.
* **Tratar o Desbalanceamento de Dados:** Comparar técnicas como pesos de classe (`class_weight='balanced'`) e ajuste de limiar de decisão (*decision threshold*).
* **Entregar Valor ao Negócio:** Identificar os principais ofensores operacionais (ex: tipos de contrato, meios de pagamento e planos).

---

## 📁 Estrutura do Projeto

O projeto está organizado nas seguintes etapas:

1. **Formulação do problema** – Definição dos objetivos de negócio e escolha das métricas principais (*Recall*, *ROC AUC*).
2. **Leitura e compreensão dos dados** – Mapeamento inicial das 21 colunas brutas e 7.043 registros.
3. **Limpeza e Pré-processamento:**
   * Remoção de colunas não preditivas ou com risco de *data leakage* (`customerID`, `gender`, `TotalCharges`).
   * Verificação e tratamento de dados ausentes.
   * *One-Hot Encoding* (`pd.get_dummies`) para variáveis categóricas.
   * Padronização de variáveis contínuas com `StandardScaler` (`tenure`, `MonthlyCharges`).
4. **Análise Exploratória (EDA):**
   * Análise de distribuição das variáveis numéricas contínuas.
   * Cálculo da Correlação de Pearson em relação à variável alvo `Churn_Yes`.
   * Matriz de correlação cruzada (*Heatmap*).
5. **Modelagem e Avaliação:**
   * **Regressão Logística (Baseline):** Configuração padrão (limiar 0.50).
   * **Regressão Logística (`balanced`):** Ajuste de pesos por classe desbalanceada.
   * **Regressão Logística (Limiar Customizado - 35%):** Calibração operacional para contenção de alarmes falsos.
   * **Random Forest (`balanced`):** Modelo *ensemble* para capturar relações não lineares complexas.
6. **Conclusão e Diagnóstico de Negócio** – Interpretação dos resultados operacionais.

---

## 🛠️ Pré-requisitos e Instalação

Para executar o notebook localmente ou no Google Colab, você precisará das seguintes bibliotecas:

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
