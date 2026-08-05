# 📶 Telecom Churn - Projeto de Ciência de Dados

Classificação e previsão de cancelamento de clientes (*Churn*) em telecomunicações — um projeto completo de machine learning focado na retenção de clientes, cobrindo análise exploratória, tratamento do desbalanceamento de dados e comparação de modelos.

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

1. **Formulação do Problema e Leitura dos Dados:** Definição dos objetivos de negócio e mapeamento inicial do dataset (21 colunas brutas e 7.043 registros).
2. **Limpeza e Pré-processamento:**
   * Remoção de colunas não preditivas ou com risco de *data leakage* (`customerID`, `gender`, `TotalCharges`).
   * Verificação e confirmação de ausência de dados nulos.
   * Aplicação de *One-Hot Encoding* (`pd.get_dummies`) para conversão de variáveis categóricas em binárias.
3. **Divisão dos Dados (Treino e Teste):** Divisão da base em 80% para treinamento e 20% para teste, utilizando amostragem estratificada (`stratify=y`) para preservar a proporção da classe alvo.
4. **Normalização (Escalonamento):** Padronização das variáveis numéricas contínuas (`tenure` e `MonthlyCharges`) via `StandardScaler`, ajustando o *scaler* exclusivamente nos dados de treino.
5. **Análise Descritiva e Correlação:**
   * Análise de distribuição das variáveis numéricas (*tenure* e *MonthlyCharges*).
   * Cálculo da Correlação de Pearson individual em relação à variável alvo (`Churn_Yes`).
   * Visualização da Matriz de Correlação Cruzada (*Heatmap*).
6. **Modelagem e Avaliação:**
   * **Regressão Logística (Baseline):** Configuração padrão com limiar de decisão em 0.50.
   * **Regressão Logística (`balanced`):** Penalização por pesos de classe para compensar o desbalanceamento dos dados.
   * **Regressão Logística (Limiar Customizado - 35%):** Reajuste do limiar operacional para redução de falsos negativos.
   * **Random Forest (`balanced`):** Algoritmo *ensemble* não linear para capturar interações complexas entre atributos.
7. **Conclusão e Diagnóstico de Negócio:** Avaliação comparativa das métricas (*Recall* e *ROC AUC*) e recomendação de ações operacionais para retenção de clientes.

---

## 📂 Dados

1. Faça o download dos arquivos **`train.csv`** e **`test.csv`** disponíveis na página da competição no [Kaggle](https://www.kaggle.com/competitions/titanic/data).
2. Salve os arquivos em uma pasta no seu **Google Drive**.
* **Estrutura recomendada no Drive:** `dados/titanic/` (ficando no caminho `/content/drive/My Drive/dados/titanic/`).
3. **Atenção:** Se optar por salvar em um diretório diferente, lembre-se de atualizar os caminhos no notebook onde os arquivos são lidos e salvos:
* **Leitura dos dados:** `pd.read_csv(...)`
* **Exportação dos resultados:** `.to_csv(...)`

---

## 🛠️ Pré-requisitos e Instalação

Para executar o notebook localmente ou no Google Colab, você precisará das seguintes bibliotecas:

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn

````

---

## 👤 Autor
Caio Santos de Almeida

📧 caiosantal.cd@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/caiosantal/)
🐙 [GitHub](https://github.com/caiosantal)
