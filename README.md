# Credit Card Clustering - Kaggle

## Objetivo

Aplicar técnicas de clusterização hierárquica e não hierárquica para segmentação de clientes de cartão de crédito. Baseado no dataset Credit Card Dataset for Clustering do Arjun Bhasin disponibilizado no kaggle (https://www.kaggle.com/datasets/arjunbhasin2013/ccdata).

## Modelos utilizados

- KMeans
- Hierarchical Clustering

## Métricas

- Silhouette Score
- Elbow method
## Bibliotecas principais

- pandas
- numpy
- scikit-learn
- matplotlib
- scipy
- pingouin

## Estrutura do Projeto

- data/
- notebooks/
- reports/
- src/

## Problema de negócio
Segmentar clientes de cartão de crédito com base no comportamento de uso (gastos, cash advance, pagamentos, limites, frequência etc.) para apoiar decisões de negócio (campanhas, risco, retenção, limites, benefícios).

- Pipeline reprodutível (EDA → preparação → modelagem → avaliação → interpretação).
- Comparação entre K-Means e Hierárquico.
- Perfis de segmentos (personas/descrições) e recomendações de ação.

## Dataset
O conjunto de dados resume o comportamento de uso de cerca de 9.000 titulares de cartões de crédito ativos durante os últimos 6 meses. O arquivo está em nível de cliente com 18 variáveis ​​comportamentais.

## Análise exploratória de Dados (EDA)

- Estrutura Geral do Dataset:
17 variáveis numéricas (após remover CUST_ID). Apenas 2 variáveis com missing.
Dataset limpo, sem problema sério de completude.
Não há duplicados.
A base é adequada para modelagem não supervisionada.

- Assimetria Extrema, Outliers e Escala:
Variáveis extremamente assimétricas (MINIMUM_PAYMENTS; ONEOFF_PURCHASES; PURCHASES; INSTALLMENTS_PURCHASES;PAYMENTS; CASH_ADVANCE_TRX; CASH_ADVANCE).
Distribuição heavy-tailed. Concentração massiva próxima de zero e poucos clientes com valores muito altos.
Risco de clusters serem formados por magnitude financeira extrema e não por padrão comportamental.
Sem padronização Clusters serão definidos apenas por limite e pagamentos e Frequência será irrelevante, então a padronização é obrigatória.

Obs:
- Existem subestruturas latentes naturais.

## Pré-processamento dos Dados

- creditcard_risk_zscore: Risco - Identificar bons e maus pagadores.
- creditcard_credit_behavior_zscore: Comportamento de Crédito - Identificar exposição e dependência de crédito.
- creditcard_consumption_zscore: Padrão de Consumo - Identificar padrão de consumo.
- creditcard_full_zscore: Dataset Completo - Clusterização global exploratória.

Encapsulamento: preprocessing.py
