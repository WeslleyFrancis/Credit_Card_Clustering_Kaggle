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
