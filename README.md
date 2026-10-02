# Previsão de Risco de Crédito — Home Credit Default Risk

Projeto de portfólio em Data Science que prevê, no momento da solicitação de um empréstimo, a probabilidade de um cliente enfrentar dificuldades de pagamento no futuro. Esse é o problema proposto pela competição [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk), disponibilizada pela Home Credit Group no Kaggle.

## O problema de negócio

A Home Credit é uma instituição financeira voltada principalmente para clientes com pouco ou nenhum histórico de crédito formal, o público conhecido como "unbanked" ou "underbanked". Prever o risco desses clientes ajuda a instituição a tomar decisões mais informadas sobre aprovação de crédito, equilibrando a inclusão financeira com a gestão de risco.

A métrica oficial de avaliação da competição é o **ROC-AUC**, que mede a capacidade do modelo de ordenar corretamente os clientes por nível de risco, independente do ponto de corte escolhido depois para aprovação ou recusa.

## O dataset

O dataset é composto por sete tabelas relacionadas entre si, cada uma representando uma fonte diferente de informação sobre o cliente:

- **application_train/test**: tabela principal, uma linha por solicitação de empréstimo, com dados demográficos, financeiros e profissionais do solicitante.
- **bureau**: histórico de créditos que o cliente teve em outras instituições financeiras, reportado a um bureau de crédito externo.
- **bureau_balance**: saldo mensal de cada um desses créditos externos.
- **previous_application**: histórico de solicitações anteriores que o próprio cliente já fez à Home Credit.
- **POS_CASH_balance**: saldo mensal de créditos anteriores do tipo parcelado (ponto de venda ou dinheiro) com a Home Credit.
- **credit_card_balance**: saldo mensal de cartões de crédito anteriores com a Home Credit.
- **installments_payments**: histórico de pagamentos de parcelas de créditos anteriores.

A variável alvo é desbalanceada: cerca de 92% dos clientes são classificados como Baixo Risco e 8% como Alto Risco, uma proporção de aproximadamente 11:1.

## Os principais desafios

Essa estrutura relacional com sete tabelas foi o maior desafio do projeto. Diferente de um dataset de uma tabela só, aqui era preciso agregar informações de várias fontes, em diferentes níveis de granularidade (mês a mês, crédito a crédito), até chegar numa única linha por cliente. Isso significou tomar decisões constantes sobre como resumir um histórico inteiro em features que fizessem sentido de negócio, e não só estatístico.

Alguns exemplos dessas decisões ao longo do projeto:

- Separar o status de um crédito em `Aberto`, `Fechado` ou `Prejuízo` com base em evidência real dos dados (comparando dias de atraso entre categorias), em vez de supor pelo nome da categoria.
- Perceber que uma coluna como `CNT_INSTALMENT_MATURE_CUM` era cumulativa, e que por isso não fazia sentido somar ou tirar a média dela ao longo do tempo, só pegar o valor final.
- Calcular uma média ponderada corretamente ao agregar o histórico de meses de um crédito até o nível de cliente, para que um crédito com pouco histórico não pesasse o mesmo que um com anos de dados.
- Construir, a certa altura, uma função de agregação genérica para não precisar repetir manualmente o mesmo tipo de decisão em cada uma das sete tabelas.

Também valeu a pena testar hipóteses antes de assumi-las: uma das investigações do projeto esperava que clientes com múltiplas "versões" de calendário de pagamento fossem piores pagadores (por renegociação de dívida), mas os dados mostraram o oposto, esses clientes pagavam, em média, mais adiantado.

## EDA 

- O dataset apresenta desbalanceamento das classes: ~92% dos clientes classificados com Baixo Risco e ~8% classificados como Alto Risco, uma proporção de ~11:1, o que é normal nesse tipo de situação. A classificação de pagamento dos clientes depende muito do país/região, da condição dos clientes e do quanto a própria instituição está disposta a se expor.
- Pessoas mais jovens tendem a apresentar um risco de crédito maior para a instituição, e a mediana confirma essa tendência. Isso também é um comportamento normal, já que pessoas mais jovens possivelmente ainda estão começando uma carreira e não estão bem estabilizadas profissionalmente.
- Quanto maior o tempo empregado, menor a tendência de risco do cliente. Clientes bem estabilizados profissionalmente apresentam uma tendência de pagamento maior.
- Os scores externos (`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`) mostram claramente que, quanto maior o score do cliente, maior a tendência de ele ser um bom pagador e menor o risco que ele apresenta. Esses scores pesam bastante nas decisões dos modelos.

## Abordagem

1. **Feature Engineering**: agregação das sete tabelas em múltiplas camadas (mês a mês até crédito, crédito até cliente), com tratamento diferenciado de missing values estruturais (ausência de histórico) e missing values genuínos. Criação de features de razão e interação sobre a tabela principal (comprometimento de renda, idade por tempo de emprego, combinações dos scores externos).
2. **Pré-processamento**: tratamento de missing values por faixa de percentual, encoding de categóricas, padronização numérica.
3. **Seleção de features**: redução de dimensionalidade em duas etapas, por importância via modelo de árvore e por poda de colunas altamente correlacionadas entre si.
4. **Modelagem**: comparação entre um modelo de árvore (LightGBM) e uma rede neural (Keras/TensorFlow), com uma bateria de testes de arquitetura e hiperparâmetros documentada ao longo do processo.
5. **Avaliação final**: teste único no conjunto reservado desde o início, análise de matriz de confusão e do trade-off entre precisão e recall em diferentes limiares de decisão.

## Resultados

A arquitetura final da rede neural ficou assim:

```
Input (176 features)
Dense(64, relu) → BatchNorm → Dropout(0.4)
Dense(32, relu) → BatchNorm → Dropout(0.4)
Dense(16, relu) → Dropout(0.3)
Dense(1, sigmoid)
```

Chegar nessa configuração exigiu uma bateria de testes, ajustando um parâmetro de cada vez e comparando o `val_roc_auc` de pico:

| Configuração | val_roc_auc (pico) |
|---|---|
| 128-64-32 (Dropout 0.3/0.3/0.2) | 0.7618 |
| 64-32-16 (Dropout 0.3/0.3/0.2) | 0.7629 |
| 64-32-16 (Dropout 0.35/0.35/0.25) | 0.7634 |
| **64-32-16 (Dropout 0.4/0.4/0.3)** | **0.7652** |
| 64-32-16 (0.4/0.4/0.3) + L2 (0.001) | 0.7623 |
| 64-32-16 (0.4/0.4/0.3) + L2 (0.005) | 0.7559 |
| 64-32-16 (0.4/0.4/0.3) + ReduceLROnPlateau | 0.7642 |
| 64-32-16 (0.4/0.4/0.3) + batch_size 128 | 0.7640 |
| 64-32-16 (0.4/0.4/0.3) + batch_size 512 | 0.7635 |

Reduzir o tamanho da rede e aumentar o Dropout ajudaram a combater o overfitting observado nas primeiras versões. Já a regularização L2 piorou o resultado, sinal de que a combinação de rede menor com Dropout mais forte já era regularização suficiente para esse problema.

**No conjunto de teste final**, nunca usado até essa etapa, o modelo alcançou:

- **ROC-AUC: 0.7705**
- Com limiar de decisão em 0.5: **Recall de ~74,6%** e **Precisão de ~16%** para a classe Alto Risco.
> Importante: Os valores do modelo podem alterar levemente de acordo com a SEED estabelecida no começo do Notebook.

O ROC-AUC não depende do limiar escolhido, já que a métrica avalia o modelo considerando todos os limiares possíveis. O que depende do limiar, e consequentemente da regra de negócio, é o resultado prático da classificação: o quanto a empresa está disposta a recusar clientes classificados como Baixo Risco a fim de se proteger de possíveis clientes de Alto Risco. Essa relação foi explorada através da curva de precisão-recall e da comparação de matrizes de confusão em diferentes limiares, deixando claro que a escolha do ponto de corte ideal é uma decisão de negócio, não uma constante técnica fixa.

## Tecnologias

Python, pandas, numpy, scikit-learn, LightGBM, TensorFlow/Keras, matplotlib, seaborn, networkx.

Neste projeto tive o meu primeiro contato com redes neurais usando TensorFlow/Keras. Foi um processo de aprendizado construído do zero, desde entender a diferença entre montar uma arquitetura e usar um modelo pronto, até compreender na prática o que cada camada, cada callback e cada hiperparâmetro faz no comportamento do treino. Durante a execução do projeto precisei repensar minha própria abordagem no meio do caminho, migrando de um processo de feature engineering inteiramente manual para uma função de agregação mais genérica, justamente por sentir que o volume de tabelas e decisões repetitivas pediam uma solução mais escalável.
