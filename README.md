# Autoencoder coupled with a region mask

## Relatório de Andamento

### Dia 24 de janeiro de 2024

* Objetivo:

1. Verificar se a rede consegue realizar um noise removal dos dados de entrada
2. Testar a regressão dos mesmos dados de entrada
3. Testar utilizando somente MSE e MAE pra avaliação inicial, registrar os resultados.

* Testes realizados:

0. Teste #1:
batch_size=64
loss=mse
metric=mse
out=mask+fecg

Mesmo utilizando o scheduler, quando usamos mais arquivos a loss aumenta e não vê-se aprendizado ao longo do tempo (mesmo fazendo a troca de LR_INIT). A loss caiu de 0.49 para 0.46 ao longo de 20 épocas de treinamento.

1. Teste #2
batch_size=32
loss=mse
metric=mse
out=mask+fecg

2. Máscara como entrada:
batch_size=32
loss=mse
metric=mse
out=fecg

3. Máscara como entrada, loss modificada:
batch_size=32
loss=custom
metric=mse
out=fecg


* Resultados obtidos: 

* Ações a serem realizadas: