
# Previsão de Tempos de Volta na F1 - Script `predict.py`

## Objetivo
Este script visa prever os tempos de volta de pilotos da Fórmula 1 (especificamente Verstappen e Norris na corrida de 2025) utilizando dados de sessões anteriores como FP1, FP2, FP3 e Qualifying. Ele emprega modelos de machine learning combinados com modelagem de resíduos via ARIMA para melhorar a acurácia das previsões.

---

##  Bibliotecas Utilizadas

- `fastf1`: Carregamento de dados de sessões da Fórmula 1.
- `pandas`, `numpy`: Manipulação de dados.
- `matplotlib.pyplot`: Visualizações.
- `sklearn`:
  - `RandomForestRegressor`, `GradientBoostingRegressor`, `VotingRegressor`, `Ridge`: Modelos de regressão.
  - `StandardScaler`, `Pipeline`: Pré-processamento.
  - `TimeSeriesSplit`, `mean_absolute_error`, `mean_squared_error`, `r2_score`: Validação e métricas.
- `statsmodels.tsa.arima.model.ARIMA`: Modelagem de resíduos.

---

##  Fluxo Principal do Script

### 1. Carregamento de Sessões
Função: `load_sessions()`  
Carrega sessões FP1, FP2, FP3, Qualifying e Corrida do GP da Austrália 2025 via `fastf1`.

### 2. Enriquecimento de Features
Função: `extract_enhanced_features(laps)`  
Gera variáveis derivadas para melhorar o poder preditivo, como:
- `LapTimeSec`, `TyreLife`, `CompoundValue`, `FuelLoad`, `Avg3Laps`, entre outras.

### 3. Criação de Modelo Ensemble Melhorado
Função: `create_improved_model()`  
Modelo de regressão composto com pesos otimizados usando:
- Random Forest
- Gradient Boosting
- Ridge Regression

### 4. Treinamento com Validação Temporal
Função: `treinar_modelo_com_validacao_temporal(X, y)`  
Aplica validação cruzada em séries temporais (`TimeSeriesSplit`) e treina modelo final com todos os dados.

### 5. Modelos Específicos por Piloto
Função: `criar_modelos_por_piloto(sessions)`  
Cria modelos personalizados para cada piloto com dados históricos.

### 6. Previsão Combinada
Função: `prever_com_modelos_combinados(X, piloto, modelo_geral, modelos_pilotos)`  
Combina modelo geral e modelo específico ponderado pelo número de voltas disponíveis.

### 7. Modelagem de Resíduos com ARIMA
Função: `modelar_residuos_avancado(residuos)`  
Modela resíduos das previsões com ARIMA para capturar padrões não lineares restantes.

### 8. Previsão para Voltas Futuras
Função: `criar_futuras_voltas_melhoradas(X_base, tempo, lap)`  
Gera 15 voltas futuras com evolução do stint, combustível e outras features.

### 9. Execução Final
Função: `executar_analise_melhorada()`  
Executa todo o fluxo anterior:
- Treinamento dos modelos
- Previsões para Verstappen e Norris
- Visualizações dos tempos e gap acumulado

---

##  Métricas Apresentadas

- **MAE (Erro Absoluto Médio)**
- **MAPE (Erro Percentual Absoluto Médio)**
- **Acurácia (%)**
- Visualização do tempo de volta futuro e gap acumulado

---

##  Saídas Visuais

- Gráfico 1: Previsão de tempo de volta para próximas 15 voltas.
- Gráfico 2: Gap acumulado entre Verstappen e Norris.
- Mensagem final: previsão de ultrapassagem (se ocorrer).

---

##  Requisitos de Entrada

- Dados da biblioteca `fastf1` com colunas como: `LapTime`, `LapNumber`, `Compound`, `TyreLife`, `Driver`, `Stint`, etc.

---
