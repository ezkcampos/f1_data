import fastf1
import os
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# 1. ENRIQUECIMENTO DE FEATURES

def load_sessions():
    sessions = {}
    for sess in ['FP1', 'FP2', 'FP3', 'Q', 'R']:
        print(f"Carregando {sess}...")
        session = fastf1.get_session(2025, 'australia', sess)
        session.load()
        sessions[sess] = session
    return sessions




def extract_enhanced_features(laps):
    laps = laps.copy()
    laps['LapTimeSec'] = laps['LapTime'].dt.total_seconds()

    # Features básicas
    features = ['LapNumber', 'Stint']

    # Tratamento de features existentes
    if 'TyreLife' in laps.columns:
        laps['TyreLife'] = laps['TyreLife'].fillna(0)
        features.append('TyreLife')

        # Feature nova: degradação quadrática dos pneus
        laps['TyreLifeSquared'] = laps['TyreLife'] ** 2
        features.append('TyreLifeSquared')

    if 'Compound' in laps.columns:
        # Encoding mais informativo dos compostos
        compound_mapping = {
            'SOFT': 1.0,
            'MEDIUM': 0.7,
            'HARD': 0.4,
            'INTERMEDIATE': 0.3,
            'WET': 0.2
        }
        laps['CompoundValue'] = laps['Compound'].map(compound_mapping).fillna(0.5)
        features.append('CompoundValue')

    # Histórico de tempos
    laps['PrevLapTime'] = laps.groupby('Driver')['LapTimeSec'].shift(1).fillna(0)
    features.append('PrevLapTime')

    # Média móvel das últimas 3 voltas (tendência de curto prazo)
    laps['Avg3Laps'] = laps.groupby('Driver')['LapTimeSec'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    features.append('Avg3Laps')

    # Diferença entre tempo atual e média das 3 últimas voltas
    laps['DiffFromAvg'] = laps['LapTimeSec'] - laps['Avg3Laps']
    features.append('DiffFromAvg')

    # Combustível com modelagem não-linear
    total_laps = laps['LapNumber'].max()
    laps['FuelLoad'] = 1 - (laps['LapNumber'] / total_laps)
    features.append('FuelLoad')

    # Efeito quadrático do combustível (maior no início)
    laps['FuelLoadSquared'] = laps['FuelLoad'] ** 2
    features.append('FuelLoadSquared')

    # Feature de posição na pista (se disponível)
    if 'Position' in laps.columns:
        laps['Position'] = laps['Position'].fillna(laps['Position'].median())
        features.append('Position')

    # Features de tempo (se disponível) - CORRIGIDO
    if 'Time' in laps.columns:
        # Verificar se é datetime ou timedelta
        if hasattr(laps['Time'].dt, 'hour'):
            # É datetime
            laps['TimeFeature'] = laps['Time'].dt.hour + laps['Time'].dt.minute / 60
        else:
            # É timedelta - converter para segundos desde o início
            laps['TimeFeature'] = laps['Time'].dt.total_seconds() / 3600  # Em horas
        features.append('TimeFeature')

    # Interações entre features (TyreLife × Compound)
    if 'TyreLife' in laps.columns and 'CompoundValue' in laps.columns:
        laps['TyreLifeByCompound'] = laps['TyreLife'] * laps['CompoundValue']
        features.append('TyreLifeByCompound')

    X = laps[features].copy()
    for col in X.columns:
        X[col] = X[col].fillna(X[col].mean() if not X[col].isna().all() else 0)

    y = laps['LapTimeSec']
    return X, y


# 2. MODELO ENSEMBLE MELHORADO
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge


def create_improved_model():
    # Preprocessamento
    scaler = StandardScaler()

    # Modelos base aprimorados
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    gb = GradientBoostingRegressor(
        n_estimators=600,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )

    # Versão sem XGBoost e LightGBM para evitar dependências extras
    ridge = Ridge(alpha=0.5, random_state=42)

    # Ensemble com pesos otimizados
    ensemble = VotingRegressor([
        ('rf', rf),
        ('gb', gb),
        ('ridge', ridge)
    ], weights=[0.4, 0.5, 0.1])

    # Pipeline com preprocessamento
    pipeline = Pipeline([
        ('scaler', scaler),
        ('ensemble', ensemble)
    ])

    return pipeline


# 3. TRATAMENTO ESPECÍFICO PARA SÉRIES TEMPORAIS

def treinar_modelo_com_validacao_temporal(X, y):
    # Time Series Cross Validation
    tscv = TimeSeriesSplit(n_splits=5)

    model = create_improved_model()

    # Armazenar métricas de cada fold
    maes = []
    mapes = []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)

        mae = mean_absolute_error(y_val, preds)
        mape = np.mean(np.abs((y_val - preds) / y_val)) * 100

        maes.append(mae)
        mapes.append(mape)

    print(f"Validação temporal - MAE médio: {np.mean(maes):.3f}s")
    print(f"Validação temporal - MAPE médio: {np.mean(mapes):.2f}%")

    # Treinar modelo final com todos os dados
    model.fit(X, y)

    return model


# 4. AJUSTE ESPECÍFICO PARA CADA PILOTO

def criar_modelos_por_piloto(sessions):
    modelos_pilotos = {}

    # Identificar pilotos únicos
    todos_pilotos = set()
    for sess_name, sess in sessions.items():
        if sess_name != 'R':  # Excluir corrida
            pilotos = sess.laps['Driver'].unique()
            todos_pilotos.update(pilotos)

    print(f"Treinando modelos específicos para {len(todos_pilotos)} pilotos...")

    # Criar modelo específico para cada piloto
    for piloto in todos_pilotos:
        # Extrair voltas apenas deste piloto
        laps_piloto = pd.concat([
            s.laps.pick_drivers([piloto]).pick_not_deleted().dropna(subset=['LapTime'])
            for k, s in sessions.items() if k != 'R'
        ])

        if len(laps_piloto) < 10:  # Ignorar se houver poucos dados
            continue

        X_piloto, y_piloto = extract_enhanced_features(laps_piloto)

        # Treinar modelo específico
        modelo_piloto = create_improved_model()
        modelo_piloto.fit(X_piloto, y_piloto)

        modelos_pilotos[piloto] = {
            'model': modelo_piloto,
            'n_laps': len(laps_piloto)
        }

    return modelos_pilotos


# 5. PREVISÃO COMBINADA (MODELO GERAL + MODELO ESPECÍFICO DO PILOTO)

def prever_com_modelos_combinados(X_piloto, piloto, modelo_geral, modelos_pilotos):
    # Previsão do modelo geral
    pred_geral = modelo_geral.predict(X_piloto)

    # Se existe modelo específico para o piloto, faça previsão combinada
    if piloto in modelos_pilotos:
        modelo_piloto = modelos_pilotos[piloto]['model']
        n_laps = modelos_pilotos[piloto]['n_laps']

        # Previsão do modelo específico
        pred_piloto = modelo_piloto.predict(X_piloto)

        # Peso do modelo específico baseado na quantidade de dados
        # Quanto mais voltas, maior o peso do modelo específico
        peso_especifico = min(0.7, n_laps / 100)  # Máximo de 70% de peso


        pred_final = (1 - peso_especifico) * pred_geral + peso_especifico * pred_piloto

        return pred_final
    else:
        return pred_geral


# 6. MELHORIA NA MODELAGEM DE RESÍDUOS

def modelar_residuos_avancado(residuos, ordem_base=(2, 1, 2)):
    try:
        modelo = ARIMA(residuos, order=ordem_base).fit()
        return modelo.forecast(steps=15).values
    except:
        try:
            modelo = ARIMA(residuos, order=(1, 0, 1)).fit()
            return modelo.forecast(steps=15).values
        except:
            return np.full(15, residuos.mean())




def executar_analise_melhorada():
    # 1. Carregar sessões
    sessions = load_sessions()
    race = sessions['R']

    # 2. Extrair voltas melhoradas de todos os pilotos
    laps_all = pd.concat([
        s.laps.pick_not_deleted().dropna(subset=['LapTime'])
        for k, s in sessions.items() if k != 'R'
    ])
    X_all, y_all = extract_enhanced_features(laps_all)

    # 3. Treinar modelo geral melhorado
    print("Treinando modelo geral melhorado...")
    modelo_geral = treinar_modelo_com_validacao_temporal(X_all, y_all)

    # 4. Treinar modelos específicos por piloto
    print("Treinando modelos específicos por piloto...")
    modelos_pilotos = criar_modelos_por_piloto(sessions)

    # 5. Carregar voltas de Verstappen e Norris da corrida
    laps_ver = race.laps.pick_drivers(['VER']).pick_not_deleted().dropna(subset=['LapTime'])
    laps_nor = race.laps.pick_drivers(['NOR']).pick_not_deleted().dropna(subset=['LapTime'])

    laps_ver['LapTimeSec'] = laps_ver['LapTime'].dt.total_seconds()
    laps_nor['LapTimeSec'] = laps_nor['LapTime'].dt.total_seconds()

    # 6. Extrair features melhoradas
    X_ver, y_ver = extract_enhanced_features(laps_ver)
    X_nor, y_nor = extract_enhanced_features(laps_nor)


    y_ver_pred_geral = modelo_geral.predict(X_ver)
    y_nor_pred_geral = modelo_geral.predict(X_nor)


    y_ver_pred = prever_com_modelos_combinados(X_ver, 'VER', modelo_geral, modelos_pilotos)
    y_nor_pred = prever_com_modelos_combinados(X_nor, 'NOR', modelo_geral, modelos_pilotos)

    # Calcular métricas
    mae_ver = mean_absolute_error(y_ver, y_ver_pred)
    mape_ver = np.mean(np.abs((y_ver - y_ver_pred) / y_ver)) * 100
    accuracy_ver = 100 - mape_ver

    mae_nor = mean_absolute_error(y_nor, y_nor_pred)
    mape_nor = np.mean(np.abs((y_nor - y_nor_pred) / y_nor)) * 100
    accuracy_nor = 100 - mape_nor

    mae_medio = (mae_ver + mae_nor) / 2
    mape_medio = (mape_ver + mape_nor) / 2
    accuracy_media = (accuracy_ver + accuracy_nor) / 2

    print("\n===== MÉTRICAS DE DESEMPENHO COM MODELO MELHORADO =====")
    print(f"Verstappen - MAE: {mae_ver:.3f} segundos | MAPE: {mape_ver:.2f}% | Acurácia: {accuracy_ver:.2f}%")
    print(f"Norris - MAE: {mae_nor:.3f} segundos | MAPE: {mape_nor:.2f}% | Acurácia: {accuracy_nor:.2f}%")
    print(f"Média - MAE: {mae_medio:.3f} segundos | MAPE: {mape_medio:.2f}% | Acurácia: {accuracy_media:.2f}%")

    # 8. Gerar previsões para voltas futuras
    last_lap = int(max(laps_ver['LapNumber'].max(), laps_nor['LapNumber'].max()))
    ver_last_time = laps_ver[laps_ver['LapNumber'] == last_lap]['LapTimeSec'].iloc[0]
    nor_last_time = laps_nor[laps_nor['LapNumber'] == last_lap]['LapTimeSec'].iloc[0]
    gap = 0.895  # gap atual

    # Criar dados futuros com features melhoradas
    def criar_futuras_voltas_melhoradas(X_base, ultimo_tempo, ultimo_lap):
        futuras = []
        features_base = X_base.iloc[-1].copy()

        # Armazenar último tempo para usar como PrevLapTime
        tempo_anterior = ultimo_tempo

        for i in range(1, 16):
            nova = features_base.copy()
            nova['LapNumber'] = ultimo_lap + i
            nova['PrevLapTime'] = tempo_anterior

            if 'TyreLife' in nova:
                nova['TyreLife'] += i
                if 'TyreLifeSquared' in nova:
                    nova['TyreLifeSquared'] = nova['TyreLife'] ** 2

            # Atualizar combustível
            total_laps = 58  # Aproximação do total de voltas
            nova['FuelLoad'] = max(0, 1 - ((ultimo_lap + i) / total_laps))
            if 'FuelLoadSquared' in nova:
                nova['FuelLoadSquared'] = nova['FuelLoad'] ** 2

            # Atualizar média móvel se existir
            if 'Avg3Laps' in nova:
                nova['Avg3Laps'] = tempo_anterior  # Simplificação

            if 'DiffFromAvg' in nova:
                nova['DiffFromAvg'] = 0  # Simplificação

            # Se tiver interação TyreLife × Compound
            if 'TyreLifeByCompound' in nova and 'TyreLife' in nova and 'CompoundValue' in nova:
                nova['TyreLifeByCompound'] = nova['TyreLife'] * nova['CompoundValue']

            futuras.append(nova)
            tempo_anterior = tempo_anterior  # Poderia ser atualizado com previsão

        return pd.DataFrame(futuras)

    X_ver_future = criar_futuras_voltas_melhoradas(X_ver, ver_last_time, last_lap)
    X_nor_future = criar_futuras_voltas_melhoradas(X_nor, nor_last_time, last_lap)

    # Prever com modelos combinados
    ver_preds = prever_com_modelos_combinados(X_ver_future, 'VER', modelo_geral, modelos_pilotos)
    nor_preds = prever_com_modelos_combinados(X_nor_future, 'NOR', modelo_geral, modelos_pilotos)

    # 9. Modelagem avançada de resíduos
    res_ver = y_ver - y_ver_pred
    res_nor = y_nor - y_nor_pred

    ver_resid_forecast = modelar_residuos_avancado(res_ver)
    nor_resid_forecast = modelar_residuos_avancado(res_nor)

    ver_final = ver_preds + ver_resid_forecast
    nor_final = nor_preds + nor_resid_forecast

    # 10. Resultados e visualização
    df = pd.DataFrame({
        'Lap': np.arange(last_lap + 1, last_lap + 16),
        'VER_Tempo': ver_final,
        'NOR_Tempo': nor_final,
    })

    df['Diferença'] = df['VER_Tempo'] - df['NOR_Tempo']
    df['Gap_Acumulado'] = gap + df['Diferença'].cumsum()

    print("\n===== PREVISÃO DE TEMPOS PARA AS PRÓXIMAS 15 VOLTAS (MODELO MELHORADO) =====")
    print(df.round(3))

    if any(df['Gap_Acumulado'] < 0):
        volta = df[df['Gap_Acumulado'] < 0].iloc[0]['Lap']
        print(f"\nPrevisão: Verstappen ultrapassaria Norris na volta {int(volta)}")
    else:
        min_gap = df['Gap_Acumulado'].min()
        print(f"\nPrevisão: Verstappen NÃO ultrapassaria Norris. Gap mínimo: {min_gap:.3f}s")

    # Gráficos aprimorados
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    plt.plot(df['Lap'], df['VER_Tempo'], 'b-', label='Verstappen', marker='o')
    plt.plot(df['Lap'], df['NOR_Tempo'], 'o-', label='Norris', marker='o', color='orange')
    plt.ylabel('Tempo de Volta (s)')
    plt.title('Previsão de Tempos de Volta')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df['Lap'], df['Gap_Acumulado'], 'r-', marker='o')
    plt.axhline(0, color='green', linestyle='--')
    plt.ylabel('Gap Acumulado (s)')
    plt.xlabel('Volta')
    plt.title('Gap Acumulado (Negativo = VER à frente)')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    return modelo_geral, modelos_pilotos, df



modelo_geral, modelos_pilotos, resultados = executar_analise_melhorada()