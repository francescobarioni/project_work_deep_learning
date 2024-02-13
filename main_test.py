import librerie as lib
import utils

def main():
    # carico il dataset
    df = lib.pd.read_csv('precipitazioni Ferrara e limitrofi.csv',sep=';')

    # controllo se ci sono valori mancanti nel dataset
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0: # se ci sono valori mancanti
        print('Ci sono valori mancanti nel dataset:')
        print(str(missing_values) + '\n')
        # calcolo la percentuale di valori mancanti per colonna
        missing_percentage = (missing_values / len(df)) * 100
        print('Percentuale di valori mancanti per colonna:')
        print(missing_percentage.astype(str))
        # Creazione del grafico a barre orizzontali
        lib.plt.figure(figsize=(10, 6))
        missing_percentage.plot(kind='barh', color='skyblue')
        lib.plt.xlabel('Percentuale di dati mancanti', fontsize=12)
        lib.plt.ylabel('Colonna', fontsize=12)
        lib.plt.title('Percentuale di dati mancanti per colonna', fontsize=16)
        lib.plt.gca().invert_yaxis()  # Inverto l'ordine delle colonne
        #lib.plt.show()
    elif missing_values.sum() == 0: # se non ci sono valori mancanti
        print('Non ci sono valori mancanti nel dataset\n')

    # Pre-processing del "Forecast timestamp"
    # non ci sono dati mancanti
    df['Forecast timestamp'] = lib.pd.to_datetime(df['Forecast timestamp'])
    df.sort_values(by='Forecast timestamp', inplace=True)

    # Pre-processing della "Position"
    # non ci sono dati mancanti
    df[['Latitudine', 'Longitudine']] = df['Position'].str.split(',', expand=True).astype(float)
    serie_latitudine = df['Latitudine'] # serie separata per le latitudini
    serie_longitudine = df['Longitudine'] # serie separata per le longitudini

    # Pre-processing della "Temperature"
    # non ci sono dati mancanti
    utils.shapiro_wilk_test(df['Temperature'], "temperatura") # test di Shapiro-Wilk per la temperatura
    utils.kolmogorov_smirnov_test(df['Temperature'], "temperatura") # test di Kolmogorov-Smirnov per la temperatura
    # dai 2 test precedenti si ricava che la distribuzione della temperatura non è normale >> non conviene applicare la z-score
    # applico quindi la normalizzazione min-max
    min_temp = df['Temperature'].min()
    max_temp = df['Temperature'].max()
    df['Normalized_Temperature'] = (df['Temperature'] - min_temp) / (max_temp - min_temp)
    #utils.plot_normalized_data(df['Temperature'], df['Normalized_Temperature'], 'Temperature') # grafico per visualizzare la normalizzazione min-max

    # Pre-processing della "2 metre temperature"
    # non ci sono dati mancanti
    utils.shapiro_wilk_test(df['2 metre temperature'], "temperatura a 2 metri") # test di Shapiro-Wilk per la temperatura a 2 metri
    utils.kolmogorov_smirnov_test(df['2 metre temperature'], "temperatura a 2 metri") # test di Kolmogorov-Smirnov per la temperatura a 2 metri
    # dai 2 test precedenti si ricava che la distribuzione della 2 metre temperature non è normale >> non conviene applicare la z-score
    # applico quindi la normalizzazione min-max
    min_temp_2m = df['2 metre temperature'].min()
    max_temp_2m = df['2 metre temperature'].max()
    df['Normalized_2m_Temperature'] = (df['2 metre temperature'] - min_temp_2m) / (max_temp_2m - min_temp_2m)
    #utils.plot_normalized_data(df['2 metre temperature'], df['Normalized_2m_Temperature'], '2 Metre Temperature') # grafico per visualizzare la normalizzazione min-max

    # Pre-processing della "Total Precipitation"
    # dai test sui dati mancanti si osserva che in questo dato sono presenti valori mancanti
    # applico l'imputazione con la mediana (media sensibile ai valori di outlier)
    median_value = df['Total Precipitation'].median()
    df['Total Precipitation'] = df['Total Precipitation'].fillna(median_value)
    utils.shapiro_wilk_test(df['Total Precipitation'], "precipitazione totale") # test di Shapiro-Wilk per la precipitazione totale
    utils.kolmogorov_smirnov_test(df['Total Precipitation'], "precipitazione totale") # test di Kolmogorov-Smirnov per la precipitazione totale
    # la distribuzione della precipitazione totale non è normale >> non conviene applicare la z-score
    # applico quindi la normalizzazione min-max
    min_precipitation = df['Total Precipitation'].min()
    max_precipitation = df['Total Precipitation'].max()
    df['Normalized_Total_Precipitation'] = (df['Total Precipitation'] - min_precipitation) / (max_precipitation - min_precipitation)
    #utils.plot_normalized_data(df['Total Precipitation'], df['Normalized_Total_Precipitation'], 'Total Precipitation') # grafico per visualizzare la normalizzazione min-max

    # Pre-processing della "Relative humidity"
    # non ci sono dati mancanti
    utils.shapiro_wilk_test(df['Relative humidity'], "umidità relativa") # test di Shapiro-Wilk per l'umidità relativa
    utils.kolmogorov_smirnov_test(df['Relative humidity'], "umidità relativa") # test di Kolmogorov-Smirnov per l'umidità relativa
    # non seguono una distribuzione normale >> non conviene applicare la z-score
    # applico quindi la normalizzazione min-max
    min_humidity = df['Relative humidity'].min()
    max_humidity = df['Relative humidity'].max()
    df['Normalized_Relative_Humidity'] = (df['Relative humidity'] - min_humidity) / (max_humidity - min_humidity)
    #utils.plot_normalized_data(df['Relative humidity'], df['Normalized_Relative_Humidity'], 'Relative Humidity') # grafico per visualizzare la normalizzazione min-max

    # Pre-processing del "Wind speed (gust)"
    # ci sono valori mancanti
    # applico l'imputazione con la mediana (media sensibile ai valori di outlier)
    median_value = df['Wind speed (gust)'].median()
    df['Wind speed (gust)'] = df['Wind speed (gust)'].fillna(median_value)
    utils.shapiro_wilk_test(df['Wind speed (gust)'], "velocità del vento (raffica)") # test di Shapiro-Wilk per la velocità del vento (raffica)
    utils.kolmogorov_smirnov_test(df['Wind speed (gust)'], "velocità del vento (raffica)") # test di Kolmogorov-Smirnov per la velocità del vento (raffica)
    # non seguono una distribuzione normale >> non conviene applicare la z-score
    # applico quindi la normalizzazione min-max
    min_wind_speed = df['Wind speed (gust)'].min()
    max_wind_speed = df['Wind speed (gust)'].max()
    df['Normalized_Wind_Speed'] = (df['Wind speed (gust)'] - min_wind_speed) / (max_wind_speed - min_wind_speed)
    #utils.plot_normalized_data(df['Wind speed (gust)'], df['Normalized_Wind_Speed'], 'Wind Speed') # grafico per visualizzare la normalizzazione min-max

    # Pre-processing del "u-component of wind (gust)"
    # ci sono valori mancanti
    # applico l'imputazione con la mediana (media sensibile ai valori di outlier)
    median_value = df['u-component of wind (gust)'].median()
    df['u-component of wind (gust)'] = df['u-component of wind (gust)'].fillna(median_value)
    utils.shapiro_wilk_test(df['u-component of wind (gust)'], "componente u del vento (raffica)") # test di Shapiro-Wilk per la componente u del vento (raffica)
    utils.kolmogorov_smirnov_test(df['u-component of wind (gust)'], "componente u del vento (raffica)") # test di Kolmogorov-Smirnov per la componente u del vento (raffica)
    min_u_wind = df['u-component of wind (gust)'].min()
    max_u_wind = df['u-component of wind (gust)'].max()
    df['Normalized_U_Wind'] = (df['u-component of wind (gust)'] - min_u_wind) / (max_u_wind - min_u_wind)
    #utils.plot_normalized_data(df['u-component of wind (gust)'], df['Normalized_U_Wind'], 'U Wind') # grafico per visualizzare la normalizzazione min-max

    # Pre-processing del "v-component of wind (gust)"
    # ci sono valori mancanti
    # applico l'imputazione con la mediana (media sensibile ai valori di outlier)
    median_value = df['v-component of wind (gust)'].median()
    df['v-component of wind (gust)'] = df['v-component of wind (gust)'].fillna(median_value)
    utils.shapiro_wilk_test(df['v-component of wind (gust)'], "componente v del vento (raffica)") # test di Shapiro-Wilk per la componente v del vento (raffica)
    utils.kolmogorov_smirnov_test(df['v-component of wind (gust)'], "componente v del vento (raffica)") # test di Kolmogorov-Smirnov per la componente v del vento (raffica)
    min_v_wind = df['v-component of wind (gust)'].min()
    max_v_wind = df['v-component of wind (gust)'].max()
    df['Normalized_V_Wind'] = (df['v-component of wind (gust)'] - min_v_wind) / (max_v_wind - min_v_wind)
    #utils.plot_normalized_data(df['v-component of wind (gust)'], df['Normalized_V_Wind'], 'V Wind') # grafico per visualizzare la normalizzazione min-max

    # Pre-processing del "2 metre dewpoint temperature"
    # non ci sono dati mancanti
    utils.shapiro_wilk_test(df['2 metre dewpoint temperature'], "temperatura di rugiada a 2 metri") # test di Shapiro-Wilk per la temperatura di rugiada a 2 metri
    utils.kolmogorov_smirnov_test(df['2 metre dewpoint temperature'], "temperatura di rugiada a 2 metri") # test di Kolmogorov-Smirnov per la temperatura di rugiada a 2 metri
    # non seguono una distribuzione normale >> non conviene applicare la z-score
    # applico quindi la normalizzazione min-max
    min_dewpoint_temp = df['2 metre dewpoint temperature'].min()
    max_dewpoint_temp = df['2 metre dewpoint temperature'].max()
    df['Normalized_Dewpoint_Temperature'] = (df['2 metre dewpoint temperature'] - min_dewpoint_temp) / (max_dewpoint_temp - min_dewpoint_temp)
    #utils.plot_normalized_data(df['2 metre dewpoint temperature'], df['Normalized_Dewpoint_Temperature'], 'Dewpoint Temperature') # grafico per visualizzare la normalizzazione min-max
    
    # Pre-processing del "Minimum temperature at 2 metres in the last 6 hours"
    # ci sono valori mancanti
    # applico l'imputazione con la mediana (media sensibile ai valori di outlier)
    median_value = df['Minimum temperature at 2 metres in the last 6 hours'].median()
    df['Minimum temperature at 2 metres in the last 6 hours'] = df['Minimum temperature at 2 metres in the last 6 hours'].fillna(median_value)
    utils.shapiro_wilk_test(df['Minimum temperature at 2 metres in the last 6 hours'], "temperatura minima a 2 metri nelle ultime 6 ore") # test di Shapiro-Wilk per la temperatura minima a 2 metri nelle ultime 6 ore
    utils.kolmogorov_smirnov_test(df['Minimum temperature at 2 metres in the last 6 hours'], "temperatura minima a 2 metri nelle ultime 6 ore") # test di Kolmogorov-Smirnov per la temperatura minima a 2 metri nelle ultime 6 ore
    # non seguono una distribuzione normale >> non conviene applicare la z-score
    # applico quindi la normalizzazione min-max
    min_temp_2m_6h = df['Minimum temperature at 2 metres in the last 6 hours'].min()
    max_temp_2m_6h = df['Minimum temperature at 2 metres in the last 6 hours'].max()
    df['Normalized_Min_Temperature_6h'] = (df['Minimum temperature at 2 metres in the last 6 hours'] - min_temp_2m_6h) / (max_temp_2m_6h - min_temp_2m_6h)
    #utils.plot_normalized_data(df['Minimum temperature at 2 metres in the last 6 hours'], df['Normalized_Min_Temperature_6h'], 'Minimum Temperature 6h') # grafico per visualizzare la normalizzazione min-max
    
    # Pre-processing del "Maximum temperature at 2 metres in the last 6 hours"
    # ci sono valori mancanti
    # applico l'imputazione con la mediana (media sensibile ai valori di outlier)
    median_value = df['Maximum temperature at 2 metres in the last 6 hours'].median()
    df['Maximum temperature at 2 metres in the last 6 hours'] = df['Maximum temperature at 2 metres in the last 6 hours'].fillna(median_value)
    utils.shapiro_wilk_test(df['Maximum temperature at 2 metres in the last 6 hours'], "temperatura massima a 2 metri nelle ultime 6 ore") # test di Shapiro-Wilk per la temperatura massima a 2 metri nelle ultime 6 ore
    utils.kolmogorov_smirnov_test(df['Maximum temperature at 2 metres in the last 6 hours'], "temperatura massima a 2 metri nelle ultime 6 ore") # test di Kolmogorov-Smirnov per la temperatura massima a 2 metri nelle ultime 6 ore
    # non seguono una distribuzione normale >> non conviene applicare la z-score
    # applico quindi la normalizzazione min-max
    min_temp_2m_6h = df['Maximum temperature at 2 metres in the last 6 hours'].min()
    max_temp_2m_6h = df['Maximum temperature at 2 metres in the last 6 hours'].max()
    df['Normalized_Max_Temperature_6h'] = (df['Maximum temperature at 2 metres in the last 6 hours'] - min_temp_2m_6h) / (max_temp_2m_6h - min_temp_2m_6h)
    #utils.plot_normalized_data(df['Maximum temperature at 2 metres in the last 6 hours'], df['Normalized_Max_Temperature_6h'], 'Maximum Temperature 6h') # grafico per visualizzare la normalizzazione min-max
    
    # Pre-processing del "Total Cloud Cover"
    # non ci sono valori mancanti
    utils.shapiro_wilk_test(df['Total Cloud Cover'], "copertura nuvolosa totale") # test di Shapiro-Wilk per la copertura nuvolosa totale
    utils.kolmogorov_smirnov_test(df['Total Cloud Cover'], "copertura nuvolosa totale") # test di Kolmogorov-Smirnov per la copertura nuvolosa totale
    # non seguono una distribuzione normale >> non conviene applicare la z-score
    # applico quindi la normalizzazione min-max
    min_cloud_cover = df['Total Cloud Cover'].min()
    max_cloud_cover = df['Total Cloud Cover'].max()
    df['Normalized_Cloud_Cover'] = (df['Total Cloud Cover'] - min_cloud_cover) / (max_cloud_cover - min_cloud_cover)
    #utils.plot_normalized_data(df['Total Cloud Cover'], df['Normalized_Cloud_Cover'], 'Total Cloud Cover') # grafico per visualizzare la normalizzazione min-max

    # Creazione degli array delle features e del target
    features = df[['Normalized_Temperature', 'Normalized_2m_Temperature', 'Normalized_Total_Precipitation', 
                   'Normalized_Relative_Humidity', 'Normalized_Wind_Speed', 'Normalized_U_Wind', 
                   'Normalized_V_Wind', 'Normalized_Dewpoint_Temperature', 'Normalized_Min_Temperature_6h', 
                   'Normalized_Max_Temperature_6h', 'Normalized_Cloud_Cover']].values
    target = df['Normalized_Total_Precipitation'].values

    # (n_samples, n_timesteps, n_features)
    # # n_samples: numero di campioni
    # time_steps: lunghezza della serie temporale
    # n_features: numero di features per ogni serie temporale

    # Definizione degli iperparametri
    param_dist = {
        #'units': [50, 100, 150],
        'batch_size': [32, 64, 128],
        #'dropout_rate': [0.1, 0.2, 0.3],
        'epochs': [50, 100, 150],
        'optimizer': ['adam', 'rmsprop']
    }

    # reshape dei dati per renderli tridimensionali per la lstm
    features = features.reshape(features.shape[0], features.shape[1], 1)

    # Definizione dello spazio di ricerca
    random_search = lib.RandomizedSearchCV(
        estimator=lib.KerasRegressor(build_fn=utils.create_lstm_model, input_shape=(features.shape[1],1),verbose=0),
        param_distributions=param_dist,
        n_iter=10, # Numero di iterazioni per la random search
        cv=lib.TimeSeriesSplit(n_splits=5).split(features),
        verbose=2
    )

    # esecuzione della random search
    #random_search.fit(features, target)

    # identificazione delle migliori combinazioni di iperparametri secondo Random Search
    #best_params = random_search.best_params_
    #print("Migliore combinazione di iperparametri secondo Random Search:")
    #print(best_params) 
    # Random Search + Cross Validation (k = 5):
    # {'optimizer': 'rmsprop', 'epochs': 150, 'batch_size': 32} con 50 allenamenti (17 minuti circa)

    # definizione dello spazio di ricerca
    grid_search = lib.GridSearchCV(
        estimator= lib.KerasRegressor(build_fn=utils.create_lstm_model, input_shape=(features.shape[1], 1), verbose=0),
        param_grid=param_dist,
        cv=lib.TimeSeriesSplit(n_splits=5).split(features),
        verbose=2
    )

    # esecuzione della grid search
    #grid_search.fit(features, target)

    # identificazione delle migliori combinazioni di iperparametri secondo Grid Search
    #best_params = grid_search.best_params_
    #print("Migliore combinazione di iperparametri secondo Grid Search:")
    #print(best_params)
    # Grid Search + Cross Validation (k = 5):
    # {'batch_size': 32, 'epochs': 150, 'optimizer': 'adam'} con 90 allenamenti (42 minuti circa)

    best_params_random_search = {
        'batch_size': 32,
        'epochs': 150,
        'optimizer': 'rmsprop'
    }

    best_params_grid_search = {
        'batch_size': 32,
        'epochs': 150,
        'optimizer': 'adam'
    }

    # creazione del modello LSTM con la migliore combinazione di iperparametri da random search
    best_model_random_search = utils.create_lstm_model(input_shape=(features.shape[1], 1), optimizer=best_params_random_search['optimizer'])
    # allenamento del modello
    best_model_random_search.fit(features, target, epochs=best_params_random_search['epochs'], batch_size=best_params_random_search['batch_size'], verbose=2)

    # Previsione dei valori target utilizzando il modello addestrato
    predictions_random_search = best_model_random_search.predict(features)
    # ripristino dei valori originali dalla normalizzazione min-max
    min_prediction = min(df['Total Precipitation'])
    max_prediction = max(df['Total Precipitation'])
    predicted_precipitation_random_search = predictions_random_search * (max_prediction - min_prediction) + min_prediction
    print("Previsioni (Random Search):")
    print(predicted_precipitation_random_search)

    mse_random_search = lib.mean_squared_error(target, predictions_random_search)
    print("Mean Squared Error (Random Search):", mse_random_search)

    rmse_random_search = lib.np.sqrt(mse_random_search)
    print("Root Mean Squared Error (Random Search):", rmse_random_search)

    mae_random_search = lib.mean_absolute_error(target, predictions_random_search)
    print("Mean Absolute Error (Random Search):", mae_random_search)

    # creazione del modello LSTM con la migliore combinazione di iperparametri da grid search
    best_model_grid_search = utils.create_lstm_model(input_shape=(features.shape[1], 1), optimizer=best_params_grid_search['optimizer'])
    # allenamento del modello
    best_model_grid_search.fit(features, target, epochs=best_params_grid_search['epochs'], batch_size=best_params_grid_search['batch_size'], verbose=2)

    # Previsione dei valori target utilizzando il modello addestrato
    predictions_grid_search = best_model_grid_search.predict(features)
    # ripristino dei valori originali dalla normalizzazione min-max
    min_prediction = min(df['Total Precipitation'])
    max_prediction = max(df['Total Precipitation'])
    predicted_precipitation_grid_search = predictions_grid_search * (max_prediction - min_prediction) + min_prediction
    print("Previsioni (Grid Search):")
    print(predicted_precipitation_grid_search)

    mse_grid_search = lib.mean_squared_error(target, predictions_grid_search)
    print("Mean Squared Error (Grid Search):", mse_grid_search)

    rmse_grid_search = lib.np.sqrt(mse_grid_search)
    print("Root Mean Squared Error (Grid Search):", rmse_grid_search)

    mae_grid_search = lib.mean_absolute_error(target, predictions_grid_search)
    print("Mean Absolute Error (Grid Search):", mae_grid_search)

    # Grafico delle previsioni
    lib.plt.figure(figsize=(10, 5))
    lib.plt.plot(lib.np.arange(len(predicted_precipitation_random_search)), predicted_precipitation_random_search, label='Random Search', color='blue')
    lib.plt.plot(lib.np.arange(len(predicted_precipitation_grid_search)), predicted_precipitation_grid_search, label='Grid Search', color='red')
    lib.plt.xlabel('Time')
    lib.plt.ylabel('Predicted Precipitation')
    lib.plt.title('Comparison of Predictions')
    lib.plt.legend()
    lib.plt.grid(True)
    lib.plt.show()

    # grafico delle metriche di valutazione
    labels = ['MSE', 'RMSE', 'MAE']
    random_search_values = [mse_random_search, rmse_random_search, mae_random_search]
    grid_search_values = [mse_grid_search, rmse_grid_search, mae_grid_search]
    x = lib.np.arange(len(labels))
    width = 0.35
    fig, ax = lib.plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, random_search_values, width, label='Random Search')
    rects2 = ax.bar(x + width/2, grid_search_values, width, label='Grid Search')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Evaluation Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    lib.plt.show()   

if __name__ == '__main__':
    main()