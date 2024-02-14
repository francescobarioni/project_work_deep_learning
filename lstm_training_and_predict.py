import utils
import librerie as lib
import pre_processing_data as ppd

def main():
    
    features, target, df = ppd.main()

    # (n_samples, n_timesteps, n_features)
    # # n_samples: numero di campioni
    # time_steps: lunghezza della serie temporale
    # n_features: numero di features per ogni serie temporale

    # Definizione degli iperparametri
    lstm_param_dist = {
        'batch_size': [32, 64, 128],
        'epochs': [50, 100, 150],
        'optimizer': ['adam', 'rmsprop'],
    }

    # reshape dei dati per renderli tridimensionali per la lstm
    features = features.reshape(features.shape[0], features.shape[1], 1)

    # Definizione dello spazio di ricerca
    random_search = lib.RandomizedSearchCV(
        estimator=lib.KerasRegressor(build_fn=utils.create_lstm_model, input_shape=(features.shape[1],1),verbose=0),
        param_distributions=lstm_param_dist,
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
        param_grid=lstm_param_dist,
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