import utils
import librerie as lib
import pre_processing_data as ppd

def main():

    features, target, df = ppd.main()

    # definizione degli iperparametri
    gru_param_dist = {
        'batch_size': [32, 64, 128],
        'epochs': [50, 100, 150],
        'optimizer': ['adam', 'rmsprop'],
    }

    # reshape dei dati per renderli tridimensionali per la lstm
    features = features.reshape(features.shape[0], features.shape[1], 1)

    # definizione dello spazio di ricerca per la random search
    random_search = lib.RandomizedSearchCV(
        estimator=lib.KerasRegressor(build_fn=utils.create_gru_model, input_shape=(features.shape[1], 1), verbose=0),
        param_distributions=gru_param_dist,
        n_iter=10, # Numero di iterazioni per la random search
        cv=lib.TimeSeriesSplit(n_splits=5).split(features),
        verbose=2
    )

    # esecuzione della random search
    random_search.fit(features, target)

    # identificazione delle migliori combinazioni di iperparametri secondo Random Search
    best_params = random_search.best_params_
    print("Migliore combinazione di iperparametri secondo Random Search (GRU):")
    print(best_params)
