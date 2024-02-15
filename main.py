import lstm_training_and_predict as ltp
import gru_training_and_predict as gtp
import utils

def main():

    # esecuzione delle predizioni con modello LSTM
    lstm_mse_random_search, lstm_rmse_random_search, lstm_mae_random_search, \
        lstm_mse_grid_search, lstm_rmse_grid_search, lstm_mae_grid_search = ltp.main() 

    # esecuzione delle predizioni con modello GRU
    gru_mse_random_search, gru_rmse_random_search, gru_mae_random_search, \
        gru_mse_grid_search, gru_rmse_grid_search, gru_mae_grid_search = gtp.main()
    
    # confronto dei due modelli LSTM e GRU con un grafico
    utils.comparte_lstm_and_lstm_model(lstm_mse_random_search, lstm_rmse_random_search, lstm_mae_random_search, 
                            lstm_mse_grid_search, lstm_rmse_grid_search, lstm_mae_grid_search, 
                            gru_mse_random_search, gru_rmse_random_search, gru_mae_random_search, 
                            gru_mse_grid_search, gru_rmse_grid_search, gru_mae_grid_search)

if __name__ == '__main__':
    main()