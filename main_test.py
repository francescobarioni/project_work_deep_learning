import librerie as lib
import utils
import pre_processing_data as ppd
import lstm_training_and_predict as ltp
import gru_training_and_predict as gtp

def main():

    # esecuzione delle predizioni con modello LSTM
    #ltp.main() 

    # esecuzione delle predizioni con modello GRU
    gtp.main()

if __name__ == '__main__':
    main()