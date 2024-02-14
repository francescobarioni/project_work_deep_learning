import librerie as lib
import utils
import pre_processing_data as ppd
import lstm_training_and_predict as ltp

def main():

    # esecuzione delle predizioni con modello LSTM
    ltp.main() 

if __name__ == '__main__':
    main()