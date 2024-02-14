import utils
import librerie as lib
import pre_processing_data as ppd

def main():
    
    # definizione degli iperparametri
    gru_param_dist = {
        'batch_size': [32, 64, 128],
        'epochs': [50, 100, 150],
        'optimizer': ['adam', 'rmsprop'],
    }
