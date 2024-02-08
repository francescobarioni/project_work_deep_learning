import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def main():
    # carico il dataset CSV
    df = pd.read_csv('precipitazioni Ferrara e limitrofi.csv')

if __name__ == "__main__":
    main()

"""
# rimuovo righe con dati mancanti
dataset = dataset.dropna()

# seleziono le colonne di interesse per l'addestramento
features = [
    'Forecast timestamp',
    'Position',
    'Forecast base',
    'Temperature',
    '2 metre temperature',
    'Total Precipitation',
    'Relative humidity',
    'Downward long-wave radiation flux',
    'Downward short-wave radiation flux',
    'Wind speed (gust)',
    'u-component of wind (gust)',
    'v-component of wind (gust)',
    '2 metre dewpoint temperature',
    'Minimum temperature at 2 metres in the last 6 hours',
    'Maximum temperature at 2 metres in the last 6 hours',
    'Total Cloud Cover'
]

dataset = dataset[features]

# normalizzo i dati
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

# definisco la finestra di tempo per le sequenze di input
window_size = 10 # si regola con la sperimentazione 
"""



