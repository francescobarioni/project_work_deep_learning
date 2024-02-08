import librerie as lib

"""
La funzione `get_column_structure` accetta un DataFrame contenente dati meteorologici 
e restituisce una struttura dati che mappa ciascuna colonna del DataFrame al relativo tipo di dato meteorologico. 
Utilizza un dizionario di mapping predefinito per associare i nomi originali delle colonne ai nomi desiderati. 
Successivamente, itera attraverso il dizionario di mapping per creare dinamicamente le colonne desiderate nel formato `col_nomeTipoDato`. 
Se una colonna corrispondente al tipo di dato esiste nel DataFrame, viene inclusa nella struttura dati restituita.
"""
def get_column_structure(df):
    column_mapping = {
        'Forecast timestamp': 'forecasting_timestamp',
        'Position': 'position',
        'Forecast base': 'forecast_base',
        'Temperature': 'temperature',
        '2 metre temperature': 'temperature_2m',
        'Total Precipitation': 'total_precipitation',
        'Relative humidity': 'relative_humidity',
        'Downward long-wave radiation flux': 'long_wave_radiation_flux',
        'Downward short-wave radiation flux': 'short_wave_radiation_flux',
        'Wind speed (gust)': 'wind_speed',
        'u-component of wind (gust)': 'u_wind_component',
        'v-component of wind (gust)': 'v_wind_component',
        '2 metre dewpoint temperature': 'dewpoint_temperature',
        'Minimum temperature at 2 metres in the last 6 hours': 'min_temperature_last_6h',
        'Maximum temperature at 2 metres in the last 6 hours': 'max_temperature_last_6h',
        'Total Cloud Cover': 'total_cloud_cover'
    }
    
    column_structure = {}
    for i in range(len(column_mapping)):
        col_original = list(column_mapping.keys())[i]
        col_type = list(column_mapping.values())[i]
        col_name = 'col_' + col_type
        if col_original in df.columns:
            column_structure[col_name] = df.iloc[:, i]
    
    return column_structure

# Rimuove i punti da una stringa che sarebbe un numero
def remove_points_from_vector(vector):
    if not isinstance(vector, (list, lib.pd.Series)):
        raise ValueError("Input must be a list or pandas Series")

    if isinstance(vector, lib.pd.Series):
        vector = vector.tolist()

    result = []
    for number in vector:
        if isinstance(number, str):
            number = number.replace('.', '')
        elif isinstance(number, (int, float)):
            number = str(number).replace('.', '')
        else:
            raise ValueError("Input elements must be strings or numbers")
        result.append(number)
    return result

def normalize_values(data_values):
    """
    Normalizza i valori nell'intervallo 0-1 dopo aver applicato il logaritmo con l'aggiunta di un valore costante.

    Args:
    data_values (list): Una lista di valori da normalizzare.

    Returns:
    list: Una lista di valori normalizzati nell'intervallo 0-1.
    """
    # Aggiungi un valore costante per evitare problemi con valori zero o negativi
    const = 1
    values = [float(val) for val in data_values]  # Converte i valori in float

    # Applica il logaritmo con il valore costante
    values_log = lib.np.log(lib.np.array(values) + const)

    # Normalizza i valori logaritmici nell'intervallo 0-1
    values_log_normalized = (values_log - values_log.min()) / (values_log.max() - values_log.min())

    return values_log_normalized



