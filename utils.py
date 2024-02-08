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