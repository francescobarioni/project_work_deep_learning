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
        lib.plt.show()
    elif missing_values.sum() == 0: # se non ci sono valori mancanti
        print('Non ci sono valori mancanti nel dataset\n')

    # Pre-processing del "Forecast timestamp"
    df['Forecast timestamp'] = lib.pd.to_datetime(df['Forecast timestamp'])
    df.sort_values(by='Forecast timestamp', inplace=True)

    # Pre-processing della "Position"
    df[['Latitudine', 'Longitudine']] = df['Position'].str.split(',', expand=True).astype(float)
    serie_latitudine = df['Latitudine'] # serie separata per le latitudini
    serie_longitudine = df['Longitudine'] # serie separata per le longitudini

    # Pre-processing della "Temperature"
    utils.shapiro_wilk_test(df['Temperature'], "temperatura") # test di Shapiro-Wilk per la temperatura
    utils.kolmogorov_smirnov_test(df['Temperature'], "temperatura") # test di Kolmogorov-Smirnov per la temperatura
    # dai 2 test precedenti si ricava che la distribuzione della temperatura non è normale >> non conviene applicare la z-score
    # applico quindi la normalizzazione min-max
    min_temp = df['Temperature'].min()
    max_temp = df['Temperature'].max()
    df['Normalized_Temperature'] = (df['Temperature'] - min_temp) / (max_temp - min_temp)
    utils.plot_normalized_data(df['Temperature'], df['Normalized_Temperature'], 'Temperature') # grafico per visualizzare la normalizzazione min-max

    # Pre-processing della "2 metre temperature"
    utils.shapiro_wilk_test(df['2 metre temperature'], "temperatura a 2 metri") # test di Shapiro-Wilk per la temperatura a 2 metri
    utils.kolmogorov_smirnov_test(df['2 metre temperature'], "temperatura a 2 metri") # test di Kolmogorov-Smirnov per la temperatura a 2 metri
    # dai 2 test precedenti si ricava che la distribuzione della 2 metre temperature non è normale >> non conviene applicare la z-score
    # applico quindi la normalizzazione min-max
    min_temp_2m = df['2 metre temperature'].min()
    max_temp_2m = df['2 metre temperature'].max()
    df['Normalized_2m_Temperature'] = (df['2 metre temperature'] - min_temp_2m) / (max_temp_2m - min_temp_2m)
    utils.plot_normalized_data(df['2 metre temperature'], df['Normalized_2m_Temperature'], '2 Metre Temperature') # grafico per visualizzare la normalizzazione min-max

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
    utils.plot_normalized_data(df['Total Precipitation'], df['Normalized_Total_Precipitation'], 'Total Precipitation') # grafico per visualizzare la normalizzazione min-max

    # Pre-processing della "Relative humidity"

    # Pre-processing del "Wind speed (gust)"

    # Pre-processing del "u-component of wind (gust)"

    # Pre-processing del "v-component of wind (gust)"

    # Pre-processing del "2 metre dewpoint temperature"

    # Pre-processing del "Minimum temperature at 2 metres in the last 6 hours"

    # Pre-processing del "Maximum temperature at 2 metres in the last 6 hours"

    # Pre-processing del "Total Cloud Cover"


if __name__ == '__main__':
    main()