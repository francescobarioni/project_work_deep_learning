import utils
import librerie as lib

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
    # non ci sono dati mancanti
    df['Forecast timestamp'] = lib.pd.to_datetime(df['Forecast timestamp'])
    df.sort_values(by='Forecast timestamp', inplace=True)

    # Pre-processing della "Position"
    # non ci sono dati mancanti
    df[['Latitudine', 'Longitudine']] = df['Position'].str.split(',', expand=True).astype(float)
    serie_latitudine = df['Latitudine'] # serie separata per le latitudini
    serie_longitudine = df['Longitudine'] # serie separata per le longitudini

    # Pre-processing della "Temperature"
    # non ci sono dati mancanti
    utils.shapiro_wilk_test(df['Temperature'], "temperatura") # test di Shapiro-Wilk per la temperatura
    utils.kolmogorov_smirnov_test(df['Temperature'], "temperatura") # test di Kolmogorov-Smirnov per la temperatura
    # dai 2 test precedenti si ricava che la distribuzione della temperatura non è normale >> non conviene applicare la z-score
    # applico quindi la normalizzazione min-max
    min_temp = df['Temperature'].min()
    max_temp = df['Temperature'].max()
    df['Normalized_Temperature'] = (df['Temperature'] - min_temp) / (max_temp - min_temp)
    utils.plot_normalized_data(df['Temperature'], df['Normalized_Temperature'], 'Temperature') # grafico per visualizzare la normalizzazione min-max

    # Pre-processing della "2 metre temperature"
    # non ci sono dati mancanti
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
    # non ci sono dati mancanti
    utils.shapiro_wilk_test(df['Relative humidity'], "umidità relativa") # test di Shapiro-Wilk per l'umidità relativa
    utils.kolmogorov_smirnov_test(df['Relative humidity'], "umidità relativa") # test di Kolmogorov-Smirnov per l'umidità relativa
    # non seguono una distribuzione normale >> non conviene applicare la z-score
    # applico quindi la normalizzazione min-max
    min_humidity = df['Relative humidity'].min()
    max_humidity = df['Relative humidity'].max()
    df['Normalized_Relative_Humidity'] = (df['Relative humidity'] - min_humidity) / (max_humidity - min_humidity)
    utils.plot_normalized_data(df['Relative humidity'], df['Normalized_Relative_Humidity'], 'Relative Humidity') # grafico per visualizzare la normalizzazione min-max

    # Pre-processing del "Wind speed (gust)"
    # ci sono valori mancanti
    # applico l'imputazione con la mediana (media sensibile ai valori di outlier)
    median_value = df['Wind speed (gust)'].median()
    df['Wind speed (gust)'] = df['Wind speed (gust)'].fillna(median_value)
    utils.shapiro_wilk_test(df['Wind speed (gust)'], "velocità del vento (raffica)") # test di Shapiro-Wilk per la velocità del vento (raffica)
    utils.kolmogorov_smirnov_test(df['Wind speed (gust)'], "velocità del vento (raffica)") # test di Kolmogorov-Smirnov per la velocità del vento (raffica)
    # non seguono una distribuzione normale >> non conviene applicare la z-score
    # applico quindi la normalizzazione min-max
    min_wind_speed = df['Wind speed (gust)'].min()
    max_wind_speed = df['Wind speed (gust)'].max()
    df['Normalized_Wind_Speed'] = (df['Wind speed (gust)'] - min_wind_speed) / (max_wind_speed - min_wind_speed)
    utils.plot_normalized_data(df['Wind speed (gust)'], df['Normalized_Wind_Speed'], 'Wind Speed') # grafico per visualizzare la normalizzazione min-max

    # Pre-processing del "u-component of wind (gust)"
    # ci sono valori mancanti
    # applico l'imputazione con la mediana (media sensibile ai valori di outlier)
    median_value = df['u-component of wind (gust)'].median()
    df['u-component of wind (gust)'] = df['u-component of wind (gust)'].fillna(median_value)
    utils.shapiro_wilk_test(df['u-component of wind (gust)'], "componente u del vento (raffica)") # test di Shapiro-Wilk per la componente u del vento (raffica)
    utils.kolmogorov_smirnov_test(df['u-component of wind (gust)'], "componente u del vento (raffica)") # test di Kolmogorov-Smirnov per la componente u del vento (raffica)
    min_u_wind = df['u-component of wind (gust)'].min()
    max_u_wind = df['u-component of wind (gust)'].max()
    df['Normalized_U_Wind'] = (df['u-component of wind (gust)'] - min_u_wind) / (max_u_wind - min_u_wind)
    utils.plot_normalized_data(df['u-component of wind (gust)'], df['Normalized_U_Wind'], 'U Wind') # grafico per visualizzare la normalizzazione min-max

    # Pre-processing del "v-component of wind (gust)"
    # ci sono valori mancanti
    # applico l'imputazione con la mediana (media sensibile ai valori di outlier)
    median_value = df['v-component of wind (gust)'].median()
    df['v-component of wind (gust)'] = df['v-component of wind (gust)'].fillna(median_value)
    utils.shapiro_wilk_test(df['v-component of wind (gust)'], "componente v del vento (raffica)") # test di Shapiro-Wilk per la componente v del vento (raffica)
    utils.kolmogorov_smirnov_test(df['v-component of wind (gust)'], "componente v del vento (raffica)") # test di Kolmogorov-Smirnov per la componente v del vento (raffica)
    min_v_wind = df['v-component of wind (gust)'].min()
    max_v_wind = df['v-component of wind (gust)'].max()
    df['Normalized_V_Wind'] = (df['v-component of wind (gust)'] - min_v_wind) / (max_v_wind - min_v_wind)
    utils.plot_normalized_data(df['v-component of wind (gust)'], df['Normalized_V_Wind'], 'V Wind') # grafico per visualizzare la normalizzazione min-max

    # Pre-processing del "2 metre dewpoint temperature"
    # non ci sono dati mancanti
    utils.shapiro_wilk_test(df['2 metre dewpoint temperature'], "temperatura di rugiada a 2 metri") # test di Shapiro-Wilk per la temperatura di rugiada a 2 metri
    utils.kolmogorov_smirnov_test(df['2 metre dewpoint temperature'], "temperatura di rugiada a 2 metri") # test di Kolmogorov-Smirnov per la temperatura di rugiada a 2 metri
    # non seguono una distribuzione normale >> non conviene applicare la z-score
    # applico quindi la normalizzazione min-max
    min_dewpoint_temp = df['2 metre dewpoint temperature'].min()
    max_dewpoint_temp = df['2 metre dewpoint temperature'].max()
    df['Normalized_Dewpoint_Temperature'] = (df['2 metre dewpoint temperature'] - min_dewpoint_temp) / (max_dewpoint_temp - min_dewpoint_temp)
    utils.plot_normalized_data(df['2 metre dewpoint temperature'], df['Normalized_Dewpoint_Temperature'], 'Dewpoint Temperature') # grafico per visualizzare la normalizzazione min-max
    
    # Pre-processing del "Total Cloud Cover"
    # non ci sono valori mancanti
    utils.shapiro_wilk_test(df['Total Cloud Cover'], "copertura nuvolosa totale") # test di Shapiro-Wilk per la copertura nuvolosa totale
    utils.kolmogorov_smirnov_test(df['Total Cloud Cover'], "copertura nuvolosa totale") # test di Kolmogorov-Smirnov per la copertura nuvolosa totale
    # non seguono una distribuzione normale >> non conviene applicare la z-score
    # applico quindi la normalizzazione min-max
    min_cloud_cover = df['Total Cloud Cover'].min()
    max_cloud_cover = df['Total Cloud Cover'].max()
    df['Normalized_Cloud_Cover'] = (df['Total Cloud Cover'] - min_cloud_cover) / (max_cloud_cover - min_cloud_cover)
    utils.plot_normalized_data(df['Total Cloud Cover'], df['Normalized_Cloud_Cover'], 'Total Cloud Cover') # grafico per visualizzare la normalizzazione min-max

    # Creazione degli array delle features e del target
    features = df[['Normalized_Temperature', 'Normalized_2m_Temperature', 'Normalized_Total_Precipitation', 
                   'Normalized_Relative_Humidity', 'Normalized_Wind_Speed', 'Normalized_U_Wind', 
                   'Normalized_V_Wind', 'Normalized_Dewpoint_Temperature','Normalized_Cloud_Cover']].values
    target = df['Normalized_Total_Precipitation'].values
    return features, target, df