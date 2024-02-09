import librerie as lib
import utils

def main():

    df = lib.pd.read_csv('precipitazioni Ferrara e limitrofi.csv',sep=';')
    # Prendo solo le colonne necessarie per la previsione
    colonne_necessarie = ['Forecast timestamp','Position', 'Forecast base', 'Temperature', '2 metre temperature', 'Total Precipitation', 
                          'Relative humidity', 'Wind speed (gust)', 'u-component of wind (gust)', 'v-component of wind (gust)', 
                          '2 metre dewpoint temperature', 'Minimum temperature at 2 metres in the last 6 hours', 
                          'Maximum temperature at 2 metres in the last 6 hours', 'Total Cloud Cover']
    dati_necessari = df[colonne_necessarie]

    # PRE PROCESSING DEL FORECAST TIMESTAMP
    # mi assicuro che la colonna 'Forecast timestamp' sia di tipo datetime
    dati_necessari['Forecast timestamp'] = lib.pd.to_datetime(dati_necessari['Forecast timestamp'])
    # ordino i dati all'interno di dati_necessari in base al timestamp del forecast
    dati_necessari.sort_values(by='Forecast timestamp', inplace=True)

    # PRE PROCESSING DELLA POSIZIONE
    coordinates = dati_necessari['Position'].str.split(',', expand=True).astype(float)

    # Test necessari per verificare la distribuzione normale delle latitudini
    #utils.grafico_istogramma(coordinates[0], "latitudini") # istogramma delle latitudini
    #utils.grafico_qqplot(coordinates[0], "latitudini") # qqplot delle latitudini
    utils.shapiro_wilk_test(coordinates[0], "latitudini") # test di Shapiro-Wilk per le latitudini
    utils.kolmogorov_smirnov_test(coordinates[0], "latitudini") # test di Kolmogorov-Smirnov per le latitudini

    # Test necessari per verificare la distribuzione normale delle longitudini
    #utils.grafico_istogramma(coordinates[1], "longitudini") # istogramma delle longitudini
    #utils.grafico_qqplot(coordinates[1], "longitudini") # qqplot delle longitudini
    utils.shapiro_wilk_test(coordinates[1], "longitudini") # test di Shapiro-Wilk per le longitudini
    utils.kolmogorov_smirnov_test(coordinates[1], "longitudini") # test di Kolmogorov-Smirnov per le longitudini

    # Normalizzazione con Min-Max delle latitudini e delle longitudini
    scaler = lib.sk.preprocessing.MinMaxScaler()
    normalized_coordinates = scaler.fit_transform(coordinates)
    dati_necessari['Position'] = normalized_coordinates

    #utils.grafico_istogramma(dati_necessari['Temperature'], "temperatura") # istogramma delle latitudini
    #utils.shapiro_wilk_test(dati_necessari['Temperature'], "temperatura") # test di Shapiro-Wilk per le latitudini


    #dati_necessari['Latitudine'], df['Longitudine'] = dati_necessari['Position'].str.split(',', 1).str
    #dati_necessari['Latitudine'] = lib.pd.to_numeric(dati_necessari['Latitudine']) # converto la latitudine in numerico
    #dati_necessari['Longitudine'] = lib.pd.to_numeric(dati_necessari['Longitudine']) # converto la longitudine in numerico
    # Normalizzo i valori della latitudine e della longitudine con normalizzazione min-max
    #dati_necessari['Latitudine'] = (dati_necessari['Latitudine'] - dati_necessari['Latitudine'].min()) / (dati_necessari['Latitudine'].max() - dati_necessari['Latitudine'].min())
    #dati_necessari['Longitudine'] = (dati_necessari['Longitudine'] - dati_necessari['Longitudine'].min()) / (dati_necessari['Longitudine'].max() - dati_necessari['Longitudine'].min())    

    # PRE PROCESSING DELLA TEMPERATURA
    # Normalizzazione della temperatura tra 0 e 1 con min-max
    #dati_necessari['Temperature'] = (dati_necessari['Temperature'] - dati_necessari['Temperature'].min()) / (dati_necessari['Temperature'].max() - dati_necessari['Temperature'].min())

    # PRE PROCESSING DELLA TEMPERATURA A 2 METRI
    # Normalizzazione della colonna '2 metre temperature' tra 0 e 1 con min-max
    #dati_necessari['2 metre temperature'] = (dati_necessari['2 metre temperature'] - dati_necessari['2 metre temperature'].min()) / (dati_necessari['2 metre temperature'].max() - dati_necessari['2 metre temperature'].min())




if __name__ == "__main__":
    main()


