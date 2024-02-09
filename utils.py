import librerie as lib

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

def plot_of_dates(x_data, y_data, x_label, y_label, title, rotation, x_ticks ,y_ticks,i,j):
    """
    Crea un grafico dei dati che vengono passati come parametri con i relativi dettagi.

    Args:
    - x_data (list): Una lista di valori per l'asse x.
    - y_data (list): Una lista di valori per l'asse y.
    - x_label (str): Una stringa per l'etichetta dell'asse x.
    - y_label (str): Una stringa per l'etichetta dell'asse y.
    - title (str): Una stringa per il titolo del grafico.
    - rotation (int): Un intero per la rotazione delle etichette dell'asse x.
    - x_ticks (list): Una lista di valori per le etichette dell'asse x.
    - y_ticks (list): Una lista di valori per le etichette dell'asse y.
    - i (int): numero di punti da saltare per l'asse x
    - j (int): numero di punti da saltare per l'asse y
    
    Returns: None
    """

    lib.plt.figure(figsize=(10, 6))
    lib.plt.plot(x_data[::i], y_data[::j], marker='o', linestyle='-', color='b', alpha=0.5)
    lib.plt.xlabel(x_label)
    lib.plt.ylabel(y_label)
    lib.plt.title(title)
    lib.plt.grid(True)
    lib.plt.xticks(rotation=rotation)
    lib.plt.xticks(x_ticks[::10])
    lib.plt.yticks(rotation=rotation)
    #lib.plt.yticks(y_ticks[::10])
    lib.plt.tight_layout()
    lib.plt.show()

def shapiro_wilk_test(data, nome_dati):
    # Esegui il test di Shapiro-Wilk per la normalità
    stat, p_value = lib.stats.shapiro(data)
    print("Test di Shapiro-Wilk per " + str(nome_dati) + ":")
    print("Statistiche test:", stat)
    print("Valore p:", p_value)
    if p_value > 0.05:
        print("Non c'è sufficiente evidenza per rifiutare l'ipotesi che i dati" + str(nome_dati) + "siano distribuiti normalmente.\n")
    else:
        print("I dati " + str(nome_dati) +" non seguono una distribuzione normale.\n")

def kolmogorov_smirnov_test(data, data_name):
    # test di Kolmogorov-Smirnov per confrontare i dati con una distribuzione normale
    # utilizzare la distribuzione normale come distribuzione di riferimento
    ks_statistic, p_value = lib.stats.kstest(data, 'norm')
    
    print("Test di Kolmogorov-Smirnov per", data_name + ":")
    print("Statistiche test:", ks_statistic)
    print("Valore p:", p_value)
    
    # Confrontare il valore p con un livello di significatività predefinito (ad esempio, 0.05)
    if p_value > 0.05:
        print("Non c'è sufficiente evidenza per rifiutare l'ipotesi che i dati", data_name, "siano distribuiti normalmente.\n")
    else:
        print("I dati", data_name, "non seguono una distribuzione normale.\n")

def grafico_qqplot(data, nome_dati):
    # Traccia il grafico Q-Q
    lib.stats.probplot(data, dist="norm", plot=lib.plt)
    lib.plt.title("Grafico Q-Q di " + str(nome_dati))
    lib.plt.show()

def grafico_istogramma(data, nome_dati):
    # Traccia l'istogramma
    lib.plt.hist(data, bins=20, density=True, alpha=0.6, color='g')

    # Aggiungi la curva di densità normale per confronto
    mu, sigma = lib.stats.norm.fit(data)
    xmin, xmax = lib.plt.xlim()
    x = lib.np.linspace(xmin, xmax, 100)
    p = lib.stats.norm.pdf(x, mu, sigma)
    lib.plt.plot(x, p, 'k', linewidth=2)
    lib.plt.title("Distribuzione di " + str(nome_dati))
    lib.plt.show()


