import librerie as lib

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

def plot_normalized_data(original_data, normalized_data, name_data):
    """
    Funzione per visualizzare i punti prima e dopo la normalizzazione min-max.
    
    Parametri:
    - original_data: Array numpy contenente i dati originali.
    - normalized_data: Array numpy contenente i dati normalizzati.
    """
    lib.plt.figure(figsize=(10, 5))
    
    # Visualizzazione dei dati originali
    lib.plt.subplot(1, 2, 1)
    lib.plt.scatter(range(len(original_data)), original_data, color='blue', label='Dati Originali')
    lib.plt.xlabel('Indice dei dati')
    lib.plt.ylabel('Valore')
    lib.plt.title(str(name_data) + ' - Dati Originali')
    lib.plt.legend()
    
    # Visualizzazione dei dati normalizzati
    lib.plt.subplot(1, 2, 2)
    lib.plt.scatter(range(len(normalized_data)), normalized_data, color='red', label='Dati Normalizzati')
    lib.plt.xlabel('Indice dei dati')
    lib.plt.ylabel('Valore normalizzato')
    lib.plt.title(str(name_data) + ' - Dati Normalizzati')
    lib.plt.legend()
    
    lib.plt.tight_layout()
    lib.plt.show()

def build_lstm_model(input_shape, units=50):
    model = lib.Sequential()
    model.add(lib.LSTM(units,input_shape=input_shape))
    model.add(lib.Dense(1)) # funzione di attivazione utilizzata lineare
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_lstm_model(input_shape, dropout_rate=0.2, units=50, optimizer='adam'):
    model = lib.Sequential()
    model.add(lib.LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(lib.Dropout(dropout_rate))
    model.add(lib.LSTM(units=units, return_sequences=True))
    model.add(lib.Dropout(dropout_rate))
    model.add(lib.LSTM(units=units))
    model.add(lib.Dropout(dropout_rate))
    model.add(lib.Dense(units=1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model
