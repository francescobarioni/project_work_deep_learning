import librerie as lib
import utils

def main():
    df = lib.pd.read_csv('precipitazioni Ferrara e limitrofi.csv',sep=';')
    coloumn_structure = utils.get_column_structure(df) # separo il df in colonne per tipo di dato

    # Processo di normalizzazione della temperatura
    coloumn_structure['col_temperature'] = utils.normalize_values(utils.remove_points_from_vector(coloumn_structure['col_temperature']))

    lib.plt.figure(figsize=(10, 6))
    lib.plt.plot(coloumn_structure['col_forecasting_timestamp'][::5], coloumn_structure['col_temperature'][::5],
                 marker='o', linestyle='-', color='b',alpha=0.5) 
    lib.plt.xlabel('forecasting timestap')
    lib.plt.ylabel('temperature')
    lib.plt.title('Temperature trend')
    lib.plt.grid(True)
    lib.plt.xticks(rotation=45)
    lib.plt.xticks(coloumn_structure['col_forecasting_timestamp'][::10])
    lib.plt.tight_layout()
    lib.plt.show()

if __name__ == "__main__":
    main()


