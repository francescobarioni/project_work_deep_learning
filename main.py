import librerie as lib
import utils

def main():
    df = lib.pd.read_csv('precipitazioni Ferrara e limitrofi.csv',sep=';')
    coloumn_structure = utils.get_column_structure(df) # separo il df in colonne per tipo di dato

    # Processo di normalizzazione della temperatura
    coloumn_structure['col_temperature'] = utils.normalize_values(utils.remove_points_from_vector(coloumn_structure['col_temperature']))
    # Plot della temperatura normalizzata
    utils.plot_of_dates(coloumn_structure['col_forecasting_timestamp'], coloumn_structure['col_temperature'],
                        'forecasting timestap', 'temperature', 'Temperature trend', 45, coloumn_structure['col_forecasting_timestamp'],
                        coloumn_structure['col_temperature'], 5, 5)

if __name__ == "__main__":
    main()


