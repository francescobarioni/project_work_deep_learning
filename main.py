import librerie as lib
import utils

def main():
    df = lib.pd.read_csv('precipitazioni Ferrara e limitrofi.csv',sep=';')
    coloumn_structure = utils.get_column_structure(df) # separo il df in colonne per tipo di dato

    # Processo di normalizzazione della temperatura
    coloumn_structure['col_temperature'] = utils.normalize_values(utils.remove_points_from_vector(coloumn_structure['col_temperature']))

if __name__ == "__main__":
    main()


