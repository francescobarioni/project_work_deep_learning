import librerie as lib
import utils

def main():
    df = lib.pd.read_csv('precipitazioni Ferrara e limitrofi.csv',sep=';')
    coloumn_structure = utils.get_column_structure(df) # separo il df in colonne per tipo di dato
    #print(coloumn_structure['col_relative_humidity'].head(510))

if __name__ == "__main__":
    main()


