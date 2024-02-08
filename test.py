import numpy as np
import pandas as pd

# Dati di esempio
temperatura_fuori_scala = 566015625

# Applica la trasformazione logaritmica
temperatura_trasformata = np.log(temperatura_fuori_scala)

print("Dato originale originale:", temperatura_fuori_scala)
print("Dato trasformata:", temperatura_trasformata)
