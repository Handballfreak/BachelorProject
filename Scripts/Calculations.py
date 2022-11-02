import pandas as pd
import numpy as np

#Einfügen des Pfades zur CSV der Ergebnisse des Sweeps
csv_path = ""

df = pd.read_csv(csv_path)
std = np.std(df)
varianz = np.var(df)


print()