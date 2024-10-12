import pandas as pd
from sklearn.preprocessing import MinMaxScaler


caminho_arquivo_CSV = r'C:\Users\clara\3W\dataset\folds\folds_clf_02.csv'


df = pd.read_csv(caminho_arquivo_CSV)


df.ffill(inplace=True)


print("Colunas disponíveis no dataset:", df.columns)




scaler = MinMaxScaler()

print(df.head())


print(df.info())


print(df.describe())

