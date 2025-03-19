import pandas as pd
import re as re

df = pd.read_csv('iris_with_errors.csv')
# print(df)

def stats(df):
    missing_data = df.isna().sum()
    print("Brakujące dane w każdej kolumnie:")
    print(missing_data)

    print("\nStatystyki bazy danych z błędami:")
    print(df.describe(include='all'))

def fixDeviatedNums(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

        outOfRange = df[(df[col] < 0) | (df[col] > 15)].index

        meanVal = df[col].mean()
        df.loc[outOfRange, col] = meanVal

    print("\nStatystyki po naprawie zakresu:")
    print(df.describe(include='all'))

def fixWrongSpelledNames(df):
    for col in df['variety']:
        if col not in ['Setosa', 'Versicolor', 'Virginica']:
            if re.match(r"[sS][a-zA-Z]+", col):
                df['variety'] = df['variety'].replace(col, 'Setosa')
            elif re.match(r"^[vV]e[a-zA-Z]+", col):
                df['variety'] = df['variety'].replace(col, 'Versicolor')
            elif re.match(r"[vV]i[a-zA-Z]+", col):
                df['variety'] = df['variety'].replace(col, 'Virginica')
            else:
                df['variety'] = df['variety'].replace(col, 'Setosa')

    print("\nStatystyki po naprawie błędnych nazw:")
    # print(df.describe(include='all'))
    print(df['variety'].unique())

def main():
    stats(df)
    fixDeviatedNums(df, columns)
    fixWrongSpelledNames(df)

if __name__ == '__main__':
    columns = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
    main()

