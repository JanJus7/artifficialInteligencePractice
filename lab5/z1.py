'''
a) StandardScaler standaryzuje dane tak, aby miały średnią 0 i odchylenie standardowe 1.
b) OneHotEncoder koduje etykiety jako wektory binarne.
c) Warstwa wejściowa ma liczbę neuronów równą X_train.shape[1] czyli liczbie cech w zbiorze danych.
   Warstwy ukryte mają 64 neurony
   Warstwa wyjściowa ma liczbę neuronów równą  y_encoded.shape[1] czyli liczbie unikalnych klas w zbiorze danych.
d) Funkcja Relu jest bardzo popularna ale nie zawsze najlepsza.
e) Można zmienić w optymalizatorze, np. optimizer=Adam(learning_rate=0.01)
f) Tak, można zmienić batch size.
   Małe batch (np 4) daje więcej iteracji ale mniej dokładne gradienty.
   Duże batrch (np 32) daje mniej iteracji ale dokładniejsze gradienty.
g) Sieć osiągneła najlepszą wydajność w 40 epoce. POtem mamy do czynienia z przeuczeniem.
h)Kod wczytuje zbiór danych Iris, standaryzuje cechy i koduje etykiety klas. 
  Dane są dzielone na zbiór treningowy i testowy.
  Wczytuje wstępnie wytrenowany model, dokłada 10 epok treningu, zapisuje zaktualizowany model i ocenia jego dokładność na zbiorze testowym. 
'''