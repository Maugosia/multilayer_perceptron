## Perceptron wielowarstwowy
# Wykonawcy - Sofiia Levchenko [sofiia2002], Małgorzata Sikora [Maugosia]
# Wstęp
Celem zadania była implementacja sieci perceptronu wielowarstwowego o wybranym
algorytmie optymalizacji gradientowej z algorytmem propagacji wstecznej i przetestowanie jego
działania na ogólnodostępnym zbiorze danych MNIST.
Zadanie zrealizowano w sposób ogólny tak, aby sieć dało się testować również na innych
zbiorach danych.
# Struktura projektu  
Ze względu na specyfikę wykonywanego zadania, zdecydowałyśmy się, że struktura projektu
będzie wyglądała w następujący sposób:
➔ main.py - zawiera wywołania eksperymentów przeprowadzonych w trakcie realizacji
zadania
➔ experiments.py - funkcje z definicjami eksperymentów przeprowadzonych w trakcie
zadania
➔ multilayer_perceptron.py - implementacja wielowarstwowego perceptronu (naszej sieci
neuronowej) w postaci klasy MultilayerPerceptron.
➔ neuron.py - implementacja pojedynczego węzła warst ukrytych oraz warstwy wyjściowej
sieci neuronowej w postaci klasy Neuron.
➔ sgd.py - implementacja funkcji pomocniczych, wykorzystywanych do obliczania
gradientu na potrzeby zmiany wag węzłów sieci neuronowej.
➔ activation_funcions.py - reprezentacja używanych funkcji aktywacji przy pomocy
biblioteki sympy (zostały zaimplementowane takie funkcje jak tnh, sigmoid oraz ReLU).
➔ get_data.py - funckje pomocnicze, wykorzystywane w celu załadowania oraz
przetworzenia (normalizacji wartości oraz oddzielenia od zbioru danych nagłówka z
nazwami kolumn) zbioru danych, na których następnie trenowaliśmy oraz sprawdzaliśmy
sieć neuronową.
➔ plot_data.py - funkcije pomocnicze pozwalające na stworzenie wykresów - krzywej
uczenia oraz macierzy pomyłek
W ramach projektu były wykorzystywane różne biblioteki zewnętrzne, ułatwiające wykonanie
zadania:
● numpy, math - do działań na różnych strukturach danych oraz przeprowadzenia na nich
obliczeń/działań matematycznych.
● sympy - do przedstawienia w czytelny sposób funkcji aktywacji oraz obliczanie ich gradientu
● random - do generowania liczb losowych na podstawie których wybierane są próbki do
uczenia sieci neuronowej
● sklearn, pandas - do przetwarzania/załadowania zbiorów danych
● mlextend - biblioteka zawierająca narzędzie pozwalające w łatwy sposób automatycznie
generować ilustrację macierz pomyłek.
● matplotlib - biblioteka ułatwiająca tworzenie wykresów
# Decyzje projektowe
# Opis wykonanych eksperymentów
# Wyniki
# Wnioski
