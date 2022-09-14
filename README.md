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
* main.py - zawiera wywołania eksperymentów przeprowadzonych w trakcie realizacji
zadania
* experiments.py - funkcje z definicjami eksperymentów przeprowadzonych w trakcie
zadania
* multilayer_perceptron.py - implementacja wielowarstwowego perceptronu (naszej sieci
neuronowej) w postaci klasy MultilayerPerceptron.
* neuron.py - implementacja pojedynczego węzła warst ukrytych oraz warstwy wyjściowej
sieci neuronowej w postaci klasy Neuron.
* sgd.py - implementacja funkcji pomocniczych, wykorzystywanych do obliczania
gradientu na potrzeby zmiany wag węzłów sieci neuronowej.
* activation_funcions.py - reprezentacja używanych funkcji aktywacji przy pomocy
biblioteki sympy (zostały zaimplementowane takie funkcje jak tnh, sigmoid oraz ReLU).
* get_data.py - funckje pomocnicze, wykorzystywane w celu załadowania oraz
przetworzenia (normalizacji wartości oraz oddzielenia od zbioru danych nagłówka z
nazwami kolumn) zbioru danych, na których następnie trenowaliśmy oraz sprawdzaliśmy
sieć neuronową.
* plot_data.py - funkcije pomocnicze pozwalające na stworzenie wykresów - krzywej
uczenia oraz macierzy pomyłek 
\
&nbsp; 
\
&nbsp;
W ramach projektu były wykorzystywane różne biblioteki zewnętrzne, ułatwiające wykonanie
zadania: 
\
&nbsp;
* numpy, math - do działań na różnych strukturach danych oraz przeprowadzenia na nich
obliczeń/działań matematycznych.
* sympy - do przedstawienia w czytelny sposób funkcji aktywacji oraz obliczanie ich gradientu
* random - do generowania liczb losowych na podstawie których wybierane są próbki do
uczenia sieci neuronowej
* sklearn, pandas - do przetwarzania/załadowania zbiorów danych
* mlextend - biblioteka zawierająca narzędzie pozwalające w łatwy sposób automatycznie
generować ilustrację macierz pomyłek.
* matplotlib - biblioteka ułatwiająca tworzenie wykresów
# Decyzje projektowe

Główna część implementacji problemu zawarta jest w dwóch klasach - Neuron oraz
MultilayerPerceptron.
\
&nbsp; 
***Klasa Neuron*** stanowi reprezentację pojedynczej jednostki sieci przetwarzającej informację, jego
główne atrybuty to:
* używana funkcja aktywacji
* wagi wejściowe (losowane podczas inicjalizacji z rozkładu jednostajnego)
* wartość bias
\
&nbsp; 
\
&nbsp; 
Również zostały wprowadzone dodatkowe atrybuty na potrzeby przeprowadzenia propagacji
wstecznej:
* wartość wyjścia (po przejściu przez funkcję aktywacji)
* wartość wyjścia (przez przejściem przez funkcję aktywacji)

\
&nbsp; 
Wspomniana klasa Neuron również posiada również kilka metod (oprócz funkcji do
inicjalizacji bias’u oraz wag), pozwalających na zmianę wartości wag oraz bias’u podczas
przeprowadzenia propagacji wstecz. Do tych funkcji wprowadzany są obliczony gradient oraz wartość
kroku (parametr beta), na podstawie których obliczane są nowe wartości wag oraz bias’u.
\
&nbsp;
\
&nbsp;
***Klasa MultilayerPerceptron*** stanowi reprezentację całej sieci. Zawiera metody pozwalające
na inicjalizację sieci, trenowanie sieci, czy uzyskiwanie przewidywań na podstawie próbki danych
wejściowych. Dana klasa pozwala na stworzenia sieci neuronowej o dowolnej liczbie wejść, wyjść,
warstw ukrytych oraz liczbie węzłów w środku takiej warstwy ukrytej.
Najważniejsze z punktu widzenia działania programu atrybut tej klasy to:
* network - lista list reprezentująca sieć neuronową. Każdy jej wiersz odpowiada jednej
warstwie. Elementy warstwy stanowią obiekty klasy Neuron, przykładowo dla 3 warstw i 2
neuronów w każdej warstwie:\
&nbsp;
[[neuron_11, neuron_12], [neuron_21, neuron_22], [neuron_31, neuron_32]]
\
&nbsp;
Funkcja inicjalizująca pozwala wybrać dowolną liczbę warstw i ustawić różną liczbę
neuronów na każdej z nich (poza ostatnią, w której liczba neuronów jest automatycznie
dostosowywana do wymiaru wyjścia z sieci). Można też każdej warstwie ustawić inną funkcję
aktywacji (identyczną dla wszystkich neuronów w tej warstwie).\
&nbsp;
Trenowanie sieci, reprezentowanej za pomocą klasy MultilayerPerceptron odbywa się na
zasadzie propagacji wstecznej, gdzie najpierw jest obliczana pomyłka dla warstw wyjściowych, a
następnie na jej podstawie jest obliczana odchyłka dla każdej z wag oraz biasów (którą uśredniamy
dla N próbek z paczki) dla każdej kolejnej warstwy (od ostatniej do pierwszej) na zasadzie reakcji
łańcuchowej

# Opis wykonanych eksperymentów

Ponieważ na chwilę obecną nie istnieją metody automatycznej generacji hiperparametrów
sieci neuronowej dla wszystkich możliwych zadań (typu regresji czy klasyfikacji) i wszystkich
możliwych danych, konfiguracja odbyła się metodą prób i błędów.\
&nbsp;
Zaczęłyśmy od testowania naszej sieci dla funkcji prostszych, wymagających prostszych
konfiguracji sieci neuronowej (funkcji OR oraz AND), następnie dla bardziej skomplikowanego
zbioru danych, gdzie sieć miałaby kilka wejść i jedno wyjście (klasyfikator zero-jedynkowy
wykorzystywany do predykcji czy jakieś mieszkanie o określonych cechach ma cenę wyższą od
średniej, czy jednak nie), na koniec ukształtowałyśmy naszą sieć w taki sposób aby mogła
przetworzyć i przewidywać wartości dla docelowego zbioru MNIST.\
&nbsp;
\
&nbsp;
Dla pierwszego typu eksperymentów, po przeprowadzeniu kilku prób, najlepszym zestawem
parametrów był:
* Ustawienie 1 warstwy ukrytej o 3 węzłach
* Wykorzystanie funkcji aktywacji tanh dla warstwy ukrytej oraz logistic dla warstwy
wyjściowej
* Wykorzystanie parametru beta równego 0.9
 \
&nbsp;
\
&nbsp;
Dla drugiego typu eksperymentów, po przeprowadzeniu kilku prób, najlepszym zestawem
parametrów był:
* Wykorzystanie 2 warstw ukrytych o 32 węzłach
* Wykorzystanie funkcji aktywacji tanh dla pierwszej warstwy ukrytej, ReLU dla drugie
warstwy ukrytej oraz logistic dla warstwy wyjściowej
* Wykorzystanie parametru beta równego 0.4\
&nbsp;
\
&nbsp;
Dla trzeciego i ostatniego eksperymentu, po przeprowadzeniu kilku prób, najlepszym
zestawem parametrów był:
* Wykorzystanie 2 warstw ukrytych o 192 węzłach
* Wykorzystanie funkcji aktywacji tanh dla pierwszej warstwy ukrytej, ReLU dla drugiej
warstwy ukrytej oraz logistic dla warstwy wyjściowej
* Wykorzystanie parametru beta równego 0.4

# Wyniki
# Wnioski
