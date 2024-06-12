
Concepte si Aplicatii in Vederea Artificiala

Versiuni
Python 3.11.5
Numpy  1.26.2
Opencv 4.8.1.78

(Toate modulele se regasesc in environment.yaml)

Codul este alcatuit dintr-un jupyter notebook si un python package. In fisierul .ipynb se afla functiile finale pentru rezolvarea cerintelor. In pachetul additional se regaseste un fisier in care sunt functiile pentru diverse tipuri de citire a unei imagini.
Fisierul projectFunctions contine multe dintre functiile implicate in rezolvarea cerintelor, precum extragerea careului, aplicarea preprocesarilor, template matching-ul, etc. dar si unele functii care au fost folosite pentru testarea solutiei sau salvarea unor rezultate(salvare template, identificare numar cercuri, testarea unui singur domino, etc.)

dict_nr_circles.pkl contine un dictionar cu perechi avand _key_: numele fisierului template si _value_: numarul de pe domino

(template_X, _numar_)

Primele doua cerinte sunt rezolvate concomitent de functia genereaza_solutie. Aceasta poate sa fie apelata cu path-ul pentru folderul datelor care trebuie prezise ca prim parametru, si un folder in care sa scrie rezultatele:\
genereaza_solutie("C:/Users/Laurentiu/PycharmProjects/CAVAtest/antrenare", "rezultate")


Daca folderul pentru rezultate nu este gol, sunt sterse datele din el.


In cadrul acesteia(ultima linie) se apeleaza si functia care calculeaza scorul. Pentru a functiona trebuie ca fisierele cu rezultatele pentru cerinta 1 si 2 sa fie deja generate, deoarece sunt folosite pentru citire in cadrul generarii scorului.
Fisierele cu primele rezultate sunt generate daca codul este rulat in ordinea din notebook.

Timpul de executie este de aproximativ 5 minute.

Structura proiectului:
ProjectCAVA\
│\
├── additional\
│ ├── init.py\
│ ├── universal.py\
│ └── projectFunctions.py\
│\
├── templates\
│\
├── dict_nr_circles.pkl\
├── solutie.ipynb