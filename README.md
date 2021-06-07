# msid-4
Metody Systemowe i Decyzyjne zadanie 4


#Fashion-MNIST ##Wprowadzenie Istotą zadania jest zaimplementowanie modelu pozwalającego na klasyfikację miniaturek zdjęć przedstawiających ubrania z Fashion-MNIST. Fashion-MNIST to zbiór danych obrazów artykułów Zalando, składający się z zestawu treningowego 60 000 przykładów i zestawu testowego 10 000 przykładów. Każdy przykład to obrazek w skali szarości 28x28, powiązany z etykietą z 10 klas.

Oto przykład, jak wyglądają dane:

![fashion-mnist](./image/fashion-mnist.png)


##Pierwszym klasyfikatorem, który omówimy, jest KNN (k-najbliższych sąsiedów). Metryką odległości, którą zdecydowałem się użyć, jest odległość Manhattan. Algorytm KNN zakłada, że podobne rzeczy istnieją w bliskiej odległości. Parametr K reprezentuje liczbę sąsiadów, których bierzemy pod uwagę.

