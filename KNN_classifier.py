import file_reader
import numpy as np


def load_data():
    X_train, y_train = file_reader.load_mnist('fashion', kind='train')
    X_test, y_test = file_reader.load_mnist('fashion', kind='t10k')
    data = {'Xval': X_test,
            'Xtrain': X_train,
            'yval': y_test,
            'ytrain': y_train}
    return data


def manhattan_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. Odleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """
    X = X.astype(int)
    X_train = X_train.astype(int)
    Dist = np.zeros((len(X), len(X_train)))
    for i in range(0, len(X)):
        Dist[i] = np.sum(abs(X_train - X[i]), axis=1)
        print(i)
    return Dist


def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.
    """
    return (y[Dist.argsort(kind='mergesort')])
    pass


def p_y_x_knn(y, k):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
    """
    result = np.zeros((len(y), 10))
    for i in range(len(y)):
        for j in range(k):
            result[i, y[i, j]] += 1

    return result / k
    pass


def classification_error(p_y_x, y_true):
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """
    sum = 0
    for i in range(y_true.shape[0]):
        biggestProbIndex = 0
        best_p = 0
        for j in range(p_y_x[i].shape[0]):
            if p_y_x[i, j] >= best_p:
                biggestProbIndex = j
                best_p = p_y_x[i, j]
        if y_true[i] != biggestProbIndex:
            sum += 1

    sum = sum / y_true.shape[0]
    return sum

    pass


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """
    labeled = sort_train_labels_knn(np.array(manhattan_distance(X_val, X_train)), y_train)
    best_k = k_values[0]
    best_error = classification_error(p_y_x_knn(labeled, k_values[0]), y_val)
    errors = [best_error]
    for i in range(1, len(k_values)):
        curr_error = classification_error(p_y_x_knn(labeled, k_values[i]), y_val)
        errors.append(curr_error)
        if curr_error < best_error:
            best_error = curr_error
            best_k = k_values[i]
    return best_error, best_k, errors
    pass


def run_training():
    data = load_data()
    print(data['Xtrain'].shape)
    # KNN model selection
    k_values = range(1, 10)
    print('\n------------- Selekcja liczby sasiadow dla modelu dla KNN -------------')
    print('-------------------- Wartosci k: 1, 2, 3, 4, 5, 6, 7, 8, 9 -----------------------')

    error_best, best_k, errors = model_selection_knn(data['Xval'],
                                                     data['Xtrain'],
                                                     data['yval'],
                                                     data['ytrain'],
                                                     k_values)
    print(f'Najlepsze k: {best_k} i najlepszy blad: {error_best:.4f}')


run_training()
