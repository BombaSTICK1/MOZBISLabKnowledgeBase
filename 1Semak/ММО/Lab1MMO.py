# Вариант 5
import numpy as np
from collections import Counter

# Данные обучающей выборки (Вариант 5)
X_train = np.array([
    [88, 12], [96, 4], [70, 30], [76, 24], 
    [65, 35], [98, 2], [90, 10]
])
y_train = np.array(['J7', 'Rich', 'Gardens', 'I', 'Gardens', 'Rich', 'J7'])
X_test = np.array([87, 13]) # Немаркированный объект

# Функции вычисления расстояний
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def chebyshev_distance(x1, x2):
    return np.max(np.abs(x1 - x2))

# Детальная реализация KNN
def knn_predict_detailed(X_train, y_train, X_test, k=3, metric='euclidean'):
    print(f"\n{'='*50}\nМЕТРИКА: {metric.upper()}\n{'='*50}")
    print(f"Немаркированный объект: {X_test}")
    
    print("\n--- ШАГ 1: Вычисление расстояний до всех известных объектов ---")
    distances = []
    for i in range(len(X_train)):
        if metric == 'manhattan':
            dist = manhattan_distance(X_test, X_train[i])
            formula = f"|{X_test[0]}-{X_train[i][0]}| + |{X_test[1]}-{X_train[i][1]}|"
        elif metric == 'euclidean':
            dist = euclidean_distance(X_test, X_train[i])
            formula = f"sqrt(({X_test[0]}-{X_train[i][0]})^2 + ({X_test[1]}-{X_train[i][1]})^2)"
        elif metric == 'chebyshev':
            dist = chebyshev_distance(X_test, X_train[i])
            formula = f"max(|{X_test[0]}-{X_train[i][0]}|, |{X_test[1]}-{X_train[i][1]}|)"
            
        print(f"Объект {i+1} {X_train[i]} (Класс: {y_train[i]:<7}):")
        print(f"  Формула: {formula}")
        print(f"  Расстояние = {dist:.4f}")
        distances.append((dist, y_train[i], i+1))
    
    print("\n--- ШАГ 2: Сортировка объектов по возрастанию расстояния ---")
    distances.sort(key=lambda x: x[0])
    for rank, (d, label, idx) in enumerate(distances):
        print(f"{rank+1}-е место: Объект {idx} (Расстояние: {d:.4f}, Класс: {label})")
        
    print(f"\n--- ШАГ 3: Выбор k={k} ближайших соседей ---")
    k_nearest = distances[:k]
    k_nearest_labels = [label for _, label, _ in k_nearest]
    print(f"Берем первые {k} объекта. Их классы: {k_nearest_labels}")
    
    print("\n--- ШАГ 4: Голосование (поиск самого частого класса) ---")
    counts = Counter(k_nearest_labels)
    for label, count in counts.items():
        print(f"Голосов за '{label}': {count}")
        
    most_common = counts.most_common(1)[0][0]
    print(f"\n>>> ИТОГОВЫЙ ПРЕДСКАЗАННЫЙ КЛАСС: {most_common} <<<")
    return most_common

# Запуск с детальным выводом
knn_predict_detailed(X_train, y_train, X_test, k=3, metric='manhattan')
knn_predict_detailed(X_train, y_train, X_test, k=3, metric='euclidean')
knn_predict_detailed(X_train, y_train, X_test, k=3, metric='chebyshev')