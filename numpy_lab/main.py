"""
Модуль для работы с массивами NumPy, анализа данных и визуализации.
"""

import os
from typing import Union

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create_vector() -> np.ndarray:
    """
    Создать массив от 0 до 9.

    Returns:
        np.ndarray: Массив чисел от 0 до 9 включительно.
    """
    return np.arange(10)


def create_matrix() -> np.ndarray:
    """
    Создать матрицу 5x5 со случайными числами [0, 1).

    Returns:
        np.ndarray: Матрица 5x5 со случайными значениями от 0 до 1.
    """
    return np.random.rand(5, 5)


def reshape_vector(vec: np.ndarray) -> np.ndarray:
    """
    Преобразовать вектор формы (10,) в матрицу формы (2, 5).

    Args:
        vec (np.ndarray): Входной массив формы (10,).

    Returns:
        np.ndarray: Преобразованный массив формы (2, 5).
    """
    return vec.reshape(2, 5)


def transpose_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Транспонировать матрицу.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.transpose.html

    Args:
        mat (np.ndarray): Входная матрица.

    Returns:
        np.ndarray: Транспонированная матрица.
    """
    return mat.T


def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Поэлементное сложение векторов одинаковой длины (векторизация без циклов).

    Args:
        a (np.ndarray): Первый вектор.
        b (np.ndarray): Второй вектор.

    Returns:
        np.ndarray: Результат поэлементного сложения.
    """
    return a + b


def scalar_multiply(vec: np.ndarray, scalar: Union[int, float]) -> np.ndarray:
    """
    Умножение вектора на скаляр.

    Args:
        vec (np.ndarray): Входной вектор.
        scalar (int | float): Число для умножения.

    Returns:
        np.ndarray: Результат умножения вектора на скаляр.
    """
    return vec * scalar


def elementwise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Поэлементное умножение двух массивов одинаковой формы.

    Args:
        a (np.ndarray): Первый вектор или матрица.
        b (np.ndarray): Второй вектор или матрица.

    Returns:
        np.ndarray: Результат поэлементного умножения.
    """
    return a * b


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Вычислить скалярное произведение двух векторов.

    Args:
        a (np.ndarray): Первый вектор.
        b (np.ndarray): Второй вектор.

    Returns:
        float: Скалярное произведение векторов.
    """
    return float(np.dot(a, b))


def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Умножение двух матриц.

    Args:
        a (np.ndarray): Первая матрица.
        b (np.ndarray): Вторая матрица.

    Returns:
        np.ndarray: Результат умножения матриц.
    """
    return a @ b


def matrix_determinant(a: np.ndarray) -> float:
    """
    Вычислить определитель квадратной матрицы.

    Args:
        a (np.ndarray): Квадратная матрица.

    Returns:
        float: Определитель матрицы.
    """
    return float(np.linalg.det(a))


def matrix_inverse(a: np.ndarray) -> np.ndarray:
    """
    Вычислить обратную матрицу.

    Args:
        a (np.ndarray): Квадратная невырожденная матрица.

    Returns:
        np.ndarray: Обратная матрица.
    """
    return np.linalg.inv(a)


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решить систему линейных уравнений Ax = b.

    Args:
        a (np.ndarray): Матрица коэффициентов A.
        b (np.ndarray): Вектор свободных членов b.

    Returns:
        np.ndarray: Вектор решения x.
    """
    return np.linalg.solve(a, b)


def load_dataset(path: str = "data/students_scores.csv") -> np.ndarray:
    """
    Загрузить CSV-файл и вернуть данные в виде NumPy-массива.

    Args:
        path (str): Путь к CSV-файлу.

    Returns:
        np.ndarray: Загруженные данные в виде массива.
    """
    return pd.read_csv(path).to_numpy()


def statistical_analysis(data: np.ndarray) -> dict[str, float]:
    """
    Выполнить статистический анализ одномерного массива данных.

    Вычисляет: среднее, медиану, стандартное отклонение, минимум,
    максимум, 25-й и 75-й перцентили.

    Args:
        data (np.ndarray): Одномерный массив числовых данных.

    Returns:
        dict[str, float]: Словарь со статистическими показателями.
    """
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "percentile25": float(np.percentile(data, 25)),
        "percentile75": float(np.percentile(data, 75)),
    }


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Выполнить Min-Max нормализацию данных.

    Формула: (x - min) / (max - min)

    Args:
        data (np.ndarray): Входной массив данных.

    Returns:
        np.ndarray: Нормализованный массив в диапазоне [0, 1].
                   Если все значения одинаковы, возвращает нулевой массив.
    """
    data_min = np.min(data)
    data_max = np.max(data)
    data_range = data_max - data_min

    if data_range != 0:
        return (data - data_min) / data_range
    return np.zeros_like(data, dtype=float)


def plot_histogram(data: np.ndarray) -> None:
    """
    Построить и сохранить гистограмму распределения данных.

    Args:
        data (np.ndarray): Данные для построения гистограммы.
    """
    os.makedirs("plots", exist_ok=True)

    # Преобразуем 2D-массив в 1D при необходимости
    if data.ndim > 1:
        data = data.flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, color="skyblue", edgecolor="black", alpha=0.7)

    plt.title("Гистограмма распределения")
    plt.xlabel("Значение")
    plt.ylabel("Частота")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/histogram.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_heatmap(matrix: np.ndarray) -> None:
    """
    Построить и сохранить тепловую карту корреляции.

    Args:
        matrix (np.ndarray): Матрица корреляции (квадратная).
    """
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Корреляция"},
    )

    plt.title("Тепловая карта корреляции", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Признаки", fontsize=12)
    plt.ylabel("Признаки", fontsize=12)

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig("plots/heatmap.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_line(x: np.ndarray, y: np.ndarray) -> None:
    """
    Построить и сохранить линейный график: студент → оценка.

    Args:
        x (np.ndarray): Номера студентов (ось X).
        y (np.ndarray): Оценки студентов (ось Y).
    """
    os.makedirs("plots", exist_ok=True)

    # Адаптивный размер фигуры под количество точек
    width = max(12, len(x) * 0.5)
    plt.figure(figsize=(width, 6))

    plt.plot(
        x,
        y,
        marker="o",
        linestyle="-",
        color="steelblue",
        markersize=5,
        linewidth=1,
        markerfacecolor="red",
        markeredgecolor="darkred",
        label="Оценка по математике",
    )

    plt.title(
        "Зависимость оценки по математике от номера студента",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Номер студента", fontsize=12)
    plt.ylabel("Оценка", fontsize=12)

    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=10)

    # Адаптивное отображение подписей оси X
    n_students = len(x)
    if n_students <= 25:
        plt.xticks(x, rotation=45, ha="right", fontsize=9)
    elif n_students <= 50:
        plt.xticks(x, rotation=90, fontsize=8)
    else:
        step = max(1, n_students // 25)
        plt.xticks(x[::step], rotation=45, ha="right", fontsize=9)

    plt.tight_layout()
    plt.savefig("plots/line_plot.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    math_data = load_dataset()[:, 0]
    student_ids = np.arange(1, len(math_data) + 1)
    plot_line(student_ids, math_data)