from main import *
import pytest


def test_create_vector():
    v = create_vector()
    assert isinstance(v, np.ndarray)
    assert v.shape == (10,)
    assert np.array_equal(v, np.arange(10))


def test_create_matrix():
    m = create_matrix()
    assert isinstance(m, np.ndarray)
    assert m.shape == (5, 5)
    assert np.all((m >= 0) & (m < 1))


def test_reshape_vector():
    v = np.arange(10)
    reshaped = reshape_vector(v)
    assert reshaped.shape == (2, 5)
    assert reshaped[0, 0] == 0
    assert reshaped[1, 4] == 9


def test_transpose_matrix():
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    expected = np.array([[1, 4],
                         [2, 5],
                         [3, 6]])
    assert np.array_equal(transpose_matrix(A), expected)
    B = np.array([[1, 2],
                  [3, 4]])
    assert np.array_equal(transpose_matrix(B), B.T)
    v = np.array([1, 2, 3])
    assert np.array_equal(transpose_matrix(v), v)


def test_vector_add():
    assert np.array_equal(
        vector_add(np.array([1,2,3]), np.array([4,5,6])),
        np.array([5,7,9])
    )
    assert np.array_equal(
        vector_add(np.array([0,1]), np.array([1,1])),
        np.array([1,2])
    )


def test_scalar_multiply():
    assert np.array_equal(
        scalar_multiply(np.array([1,2,3]), 2),
        np.array([2,4,6])
    )


def test_elementwise_multiply():
    assert np.array_equal(
        elementwise_multiply(np.array([1,2,3]), np.array([4,5,6])),
        np.array([4,10,18])
    )


def test_dot_product():
    assert dot_product(np.array([1,2,3]), np.array([4,5,6])) == 32
    assert dot_product(np.array([2,0]), np.array([3,5])) == 6


def test_matrix_multiply():
    A = np.array([[1,2],[3,4]])
    B = np.array([[2,0],[1,2]])
    assert np.array_equal(matrix_multiply(A,B), A @ B)


def test_matrix_determinant():
    A = np.array([[1,2],[3,4]])
    assert round(matrix_determinant(A),5) == -2.0


def test_matrix_inverse():
    A = np.array([[1,2],[3,4]])
    invA = matrix_inverse(A)
    assert np.allclose(A @ invA, np.eye(2))


def test_solve_linear_system():
    A = np.array([[2,1],[1,3]])
    b = np.array([1,2])
    x = solve_linear_system(A,b)
    assert np.allclose(A @ x, b)


def test_load_dataset():
    # Для теста создадим временный файл
    test_data = "math,physics,informatics\n78,81,90\n85,89,88"
    with open("test_data.csv", "w") as f:
        f.write(test_data)
    try:
        data = load_dataset("test_data.csv")
        assert data.shape == (2, 3)
        assert np.array_equal(data[0], [78,81,90])
    finally:
        os.remove("test_data.csv")


def test_statistical_analysis():
    data = np.array([1, 2, 3, 4, 100])
    result = statistical_analysis(data)
    assert result["mean"] == 22
    assert result["min"] == 1
    assert result["max"] == 100
    assert result["median"] == 3


def test_statistical_analysis_percentiles():
    data = np.arange(1, 101)  # 1..100
    result = statistical_analysis(data)
    assert np.isclose(result["percentile25"], 25.75)  # Q1
    assert np.isclose(result["percentile75"], 75.25)  # Q3


def test_statistical_analysis_std():
    data = np.array([2, 4, 4, 4, 5, 5, 7, 9])
    result = statistical_analysis(data)
    # population std (ddof=0 по умолчанию в np.std)
    assert np.isclose(result["std"], 2.0, atol=0.01)


def test_normalization():
    data = np.array([0,5,10])
    norm = normalize_data(data)
    assert np.allclose(norm, np.array([0,0.5,1]))

def test_normalize_data_constant():
    data = np.array([5, 5, 5, 5])
    result = normalize_data(data)
    # Ожидаем нулевой массив (или массив с NaN, но лучше обработать)
    assert np.allclose(result, np.zeros_like(data, dtype=float))


def test_normalize_data_negative():
    data = np.array([-10, 0, 10])
    result = normalize_data(data)
    assert np.allclose(result, [0, 0.5, 1.0])


def test_normalize_data_single_element():
    data = np.array([42])
    result = normalize_data(data)
    assert result.shape == (1,)
    assert np.isclose(result[0], 0.0)


def test_plot_histogram():
    # Просто проверяем, что функция не падает
    data = np.array([1,2,3,4,5])
    plot_histogram(data)


def test_plot_heatmap():
    matrix = np.array([[1,0.5],[0.5,1]])
    plot_heatmap(matrix)


def test_plot_line():
    x = np.array([1,2,3])
    y = np.array([4,5,6])
    plot_line(x, y)


def test_plot_histogram_creates_file(tmp_path):
    """Проверяем, что гистограмма действительно сохраняется"""
    import pathlib

    data = np.random.randn(100)
    # Временная папка для тестов
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        plot_histogram(data)
        assert pathlib.Path("plots/histogram.png").exists()
        assert pathlib.Path("plots/histogram.png").stat().st_size > 0
    finally:
        os.chdir(original_cwd)


def test_plot_heatmap_creates_file(tmp_path):
    matrix = np.random.rand(5, 5)
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        plot_heatmap(matrix)
        assert (tmp_path / "plots" / "heatmap.png").exists()
    finally:
        os.chdir(original_cwd)


def test_plot_line_creates_file(tmp_path):
    x = np.arange(10)
    y = np.random.randn(10)
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        plot_line(x, y)
        assert (tmp_path / "plots" / "line_plot.png").exists()
    finally:
        os.chdir(original_cwd)



def test_reshape_vector_wrong_size():
    v = np.arange(9)  # 9 элементов, а нужно 10
    with pytest.raises((ValueError, TypeError)):
        reshape_vector(v)


def test_matrix_inverse_singular():
    A = np.array([[1, 2],
                  [2, 4]])  # det = 0
    with pytest.raises(np.linalg.LinAlgError):
        matrix_inverse(A)


def test_solve_linear_system_incompatible_shapes():
    A = np.array([[1, 2, 3]])  # 1x3
    b = np.array([1, 2])        # 2 элемента
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        solve_linear_system(A, b)


def test_dot_product_different_lengths():
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    with pytest.raises(ValueError):
        dot_product(a, b)



