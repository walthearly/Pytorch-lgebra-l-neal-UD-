import torch

# Suma de matrices
def sumar_matrices(matriz1, matriz2):
    return matriz1 + matriz2

# Resta de matrices
def restar_matrices(matriz1, matriz2):
    return matriz1 - matriz2

# Multiplicación de matrices
def multiplicar_matrices(matriz1, matriz2):
    return torch.matmul(matriz1, matriz2)

# Cálculo del determinante
def calcular_determinante(matriz):
    return torch.linalg.det(matriz)

# Cálculo de la inversa
def calcular_inversa(matriz):
    # Verificar si la matriz es invertible antes de calcular la inversa
    if es_invertible(matriz):
        return torch.linalg.inv(matriz)
    else:
        return "La matriz no es invertible."

# Verificar si una matriz es invertible
def es_invertible(matriz):
    if matriz.shape[0] != matriz.shape[1]:
        return False  # Solo matrices cuadradas pueden ser invertidas
    determinante = torch.linalg.det(matriz)
    return not torch.isclose(determinante, torch.tensor(0.0))

# Función de ejemplo para mostrar cómo usar estas funciones
def ejemplo():
    # Crear matrices de ejemplo
    matriz1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    matriz2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    # Operaciones
    print("Suma de matrices:")
    print(sumar_matrices(matriz1, matriz2))

    print("\nResta de matrices:")
    print(restar_matrices(matriz1, matriz2))

    print("\nMultiplicación de matrices:")
    print(multiplicar_matrices(matriz1, matriz2))

    print("\nDeterminante de matriz 1:")
    print(calcular_determinante(matriz1))

    print("\nInversa de matriz 1:")
    print(calcular_inversa(matriz1))

# Ejecutar ejemplo
ejemplo()
