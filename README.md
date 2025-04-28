import torch

# Suma de matrices
def sumar_matrices(matriz1, matriz2):
    try:
        return matriz1 + matriz2
    except RuntimeError as e:
        return f"Error en suma de matrices: {e}"

# Resta de matrices
def restar_matrices(matriz1, matriz2):
    try:
        return matriz1 - matriz2
    except RuntimeError as e:
        return f"Error en resta de matrices: {e}"

# Multiplicación de matrices
def multiplicar_matrices(matriz1, matriz2):
    try:
        return torch.matmul(matriz1, matriz2)
    except RuntimeError as e:
        return f"Error en multiplicación de matrices: {e}"

# Cálculo del determinante
def calcular_determinante(matriz):
    try:
        return torch.linalg.det(matriz)
    except RuntimeError as e:
        return f"Error al calcular determinante: {e}"

# Cálculo de la inversa
def calcular_inversa(matriz):
    if es_invertible(matriz):
        try:
            return torch.linalg.inv(matriz)
        except RuntimeError as e:
            return f"Error al calcular la inversa: {e}"
    else:
        return "La matriz no es invertible."

# Verificar si una matriz es invertible
def es_invertible(matriz):
    if matriz.shape[0] != matriz.shape[1]:
        return False  # Solo matrices cuadradas pueden ser invertidas
    determinante = torch.linalg.det(matriz)
    return not torch.isclose(determinante, torch.tensor(0.0))

# Función para ingresar una matriz desde la entrada del usuario
def ingresar_matriz():
    filas = int(input("Ingrese el número de filas de la matriz: "))
    columnas = int(input("Ingrese el número de columnas de la matriz: "))
    datos = []
    print("Ingrese los elementos de la matriz:")
    for i in range(filas):
        fila = list(map(float, input(f"Ingrese los elementos de la fila {i + 1}: ").split()))
        if len(fila) != columnas:
            raise ValueError("La cantidad de elementos de la fila no coincide con el número de columnas.")
        datos.append(fila)
    return torch.tensor(datos)

# Función para mostrar el menú de operaciones
def menu():
    while True:
        print("\nSeleccione la operación que desea realizar:")
        print("1. Suma de matrices")
        print("2. Resta de matrices")
        print("3. Multiplicación de matrices")
        print("4. Determinante de matriz")
        print("5. Inversa de la matriz")
        print("0. Salir")

        opcion = input("Ingrese el número de la operación: ")

        if opcion == '1':
            matriz1 = ingresar_matriz()
            matriz2 = ingresar_matriz()
            print(sumar_matrices(matriz1, matriz2))
        elif opcion == '2':
            matriz1 = ingresar_matriz()
            matriz2 = ingresar_matriz()
            print(restar_matrices(matriz1, matriz2))
        elif opcion == '3':
            matriz1 = ingresar_matriz()
            matriz2 = ingresar_matriz()
            print(multiplicar_matrices(matriz1, matriz2))
        elif opcion == '4':
            matriz = ingresar_matriz()
            print(calcular_determinante(matriz))
        elif opcion == '5':
            matriz = ingresar_matriz()
            print(calcular_inversa(matriz))
        elif opcion == '0':
            print("¡Hasta luego!")
            break
        else:
            print("Opción inválida. Intente nuevamente.")

# Ejecutar el menú
menu()

