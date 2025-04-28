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

# Resolver un sistema de ecuaciones lineales
def resolver_sistema_ecuaciones(matriz_coeficientes, vector_terminos):
    if es_invertible(matriz_coeficientes):
        try:
            # Usamos la inversa de la matriz de coeficientes
            solucion = torch.matmul(torch.linalg.inv(matriz_coeficientes), vector_terminos)
            return solucion
        except RuntimeError as e:
            return f"Error al resolver el sistema de ecuaciones: {e}"
    else:
        return "La matriz de coeficientes no es invertible."

# Función para ingresar un sistema de ecuaciones
def ingresar_sistema_ecuaciones():
    num_ecuaciones = int(input("Ingrese el número de ecuaciones en el sistema: "))
    coeficientes = []
    print("Ingrese los coeficientes de las ecuaciones (debe separar los valores por espacios):")
    
    for i in range(num_ecuaciones):
        ecuacion = list(map(float, input(f"Ingrese los coeficientes de la ecuación {i + 1}: ").split()))
        coeficientes.append(ecuacion)
    
    matriz_coeficientes = torch.tensor(coeficientes)
    
    # Ingresar los términos constantes (lado derecho de las ecuaciones)
    vector_terminos = list(map(float, input("Ingrese los términos constantes (lado derecho) separados por espacios: ").split()))
    vector_terminos = torch.tensor(vector_terminos)
    
    return matriz_coeficientes, vector_terminos

# Función para mostrar el menú de operaciones
def menu():
    while True:
        print("\nSeleccione la operación que desea realizar:")
        print("1. Suma de matrices")
        print("2. Resta de matrices")
        print("3. Multiplicación de matrices")
        print("4. Determinante de matriz")
        print("5. Inversa de la matriz")
        print("6. Resolver sistema de ecuaciones lineales")
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
        elif opcion == '6':
            matriz_coeficientes, vector_terminos = ingresar_sistema_ecuaciones()
            print(resolver_sistema_ecuaciones(matriz_coeficientes, vector_terminos))
        elif opcion == '0':
            print("¡Hasta luego!")
            break
        else:
            print("Opción inválida. Intente nuevamente.")

# Ejecutar el menú
menu()

