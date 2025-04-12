# Pytorch-lgebra-l-neal-UD-
Universidad distrital Francisco José de Caldas 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Definición de la red neuronal simple
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Capa de entrada con 28x28 (imagen MNIST)
        self.fc1 = nn.Linear(28*28, 128)  # Primera capa lineal
        self.fc2 = nn.Linear(128, 64)     # Segunda capa lineal
        self.fc3 = nn.Linear(64, 10)      # Capa de salida (10 clases)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Aplanar la imagen
        x = torch.relu(self.fc1(x))  # Activación ReLU
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Salida sin activación (se hace softmax en la pérdida)
        return x

# Configuración de transformaciones para normalizar imágenes
transform = transforms.Compose([
    transforms.ToTensor(),  # Convertir las imágenes a tensores
    transforms.Normalize((0.5,), (0.5,))  # Normalizar las imágenes
])

# Cargar el dataset MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Crear el modelo, la función de pérdida y el optimizador
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # Pérdida para clasificación multi-clase
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # Borrar los gradientes de iteraciones previas

        # Forward pass
        outputs = model(inputs)
        
        # Calcular la pérdida
        loss = criterion(outputs, labels)
        
        # Backward pass y optimización
        loss.backward()
        optimizer.step()

        # Estadísticas del entrenamiento
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0

print("Entrenamiento completado.")

# Guardar el modelo entrenado
torch.save(model.state_dict(), 'modelo_mnist.pth')
