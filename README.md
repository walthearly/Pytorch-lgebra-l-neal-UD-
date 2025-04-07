# Pytorch-lgebra-l-neal-UD-
Universidad distrital Francisco José de Caldas 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Cargar el conjunto de datos MNIST
transformacion = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataloader = DataLoader(
    datasets.MNIST('./datos', train=True, download=True, transform=transformacion),
    batch_size=128, shuffle=True
)

# 2. Definimos el Generador
class Generador(nn.Module):
    def __init__(self, dimension_ruido=100):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(dimension_ruido, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        return self.red(z).view(-1, 1, 28, 28)

# 3. Definimos el Discriminador
class Discriminador(nn.Module):
    def __init__(self):
        super().__init__()
        self.red = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, imagen):
        return self.red(imagen)

# 4. Iniciamos los modelos
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generador = Generador().to(dispositivo)
discriminador = Discriminador().to(dispositivo)

# 5. Función de pérdida y optimizadores
criterio = nn.BCELoss()
optimizador_G = optim.Adam(generador.parameters(), lr=0.0002)
optimizador_D = optim.Adam(discriminador.parameters(), lr=0.0002)

# 6. Entrenamiento
dimension_ruido = 100

for epoca in range(5):
    for imagenes_reales, _ in dataloader:
        imagenes_reales = imagenes_reales.to(dispositivo)
        tamano_lote = imagenes_reales.size(0)

        # Etiquetas para entrenamiento
        reales = torch.ones(tamano_lote, 1).to(dispositivo)
        falsas = torch.zeros(tamano_lote, 1).to(dispositivo)

        # --- Entrenamos el Discriminador ---
        z = torch.randn(tamano_lote, dimension_ruido).to(dispositivo)
        imagenes_falsas = generador(z)

        perdida_real = criterio(discriminador(imagenes_reales), reales)
        perdida_falsa = criterio(discriminador(imagenes_falsas.detach()), falsas)
        perdida_D = perdida_real + perdida_falsa

        optimizador_D.zero_grad()
        perdida_D.backward()
        optimizador_D.step()

        # --- Entrenamos el Generador ---
        z = torch.randn(tamano_lote, dimension_ruido).to(dispositivo)
        imagenes_generadas = generador(z)
        salida_discriminador = discriminador(imagenes_generadas)
        perdida_G = criterio(salida_discriminador, reales)  # Queremos engañar al discriminador

        optimizador_G.zero_grad()
        perdida_G.backward()
        optimizador_G.step()

    print(f"Época {epoca+1} - Pérdida D: {perdida_D.item():.4f}, Pérdida G: {perdida_G.item():.4f}")

    # Visualizamos algunas imágenes generadas
    with torch.no_grad():
        z_muestra = torch.randn(16, dimension_ruido).to(dispositivo)
        imagenes = generador(z_muestra).cpu()
        imagenes = imagenes.view(-1, 1, 28, 28)

        plt.figure(figsize=(4, 4))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(imagenes[i].squeeze(), cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()


