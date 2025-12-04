import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def comparar_filtros(caminho_imagem):
    # 1. Carrega em Escala de Cinza (0)
    img = cv2.imread(caminho_imagem, 0)
    if img is None:
        print(f"Erro: Imagem '{caminho_imagem}' não encontrada.")
        return

    # 2. Definição Manual dos Kernels (O coração da diferença)
    
    # Prewitt (Pesos iguais: 1, 1, 1)
    k_prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    k_prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # Sobel (Pesos Gaussianos: 1, 2, 1) -> Note o 2 no meio
    k_sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    k_sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # 3. Aplicação (Usando float64 para precisão matemática)
    # Prewitt
    px = cv2.filter2D(img, cv2.CV_64F, k_prewitt_x)
    py = cv2.filter2D(img, cv2.CV_64F, k_prewitt_y)
    mag_prewitt = np.sqrt(px**2 + py**2) # Pitágoras

    # Sobel
    sx = cv2.filter2D(img, cv2.CV_64F, k_sobel_x)
    sy = cv2.filter2D(img, cv2.CV_64F, k_sobel_y)
    mag_sobel = np.sqrt(sx**2 + sy**2)   # Pitágoras

    # 4. Normalização (0-255) para visualização
    res_prewitt = cv2.normalize(mag_prewitt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    res_sobel = cv2.normalize(mag_sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 5. Visualização
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(res_prewitt, cmap='gray')
    plt.title("Prewitt (1,1,1)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(res_sobel, cmap='gray')
    plt.title("Sobel (1,2,1)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- CONFIGURAÇÃO ---
# Basta colocar o nome da sua imagem aqui:

CAMINHO_IMAGEM = f"{os.path.dirname(__file__)}/image.png"

comparar_filtros(CAMINHO_IMAGEM)