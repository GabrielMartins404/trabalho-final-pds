import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    # --- 1. VISUALIZAR O KERNEL 3D (O SINO) ---
    
    # Criar um kernel Gaussiano 15x15 com Sigma 3
    # Usamos getGaussianKernel que retorna 1D, então multiplicamos para ter 2D
    tamanho = 15
    sigma = 2.5
    kernel_1d = cv2.getGaussianKernel(tamanho, sigma)
    kernel_2d = kernel_1d * kernel_1d.T

    # Preparar gráfico 3D
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 1: Superfície 3D
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    x = np.arange(0, tamanho)
    y = np.arange(0, tamanho)
    X, Y = np.meshgrid(x, y)
    
    ax1.plot_surface(X, Y, kernel_2d, cmap='viridis', linewidth=0)
    ax1.set_title(f'Kernel Gaussiano 3D\n(Sigma={sigma})')
    ax1.set_zlabel('Peso')

    # --- 2. COMPARAR EFEITO NA IMAGEM (Média vs Gaussiano) ---
    
    # Criar imagem com ruído e formas
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
    # Adicionar ruído "Sal e Pimenta"
    ruido = np.random.randint(0, 100, (200, 200))
    img = np.where(ruido < 5, 0, img)   # Pontos pretos
    img = np.where(ruido > 95, 255, img) # Pontos brancos

    # Aplicar Filtro de Média (Box Blur)
    # Tamanho 9x9
    img_media = cv2.blur(img, (9, 9))

    # Aplicar Filtro Gaussiano
    # Mesmo tamanho 9x9, Sigma automático
    img_gauss = cv2.GaussianBlur(img, (9, 9), 0)

    # Plot 2: Comparação Visual
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')
    
    # Montar uma imagem composta para exibição
    # Cima: Original
    # Baixo Esq: Média | Baixo Dir: Gaussiano
    linha1 = img
    linha2 = np.hstack((img_media, img_gauss))
    
    # Truque para exibir tudo junto no matplotlib (redimensionar linha 1)
    linha1_zao = cv2.resize(linha1, (400, 200), interpolation=cv2.INTER_NEAREST)
    
    # Títulos manuais na imagem seria complexo, vamos usar subplots normais melhor:
    
    plt.close() # Fecha a fig anterior para fazer uma melhor
    fig = plt.figure(figsize=(12, 8))

    # Kernel 3D
    ax_kernel = fig.add_subplot(2, 2, 1, projection='3d')
    ax_kernel.plot_surface(X, Y, kernel_2d, cmap='viridis')
    ax_kernel.set_title("O Kernel Gaussiano (Forma de Sino)")

    # Imagens
    plt.subplot(2, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.title("Original com Ruído")

    plt.subplot(2, 2, 3)
    plt.imshow(img_media, cmap='gray')
    plt.title("Filtro de Média (9x9)\nNote o 'box artifact' (quadrado)")

    plt.subplot(2, 2, 4)
    plt.imshow(img_gauss, cmap='gray')
    plt.title("Filtro Gaussiano (9x9)\nSuavização mais redonda/natural")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()