import cv2
import numpy as np
import matplotlib.pyplot as plt

def criar_imagem_sintetica():
    """Cria uma imagem com quadrado, círculo e triângulo para teste geométrico."""
    img = np.zeros((300, 300), dtype=np.uint8)
    
    # 1. Um Retângulo (Linhas verticais e horizontais puras)
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
    
    # 2. Um Círculo (Todas as direções de borda possíveis)
    cv2.circle(img, (220, 100), 50, 255, -1)
    
    # 3. Um Triângulo (Linhas diagonais)
    pt1 = (150, 200)
    pt2 = (50, 280)
    pt3 = (250, 280)
    triangle_cnt = np.array([pt1, pt2, pt3])
    cv2.drawContours(img, [triangle_cnt], 0, 255, -1)
    
    return img

def main():
    # Gera a imagem
    img = criar_imagem_sintetica()

    # --- DEFINIÇÃO DOS KERNELS PREWITT ---
    # Prewitt X: Note as colunas (-1, 0, 1)
    # Detecta variação da Esquerda para a Direita
    kernel_prewitt_x = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]])

    # Prewitt Y: Note as linhas (-1, 0, 1 transposto)
    # Detecta variação de Cima para Baixo
    kernel_prewitt_y = np.array([[-1, -1, -1],
                                 [ 0,  0,  0],
                                 [ 1,  1,  1]])

    # --- APLICAÇÃO (Convolução) ---
    # Importante: Usar float64 (CV_64F) para manter os números negativos!
    # Se usar uint8 direto, a borda que vai do branco pro preto (negativa) vira 0.
    prewitt_x = cv2.filter2D(img, cv2.CV_64F, kernel_prewitt_x)
    prewitt_y = cv2.filter2D(img, cv2.CV_64F, kernel_prewitt_y)

    # --- CÁLCULO DA MAGNITUDE ---
    # Pitágoras: sqrt(x^2 + y^2)
    magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
    
    # Normalizar para visualização (0 a 255)
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    
    # Pegamos o valor absoluto dos componentes X e Y só para visualização
    abs_x = np.absolute(prewitt_x)
    abs_x = np.uint8(255 * abs_x / np.max(abs_x))
    
    abs_y = np.absolute(prewitt_y)
    abs_y = np.uint8(255 * abs_y / np.max(abs_y))

    # --- VISUALIZAÇÃO ---
    plt.figure(figsize=(12, 10))

    # 1. Imagem Original
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("1. Original (Formas Geométricas)")
    plt.axis('off')

    # 2. Prewitt X (Vertical)
    plt.subplot(2, 2, 2)
    plt.imshow(abs_x, cmap='gray')
    plt.title("2. Prewitt X (Detecta Paredes)\nNote: Topo e base do quadrado somem!")
    plt.axis('off')

    # 3. Prewitt Y (Horizontal)
    plt.subplot(2, 2, 3)
    plt.imshow(abs_y, cmap='gray')
    plt.title("3. Prewitt Y (Detecta Teto/Chão)\nNote: Lados do quadrado somem!")
    plt.axis('off')

    # 4. Magnitude Final
    plt.subplot(2, 2, 4)
    plt.imshow(magnitude, cmap='gray')
    plt.title("4. Magnitude (Pitágoras)\nTodas as bordas reconstruídas")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()