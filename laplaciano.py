import cv2
import numpy as np
import matplotlib.pyplot as plt

def criar_imagem_com_ruido():
    # 1. Criar imagem limpa (Fundo preto, Círculo Branco)
    img = np.zeros((300, 300), dtype=np.float64)
    cv2.circle(img, (150, 150), 80, 255, -1)
    
    # 2. Adicionar Ruído Gaussiano (Granulação típica de fotos)
    media = 0
    desvio = 20 # Intensidade do ruído
    ruido = np.random.normal(media, desvio, img.shape)
    
    img_ruidosa = img + ruido
    
    # Garantir que fique entre 0 e 255
    img_ruidosa = np.clip(img_ruidosa, 0, 255).astype(np.uint8)
    return img_ruidosa

def aplicar_laplaciano(img, titulo):
    # Aplica Laplaciano (cv2.CV_64F é vital para não perder os valores negativos)
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    
    # Pega o valor absoluto para visualização
    lap = np.absolute(lap)
    
    # Normaliza para 0-255
    lap = np.uint8(255 * lap / np.max(lap))
    return lap

def main():
    # 1. Preparar o cenário
    img = criar_imagem_com_ruido()

    # --- MÉTODO 1: LAPLACIANO PURO (Sem proteção) ---
    # O filtro vai derivar cada grão de ruído
    lap_puro = aplicar_laplaciano(img, "Puro")

    # --- MÉTODO 2: LoG (Laplacian of Gaussian) ---
    # Passo A: Suavizar (Matar o ruído)
    # Usamos um kernel 5x5 para um desfoque decente
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Passo B: Detectar Bordas (Agora no ambiente limpo)
    lap_log = aplicar_laplaciano(img_blur, "LoG")

    # --- VISUALIZAÇÃO ---
    plt.figure(figsize=(15, 6))

    # Imagem Original
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("1. Original com Ruído\n(Granulação visível)")
    plt.axis('off')

    # Laplaciano Puro
    plt.subplot(1, 3, 2)
    plt.imshow(lap_puro, cmap='gray')
    plt.title("2. Laplaciano PURO\n(Inutilizável: O ruído domina)")
    plt.axis('off')

    # LoG
    plt.subplot(1, 3, 3)
    plt.imshow(lap_log, cmap='gray')
    plt.title("3. Laplaciano do Gaussiano (LoG)\n(Bordas Nítidas e Fundo Limpo)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()