import cv2
import numpy as np
import matplotlib.pyplot as plt

def carregar_imagem(caminho):
    """Carrega imagem em escala de cinza e redimensiona para tamanho fixo."""
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erro ao carregar {caminho}")
        return None
    # Redimensiona para garantir que ambas tenham o mesmo tamanho (para o transplante)
    img = cv2.resize(img, (300, 300))
    return img

def separar_mag_fase(imagem):
    """Calcula FFT e retorna Magnitude e Fase separados."""
    f = np.fft.fft2(imagem)
    fshift = np.fft.fftshift(f)
    
    # Magnitude (Valor absoluto)
    magnitude = np.abs(fshift)
    
    # Fase (Ângulo em radianos)
    fase = np.angle(fshift)
    
    return magnitude, fase

def reconstruir(magnitude, fase):
    """Reconstrói a imagem a partir de uma magnitude e fase dadas."""
    # Reconstrói o número complexo: Z = Mag * e^(j*Fase)
    # Euler: e^(j*theta) = cos(theta) + j*sen(theta)
    fshift_combinado = magnitude * np.exp(1j * fase)
    
    # Desfaz o shift
    f_ishift = np.fft.ifftshift(fshift_combinado)
    
    # FFT Inversa
    img_back = np.fft.ifft2(f_ishift)
    
    # Retorna apenas a parte real (despreza erros numéricos imaginários)
    return np.abs(img_back)

def main():
    # --- 1. Carregue duas imagens diferentes aqui ---
    # Sugestão: Uma foto de rosto e uma de um prédio/objeto geométrico
    img1 = carregar_imagem('luffy.jpg') # Coloque o nome da sua imagem
    img2 = carregar_imagem('predio.jpg') # Coloque o nome da segunda imagem

    # Cria imagens sintéticas caso não existam arquivos, para o código não quebrar
    if img1 is None:
        print("Criando imagem sintética 1 (Círculo)...")
        img1 = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(img1, (150, 150), 80, 255, -1)
    
    if img2 is None:
        print("Criando imagem sintética 2 (Retângulo)...")
        img2 = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(img2, (50, 50), (250, 250), 255, -1)

    # --- 2. Extração de Componentes ---
    mag1, fase1 = separar_mag_fase(img1)
    mag2, fase2 = separar_mag_fase(img2)

    # --- 3. Experimento A: Isolamento (Só Mag ou Só Fase) ---
    # Para ver SÓ a magnitude, zeramos a fase (fase = 0)
    rec_so_mag1 = reconstruir(mag1, np.zeros_like(fase1))
    
    # Para ver SÓ a fase, tornamos a magnitude constante (mag = 1)
    # Isso nivela o contraste de todas as frequências
    rec_so_fase1 = reconstruir(np.ones_like(mag1), fase1)

    # --- 4. Experimento B: O Transplante (Troca de Fase) ---
    # Mag da Imagem 1 + Fase da Imagem 2
    rec_mista_12 = reconstruir(mag1, fase2)
    
    # Mag da Imagem 2 + Fase da Imagem 1
    rec_mista_21 = reconstruir(mag2, fase1)

    # --- 5. Visualização ---
    plt.figure(figsize=(12, 10))

    # Linha 1: Isolamento da Imagem 1
    plt.subplot(3, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Original 1')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(np.log(rec_so_mag1 + 1), cmap='gray') 
    plt.title('Apenas Magnitude 1\n(Sem forma definida)')
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.imshow(rec_so_fase1, cmap='gray')
    plt.title('Apenas Fase 1\n(Bordas visíveis!)')
    plt.axis('off')

    # Linha 2: Imagens Originais para Comparação do Transplante
    plt.subplot(3, 3, 4)
    plt.imshow(img1, cmap='gray')
    plt.title('Fonte Mag (Img 1)')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.title('--- MISTURA ---')
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(img2, cmap='gray')
    plt.title('Fonte Fase (Img 2)')
    plt.axis('off')

    # Linha 3: O Resultado da Troca
    plt.subplot(3, 3, 7)
    plt.imshow(rec_mista_12, cmap='gray')
    plt.title('Mag(1) + Fase(2)\n(Parece a Imagem 2!)')
    plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.imshow(rec_mista_21, cmap='gray')
    plt.title('Mag(2) + Fase(1)\n(Parece a Imagem 1!)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()