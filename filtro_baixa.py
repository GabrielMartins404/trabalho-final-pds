import cv2
import numpy as np
import matplotlib.pyplot as plt

def calcular_fft(imagem):
    """
    Calcula a Transformada de Fourier (FFT) 2D.
    """
    f = np.fft.fft2(imagem)
    fshift = np.fft.fftshift(f)
    # Soma +1 para evitar log(0)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

def aplicar_filtro_sobel(imagem_gray):
    """Aplica Sobel (Passa-Altas/Bordas)."""
    sobelx = cv2.Sobel(imagem_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(imagem_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    return magnitude

def aplicar_filtro_passa_baixa(imagem_gray):
    """
    Aplica um Filtro Gaussiano Forte (Passa-Baixas/Blur).
    Isso elimina detalhes finos e deixa apenas as formas "grosseiras".
    """
    # Kernel (25, 25) é bem grande para deixar o efeito óbvio
    img_blur = cv2.GaussianBlur(imagem_gray, (25, 25), 0)
    return img_blur

def main():
    # --- 1. Carregamento ---
    # Substitua pelo caminho da sua imagem
    caminho_imagem = r'circulo_anel.jpg' 
    
    # Tenta carregar. Se falhar, cria uma imagem sintética para teste
    img_original_bgr = cv2.imread(caminho_imagem)
    
    if img_original_bgr is None:
        print(f"Aviso: Imagem '{caminho_imagem}' não encontrada. Criando uma sintética.")
        img_original_bgr = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img_original_bgr, (50, 100), (250, 200), (255, 255, 255), -1)

    img_original_rgb = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2GRAY)

    # --- 2. Aplicação dos Filtros ---
    
    # A. Passa-Baixa (Blur - Mantém o centro do espectro)
    img_baixa = aplicar_filtro_passa_baixa(img_gray)
    
    # B. Passa-Alta (Sobel - Mantém a periferia do espectro)
    img_alta = aplicar_filtro_sobel(img_gray)

    # --- 3. Cálculo das FFTs ---
    fft_original = calcular_fft(img_gray)
    fft_baixa = calcular_fft(img_baixa)
    fft_alta = calcular_fft(img_alta)

    # --- 4. Visualização (Grid 2x3) ---
    plt.figure(figsize=(16, 9))

    # --- COLUNA 1: ORIGINAL ---
    plt.subplot(2, 3, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('ORIGINAL')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(fft_original, cmap='inferno')
    plt.title('Espectro Original\n(Mistura de tudo)')
    plt.axis('off')

    # --- COLUNA 2: PASSA-BAIXA (Blur) ---
    plt.subplot(2, 3, 2)
    plt.imshow(img_baixa, cmap='gray')
    plt.title('FILTRO PASSA-BAIXA (Blur)\nSuaviza bordas')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(fft_baixa, cmap='inferno')
    plt.title('Espectro Passa-Baixa\n(Brilho concentrado no CENTRO)')
    plt.axis('off')

    # --- COLUNA 3: PASSA-ALTA (Sobel) ---
    plt.subplot(2, 3, 3)
    plt.imshow(img_alta, cmap='gray')
    plt.title('FILTRO PASSA-ALTA (Bordas)\nRemove o fundo constante')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(fft_alta, cmap='inferno')
    plt.title('Espectro Passa-Alta\n(Centro escuro, bordas brilhantes)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()