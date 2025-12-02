import cv2
import numpy as np
import matplotlib.pyplot as plt

def calcular_fft(imagem):
    """
    Calcula a Transformada de Fourier (FFT) 2D de uma imagem para visualização.
    Retorna o espectro de magnitude em escala logarítmica.
    """
    # 1. Cálculo da FFT 2D
    f = np.fft.fft2(imagem)
    
    # 2. Shift: Move a frequência zero (DC) dos cantos para o centro
    fshift = np.fft.fftshift(f)
    
    # 3. Magnitude: Calcula o valor absoluto (módulo) dos números complexos
    # 4. Log: Aplica logaritmo para reduzir a diferença dinâmica (o centro é muito brilhante)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    return magnitude_spectrum

def aplicar_filtro_sobel(imagem_gray):
    """Aplica o filtro de Sobel (Passa-Altas) usando funções otimizadas do OpenCV."""
    # Sobel X (Bordas verticais) - cv2.CV_64F permite números negativos
    sobelx = cv2.Sobel(imagem_gray, cv2.CV_64F, 1, 0, ksize=3)
    
    # Sobel Y (Bordas horizontais)
    sobely = cv2.Sobel(imagem_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude do Gradiente (Teorema de Pitágoras: sqrt(x^2 + y^2))
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normaliza para 0-255 para exibição
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    
    return magnitude

def aplicar_filtro_prewitt_manual(imagem_gray):
    """
    Aplica o filtro Prewitt via Convolução manual (filter2D) 
    para demonstrar o conceito de Kernel.
    """
    # Definição manual dos Kernels de Prewitt
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
                         
    kernel_y = np.array([[-1, -1, -1],
                         [ 0,  0,  0],
                         [ 1,  1,  1]])

    # Aplicação da Convolução 2D
    img_prewittx = cv2.filter2D(imagem_gray, -1, kernel_x)
    img_prewitty = cv2.filter2D(imagem_gray, -1, kernel_y)
    
    # Combinação (aproximada pela soma dos valores absolutos para simplificar)
    prewitt_combinado = img_prewittx + img_prewitty
    return prewitt_combinado

def main():
    # --- 1. Carregamento da Imagem ---
    #caminho_imagem = r'C:\Users\Gabriel\Desktop\Emba\imagem_2.jpg'
    caminho_imagem = r'circulo_pret.jpg'
    
    # Carrega em cor para visualização final
    img_original_bgr = cv2.imread(caminho_imagem)
    
    if img_original_bgr is None:
        print(f"Erro: Não foi possível carregar a imagem '{caminho_imagem}'. Verifique o nome.")
        return

    # Converte para RGB (para o matplotlib mostrar as cores certas)
    img_original_rgb = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB)
    
    # Converte para Escala de Cinza (para o processamento matemático)
    img_gray = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2GRAY)

    # --- 2. Pré-processamento (Remoção de Ruído) ---
    # Aplicamos um filtro Gaussiano (Passa-Baixas) suave antes
    img_gray_suave = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # --- 3. Aplicação dos Filtros (Detecção de Bordas) ---
    # Usando Sobel (Passa-Altas)
    img_bordas = aplicar_filtro_sobel(img_gray_suave)
    
    # --- 4. Análise de Frequência (FFT) ---
    # Espectro da imagem ORIGINAL (suavizada)
    espectro_original = calcular_fft(img_gray_suave)
    
    # Espectro da imagem FILTRADA (só bordas)
    espectro_bordas = calcular_fft(img_bordas)

    # --- 5. Visualização dos Resultados ---
    plt.figure(figsize=(12, 8))

    # Imagem Original
    plt.subplot(2, 2, 1)
    plt.imshow(img_original_rgb)
    plt.title('1. Imagem Original')
    plt.axis('off')

    # Espectro Original
    plt.subplot(2, 2, 2)
    plt.imshow(espectro_original, cmap='inferno')
    plt.title('2. Espectro Original (FFT)\nObserve o centro brilhante (Baixas Freq)')
    plt.axis('off')

    # Imagem de Bordas (Sobel)
    plt.subplot(2, 2, 3)
    plt.imshow(img_bordas, cmap='gray')
    plt.title('3. Resultado Sobel (Passa-Altas)\nBordas detectadas')
    plt.axis('off')

    # Espectro das Bordas
    plt.subplot(2, 2, 4)
    plt.imshow(espectro_bordas, cmap='inferno')
    plt.title('4. Espectro das Bordas\nObserve a dispersão (Altas Freq)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()