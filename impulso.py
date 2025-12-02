import cv2
import numpy as np
import matplotlib.pyplot as plt

def gerar_espectro(imagem):
    """Gera o espectro de frequência centralizado e logarítmico"""
    f = np.fft.fft2(imagem)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-5) # +1e-5 evita log(0)
    return magnitude

def main():
    # 1. CRIAR O IMPULSO (Delta de Dirac)
    tamanho = 64 # Imagem 64x64
    impulso = np.zeros((tamanho, tamanho), dtype=np.float32)
    
    # Coloca um único pixel branco (valor 1.0) EXATAMENTE no centro
    centro = tamanho // 2
    impulso[centro, centro] = 1.0

    # Definindo Kernels Manuais para teste
    
    # A. Média (Box Blur) - Suaviza tudo
    k_media = np.ones((5, 5), np.float32) / 25
    
    # B. Gaussiano - Suaviza suavemente (redondo)
    # Usamos uma função do OpenCV para gerar a matriz 5x5
    k_gauss = cv2.getGaussianKernel(5, -1)
    k_gauss = k_gauss * k_gauss.T # Torna 2D
    
    # C. Laplaciano (Bordas em todas as direções) - Passa-Altas
    k_laplace = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=np.float32)

    # Lista de filtros para iterar
    filtros = [
        ("Impulso Original", None), # Caso base
        ("Filtro Média (Passa-Baixa)", k_media),
        ("Filtro Gaussiano (Passa-Baixa)", k_gauss),
        ("Laplaciano (Passa-Altas)", k_laplace)
    ]

    plt.figure(figsize=(10, 12))

    for i, (nome, kernel) in enumerate(filtros):
        # Processamento
        if kernel is None:
            resultado_espacial = impulso
        else:
            # Aplica a convolução do kernel no impulso
            resultado_espacial = cv2.filter2D(impulso, -1, kernel)
            
        # Cálculo da Frequência (FFT)
        espectro = gerar_espectro(resultado_espacial)

        # --- PLOTAGEM ---
        # Coluna da Esquerda: Domínio do Espaço (O desenho do filtro)
        plt.subplot(4, 2, i*2 + 1)
        plt.imshow(resultado_espacial, cmap='gray')
        plt.title(f"{nome} - Espacial")
        plt.axis('off')
        
        # Coluna da Direita: Domínio da Frequência (O que ele deixa passar)
        plt.subplot(4, 2, i*2 + 2)
        plt.imshow(espectro, cmap='inferno')
        plt.title(f"Espectro de Frequência")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()