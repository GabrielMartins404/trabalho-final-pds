import cv2
import numpy as np
import matplotlib.pyplot as plt

def adicionar_ruido_sal_pimenta(imagem, quantidade=0.05):
    """
    Adiciona ruído 'Sal e Pimenta' a uma imagem.
    quantidade: porcentagem de pixels que serão afetados (ex: 0.05 = 5%)
    """
    row, col = imagem.shape
    s_vs_p = 0.5 # Divisão meio a meio entre sal e pimenta
    out = np.copy(imagem)
    
    # Adicionar Sal (pixels brancos)
    num_salt = np.ceil(quantidade * imagem.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in imagem.shape]
    out[tuple(coords)] = 255

    # Adicionar Pimenta (pixels pretos)
    num_pepper = np.ceil(quantidade * imagem.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in imagem.shape]
    out[tuple(coords)] = 0
    return out

def normalizar_para_exibicao(img_float):
    """Converte imagem float (com negativos) para uint8 (0-255) para exibir"""
    img_abs = np.absolute(img_float)
    # Evitar divisão por zero se a imagem for toda preta
    if np.max(img_abs) == 0: return np.uint8(img_abs)
    img_norm = np.uint8(255 * img_abs / np.max(img_abs))
    return img_norm

def main():
    # 1. CRIAR IMAGEM SINTÉTICA LIMPA
    img_limpa = np.zeros((300, 300), dtype=np.uint8)
    # Desenhando um quadrado branco no centro
    cv2.rectangle(img_limpa, (75, 75), (225, 225), 255, -1)

    # 2. ADICIONAR RUÍDO PESADO (O Teste de Estresse)
    # Vamos sujar 10% da imagem. É bastante coisa.
    img_ruidosa = adicionar_ruido_sal_pimenta(img_limpa, quantidade=0.10)

    # --- PROCESSAMENTO PREWITT (Manual) ---
    # Kernels com pesos iguais (1, 1, 1)
    k_prewitt_x = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])
    k_prewitt_y = np.array([[-1, -1, -1],
                            [ 0,  0,  0],
                            [ 1,  1,  1]])

    # Aplicar filtros (usando float64 para precisão)
    prewitt_x = cv2.filter2D(img_ruidosa, cv2.CV_64F, k_prewitt_x)
    prewitt_y = cv2.filter2D(img_ruidosa, cv2.CV_64F, k_prewitt_y)
    # Magnitude (Pitágoras)
    mag_prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)
    res_prewitt = normalizar_para_exibicao(mag_prewitt)

    # --- PROCESSAMENTO SOBEL (Automático) ---
    # Kernels com pesos gaussianos (1, 2, 1) implícitos na função
    sobel_x = cv2.Sobel(img_ruidosa, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_ruidosa, cv2.CV_64F, 0, 1, ksize=3)
    # Magnitude (Pitágoras)
    mag_sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    res_sobel = normalizar_para_exibicao(mag_sobel)

    # --- VISUALIZAÇÃO LADO A LADO ---
    plt.figure(figsize=(14, 8))

    # Imagem Original Ruidosa
    plt.subplot(1, 3, 1)
    plt.imshow(img_ruidosa, cmap='gray')
    plt.title("1. Entrada com Ruído (Sal e Pimenta)")
    plt.axis('off')

    # Resultado Prewitt
    plt.subplot(1, 3, 2)
    plt.imshow(res_prewitt, cmap='gray')
    plt.title("2. Resultado PREWITT\n(Pesos iguais: 1,1,1)")
    plt.axis('off')

    # Resultado Sobel
    plt.subplot(1, 3, 3)
    plt.imshow(res_sobel, cmap='gray')
    plt.title("3. Resultado SOBEL\n(Pesos gaussianos: 1,2,1)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()