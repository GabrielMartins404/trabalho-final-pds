import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    caminho_imagem = r'C:\Users\Gabriel\Desktop\Emba\imagem_2.jpg' # TROQUE PELO NOME DA SUA IMAGEM
    img = cv2.imread(caminho_imagem)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converter para RGB para exibir corretamente

    # Definindo um Kernel de 'Sharpen' (Realce)
    # O centro é positivo (preserva o pixel) e os vizinhos negativos (removem a média)
    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])

    # Aplicando a convolução na imagem COLORIDA
    # O valor -1 indica que a profundidade da imagem de saída é igual à de entrada
    img_realcada = cv2.filter2D(img, -1, kernel_sharpen)

    # Visualização
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_realcada)
    plt.title("Com Filtro Sharpen (Convolução Colorida)")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()