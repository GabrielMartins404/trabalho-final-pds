import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. CRIAR O "RUÍDO" (Um único pixel branco no meio do nada)
    # Usamos uma imagem 7x7 para ficar fácil de ler no terminal
    img_ponto = np.zeros((7, 7), dtype=np.float32)
    img_ponto[3, 3] = 255  # Pixel central (o ruído)

    # 2. DEFINIR KERNELS MANUAIS
    
    # Kernel Prewitt X (Pesos iguais nas colunas: 1, 0, -1)
    k_prewitt_x = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]], dtype=np.float32)
                            
    # Kernel Sobel X (Pesos Gaussianos: 1, 2, 1)
    k_sobel_x = np.array([[-1, 0, 1],
                          [-2, 0, 2],  # <--- Note o 2 aqui!
                          [-1, 0, 1]], dtype=np.float32)

    # 3. APLICAR FILTROS
    res_prewitt = cv2.filter2D(img_ponto, -1, k_prewitt_x)
    res_sobel = cv2.filter2D(img_ponto, -1, k_sobel_x)

    # 4. IMPRIMIR OS VALORES (A PROVA MATEMÁTICA)
    print("--- ANÁLISE DO IMPACTO DO RUÍDO ---")
    print("Imagine que o centro (3,3) é um defeito na imagem.\n")

    print("1. RESPOSTA DO PREWITT (Olhe a coluna da direita):")
    # Vamos olhar apenas o recorte central 3x3 onde o filtro atuou
    recorte_p = res_prewitt[2:5, 2:5]
    print(recorte_p)
    print(">> Note: Os valores 255 nas pontas (diagonais) e no meio são IGUAIS.")
    print(">> Conclusão: O ruído se espalhou com força total para os cantos.\n")

    print("-" * 40)

    print("2. RESPOSTA DO SOBEL (Olhe a coluna da direita):")
    recorte_s = res_sobel[2:5, 2:5]
    print(recorte_s)
    print(">> Note: O valor do meio é 510, mas as pontas são 255.")
    print(">> Conclusão: O Sobel concentrou o sinal no eixo principal.")
    print(">> RELATIVAMENTE, as diagonais (cantos) ficaram mais fracas que o centro.")

    # 5. VISUALIZAÇÃO GRÁFICA ZOOMADA
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(res_prewitt), cmap='gray')
    plt.title("Prewitt (Visual)\nO 'Ruído' vira um Quadrado\n(Forte nas diagonais)")
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(res_sobel), cmap='gray')
    plt.title("Sobel (Visual)\nO 'Ruído' vira um Diamante\n(Fraco nas diagonais)")
    
    plt.show()

if __name__ == "__main__":
    main()