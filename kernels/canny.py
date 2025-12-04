import cv2
import numpy as np
import os

def nada(x):
    # Função de callback necessária para o trackbar
    pass

# 1. Carregar a imagem em escala de cinza
# Substitua 'sua_imagem.png' pelo caminho do seu arquivo
caminho_imagem = f"{os.path.dirname(__file__)}/image.png"
imagem = cv2.imread(caminho_imagem, 0)

if imagem is None:
    print("Erro: Imagem não encontrada! Verifique o nome do arquivo.")
    exit()

# 2. Criar uma janela nomeada
nome_janela = 'Ajuste Fino: Blur + Canny'
cv2.namedWindow(nome_janela)

# 3. Criar as barras deslizantes (Trackbars)
# Trackbar para o Blur (valores ímpares apenas: 1, 3, 5...)
cv2.createTrackbar('Blur (Desfoque)', nome_janela, 1, 15, nada)
# Trackbars para os Limiares do Canny
cv2.createTrackbar('Min Threshold', nome_janela, 50, 500, nada)
cv2.createTrackbar('Max Threshold', nome_janela, 150, 500, nada)

print("Ajuste as barras. O valor do Blur será sempre ímpar. Pressione 'ESC' para sair.")

while True:
    # 4. Ler as posições atuais das barras
    blur_val = cv2.getTrackbarPos('Blur (Desfoque)', nome_janela)
    min_val = cv2.getTrackbarPos('Min Threshold', nome_janela)
    max_val = cv2.getTrackbarPos('Max Threshold', nome_janela)

    # O kernel do GaussianBlur DEVE ser um número ímpar (1, 3, 5, etc.)
    # Se o valor for par, soma 1 para torná-lo ímpar. Se for 0, vira 1.
    kernel_size = blur_val if blur_val % 2 == 1 else blur_val + 1
    if kernel_size < 1: kernel_size = 1 # Garante que seja pelo menos 1

    # 5. Passo Crucial: Aplicar o Desfoque Gaussiano ANTES do Canny
    # Isso remove o ruído e a textura fina
    imagem_borrada = cv2.GaussianBlur(imagem, (kernel_size, kernel_size), 0)

    # 6. Aplicar o Canny na imagem JÁ borrada
    bordas = cv2.Canny(imagem_borrada, min_val, max_val)

    # Opcional: Mostrar o valor real do kernel do blur na imagem
    cv2.putText(bordas, f'Kernel Blur: {kernel_size}x{kernel_size}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 7. Mostrar o resultado
    cv2.imshow(nome_janela, bordas)

    # Pressione ESC (código 27) para fechar
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()