import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # --- PARTE 1: ENTENDENDO O SOBEL E A SETA ---
    
    # Criar uma imagem preta 200x200
    img_seta = np.zeros((200, 200), dtype=np.uint8)
    
    # Desenhar uma seta branca apontando para a DIREITA
    # (Ponto inicial, Ponto final, Cor, Espessura, Tipo da ponta)
    cv2.arrowedLine(img_seta, (50, 100), (150, 100), 255, 10, tipLength=0.3)

    # Calcular Sobel X e Sobel Y
    sobelx = cv2.Sobel(img_seta, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_seta, cv2.CV_64F, 0, 1, ksize=3)

    # Calcular a DIREÇÃO (Ângulo) do gradiente
    # A função 'phase' calcula o ângulo em graus para cada pixel
    direcao = cv2.phase(sobelx, sobely, angleInDegrees=True)
    
    # Mascara: Só queremos ver a direção onde EXISTE borda (magnitude alta)
    magnitude = cv2.magnitude(sobelx, sobely)
    mask_bordas = magnitude > 50 # Limiar para ignorar o fundo preto
    
    # Visualização da Direção (Usando cores HSV)
    # Matiz (Cor) = Direção (0 a 360 graus)
    hsv = np.zeros((200, 200, 3), dtype=np.uint8)
    hsv[..., 0] = direcao / 2      # O Hue no OpenCV vai de 0-179 (então dividimos 360 por 2)
    hsv[..., 1] = 255              # Saturação máxima
    hsv[..., 2] = np.where(mask_bordas, 255, 0) # Brilho só onde tem borda
    
    img_direcao_colorida = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # --- PARTE 2: O LAPLACIANO E OS PONTOS (BLOBS) ---
    
    # Criar imagem com "estrelas" (pontos)
    img_pontos = np.zeros((200, 200), dtype=np.float32)
    # Criar alguns pontos gaussianos (bolinhas suaves)
    for y, x in [(50, 50), (100, 150), (150, 50)]:
        img_pontos[y, x] = 1.0 # Ponto central
        # Um pequeno borrão para simular um ponto real (não apenas 1 pixel)
        
    img_pontos = cv2.GaussianBlur(img_pontos, (9, 9), 2)
    # Normalizar para visualização (0 a 1)
    img_pontos = img_pontos / img_pontos.max()

    # Aplicar Laplaciano
    laplaciano = cv2.Laplacian(img_pontos, cv2.CV_64F)

    # --- VISUALIZAÇÃO ---
    plt.figure(figsize=(12, 8))

    # 1. A Seta Original
    plt.subplot(2, 2, 1)
    plt.imshow(img_seta, cmap='gray')
    plt.title("Imagem Original: Seta (Dir)")

    # 2. O Que o Sobel Vê (Direção)
    plt.subplot(2, 2, 2)
    plt.imshow(img_direcao_colorida)
    plt.title("Sobel: Cores indicam a direção da BORDA\n(Não da seta inteira!)")
    # Adicionar legenda explicativa no gráfico seria complexo, 
    # mas note que as bordas de cima têm cor diferente das de baixo.

    # 3. Os Pontos Originais
    plt.subplot(2, 2, 3)
    plt.imshow(img_pontos, cmap='gray')
    plt.title("Imagem Original: Pontos (Blobs)")

    # 4. A Resposta do Laplaciano
    plt.subplot(2, 2, 4)
    plt.imshow(laplaciano, cmap='jet') # Jet ajuda a ver positivo/negativo
    plt.title("Laplaciano (Detector de Pontos)\nVermelho=Centro, Azul=Borda")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()