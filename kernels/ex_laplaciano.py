import cv2
import numpy as np
import matplotlib.pyplot as plt

def criar_ceu_estrelado():
    """Gera uma imagem sintética de céu com estrelas e ruído de sensor."""
    largura, altura = 800, 600
    ceu = np.zeros((altura, largura), dtype=np.uint8)
    
    # 1. Adicionar Estrelas (Pontos brilhantes de tamanhos variados)
    import random
    estrelas_reais = []
    for _ in range(30):
        x = random.randint(20, largura-20)
        y = random.randint(20, altura-20)
        brilho = random.randint(150, 255)
        raio = random.randint(1, 3) # Estrelas maiores e menores
        
        # Desenhamos círculos sólidos (blobs)
        cv2.circle(ceu, (x, y), raio, brilho, -1)
        # Salvar coordenada para conferência visual depois
        estrelas_reais.append((x, y))
        
    # 2. Adicionar Ruído (Simulando ISO alto)
    ruido = np.random.normal(0, 10, ceu.shape).astype(np.int16)
    ceu_ruidoso = cv2.add(ceu.astype(np.int16), ruido)
    ceu_final = np.clip(ceu_ruidoso, 0, 255).astype(np.uint8)
    
    return ceu_final

def detectar_estrelas(imagem):
    # --- PASSO 1: LoG (Laplacian of Gaussian) ---
    
    # A. Gaussian Blur
    # O sigma (desvio padrão) deve combinar com o tamanho da estrela que você quer achar.
    # Sigma=2 detecta bem estrelas pequenas/médias.
    img_blur = cv2.GaussianBlur(imagem, (9, 9), 2.0)
    
    # B. Laplaciano
    # Usamos float64 para não perder sinais negativos
    laplaciano = cv2.Laplacian(img_blur, cv2.CV_64F, ksize=3)
    
    # --- PASSO 2: FILTRAGEM DE PICOS ---
    
    # Estrelas são picos de curvatura. No kernel padrão do OpenCV, 
    # o centro brilhante num fundo escuro gera um valor NEGATIVO forte ou POSITIVO forte 
    # dependendo da implementação do kernel. Vamos pegar a Magnitude Absoluta.
    lap_abs = np.absolute(laplaciano)
    
    # Normalizar para 0-255 para podermos fazer cortes
    lap_norm = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Corte (Threshold): Só aceitamos picos muito fortes
    # Isso elimina o ruído de fundo que o Laplaciano amplificou
    # Tudo abaixo de pixel valor 100 é considerado "vazio"
    _, mask_estrelas = cv2.threshold(lap_norm, 100, 255, cv2.THRESH_BINARY)

    # --- PASSO 3: ENCONTRAR COORDENADAS (Centróides) ---
    
    # A máscara agora tem "ilhas" brancas onde estão as estrelas.
    # Usamos Contornos para achar o centro dessas ilhas.
    contornos, _ = cv2.findContours(mask_estrelas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    imagem_resultado = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
    contagem = 0
    
    for cnt in contornos:
        # Calcular o momento para achar o centro do blob
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Desenhar um círculo verde ao redor da estrela detectada
            cv2.circle(imagem_resultado, (cX, cY), 10, (0, 255, 0), 2)
            # Marcar o centro
            cv2.circle(imagem_resultado, (cX, cY), 1, (0, 0, 255), -1)
            contagem += 1
            
    return imagem_resultado, lap_norm, mask_estrelas, contagem

def main():
    # 1. Gerar imagem
    ceu = criar_ceu_estrelado()
    
    # 2. Detectar
    resultado, mapa_laplaciano, mascara, num_estrelas = detectar_estrelas(ceu)
    
    # 3. Visualizar
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(ceu, cmap='gray')
    plt.title("1. Imagem Original (Com Ruído)")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(mapa_laplaciano, cmap='inferno')
    plt.title("2. Resposta do Laplaciano (Mapa de Calor)\nVeja como as estrelas 'explodem' em energia")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(mascara, cmap='gray')
    plt.title("3. Máscara Binária (Após Threshold)\nIsolando os picos máximos")
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
    plt.title(f"4. Resultado Final ({num_estrelas} estrelas detectadas)\nCírculos Verdes = Detecção")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()