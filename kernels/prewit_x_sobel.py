import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. CRIAR O CENÁRIO (Um Círculo Perfeito)
    # Usamos uma imagem grande para minimizar erros de pixelização (aliasing)
    tamanho = 500
    centro = tamanho // 2
    raio = 150
    
    img = np.zeros((tamanho, tamanho), dtype=np.float64)
    # Desenhamos um círculo branco sólido
    cv2.circle(img, (centro, centro), raio, 255, -1)
    
    # Suavizamos levemente a borda do círculo original para simular uma imagem real
    # e reduzir o serrilhado digital (aliasing) que atrapalharia o teste
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 2. APLICAR FILTRO PREWITT (Manual)
    # Kernel sem suavização (1, 1, 1)
    k_prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    k_prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    p_x = cv2.filter2D(img, cv2.CV_64F, k_prewitt_x)
    p_y = cv2.filter2D(img, cv2.CV_64F, k_prewitt_y)
    mag_prewitt = np.sqrt(p_x**2 + p_y**2)

    # 3. APLICAR FILTRO SOBEL (Automático)
    # Kernel com suavização (1, 2, 1)
    s_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    s_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag_sobel = np.sqrt(s_x**2 + s_y**2)

    # 4. AMOSTRAGEM DOS DADOS (O "Scanner")
    # Vamos ler o valor da magnitude em cada grau do círculo (0 a 360)
    angulos = np.arange(0, 360, 1)
    valores_prewitt = []
    valores_sobel = []

    for angulo in angulos:
        # Converter graus para radianos
        rad = np.deg2rad(angulo)
        
        # Calcular a coordenada (x, y) exata na borda do círculo
        # Usamos raio da borda. Como aplicamos blur, a borda está centrada no raio
        x = int(centro + raio * np.cos(rad))
        y = int(centro + raio * np.sin(rad))
        
        # Ler o valor do pixel nessa coordenada
        valores_prewitt.append(mag_prewitt[y, x])
        valores_sobel.append(mag_sobel[y, x])

    # 5. NORMALIZAÇÃO PARA COMPARAÇÃO JUSTA
    # O Sobel gera valores maiores por natureza (por causa do peso 2).
    # Para comparar a estabilidade, vamos normalizar ambos pela sua própria média.
    # Assim, ambos ficarão em torno de 1.0 (100%).
    norm_prewitt = valores_prewitt / np.mean(valores_prewitt)
    norm_sobel = valores_sobel / np.mean(valores_sobel)

    # Calcular o Desvio Padrão (Quem oscila mais?)
    std_prewitt = np.std(norm_prewitt)
    std_sobel = np.std(norm_sobel)

    # 6. VISUALIZAÇÃO
    plt.figure(figsize=(14, 6))

    # Gráfico da Variação
    plt.plot(angulos, norm_prewitt, label=f'Prewitt (Oscilação: {std_prewitt:.4f})', color='red', alpha=0.7)
    plt.plot(angulos, norm_sobel, label=f'Sobel (Oscilação: {std_sobel:.4f})', color='blue', linewidth=2)
    
    plt.title("Teste de Isotropia: Estabilidade da Detecção vs. Ângulo da Borda")
    plt.xlabel("Ângulo da Borda (Graus)")
    plt.ylabel("Intensidade Relativa Detectada (1.0 = Média)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adicionar anotações
    plt.axhline(1.0, color='black', linestyle='--')
    plt.text(10, 1.02, "Linha Ideal (Perfeitamente Isotrópico)", color='black')

    plt.show()

if __name__ == "__main__":
    main()