import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. CRIAR O IMPULSO (Delta de Dirac)
    # Usaremos uma imagem pequena (30x30) para facilitar o entendimento
    tamanho = 30
    impulso = np.zeros((tamanho, tamanho), dtype=np.float32)
    centro = tamanho // 2
    impulso[centro, centro] = 1.0 # O único pixel branco

    # 2. CALCULAR A FFT
    f = np.fft.fft2(impulso)
    fshift = np.fft.fftshift(f)
    
    # --- AQUI ESTÁ O SEGREDO ---
    
    # A. Magnitude Linear (BRUTA)
    # É o valor real que o Fourier calculou. Para um impulso, deve ser 1.0 constante.
    magnitude_linear = np.abs(fshift)

    # B. Magnitude Logarítmica (A que usamos antes)
    # Adicionamos 1e-5 para não dar erro se tiver zero, mas no impulso é tudo 1.
    # log(1) = 0.
    magnitude_log = 20 * np.log(magnitude_linear + 1e-5)

    # Vamos imprimir os valores numéricos para provar o conceito
    print(f"--- Estatísticas do Espectro ---")
    print(f"Valor Mínimo Linear: {np.min(magnitude_linear)}")
    print(f"Valor Máximo Linear: {np.max(magnitude_linear)}")
    print(f"Valor Mínimo Log:    {np.min(magnitude_log)}")
    print(f"Valor Máximo Log:    {np.max(magnitude_log)}")

    # 3. VISUALIZAÇÃO
    plt.figure(figsize=(12, 5))

    # Plot 1: O Impulso no Espaço
    plt.subplot(1, 3, 1)
    plt.imshow(impulso, cmap='gray')
    plt.title("Espaço: Impulso Unitário\n(Um ponto no meio)")
    plt.axis('off')

    # Plot 2: Espectro LINEAR (A verdade nua e crua)
    plt.subplot(1, 3, 2)
    # Usamos vmin e vmax para forçar a escala de cinza a mostrar o valor 1 como algo visível
    plt.imshow(magnitude_linear, cmap='gray', vmin=0, vmax=2) 
    plt.title("Espectro LINEAR\n(Valor constante = 1.0)")
    plt.axis('off')

    # Plot 3: Espectro LOGARÍTMICO (O que você viu antes)
    plt.subplot(1, 3, 3)
    plt.imshow(magnitude_log, cmap='gray')
    plt.title("Espectro LOGARÍTMICO\n(log(1) = 0 -> Preto)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()