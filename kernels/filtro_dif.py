import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Carrega imagem em grayscale
# -----------------------------
img = cv2.imread("circulo_anel.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Erro ao carregar a imagem. Certifique-se que 'imagem.jpg' existe no diretório.")

# ---------------------------------------------------
# 2. Define os kernels do Sobel e do Prewitt (Gx, Gy)
# ---------------------------------------------------

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

prewitt_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

prewitt_y = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float32)

# ---------------------------------------------
# 3. Aplica convolução manual com os kernels
# ---------------------------------------------
sobel_x_res = cv2.filter2D(img, -1, sobel_x)
sobel_y_res = cv2.filter2D(img, -1, sobel_y)
sobel_mag = cv2.magnitude(sobel_x_res.astype(float), sobel_y_res.astype(float))

prewitt_x_res = cv2.filter2D(img, -1, prewitt_x)
prewitt_y_res = cv2.filter2D(img, -1, prewitt_y)
prewitt_mag = cv2.magnitude(prewitt_x_res.astype(float), prewitt_y_res.astype(float))

# ------------------------------------------------
# 4. Visualização lado a lado dos resultados
# ------------------------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Sobel Magnitude")
plt.imshow(sobel_mag, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Prewitt Magnitude")
plt.imshow(prewitt_mag, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Kernel Sobel X")
plt.imshow(sobel_x, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Kernel Prewitt X")
plt.imshow(prewitt_x, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("Destaque: Sobel usa peso 2 no centro")
plt.text(0.1, 0.5, str(sobel_x), fontsize=14)  # apenas explicativo
plt.axis('off')

plt.tight_layout()
plt.show()

# --------------------------------------------------------
# 5. Observação automática no terminal
# --------------------------------------------------------
print("\nPrincipais diferenças entre Sobel e Prewitt:")
print("- Sobel usa peso 2 no pixel central, o que suaviza ruído na direção perpendicular.")
print("- Prewitt usa pesos uniformes, mais sensível a ruídos.")
print("- Sobel tende a produzir bordas mais suaves e robustas.")
