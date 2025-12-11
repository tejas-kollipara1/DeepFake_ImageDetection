import cv2
import numpy as np
import os
import pywt
import matplotlib.pyplot as plt
# AI generates images in this process, first they start with pure gaussian noise, denoise using neural a neural network, and after many steps, an image emerges

# 1. Converting image to grayscale
# Noise appear strongest in the brightness/luminance rather than the chroma
def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # converts color image to gray color and pixel values from [0...255] to [0...1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray

# 2. Extract noise-like residual
def extract_residual(gray):
    # converting back to 8-bit as OPENCV's denoiser needs an 8-bit input
    gray_8u = (gray * 255).astype(np.uint8)
    denoised_8u = cv2.fastNlMeansDenoising(gray_8u, None, h=10)
    denoised = denoised_8u.astype(np.float32) / 255.0

    residual = gray - denoised
    residual -= np.mean(residual)
    std = np.std(residual)
    if std > 0:
        residual = residual/std
    return residual

# 3. Latent Prior Leakage (Gaussianity)
# AI diffusion images start from pure Gaussian noise --> leaves a statistical fingerprint
# Real Camera noise is NOT Gaussian
# --> Shot(light arrives in random photon packets) Noise: Poisson noise
# --> Read Noise(noise when added by camera's hardware when it reads pixel values) --> electronic noise
def gaussianity_metrics(residual):
    r = residual.flatten().astype(np.float64)
    r -= np.mean(r)
    std = np.std(r)
    if std == 0:
        return {"excess_kurtosis": 0.0, "skewness": 0.0}

    # A perfectly Gaussian has Skewness = 0 & Excess Kurtosis = 0
    # It is a measure of how far the data deviates from Gaussianity
    r /= std
    kurtosis = np.mean(r**4) - 3.0
    skewness = np.mean(r**3)

    return {
        "excess_kurtosis": float(kurtosis),
        "skewness": float(skewness)
    }

# AI-generated images lack the complex, camera-specific high-frequency noise created by real sensor physics, CFA patterns, and demosaicing.
# AI can inject noise but that is not real sensor noise as it is not gaussian and uniform
# AI-generated image lacks the physically structured high freq noise --> so their freq content is weaker and when you use a global threshold to wipe it ---> we get unnaturally smooth surface
# Because their noise is synthetic and uniform, it collapses under wavelet analysis, revealing smooth, unnatural backgrounds that real photos never produce.
# Wavelet decomposition separates the parts of an image--> smooth stuff, medium details, and fine details(you remove the noise while keeping edges and structure intact)
# If the denoised images looks "too smooth" --> the original image had fake textures
def wavelet_denoise(gray, wavelet='db8', level=None):
    h, w = gray.shape
    max_level = pywt.dwt_max_level(min(h, w), wavelet)
    if level is None or level > max_level:
        level = max(1, max_level)

    # 1. Wavelet decomposition
    coeffs = pywt.wavedec2(gray, wavelet=wavelet, level=level)

    # 2. Noise sigma estimation (finest detail coefficients)
    cHn, cVn, cDn = coeffs[-1]
    sigma_est = np.median(np.abs(cDn)) / 0.6745 if cDn.size else 0.0
    uthresh = sigma_est * np.sqrt(2 * np.log(gray.size)) if sigma_est > 0 else 0.0

    # 3. Thresholding detail coefficients, preserve structure
    coeffs_thresh = [coeffs[0]]
    for (cH, cV, cD) in coeffs[1:]:
        coeffs_thresh.append((
            pywt.threshold(cH, uthresh, mode='soft'),
            pywt.threshold(cV, uthresh, mode='soft'),
            pywt.threshold(cD, uthresh, mode='soft')
        ))

    # 4. Reconstruction
    denoised = pywt.waverec2(coeffs_thresh, wavelet)
    denoised = denoised[:h, :w]
    denoised = np.clip(denoised, 0, 1)

    # 5. PLOT the matrix instead of printing
    plt.figure(figsize=(6, 6))
    plt.imshow(denoised, cmap='viridis')   # or 'gray'
    plt.colorbar()
    plt.title("Wavelet Denoised Output")
    plt.axis("off")
    plt.show()
# Pass the above image into a CNN classifier --> This is useful for spatial patterns

if __name__ == "__main__":
    image_path = "dog_real.png"
    print(os.path.basename(image_path))

    gray = load_gray(image_path)
    residual = extract_residual(gray)

    g = gaussianity_metrics(residual)
    wavelet_denoise(gray)

    print("\n LATENT PRIOR LEAKAGE (Gaussianity)")
    print(g)

'''
Train a logistic model --> X = [c k]
W = [w1 w2]T
c --> periodicity factor
k --> excess_kurtosis

y --> {1, if p >= 0.5
  --> {0, if p >= 0.5
  
Cross Entropy Loss

z = wTx + b
p(y = 1|x) = σ(z) = 1/1+e^-z

Regularization(L2 Ridge) to prevent overfitting
'''

