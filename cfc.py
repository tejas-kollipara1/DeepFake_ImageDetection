# Features that we will use to differentiate between a real and AI generated images

# 1. Camera MetaData
# Can be modified as AI can generate realistic metadata as well

"""
2.
# It will be very hard for AI to mimic CFC patterns as these come from physics, not appearance

CFA(Color Filter Array) --> layer of tiny red, green, and blue filters placed on top of a camera sensor
# Applying a filter repeatedly across the image
# Right now all cameras use a 2 * 2 Bayer CFA Pattern
G R
B G
# The light hits the CFA filter first, and the filter only allows one color to pass and then the sensor measures brightness of that color only
# Green Wavelengths are strong --> passes a lot of light
# Red and Blue are a bit weaker --> pass lesser amount of light
# Note that only that color will get that brightness values, and the other colors will get the avg values of the neighboring pixels(demosaicing)
# Pixels that receive less light have more noise --> have higher variance
# Demosaicing --> Helps variance be similar but there will still be a difference
# Real Cameras use CFA cause camera's sensor can't see color --> They can only see brightness
# AI Images are generated mathematically by a computer

# CFA Periodicity Score --> How strong a repeating pattern is detected in the image
--> The standard deviation of the 4 patch variances
# Real Cameras --> Have a strong repeating CFA structure
# AI Images --> Have no CFA structure(it directly outputs RGB pixels)

# real images show uneven variance due to CFA sampling
# AI images show almost equal variances

# Chatgpt Generated Image -->
# Dog AI Image(by NanoBanana) --> Models like NanoBanana, Kandinsky, Stable diffusion add synthetic noise or texturing

"""

import cv2
import numpy as np

def cfa_periodicity_score(img):
    # first we convert the image to grayscale(take the brightness of each pixel)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # variance at each CFA position (2x2 grid)
    patches = {
        "00": gray[0::2, 0::2],
        "01": gray[0::2, 1::2],
        "10": gray[1::2, 0::2],
        "11": gray[1::2, 1::2],
    }

    variances = {k: np.var(v) for k, v in patches.items()}

    # real images show uneven variance due to CFA sampling
    # AI images show almost equal variances
    diffs = np.std(list(variances.values()))

    return diffs, variances

def compare_patch_variances(variances):
    """
    PURE numerical metrics showing how close/far patch variances are.
    No heuristics, no verbal interpretation.
    """
    values = np.array(list(variances.values()), dtype=np.float32)

    v_min = float(np.min(values))
    v_max = float(np.max(values))
    v_mean = float(np.mean(values))

    spread = v_max - v_min                    # absolute difference
    rel_spread = spread / v_mean if v_mean != 0 else 0.0   # normalized difference

    return {
        "relative_spread": rel_spread,
    }


img = cv2.imread("real.png")
score, var_map = cfa_periodicity_score(img)

print("CFA periodicity score:", score)
# print("Patch variances:", var_map)

metrics = compare_patch_variances(var_map)
print("\nPatch Variance Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")
