# CENG 466 THE2
# Mert Uludoğan 2380996
# Yiğitcan Özcan 2521847

import numpy as np
import cv2
import os
import pywt
from scipy.fftpack import dct, idct
from skimage.metrics import mean_squared_error


### COMMON FUNCTIONS ###
input_folder = 'THE2_Images/'
output_folder = 'THE2_Images/'


def read_image(filename, gray_scale=False):
    # CV2 is just a suggestion you can use other libraries as well
    if gray_scale:
        img = cv2.imread(input_folder + filename, cv2.IMREAD_GRAYSCALE)
        return img
    img = cv2.imread(input_folder + filename)
    return img


def write_image(img, filename):
    # CV2 is just a suggestion you can use other libraries as well
    if os.path.exists(output_folder) == False:
        os.makedirs(output_folder)
    cv2.imwrite(output_folder + filename, img)

#QUESTION 1
# Load the images
img1 = read_image("a1.png", gray_scale=True)
img2 = read_image("a2.png", gray_scale=True)

# Save grayscale images
write_image(img1, "grayscale_a1.png")
write_image(img2, "grayscale_a2.png",)

# Define custom kernels
roberts_x = np.array([[1, 0], [0, -1]])
roberts_y = np.array([[0, 1], [-1, 0]])

prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

def apply_filters(img, prefix):
    # Sobel
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)

    # Roberts
    roberts = cv2.filter2D(img, -1, roberts_x) + cv2.filter2D(img, -1, roberts_y)

    # Prewitt
    prewitt = cv2.filter2D(img, -1, prewitt_x) + cv2.filter2D(img, -1, prewitt_y)

    # Save outputs
    write_image(sobel, f"{prefix}_sobel.png")
    write_image(roberts, f"{prefix}_roberts.png")
    write_image(prewitt, f"{prefix}_prewitt.png")

apply_filters(img1, "a1")
apply_filters(img2, "a2")                     #step 1 and step 2 completed.

def blur_images(img, prefix):
    for k in [3, 5, 7]:
        blurred = cv2.GaussianBlur(img, (k, k), 0)
        write_image(blurred, f"{prefix}_blurred_k{k}.png")

blur_images(img1, "a1")
blur_images(img2, "a2")                         #step 3

#step 4
for k in [3, 5, 7]:
    img1_blurred = read_image(f"a1_blurred_k{k}.png", gray_scale=True)
    img2_blurred = read_image(f"a2_blurred_k{k}.png", gray_scale=True)
    apply_filters(img1_blurred, f"a1_blurred_k{k}_filtered")
    apply_filters(img2_blurred, f"a2_blurred_k{k}_filtered")

#step 5
def binarize_msb(img, prefix):
    msb_img = ((img >> 7) & 1) * 255  # Extract MSB and scale to 0-255
    write_image(msb_img, f"{prefix}_msb.jpg")
    return msb_img

msb_img1 = binarize_msb(img1, "a1")
msb_img2 = binarize_msb(img2, "a2")

#step 6
apply_filters(msb_img1, "a1_msb")
apply_filters(msb_img2, "a2_msb")

#step 7
blur_images(msb_img1, "a1_msb")
blur_images(msb_img2, "a2_msb")

#step 8
for k in [3, 5, 7]:
    img1_msb_blurred = read_image(f"a1_msb_blurred_k{k}.png", gray_scale=True)
    img2_msb_blurred = read_image(f"a2_msb_blurred_k{k}.png", gray_scale=True)
    apply_filters(img1_msb_blurred, f"a1_msb_blurred_k{k}")
    apply_filters(img2_msb_blurred, f"a2_msb_blurred_k{k}")



#QUESTION 2
# Load the images
img1 = read_image("b1.jpg")
img2 = read_image("b2.jpg")
img3 = read_image("b3.jpg")


blue_channel, green_channel, red_channel = cv2.split(img1)
write_image(blue_channel, "b1_blue_channel.png")
write_image(green_channel, "b1_green_channel.png")
write_image(red_channel, "b1_red_channel.png")

b1_gaus_blurred = cv2.GaussianBlur(img1, (21, 21), 0)
b1_med_blurred = cv2.medianBlur(img1, 13)
write_image(b1_med_blurred, "b1_median.png")
write_image(b1_gaus_blurred, "b1_gaussian.png")


blue_channel, green_channel, red_channel = cv2.split(img2)
write_image(blue_channel, "b2_blue_channel.png")
write_image(green_channel, "b2_green_channel.png")
write_image(red_channel, "b2_red_channel.png")

b2_gaus_blurred = cv2.GaussianBlur(img2, (11, 11), 0)
b2_med_blurred = cv2.medianBlur(img2, 9)
write_image(b2_med_blurred, "b2_median.png")
write_image(b2_gaus_blurred, "b2_gaussian.png")


blue_channel, green_channel, red_channel = cv2.split(img3)
write_image(blue_channel, "b3_blue_channel.png")
write_image(green_channel, "b3_green_channel.png")
write_image(red_channel, "b3_red_channel.png")

b3_gaus_blurred = cv2.GaussianBlur(img3, (17, 17), 0)
b3_med_blurred = cv2.medianBlur(img3, 9)
write_image(b3_med_blurred, "b3_median.png")
write_image(b3_gaus_blurred, "b3_gaussian.png")

# It might be necessary to apply different filters to different channels, since channels do not have the same type of noise.


def apply_fourier_filters(img, output_prefix):
    # Split the image into channels
    b_channel, g_channel, r_channel = cv2.split(img)

    # Function to apply Fourier filters on a single channel
    def filter_channel(channel, prefix):
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2

        # Fourier Transform
        dft = np.fft.fft2(channel)
        dft_shift = np.fft.fftshift(dft)

        # Frequency grid
        u, v = np.meshgrid(np.arange(cols), np.arange(rows))
        u = u - ccol
        v = v - crow
        distance = np.sqrt(u**2 + v**2)

        # Ideal Low-Pass Filter (ILP)
        cutoff_low = 50
        ilp_filter = (distance <= cutoff_low).astype(np.float32)

        # Band-Pass Filter (BP)
        cutoff_low_bp, cutoff_high_bp = 30, 70
        bp_filter = ((distance >= cutoff_low_bp) & (distance <= cutoff_high_bp)).astype(np.float32)

        # Band-Reject Filter (BR)
        cutoff_low_br, cutoff_high_br = 30, 70
        br_filter = ((distance < cutoff_low_br) | (distance > cutoff_high_br)).astype(np.float32)

        # Apply filters
        ilp_result = dft_shift * ilp_filter
        bp_result = dft_shift * bp_filter
        br_result = dft_shift * br_filter

        # Transform back to spatial domain
        ilp_img = np.abs(np.fft.ifft2(np.fft.ifftshift(ilp_result)))
        bp_img = np.abs(np.fft.ifft2(np.fft.ifftshift(bp_result)))
        br_img = np.abs(np.fft.ifft2(np.fft.ifftshift(br_result)))

        # Normalize results
        ilp_img = cv2.normalize(ilp_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        bp_img = cv2.normalize(bp_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        br_img = cv2.normalize(br_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return ilp_img, bp_img, br_img

    # Apply filters to each channel
    b_ilp, b_bp, b_br = filter_channel(b_channel, "b")
    g_ilp, g_bp, g_br = filter_channel(g_channel, "g")
    r_ilp, r_bp, r_br = filter_channel(r_channel, "r")

    # Merge results for each filter
    ilp_result = cv2.merge((b_ilp, g_ilp, r_ilp))
    bp_result = cv2.merge((b_bp, g_bp, r_bp))
    br_result = cv2.merge((b_br, g_br, r_br))

    # Save the results
    write_image(ilp_result, f"{output_prefix}_ilp.png")
    write_image(bp_result, f"{output_prefix}_bp.png")
    write_image(br_result, f"{output_prefix}_br.png")


apply_fourier_filters(img1, "b1")
apply_fourier_filters(img2, "b2")
apply_fourier_filters(img3, "b3")





def apply_compression(image_path, output_prefix, n_values):
    # Read grayscale image
    image = read_image(image_path, gray_scale=True)
    rows, cols = image.shape

    # Haar Wavelet Transform
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs

    # Discrete Cosine Transform (DCT)
    image_dct = dct(dct(image.T, norm='ortho').T, norm='ortho')

    results = {}

    for n in n_values:
        # Haar Wavelet Compression
        haar_coeffs = compress_wavelet((cA, (cH, cV, cD)), n)
        compressed_haar = pywt.idwt2(haar_coeffs, 'haar')

        # Discrete Cosine Transform Compression
        dct_compressed = compress_dct(image_dct, n)
        reconstructed_dct = idct(idct(dct_compressed.T, norm='ortho').T, norm='ortho')

        # Compute MSE
        mse_haar = mean_squared_error(image, compressed_haar)
        mse_dct = mean_squared_error(image, reconstructed_dct)

        # Save compressed images
        write_image(compressed_haar, f"{output_prefix}_haar_{n}.jpg")
        write_image(reconstructed_dct, f"{output_prefix}_dct_{n}.jpg")

        # Store results
        results[n] = {
            "mse_haar": mse_haar,
            "mse_dct": mse_dct,
            "haar_image": compressed_haar,
            "dct_image": reconstructed_dct
        }

    return results

def compress_wavelet(coeffs, N):
    """Compress wavelet coefficients by retaining top N%."""
    cA, (cH, cV, cD) = coeffs
    coeffs_flat = np.hstack([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])
    threshold = np.percentile(np.abs(coeffs_flat), 100 - N)
    cA = np.where(np.abs(cA) < threshold, 0, cA)
    cH = np.where(np.abs(cH) < threshold, 0, cH)
    cV = np.where(np.abs(cV) < threshold, 0, cV)
    cD = np.where(np.abs(cD) < threshold, 0, cD)
    return cA, (cH, cV, cD)

def compress_dct(dct_matrix, N):
    """Compress DCT coefficients by retaining top N%."""
    coeffs_flat = dct_matrix.flatten()
    threshold = np.percentile(np.abs(coeffs_flat), 100 - N)
    compressed_dct = np.where(np.abs(dct_matrix) < threshold, 0, dct_matrix)
    return compressed_dct


results1 = apply_compression("c1.jpg", "c1", [1, 10, 50])
results2 = apply_compression("c2.jpg", "c2", [1, 10, 50])
results3 = apply_compression("c3.jpg", "c3", [1, 10, 50])

# Analyze results
print("-----------------")
print("c1.jpg")
print("-----------------")
for n, metrics in results1.items():
    print(f"N = {n}%")
    print(f"MSE (Haar): {metrics['mse_haar']:.4f}")
    print(f"MSE (DCT): {metrics['mse_dct']:.4f}")
print("-----------------")
print("c2.jpg")
print("-----------------")
for n, metrics in results2.items():
    print(f"N = {n}%")
    print(f"MSE (Haar): {metrics['mse_haar']:.4f}")
    print(f"MSE (DCT): {metrics['mse_dct']:.4f}")
print("-----------------")
print("c3.jpg")
print("-----------------")
for n, metrics in results3.items():
    print(f"N = {n}%")
    print(f"MSE (Haar): {metrics['mse_haar']:.4f}")
    print(f"MSE (DCT): {metrics['mse_dct']:.4f}")