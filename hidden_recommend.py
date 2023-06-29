from skimage import data
from skimage.filters.rank import entropy
from skimage.morphology import disk, ball
from tqdm import tqdm as tqdm
import numpy as np
import cv2


def HOG(img):
    # Grayscale
    def BGR2GRAY(img):
        gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
        return gray

    # Magnitude and gradient
    def get_gradXY(gray):
        H, W = gray.shape

        # padding before grad
        gray = np.pad(gray, (1, 1), 'edge')

        # get grad x
        gx = gray[1:H+1, 2:] - gray[1:H+1, :W]
        # get grad y
        gy = gray[2:, 1:W+1] - gray[:H, 1:W+1]
        # replace 0 with 
        gx[gx == 0] = 1e-6

        return gx, gy

    # get magnitude and gradient
    def get_MagGrad(gx, gy):
        # get gradient maginitude
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # get gradient angle
        gradient = np.arctan(gy / gx)

        gradient[gradient < 0] = np.pi / 2 + gradient[gradient < 0] + np.pi / 2

        return magnitude, gradient

    # Gradient histogram
    def quantization(gradient):
        # prepare quantization table
        gradient_quantized = np.zeros_like(gradient, dtype=int) # np.int is deprecated after 1.20

        # quantization base
        d = np.pi / 9

        # quantization
        for i in range(9):
            gradient_quantized[np.where((gradient >= d * i) & (gradient <= d * (i + 1)))] = i

        return gradient_quantized


    # get gradient histogram
    def gradient_histogram(gradient_quantized, magnitude, N=8):
        # get shape
        H, W = magnitude.shape

        # get cell num
        cell_N_H = H // N
        cell_N_W = W // N
        histogram = np.zeros((cell_N_H, cell_N_W, 9), dtype=np.float32)

        # each pixel
        for y in range(cell_N_H):
            for x in range(cell_N_W):
                for j in range(N):
                    for i in range(N):
                        histogram[y, x, gradient_quantized[y * 4 + j, x * 4 + i]] += magnitude[y * 4 + j, x * 4 + i]

        return histogram

		# histogram normalization
    def normalization(histogram, C=3, epsilon=1):
        cell_N_H, cell_N_W, _ = histogram.shape
        ## each histogram
        for y in range(cell_N_H):
    	    for x in range(cell_N_W):
       	    #for i in range(9):
                histogram[y, x] /= np.sqrt(np.sum(histogram[max(y - 1, 0) : min(y + 2, cell_N_H),
                                                            max(x - 1, 0) : min(x + 2, cell_N_W)] ** 2) + epsilon)

        return histogram

    # 1. BGR -> Gray
    gray = BGR2GRAY(img)

    # 1. Gray -> Gradient x and y
    gx, gy = get_gradXY(gray)

    # 2. get gradient magnitude and angle
    magnitude, gradient = get_MagGrad(gx, gy)

    # 3. Quantization
    gradient_quantized = quantization(gradient)

    # 4. Gradient histogram
    histogram = gradient_histogram(gradient_quantized, magnitude)
    
    # 5. Histogram normalization
    histogram = normalization(histogram)

    return histogram

def recommend(origin,fore,mask_dilated):
    h,w,_=fore.shape
    fore_hog=HOG(fore)
    h_hog,w_hog,_=fore_hog.shape

    fore_hist=fore_hog[np.where(cv2.resize(mask_dilated,(w_hog,h_hog))>0)]

    origin_hog=HOG(origin)
    h_hog_origin,w_hog_origin,_=origin_hog.shape

    mat_idxes=np.where(cv2.resize(mask_dilated,(w_hog,h_hog))>0)
    mat_idxes_origin=np.where(mask_dilated>0)


    origin_gray=cv2.cvtColor(origin,cv2.COLOR_BGR2GRAY)
    ent = entropy(origin_gray, disk(5))


    objective_list=[]
    for y_start in tqdm(range(0,h_hog_origin-h//8)):
        for x_start in range(0,w_hog_origin-w//8):
            try:
                bg_hist=origin_hog[y_start:y_start+h//8,x_start:x_start+w//8]
                bg_hist=bg_hist[mat_idxes]
                mat_entropy=ent[y_start*8:y_start*8+h,x_start*8:x_start*8+w][mat_idxes_origin]
                objective = np.sum(np.abs(fore_hist-bg_hist)) - np.sum(mat_entropy)
                objective_list.append([objective,y_start,x_start])
            except:
                print(y_start,y_start+h//8,x_start,x_start+w//8,bg_hist.shape)

    objective_list=np.array(objective_list)
    idx=np.argmin(objective_list[:,0])
    objective_list[idx]

    _,y_start,x_start=objective_list[np.argmin(objective_list[:,0])]
    y_start,x_start=int(y_start*8),int(x_start*8)
    return y_start,x_start
    