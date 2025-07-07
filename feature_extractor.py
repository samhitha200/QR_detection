import cv2

def extract_white_area_features(image):
    features = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_pixels = gray[white_mask == 255]

    if white_pixels.size == 0:
        features.extend([0] * 16)
        return np.array(features)

    white_ratio = white_pixels.size / gray.size

  
    features.append(np.mean(white_pixels))
    features.append(np.std(white_pixels))
    features.append(np.min(white_pixels))
    features.append(np.max(white_pixels))
    features.append(white_ratio)

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_masked = lbp[white_mask == 255]
    lbp_hist, _ = np.histogram(lbp_masked, bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    features.extend(lbp_hist.tolist())

    entropy_val = shannon_entropy(white_pixels)
    features.append(entropy_val)

    return np.array(features)
