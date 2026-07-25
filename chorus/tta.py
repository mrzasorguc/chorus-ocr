"""Image preprocessing and test-time augmentation variants."""
import cv2
import numpy as np

def _unsharp(img):
    g = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.addWeighted(img, 1.6, g, -0.6, 0)

def v_up2x(img):
    up = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return _unsharp(up)

def v_binarize(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

def v_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def v_denoise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)

def estimate_skew(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = cv2.bitwise_not(b)
    coords = cv2.findNonZero(b)
    if coords is None:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    return float(angle)

def rotate(img, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def build_variants(img, fast=False):
    """[(name, image, quality_weight)] dondurur."""
    out = [("orig", img, 1.0), ("up2x", v_up2x(img), 1.0)]
    if fast:
        return out
    out.append(("bin", v_binarize(img), 0.9))
    out.append(("clahe", v_clahe(img), 0.9))
    ang = estimate_skew(img)
    if abs(ang) > 0.5:
        dsk = rotate(img, ang)
        out.append(("deskew", dsk, 1.0))
        out.append(("deskew_up2x", v_up2x(dsk), 1.0))
        out.append(("deskew_dn", v_denoise(dsk), 0.95))
    else:
        out.append(("denoise", v_denoise(img), 0.95))
    return out
