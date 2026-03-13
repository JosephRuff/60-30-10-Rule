from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from skimage import color
import numpy as np

HARMONY_TEMPLATES = {
    'triadic': [0, 120, 240],
    'analogous': [0, 30, 60],
    'split_complementary': [0, 150, 210],
    'complementary_accent': [0, 180, 30],
    'clash': [0, 90, 180],
    'accented_analogous': [0, 30, 180],
    'near_complementary': [0, 15, 180],
}

def score_harmony(hues, template):
    hues_norm = (hues - hues[0]) % 360
    diff = abs(hues_norm[1] - template[1]) + abs(hues_norm[2] - template[2])
    score = 1 - (diff / 360)

    mirrored = [0, (360 - template[1]) % 360, (360 - template[2]) % 360]
    diff_mirrored = abs(hues_norm[1] - mirrored[1]) + abs(hues_norm[2] - mirrored[2])
    score_mirrored = 1 - (diff_mirrored / 360)

    return max(score, score_mirrored)

def get_scores(pixels, labels, centers_rgb):
    # pixels: use same colour space used for clustering (used for silhouette, davies_bouldin, calinski_harabasz)
    # centers_rgb: must be in RGB 0-1 (used for harmony and ratio calculations)
    scores = {}

    idx = np.random.choice(len(pixels), min(10000, len(pixels)), replace=False)
    
    # get model scores
    scores['silhouette'] = float(silhouette_score(pixels[idx], labels[idx]))
    scores['davies_bouldin'] = float(davies_bouldin_score(pixels[idx], labels[idx]))
    scores['calinski_harabasz'] = float(calinski_harabasz_score(pixels, labels))
    
    percentages = np.bincount(labels) / len(labels) * 100
    size_order = np.argsort(percentages)[::-1]
    percentages = percentages[size_order].copy()
    centers_rgb = centers_rgb[size_order].copy()

    hues = color.rgb2hsv(centers_rgb.reshape(1, -1, 3)).reshape(-1, 3)[:, 0] * 360

    harmony_scores = {}

    for name, template in HARMONY_TEMPLATES.items():
        harmony_scores[name] = float(score_harmony(hues, template))

    harmony_scores['best_scheme'] = max(harmony_scores, key=harmony_scores.get)
    harmony_scores['best_score'] = harmony_scores[harmony_scores['best_scheme']]

    scores['ratio_score'] = float(1 - (np.abs(percentages - np.array([60, 30, 10])).sum() / 80))
    
    scores['harmony_scores'] = harmony_scores
    
    return scores