import cv2
import numpy as np
import os
from skimage import transform as trans

def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance):
    num_points = distance.shape[1] // 2
    kps = np.zeros((distance.shape[0], num_points * 2))
    for i in range(num_points):
        kps[:, i * 2] = points[:, 0] + distance[:, i * 2]
        kps[:, i * 2 + 1] = points[:, 1] + distance[:, i * 2 + 1]
    return kps

def compute_similarity(feat1, feat2):
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    return similarity

def norm_crop_image(img, landmark, image_size=112):
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    src *= image_size / 112.
    
    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]
    
    warped = cv2.warpAffine(img, M, (image_size, image_size))
    return warped


def build_targets(detector, recognizer, params):
    targets = []
    
    for class_dir in os.listdir(params.faces_dir):
        class_path = os.path.join(params.faces_dir, class_dir)
        
        if not os.path.isdir(class_path):
            continue
            
        embeddings = []
        
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            try:
                bboxes, kpss = detector.detect(img, max_num=1)
                if len(kpss) == 0:
                    continue

                embedding = recognizer(img, kpss[0])
                if embedding is not None:
                    embeddings.append(embedding)
            except Exception:
                continue

        if embeddings:
            embedding = np.mean(embeddings, axis=0)
            embedding = embedding / np.linalg.norm(embedding)
            targets.append((embedding, class_dir))

    return targets


