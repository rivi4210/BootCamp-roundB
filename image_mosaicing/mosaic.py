import cv2
import numpy as np
import os


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images


def find_keypoints_and_descriptors(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors


def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def stitch_images(img1, img2):
    keypoints1, descriptors1 = find_keypoints_and_descriptors(img1)
    keypoints2, descriptors2 = find_keypoints_and_descriptors(img2)

    matches = match_features(descriptors1, descriptors2)

    if len(matches) < 4:
        print("not enough")
        return img1

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        print("cant calc homography")
        return img1

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = cv2.perspectiveTransform(pts1, M)
    result_pts = np.concatenate((pts1, pts2), axis=0)

    [x_min, y_min] = np.int32(result_pts.min(axis=0).flatten()) - 5
    [x_max, y_max] = np.int32(result_pts.max(axis=0).flatten()) + 5

    translation_dist = [-x_min, -y_min]
    M_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    panorama = cv2.warpPerspective(img1, M_translation @ M, (x_max - x_min, y_max - y_min))
    panorama[translation_dist[1]:h2 + translation_dist[1], translation_dist[0]:w2 + translation_dist[0]] = img2

    return panorama


def create_panorama(images):
    if len(images) < 2:
        return None

    base_image = images[0]
    for img in images[1:]:
        base_image = stitch_images(base_image, img)

    return base_image


if __name__ == '__main__':

    folder_path = r"C:\BootCamp\project\01.09\offset_0_None"
    images = load_images_from_folder(folder_path)

    if images:
        images = images[:50]
        panorama = create_panorama(images)
        if panorama is not None:
            cv2.imwrite('panorama.jpg', panorama)
            print("Successfully")
    else:
        print("no img found")
