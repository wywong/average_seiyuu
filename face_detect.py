from PIL import Image
from skimage import transform
from scipy import ndimage
import dlib
import logging
import numpy as np
import os

logging.basicConfig(filename='logs/face_detect.log', level=logging.DEBUG)

RAW_IMAGE_ROOT = 'tmp/raw_images/'
DETECTED_FACES_ROOT = 'tmp/detected_faces/'

HEIGHT = 350
WIDTH = 225

"""
Left eye corner is 36
Right eye corner is 45
Nose bright is 27 to 30
"""
LEFT_EYE_CORNER_INDEX = 36
LEFT_X = WIDTH * 0.3

RIGHT_EYE_CORNER_INDEX = 45
RIGHT_X = WIDTH * 0.7

EYE_HEIGHT = HEIGHT / 3.0

image_filenames = os.listdir(RAW_IMAGE_ROOT)
image_filenames.sort()

predictor_path = os.environ['FACE_PREDICTOR']

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def get_affine_matrix_offset_pair(parts, is_rgb):
    t = get_affine_transform(parts)
    m = t[np.ix_([0, 1], [0, 1])]
    offset = t[np.ix_([0, 1], [2])].flatten()
    if is_rgb:
        return get_rgb_transform(m, offset)
    else:
        return (m, offset)


def get_affine_transform(parts):
    p1 = parts[LEFT_EYE_CORNER_INDEX]
    p2 = parts[RIGHT_EYE_CORNER_INDEX]
    old_coords = np.array([
        [p1.y, p1.x],
        [p2.y, p2.x]
    ])
    new_coords = np.array([
        [EYE_HEIGHT, LEFT_X],
        [EYE_HEIGHT, RIGHT_X]
    ])
    similarity_transformation = transform.estimate_transform(
        'similarity', new_coords, old_coords
    )
    return similarity_transformation.params


def get_rgb_transform(m, offset):
    m3 = np.zeros(shape=(3, 3))
    m3[0][0] = m[0][0]
    m3[0][1] = m[0][1]
    m3[1][0] = m[1][0]
    m3[1][1] = m[1][1]
    m3[2][2] = 1
    offset3 = np.array([offset[0], offset[1], 0])
    return (m3, offset3)

for filename in image_filenames:
    path = RAW_IMAGE_ROOT + filename
    img = np.array(Image.open(path))

    is_rgb = len(img.shape) == 3
    dets = detector(img, 1)

    try:
        d = next(iter(dets))
        shape = predictor(img, d)
        m, offset = get_affine_matrix_offset_pair(shape.parts(), is_rgb)
        transformed_image = ndimage.affine_transform(img, m, offset)
        updated_image = Image.fromarray(transformed_image)
        updated_image.save(DETECTED_FACES_ROOT + '%s.jpg' % filename)
    except StopIteration:
        logging.warning("No face detected in %s" % filename)
