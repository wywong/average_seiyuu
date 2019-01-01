from PIL import Image
import numpy as np
import dlib
import os
import logging

logging.basicConfig(filename='logs/face_detect.log', level=logging.DEBUG)

RAW_IMAGE_ROOT = 'tmp/raw_images/'
DETECTED_FACES_ROOT = 'tmp/detected_faces/'

image_filenames = os.listdir(RAW_IMAGE_ROOT)
image_filenames.sort()

predictor_path = os.environ['FACE_PREDICTOR']

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

for filename in image_filenames:
    path = RAW_IMAGE_ROOT + filename
    img = np.array(Image.open(path))

    dets = detector(img, 1)

    try:
        d = next(iter(dets))
        shape = predictor(img, d)
        for part in shape.parts():
            img[part.y][part.x] = [0, 255, 255]
        updated_image = Image.fromarray(img)
        updated_image.save(DETECTED_FACES_ROOT + '%s.jpg' % filename)
    except StopIteration:
        logging.warning("No face detected in %s" % filename)
