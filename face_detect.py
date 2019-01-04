from PIL import Image
from scipy import ndimage, spatial
from skimage import transform
from skimage.draw import polygon
import dlib
import logging
import numpy as np
import os

logging.basicConfig(filename='logs/face_detect.log', level=logging.DEBUG)

RAW_IMAGE_ROOT = 'tmp/raw_images/'
DETECTED_FACES_ROOT = 'tmp/detected_faces/'
AVERAGED_FACES_ROOT = 'tmp/average_faces/'

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


def get_rgb_transform(m, offset):
    m3 = np.zeros(shape=(3, 3))
    m3[0][0] = m[0][0]
    m3[0][1] = m[0][1]
    m3[1][0] = m[1][0]
    m3[1][1] = m[1][1]
    m3[2][2] = 1
    offset3 = np.array([offset[0], offset[1], 0])
    return (m3, offset3)


class PreprocessedImage:
    """
    image - the original image as a numpy array
    landmarks - the landmarks in the original coordinate system
    aligned_marks - landmarks with the eye's aligned
    """
    def __init__(self, image, landmarks, aligned_marks):
        self.image = image
        self.landmarks = landmarks
        self.aligned_marks = aligned_marks


class FacePreprocessor:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.environ['FACE_PREDICTOR'])

    def preprocess(self, image):
        detected_faces = self.detector(image, 1)
        try:
            detected_face = next(iter(detected_faces))
            shape = self.predictor(image, detected_face)
            landmarks = self.landmarks(detected_face, shape)
            m, offset = self.get_affine_matrix_offset_pair(landmarks)
            t = np.insert(m, 2, values=offset, axis=1)
            landmarks_padded = np.transpose(
                np.insert(landmarks, 2, values=np.ones(68), axis=1)
            )
            aligned_marks = np.transpose(np.matmul(t, landmarks_padded))
            return PreprocessedImage(image, landmarks, aligned_marks)

        except StopIteration:
            return None

    """
    returns a numpy array of landmarks [x, y]
    68 shape landmarks and 8 bounding box landmarks
    """
    def landmarks(self, detected_face, shape):
        landmarks_array = []
        for part in shape.parts():
            landmarks_array.append([part.y, part.x])

        return np.array(landmarks_array)

    def get_affine_matrix_offset_pair(self, landmarks):
        t = self.get_affine_transform(landmarks)
        m = t[np.ix_([0, 1], [0, 1])]
        offset = t[np.ix_([0, 1], [2])].flatten()
        return (m, offset)

    def get_affine_transform(self, landmarks):
        p1 = landmarks[LEFT_EYE_CORNER_INDEX]
        p2 = landmarks[RIGHT_EYE_CORNER_INDEX]
        old_coords = np.array([
            p1, p2
        ])
        new_coords = np.array([
            [EYE_HEIGHT, LEFT_X],
            [EYE_HEIGHT, RIGHT_X]
        ])
        similarity_transformation = transform.estimate_transform(
            'similarity', old_coords, new_coords
        )
        return similarity_transformation.params


class FaceMerger:
    def merge(self, preprocessed_images):
        aligned_marks = list(
            map(lambda p: p.aligned_marks, preprocessed_images)
        )
        target_landmarks = self.append_box_landmarks(
            self.average_np_arrays(aligned_marks)
        )

        result = np.zeros((HEIGHT, WIDTH, 3), dtype=float)
        for preprocessed_image in preprocessed_images:
            transformed_image = self.warp_triangles_image(
                preprocessed_image, target_landmarks
            )
            result += transformed_image
        result = np.floor(result / len(preprocessed_images)).astype(np.uint8)
        out = Image.fromarray(result)
        out.save('tmp/triangles/result.jpg')
        return None

    def warp_triangles_image(self, preprocessed_image, target_landmarks):
        src_landmarks = self.append_box_landmarks(preprocessed_image.landmarks)
        triangulation = spatial.Delaunay(src_landmarks).simplices
        result = np.zeros((HEIGHT, WIDTH, 3), dtype=float)
        n = len(triangulation)
        mask_sum = np.zeros((HEIGHT, WIDTH, 3), np.int32)
        for (i, tri) in zip(range(0, n), triangulation):
            img = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            src_r, src_c = self.marks_to_coords(tri, src_landmarks)
            mask = self.get_mask(src_r, src_c)
            src_coords = np.transpose(np.array([src_r, src_c]))
            target_r, target_c = self.marks_to_coords(tri, target_landmarks)
            target_coords = np.transpose(np.array([target_r, target_c]))
            similarity_transformation = transform.estimate_transform(
                'similarity', target_coords, src_coords
            )
            params = similarity_transformation.params
            m = params.copy()
            m[0][2] = 0
            m[1][2] = 0
            offset = [params[0][2], params[1][2], 0]
            img = mask * preprocessed_image.image
            transformed = ndimage.affine_transform(img, m, offset)
            result += transformed
            transformed_mask = np.where(transformed > 0, 1, 0).astype(np.uint8)
            mask_sum += transformed_mask

        mask_sum = mask_sum + np.where(mask_sum == 0, 1, 0)
        result = result * np.reciprocal(mask_sum)

        # out = Image.fromarray(np.floor(result).astype('uint8'))
        # out.save('tmp/triangles/result.jpg')
        return result



    def get_mask(self, r, c):
        rr, cc = polygon(r, c)
        mask = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        mask[rr, cc] = [1, 1, 1]
        return mask


    def marks_to_coords(self, tri, marks):
        row_coords = []
        col_coords = []
        for p in tri:
            y = marks[p][0]
            x = marks[p][1]
            row_coords.append(y)
            col_coords.append(x)
        return (row_coords, col_coords)

    def average_np_arrays(self, arrays):
        shape = arrays[0].shape
        count = float(len(arrays))
        avg_array = np.zeros(shape, np.float)

        for arr in arrays:
            avg_array += arr / count

        return np.round(avg_array)

    def append_box_landmarks(self, marks):
        landmarks_array = []
        bottom = HEIGHT
        top = 0
        left = 0
        right = WIDTH

        mid_x = left + (right - left) / 2
        mid_y = bottom - (bottom - top) / 2

        landmarks_array.append([mid_y, right])
        landmarks_array.append([top, right])
        landmarks_array.append([top, mid_x])
        landmarks_array.append([top, left])
        landmarks_array.append([mid_y, left])
        landmarks_array.append([bottom, left])
        landmarks_array.append([bottom, mid_x])
        landmarks_array.append([bottom, right])
        return np.insert(
            marks, len(marks), landmarks_array, axis=0
        )


preprocessor = FacePreprocessor()
merger = FaceMerger()

imgs = []
for filename in image_filenames[:3]:
    path = RAW_IMAGE_ROOT + filename
    img = np.array(Image.open(path))

    if len(img.shape) != 3:
        logging.warning('Skipping %s because it is grayscale')
        continue

    height, width, channels = img.shape

    if height != HEIGHT or width != WIDTH:
        logging.warning('Skipping %s because it is not the right size')
        continue

    preprocessed_image = preprocessor.preprocess(img)
    if preprocessed_image is None:
        logging.warning("No face detected in %s" % filename)
    else:
        imgs.append(preprocessed_image)

result = merger.merge(imgs)

# average_image = Image.fromarray(average_images(imgs))
# average_image.save(AVERAGED_FACES_ROOT + 'average.jpg')
