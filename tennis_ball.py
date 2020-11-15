from cv2 import connectedComponentsWithStats, findContours, morphologyEx, Canny
from human_attributes_estimation.utils.utils import resize, read_imgfile
import matplotlib.pyplot as plt
import human_attributes_estimation.utils.constants as cst
import numpy as np
import cv2


class NoReferentialError(Exception):
    pass


class TennisBall:
    # Referential tennis
    REF_TENNIS_BALL_MIN = 6.54 / 2
    REF_TENNIS_BALL_MAX = 6.86 / 2
    REF_TENNIS_BALL = 6.7 / 2

    @staticmethod
    def _get_fitting_circle(mask):

        _, contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # only proceed if at least one contour was found
        if len(contours) > 0:
            # find the largest contour in the mask, then use
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)

        else:
            raise NoReferentialError('No contour to fit')

        return x, y, radius

    @staticmethod
    def _get_fitting_ellipse(mask):
        # The function will return first lines then column -> Y, X
        [Y, X] = np.nonzero(mask)
        x_mean, y_mean = np.mean(X), np.mean(Y)
        X, Y = X-x_mean, Y-y_mean

        U, S, _ = np.linalg.svd(np.stack((X, Y)))

        a = np.sqrt(2/len(X))*S[0]*np.linalg.norm(U[:, 0])
        b = np.sqrt(2/len(X))*S[1]*np.linalg.norm(U[:, 1])

        return x_mean, y_mean, a, b, U

    @staticmethod
    def _get_bbox_rescaled(bbox, fx, img_shape):
        # Rescale for original image
        bbox_re = (np.array(bbox)*(1/fx)).astype(int)
        bbox_re_ = np.zeros(bbox_re.shape, dtype=int)
        # Enlarge bbox with width/height
        bbox_re_[:2] = bbox_re[:2] - 1*(bbox_re[2:]-bbox_re[:2])
        bbox_re_[2:] = bbox_re[2:] + 1*(bbox_re[2:]-bbox_re[:2])
        # Limit to bbox of original image
        bbox_re_[:2] = np.max([bbox_re_[:2], [0, 0]], axis=0)
        bbox_re_[2:] = np.min([bbox_re_[2:], [img_shape[0], img_shape[1]]], axis=0)
        return bbox_re_

    @staticmethod
    def draw_detection_details(img_src, x, y, radius, mask=None, mask2=None):

        fig, ax = plt.subplots(1, 2)

        img_ = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
        mask_ = np.zeros((img_src.shape[0], img_src.shape[1], 3), dtype=np.uint8)
        if mask is not None:
            mask_[:, :, 0] = 255 * mask
        mask_ = cv2.circle(mask_, (int(x), int(y)), radius=int(radius), color=cst.COLOR_JOINT, thickness=3)

        img_ = cv2.addWeighted(img_, 1, mask_, 0.5, 0)
        ax[0].imshow(mask2)
        ax[1].imshow(img_)
        plt.show()

    @staticmethod
    def find_ball_(img, radius_min=15, score_min=0.5):
        """
        Returns the bounding box of the tennis ball using only the opencv library.
        :param img: input image containing the tennis ball.
        :param radius_min: minimum admissible radius for the tennis ball.
        :param score_min: minimum roundness score for the ball.
        :return: bounding box and thresholded image
        """
        # Define local functions
        def get_perimeter(component, label_image):
            """
            Computes the perimeter of a given connected component.
            :param component: index of the component on which to compute the perimeter.
            :param label_image: labeled image w.r.t. each component.
            :return: connected component's perimeter.
            """
            mask = (label_image == component).astype(np.uint8)
            contours, _ = findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            return contours[0].shape[0]

        def get_bounding_box(component, stats):
            """
            Returns bounding box of a given connected component with same format as used in skimage
            (min_row, min_col, max_row, max_col).
            :param component: index of the component on which to compute the bounding box.
            :param stats: list of statistics for each connected component.
            :return: connected component's bounding box.
            """
            min_row = stats[component][cv2.CC_STAT_TOP]
            min_col = stats[component][cv2.CC_STAT_LEFT]
            max_row = stats[component][cv2.CC_STAT_TOP] + stats[component][cv2.CC_STAT_HEIGHT]
            max_col = stats[component][cv2.CC_STAT_LEFT] + stats[component][cv2.CC_STAT_WIDTH]
            return min_row, min_col, max_row, max_col

        # Define constants
        area_min = np.pi * radius_min ** 2

        # Get the connected components
        n_components, label_image, stats, centroids = connectedComponentsWithStats(img)
        score = []

        for component in range(n_components):
            if stats[component, cv2.CC_STAT_AREA] < area_min:
                score.append(0)
                continue

            # Get roundness descriptor
            perimeter = get_perimeter(component, label_image)
            roundness = (4 * np.pi) / (perimeter ** 2 / stats[component, cv2.CC_STAT_AREA])

            # Select sufficiently round components
            if roundness < score_min:
                score.append(0)
                continue
            score.append(roundness)

        if np.max(score) < score_min:
            raise NoReferentialError('No valid score: {}'.format(np.max(score)))
        id_max = int(np.argmax(score))
        bbox = get_bounding_box(id_max, stats)
        return bbox, 255 * (id_max == label_image).astype(np.uint8)

    @staticmethod
    def bgr2lab_polar(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(int)
        (id_l, id_a, id_b) = (0, 1, 2)

        theta = np.arctan((lab[:, :, id_a] - 128) / (lab[:, :, id_b] - 128 + 1e-3))
        theta = np.pi * (1 - np.sign(lab[:, :, id_a] - 128 + 1e-3)) / 2 + theta
        r = np.linalg.norm(lab[:, :, id_a:(id_b+1)]-128, axis=2)

        return lab[:, :, id_l], theta, r

    @staticmethod
    def detect_(path_image, return_mask):
        def get_disk(radius):
            """
            Creates a disk morphological element.
            :param radius: radius of the disk.
            :return: 8-bit array containing the disk.
            """
            kernel_size = 2 * radius + 1
            disk = np.zeros(shape=(kernel_size, kernel_size))
            for i in range(kernel_size):
                for j in range(kernel_size):
                    disk[i, j] = (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2
            return disk.astype(np.uint8)

        try:
            # STEP 1: Process small version of image
            # Load image and rescale it (smaller version)
            image = read_imgfile(path_image)
            image, _ = resize(image, h_max=np.max(image.shape[0:2]))
            image_low, fx = resize(image)

            # Detect ball in hsv -> contour + hue range
            # mask_low, _ = TennisBall.intersection_hsv(image_low)
            l, theta, r = TennisBall.bgr2lab_polar(image_low)
            mask_low = np.logical_and(np.logical_and(theta > 2.5, theta < 3.0), r > 30).astype(np.uint8)
            mask_low = morphologyEx(mask_low, cv2.MORPH_CLOSE, get_disk(5))

            # Extract most probable bbox for ball
            bbox, mask_low = TennisBall.find_ball_(mask_low, radius_min=13)

            # Potentially return the mask in low resolution
            if return_mask:
                if np.median(mask_low) == 255:
                    mask_low = 255 - mask_low

                # Apply a last cleaning step
                mask_low = morphologyEx(mask_low, cv2.MORPH_OPEN, get_disk(10))
                return mask_low

            bbox_re = TennisBall._get_bbox_rescaled(bbox, fx, image.shape)

            # STEP 2: Process image full resolution for precision
            image_high = image[bbox_re[0]:bbox_re[2], bbox_re[1]:bbox_re[3]]
            l, theta, r = TennisBall.bgr2lab_polar(image_high)
            mask_high = np.logical_and(np.logical_and(theta > 2.3, theta < 3.0), r > 30).astype(np.uint8)

            # Adjust size of disk according to size of original image
            mask_high = morphologyEx(mask_high, cv2.MORPH_OPEN, get_disk(2 + image_high.shape[0] // 100))
            mask_high = morphologyEx(mask_high, cv2.MORPH_CLOSE, get_disk(2 + image_high.shape[0] // 40))

            _, mask_high = TennisBall.find_ball_(mask_high)

            # # Get enclosing circle
            edge = Canny(image=mask_high, threshold1=200, threshold2=255)

            # x, y, r = TennisBall._get_fitting_circle(mask)
            x, y, a, b, U = TennisBall._get_fitting_ellipse(edge)
            r = np.sqrt(a * b)

            TennisBall.draw_detection_details(image_high, x, y, r, mask_high, mask_low)

            # Correct x,y, coordinate to full image
            x += bbox_re[1]
            y += bbox_re[0]
            return TennisBall.REF_TENNIS_BALL / (2 * r), {'name': cst.PATTERN_TENNIS_BALL,
                                                          'joints_dict': (x, y, r, a, b, U.tolist())}

        except NoReferentialError as e:
            print(e)
            pass

        except Exception as e:
            print('Error:', e)
            pass

        return 0, None
