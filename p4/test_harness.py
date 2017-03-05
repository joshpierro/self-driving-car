import utils
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimage


cam_mtx, cam_dist = utils.calibrate_camera()
image = cv.imread('test_images/test1.jpg')
masked_image = utils.mask_image(image)
# undistorted_dash = cv.undistort(image, cam_mtx, cam_dist, None, cam_mtx)
# m = cv.getPerspectiveTransform(utils.source(), utils.destination())
# m_inverse = cv.getPerspectiveTransform(utils.destination(), utils.source())
# image_size = (undistorted_dash.shape[1], undistorted_dash.shape[0])
# warped = cv.warpPerspective(image, m, image_size, flags=cv.INTER_LINEAR)

plt.subplot(1, 2, 2)
plt.imshow(masked_image)
plt.xlabel('warped Image')
plt.show(block=True)
