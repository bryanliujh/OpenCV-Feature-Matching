# Create your views here.
import cv2
import matplotlib.pyplot as plt
from django.http import HttpResponse
import io


#brute force matcher takes one feature and compare against all the other feature and return the closest one
def testView(request):
    image1 = cv2.imread('images/img_1.jpg')
    image2 = cv2.imread('images/img_3.jpg')

    def to_gray(color_img):
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        return gray

    image1_gray = to_gray(image1)
    image2_gray = to_gray(image2)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image1_gray, None)
    kp2, des2 = sift.detectAndCompute(image2_gray, None)

    # BFMatcher with default params (normType, crosscheck)
    bf = cv2.BFMatcher()
    # k = 2 returns 2 best matches
    matches = bf.knnMatch(des1, des2, k=2)

    #for x in range(0, 1000):
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.

    img3 = cv2.drawMatchesKnn(image1_gray, kp1, image2_gray, kp2, good, None, flags=2)

    #add text to the image to display no of matches
    font = cv2.FONT_HERSHEY_SIMPLEX
    msg = 'there are %d matches' % (len(good))
    cv2.putText(img3, msg, (10, 500), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

    plt.imshow(img3)

    f = io.BytesIO()  # Python 3
    plt.savefig(f, format="png", facecolor=(0.95, 0.95, 0.95))
    plt.clf()

    return HttpResponse(f.getvalue(),content_type='image/png')
