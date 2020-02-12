import cv2
import numpy as np
import bayesian_sgm

seg = bayesian_sgm.BayesianColorSGM()
seg.learn_from_dirs("datasets/alpha", "datasets/asli")
rokok = cv2.imread("rokok.png",-1)
scale_percent = 20
width = int(rokok.shape[1] * scale_percent / 100)
height = int(rokok.shape[0] * scale_percent / 100)
dsize = (width,height)
rokok = cv2.resize(rokok,dsize)
x_offset=y_offset=50
alpha_s = rokok[:, :, 3] / 255.0
alpha_l = 1.0 - alpha_s
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('filter.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
while(True):
    ret, img = cap.read()
    bins = seg.apply(img)
    bins = bins.astype(np.uint8)
    blur = cv2.GaussianBlur(bins,(5,5),0)
    ret3,threshold = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy =  cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)>0:
        cnt = max(contours, key = cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        center = (cX,cY)
        y1, y2 = cY, cY + rokok.shape[0]
        x1, x2 = cX-80, cX + rokok.shape[1]-80
        for c in range(0, 3):
            img[y1:y2, x1:x2, c] = (alpha_s * rokok[:, :, c] +alpha_l * img[y1:y2, x1:x2, c])

        # if radius>100:
        #     # cv2.circle(img,center, int(radius),(255, 0, 255), 2)
        #     # cv2.circle(img,center, 5, (0, 0, 0), -1)
        #     cv2.imshow("frame", img)
        #      # cv2.imshow("binary", res)
        #     k = cv2.waitKey(30) & 0xFF
        #     if k == 27:
        #         break
    cv2.imshow("frame", img)
    out.write(img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
out.release()
cv2.destroyAllWindows()
cap.release()
