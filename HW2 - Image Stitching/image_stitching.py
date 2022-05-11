import cv2
import numpy as np
import argparse


def extract_and_match_feature(img_1, img_2, ratio_test=0.7):
    """
    1/  extract SIFT feature from image 1 and image 2,
    2/  use a bruteforce search to find pairs of matched features:
        for each feature point in img_1, find its best matched feature point in img_2
    3/ apply ratio test to select the set of robust matched points

    :param img_1: input image 1
    :param img_2: input image 2
    :param ratio_test: ratio for the robustness test
    :return list_pairs_matched_keypoints: a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]]]
    """
    # to be completed ....

    img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    # pip install -U opencv-contrib-python==3.4.0.12
    sift = cv2.xfeatures2d.SIFT_create()
    #compute decriptors from keypoints found
    img_1_kp, img_1_des = sift.detectAndCompute(img_1_gray, None)

    img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    img_2_kp, img_2_des = sift.detectAndCompute(img_2_gray, None)

    #2 BF match for each feature point in img_1 and find its best match in img_2
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(img_1_des, img_2_des, k=2)
    #print(type(matches))
    
    #seeSee = []
    #3 apply ratio test
    list_pairs_matched_keypoints = []

    for i,j in matches:
        if i.distance < ratio_test*j.distance:
            list_pairs_matched_keypoints.append([img_1_kp[i.queryIdx].pt,img_2_kp[i.trainIdx].pt])
            #seeSee.append([i])
    
    """
    img3 = cv2.drawMatchesKnn(img_1_gray,img_1_kp,img_2_gray,img_2_kp,seeSee,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow("test", img3)
    cv2.waitKey(0)
    """
    return list_pairs_matched_keypoints


def find_homography_ransac(list_pairs_matched_keypoints,
                           threshold_ratio_inliers=0.85,
                           threshold_reprojection_error=3,
                           max_num_trial=1000):
    """
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points,
    transform the second set of feature point to the first (e.g. warp image 2 to image 1)

    :param list_pairs_matched_keypoints: a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],...]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples,
                                    accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojection_error: threshold of reprojection error (measured as euclidean distance, in pixels)
                                            to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    """
 
    #print(type(list_pairs_matched_keypoints))
    #print(list_pairs_matched_keypoints)
    #print(type(list_pairs_matched_keypoints[0][1]))

    keyA = []
    keyB = []
 
    for [i,_] in list_pairs_matched_keypoints:
        keyA.append(i)

    for [_,j] in list_pairs_matched_keypoints:
        keyB.append(j)
    
    matA = np.float32(np.array(keyA))  #could be changed for float
    matB = np.float32(np.array(keyB))
    #np.reshape(matA,(-1,1,2))
    #np.reshape(matB,(-1,1,2))
    #number = len(matA)
    #print(number)
    """
    print(matA)
    print("")
    print(matB)
    """
    best_H, status = cv2.findHomography(matA, matB, cv2.RANSAC,ransacReprojThreshold=threshold_reprojection_error, maxIters=max_num_trial, confidence=threshold_ratio_inliers)
    #print(status)
    #print(type(best_H))
    #print(best_H.shape)
    #print(best_H)
    """
    iterations = 0
    best_inlier_index = None
    countInlier = 0
    inlierStore = 0

    aStore = []
    bStore = []
    buffer = []

    while iterations < max_num_trial:
        buffer.clear()
        countInlier = 0
        A = np.random.randint(number, size = 1)
        A = np.int64(A)
        ran1 = A.item()

        B = np.random.randint(number, size = 1)
        B = np.int64(B)
        ran2 = B.item()
        #randomly select 2 points
        r_pt1 = matA[ran1]
        r_pt2 = matB[ran2]

        #fitting line
        kLine = np.subtract(r_pt2, r_pt1)  
        kLineNorm = np.linalg.norm(kLine)
        normVector = np.divide(kLine, kLineNorm)

        #compute distance between all points with fitting line
        for k in range(number):
            if k == ran2:
                continue
            
            temp = np.subtract(matB[k], r_pt1)
            tempNorm = np.linalg.norm(temp)
            tempNormVector = np.divide(temp,tempNorm)

            #angle between
            dot = np.dot(normVector, tempNormVector)

            #fix bug
            eps = 1e-6
            if 1.0<dot<1.0+eps:
                dot=1.0
            elif -1.0-eps <dot<-1.0:
                dot = -1.0

            dot = np.asarray(dot)
            angle = np.arccos(dot)
            dist = np.sin(angle)*(temp)
            distNorm = np.linalg.norm(dist)

            #compute the inliers with distance smaller than threshold
            if distNorm < threshold_reprojection_error:
                countInlier = countInlier +1
                
        #biggest set of inliers
        if countInlier>inlierStore:
            inlierStore = countInlier

        iterations = iterations + 1
    """
    #print(type(dot))
    #print(dist)

    #print(status)
    #print(best_H)
    #inverse
    best_H = np.linalg.inv(best_H)

    return best_H


def warp_blend_image(img_1, H, img_2):
    """
    1/  warp image img_2 using the homography H to align it with image img_1
        (using inverse warping and bilinear resampling)
    2/  stitch image img_2 to image img_1 and apply average blending to blend the 2 images into a single panorama image

    :param img_1:  the original first image
    :param H: estimated homography
    :param img_2:the original second image
    :return img_panorama: resulting panorama image
    """
    img_panorama = None
    
    #print(img_1.shape)
    height = img_1.shape[0]+img_2.shape[0] #if want to clip, delete img_2.shape[0]
    width = img_1.shape[1] +img_2.shape[1]

    #dst = cv2.perspectiveTransform(pts, H)
    #print(type(dst))
    #print(dst)
    #img_2 =cv2.polylines(img_2,[np.int32(dst)], True,255,3,cv2.LINE_AA)
    #cv2.imshow("original_image_overlapping.jpg", img_2)
    #cv2.waitKey(0)
    dst = cv2.warpPerspective(img_2, H, (width, height ), cv2.WARP_INVERSE_MAP)
    #return the dst[x-coor,y-coor][rgb array]

    #print(len(dst))
    #cv2.imshow("test", dst)
    #cv2.waitKey(0)
    #dst[0:img_2.shape[0], 0:img_2.shape[1]] = img_1

    for i in range(img_2.shape[0]):
        for j in range(img_2.shape[1]):
            if np.array_equal(dst[i,j],[0,0,0]):
                dst[i,j] = img_1[i,j]
            else:
                dst[i,j][0] = (int(dst[i,j][0]) + int(img_1[i,j][0]))//2  #r
                dst[i,j][1] = (int(dst[i,j][1]) + int(img_1[i,j][1]))//2  #g
                dst[i,j][2] = (int(dst[i,j][2]) + int(img_1[i,j][2]))//2  #b


    img_panorama = dst

    """ #the ultimate code
    stitcher = cv2.createStitcher(False)
    """

    return img_panorama


def stitch_images(img_1, img_2):
    """
    :param img_1: input image 1 is the reference image. We will not warp this image
    :param img_2: We warp this image to align and stich it to the image 1
    :return img_panorama: the resulting stiched image
    """
    print('==================================================================================')
    print('===== stitch two images to generate one panorama image =====')
    print('==================================================================================')

    # ===== extract and match features from image 1 and image 2
    list_pairs_matched_keypoints = extract_and_match_feature(img_1=img_1, img_2=img_2, ratio_test=0.7)

    # ===== use RANSAC algorithm to find homography to warp image 2 to align it to image 1
    H = find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85,
                               threshold_reprojection_error=3, max_num_trial=1000)

    # ===== warp image 2, blend it with image 1 using average blending to produce the resulting panorama image
    img_panorama = warp_blend_image(img_1=img_1, H=H, img_2=img_2)

    return img_panorama


if __name__ == "__main__":
    print('==================================================================================')
    print('CSCI3290, Spring 2020, Assignment 2: image stitching')
    print('==================================================================================')

    parser = argparse.ArgumentParser(description='Image Stitching')
    parser.add_argument('--im1', type=str, default='test_images/MelakwaLake1.png',
                        help='path of the first input image')
    parser.add_argument('--im2', type=str, default='test_images/MelakwaLake2.png',
                        help='path of the second input image')
    parser.add_argument('--output', type=str, default='MelakwaLake99.png',
                        help='the path of the output image')
    args = parser.parse_args()

    # ===== read 2 input images
    img_1 = cv2.imread(args.im1)
    img_2 = cv2.imread(args.im2)

    # ===== create a panorama image
    img_panorama = stitch_images(img_1=img_1, img_2=img_2)

    # ===== save panorama image
    cv2.imwrite(filename=args.output, img=img_panorama.clip(0.0, 255.0).astype(np.uint8))
    print("finished")