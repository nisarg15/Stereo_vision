# Importing the main libraries
import cv2
import numpy as np

# Function to resize the image
def resize(frames,scale):
    
    widht = int(frames.shape[1] * scale)
    height = int(frames.shape[0] * scale)
    dim = (widht,height)
    return cv2.resize(frames,dim)

# CALIBRATION
# SIFT Feature dectator
def detect_feature(imgA, imgB):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(imgA, None)
    kp2, des2 = sift.detectAndCompute(imgB, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    best_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
            best_matches.append([m])

    feature_1 = []
    feature_2 = []

    for i, match in enumerate(good):
        feature_1.append(kp1[match.queryIdx].pt)
        feature_2.append(kp2[match.trainIdx].pt)
    return kp1, kp2,good,feature_1,feature_2

# Compute the Fundamental Matrix
def fundamental_matrix(l_pts, r_pts):


    A = np.empty((8, 9))
    for i in range(1,7):
        x1 = l_pts[i][0]
        x2 = r_pts[i][0]
        y1 = l_pts[i][1]
        y2 = r_pts[i][1]
        A[i] = np.array([x1 * x2, x2 * y1, x2,
                         y2 * x1, y2 * y1, y2,
                         x1, y1, 1])
    # Compute F matrix by evaluating SVD
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # Enforce the F matrix to rank 2
    U_n, S_n, V_n = np.linalg.svd(F)
    S2 = np.array([[S_n[0], 0, 0], [0, S_n[1], 0], [0, 0, 0]])
    F = np.dot(np.dot(U_n, S2), V_n)
    return F


# Function to find the Essential Matrix which can be use to find the Rotation and Translation 
def essential_matrix(F, K):
    E = np.dot(K.T, np.dot(F, K))
    u, s, v = np.linalg.svd(E)
    # correction of singular values by reconstructing it with (1, 1, 0) singular values
    s_new = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    E_new = u @ s_new @ v
    return E_new

# Function to find the camera pose
def extract_camera_pose(E):

    u, s, v = np.linalg.svd(E, full_matrices=True)
    w = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]]).reshape(3, 3)

    # Computing 4 possible camera poses
    c1 = u[:, 2].reshape(3, 1)
    r1 = np.dot(np.dot(u, w), v).reshape(3, 3)
    c2 = -u[:, 2].reshape(3, 1)
    r2 = np.dot(np.dot(u, w), v).reshape(3, 3)
    c3 = u[:, 2].reshape(3, 1)
    r3 = np.dot(np.dot(u, w.T), v).reshape(3, 3)
    c4 = -u[:, 2].reshape(3, 1)
    r4 = np.dot(np.dot(u, w.T), v).reshape(3, 3)
    if np.linalg.det(r1) < 0:
        c1 = -c1
        r1 = -r1
    if np.linalg.det(r2) < 0:
        c2 = -c2
        r2 = -r2
    if np.linalg.det(r3) < 0:
        c3 = -c3
        r3 = -r3
    if np.linalg.det(r4) < 0:
        c4 = -c4
        r4 = -r4
    cam_center = np.array([c1, c2, c3, c4])
    cam_rotation = np.array([r1, r2, r3, r4])
    return cam_center, cam_rotation

# RECTIFICATION
# Function to draw lines to show the rectification
def draw_lines(img1, img2, lines, pts1, pts2):

    r, c, _ = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1_line = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1_circle = cv2.circle(img1_line, tuple(pt1), 5, color, -1)
        img2_circle = cv2.circle(img2, tuple(pt2), 5, color, -1)

    return img1_circle, img2_circle

# Function to find the rectification 
def rectification(img1, img2,new_feature_1,new_feature_2,Best_F_matrix):
    h1, w1, ch1 = img1.shape
    h2, w2, ch2 = img2.shape
    
    img1_copy = img1
    img2_copy = img2
    kp1_warp, kp2_warp, good_warp ,x,q = detect_feature(img1_copy, img2_copy)
    feature_1_warp = []
    feature_2_warp = []

    for i, match in enumerate(good_warp):
        feature_1_warp.append(kp1_warp[match.queryIdx].pt)
        feature_2_warp.append(kp2_warp[match.trainIdx].pt)

    feature_1_warp = np.int32(feature_1_warp)
    feature_2_warp = np.int32(feature_2_warp)

    F_warp, mask = cv2.findFundamentalMat(feature_1_warp, feature_2_warp, cv2.FM_LMEDS)

    feature_1_warp = feature_1_warp[mask.ravel() == 1]
    feature_2_warp = feature_2_warp[mask.ravel() == 1]


    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(feature_2_warp.reshape(-1, 1, 2), 2, F_warp)
    lines1 = lines1.reshape(-1, 3)
    
    img1, img3 = draw_lines(img1_copy, img2_copy, lines1, feature_1_warp, feature_2_warp)

    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(feature_1_warp.reshape(-1, 1, 2), 1, F_warp)
    lines2 = lines2.reshape(-1, 3)
    img2, img4 = draw_lines(img2_copy, img1_copy, lines2, feature_2_warp, feature_1_warp)

    return img1, img2

# CORRESPONDENCE
# Function to find the SSD
def sum_squared_distance(img1_pixel, img2_pixel):
    if img1_pixel.shape != img2_pixel.shape:
        return -1

    return np.sum((img1_pixel - img2_pixel) ** 2)

# Function to comapre the blocks that are slided on the horizontal lines 
def compare_blocks(y, x, block_left, right_array):
    # Get search range for the right image
    x_min = max(0, x - 10)
    x_max = min(right_array.shape[1], x + 10)
    min_ssd = None
    min_index = None
    for x in range(x_min, x_max):
        block_right = right_array[y: y + 50,
                      x: x + 50]

        ssd = sum_squared_distance(block_left, block_right)

        if min_ssd:
            if ssd < min_ssd:
                min_ssd = ssd
                min_index = (y, x)
        else:
            min_ssd = ssd
            min_index = (y, x)

    return min_index


# Function to find the disprity by using SSD
def get_disparity_map(img1, img2,new_feature_1,new_feature_2,Best_F_matrix):
    h1, w1, ch1 = img1.shape
    h2, w2, ch2 = img2.shape

    left_array = np.asarray(img1)
    right_array = np.asarray(img2)
    left_array = left_array.astype(int)
    right_array = right_array.astype(int)
    if left_array.shape != right_array.shape:
        raise Exception("Left-Right image shape mismatch!")
    h, w, _ = left_array.shape
    disparity_map = np.zeros((h, w))
    for y in range(50, h - 50):
        for x in range(50, w - 50):
            block_left = left_array[y: y + 50,
                         x: x + 50]
            min_index = compare_blocks(y, x, block_left,
                                       right_array)
            disparity_map[y, x] = abs(min_index[1] - x)

    return disparity_map

# Depth
# Function to compute the depth
def get_depth_map(disparity_map_gray,B,f):
    h, w = disparity_map_gray.shape
    depth_map = np.zeros_like(disparity_map_gray)
    for y in range(h):
        for x in range(w):
            if disparity_map_gray[y, x] == 0:
                depth_map[y, x] = 0
            else:
                depth_map[y, x] = int(B *  f / (disparity_map_gray[y, x]))

    return depth_map


if __name__ == '__main__':
    # Input Choice from the 3 Data sets
    input_set = int(input("Choose the dataset (1 or 2 or 3): "))
    if input_set == 1:
        
        img1_path = "C:/Users/nisar/Desktop/Projects/curule-20220416T174031Z-001/curule/im0.png"
        img2_path = "C:/Users/nisar/Desktop/Projects/curule-20220416T174031Z-001/curule/im1.png"
        K1 = [1758.23, 0, 977.42 , 0 ,  1758.23,  552.15 , 0,  0,  1]
        K2 =[1758.23, 0, 977.42 , 0, 1758.23, 552.15 , 0, 0, 1]
        K1 = np.reshape(K1, (3, 3))
        K2 = np.reshape(K2, (3, 3))
        vmin=55
        vmax=195
        B = 88.39
    elif input_set == 2:
        img1_path = "C:/Users/nisar/Desktop/Projects/octagon-20220416T174034Z-001/octagon/im0.png"
        img2_path = "C:/Users/nisar/Desktop/Projects/octagon-20220416T174034Z-001/octagon/im1.png"
        K1 = [1742.11, 0, 804.90, 0, 1742.11, 541.22, 0, 0, 1]
        K2 = [1742.11, 0, 804.90, 0, 1742.11, 541.22, 0, 0, 1]
        K1 = np.reshape(K1, (3, 3))
        K2 = np.reshape(K2, (3, 3))
        vmin=29
        vmax=61
        B = 221.76

    elif input_set == 3:
        img1_path = "C:/Users/nisar/Desktop/Projects/pendulum-20220416T174040Z-001/pendulum/im0.png"
        img2_path = "C:/Users/nisar/Desktop/Projects/pendulum-20220416T174040Z-001/pendulum/im1.png"
        K1 = [1729.05, 0, -364.24, 0, 1729.05, 552.22, 0, 0, 1]
        K2 = [1729.05, 0, -364.24, 0, 1729.05, 552.22, 0, 0, 1]
        K1 = np.reshape(K1, (3, 3))
        K2 = np.reshape(K2, (3, 3))
        vmin=25
        vmax=150
        B = 537.75
        
        
# Image Read
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
# Extract the keyoints and features    
    kp1, kp2, good, feature_1,feature_2 = detect_feature(img1, img2)


    feature_1 = []
    feature_2 = []

    for i, match in enumerate(good):
        feature_1.append(kp1[match.queryIdx].pt)
        feature_2.append(kp2[match.trainIdx].pt)


# Finding the Fundamental Matrix
    Best_F_matrix = fundamental_matrix(feature_1, feature_2)

# Finding the essential matrix
    E_matrix = essential_matrix(Best_F_matrix, K1)
    R, T = extract_camera_pose(E_matrix)
    H = []
    I = np.array([0, 0, 0, 1])
    for i, j in zip(R, T):
        h = np.hstack((i, j))
        h = np.vstack((h, I))
        H.append(h)

# Rectification
    img1_res, img2_res = rectification(img1, img2, feature_1, feature_2,Best_F_matrix)

    res = np.concatenate((img1_res, img2_res), axis=1)
    res = resize(res,0.7)
    cv2.imshow("Rectification", res)
    cv2.imwrite("C:/Users/nisar/Desktop/Projects/Output/Rectification 1.jpg",res)

# Disparity map
    disparity_map = get_disparity_map(img1, img2, feature_1, feature_2,Best_F_matrix)

    disparity_map_gray = None
    disparity_map_gray = cv2.normalize(disparity_map, disparity_map_gray, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_8U)
    disparity_map_gray = resize(disparity_map_gray,0.7)
    cv2.imshow('disparity_gray', disparity_map_gray)
    cv2.imwrite("C:/Users/nisar/Desktop/Projects/Output/disparity gray 1.jpg",disparity_map_gray)

    disparity_map_heat = None
    disparity_map_heat = cv2.normalize(disparity_map, disparity_map_heat, alpha=vmin, beta=vmax,
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disparity_map_heat = cv2.applyColorMap(disparity_map_heat, cv2.COLORMAP_JET)
    disparity_map_heat = resize(disparity_map_heat,0.7)
    cv2.imshow("disparity_heat", disparity_map_heat)
    cv2.imwrite("C:/Users/nisar/Desktop/Projects/Output/disparity heat 1.jpg",disparity_map_heat)
    
# Depth map
    depth_map_gray = None
    depth_map = get_depth_map(disparity_map_gray,B,5299.313 )
    depth_map_gray = cv2.normalize(depth_map, depth_map_gray, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_8U)
    depth_map_gray = resize(depth_map_gray,0.7)
    cv2.imshow('depth_gray', depth_map)
    cv2.imwrite("C:/Users/nisar/Desktop/Projects/Output/deapth gray 1.jpg",depth_map_gray)

    depth_map_heat = cv2.applyColorMap(depth_map_gray, cv2.COLORMAP_JET)
    depth_map_heat = resize(depth_map_heat,0.7)
    cv2.imshow("depth_heat", depth_map_heat)
    cv2.imwrite("C:/Users/nisar/Desktop/Projects/Output/deapth heat 1.jpg",depth_map_heat)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
