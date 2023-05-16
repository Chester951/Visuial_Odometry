import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp
import time

import cv2
import numpy as np
from orb import orb_features, update_image
from voc_tree import constructTree
from matcher import *

def draw_match(img1,img2,points1,points2):
    num = points1.shape[0]

    kps1 = []
    kps2 = []
    matches = []

    for point in points1:
        kps1.append(cv.KeyPoint(point[0],point[1],1))
    for point in points1:
        kps2.append(cv.KeyPoint(point[0],point[1],1))
    for i in range(num):
        matches.append(cv.DMatch(i,i,1))
    
    img_draw_match = cv.drawMatches(img1, kps1, img2, kps2, matches, None, flags=cv.DrawMatchesFlags_DEFAULT)
    cv.namedWindow('match', cv.WINDOW_NORMAL)
    #cv.resizeWindow('match', 1000, 400)
    cv.imshow('match', img_draw_match)
    cv.imwrite("match.jpg", img_draw_match)
    cv.waitKey(0)
    cv.destroyAllWindows()    




class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()] # read the camera parameters .npy file
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        
        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))

    def create_camera(self, R, t):
        camera = o3d.geometry.LineSet()
        camera.points = o3d.utility.Vector3dVector(
            [[0, 0, 0], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]])
        camera.lines = o3d.utility.Vector2iVector(
            [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]])
        camera.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]),(8, 1)))
        camera.rotate(R)
        camera.translate(t)
        return camera
    
    def get_correspond_points(self, method, img1, img2):
        
        if method=="orb":
            orb = cv.ORB_create()
            kp1, des1 = orb.detectAndCompute(img1,None)
            kp2, des2 = orb.detectAndCompute(img2,None)     

            bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)
            matches = bf.match(des1,des2)
            matches = sorted(matches, key = lambda x:x.distance)   

            points1 = np.array([kp1[m.queryIdx].pt for m in matches])
            points2 = np.array([kp2[m.trainIdx].pt for m in matches])

        if method=='sift':
            sift = cv.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1,None)
            kp2, des2 = sift.detectAndCompute(img2,None)     
            
            bf = cv.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2) 
            good_matches = []
            for m, n in matches:
                if m.distance<0.75*n.distance:
                    good_matches.append(m)

            good_matches = sorted(good_matches, key = lambda x:x.distance)   

            points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
            points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])     

        return kp1, kp2, points1, points2

    def sort_inliers(self, points1,points2,inlier):
        new_points1=[]
        new_points2=[]
        for i in range(len(inlier)):
            if inlier[i][0]!=0:
                new_points1.append(points1[i])
                new_points2.append(points2[i])
        return np.array(new_points1),np.array(new_points2)
    
    def sort_inlier3d(self, points1, points2, points3d, inlier):
        new_points1=[]
        new_points2=[]
        new_points3d = []
        for i in range(len(inlier)):
            if inlier[i][0]!=0:
                new_points1.append(points1[i])
                new_points2.append(points2[i])
                new_points3d.append(points3d[:,i])
        for i in range(len(new_points3d)):
            point = new_points3d[i]
            point = point/point[3]
            point = point[:3]
            new_points3d[i] = point

        return np.array(new_points1),np.array(new_points2),np.array(new_points3d)

    def get_scale_factor(self, points_prev, points3d_prev, points_curr, points3d_curr):
        same_points = []
        scale_factor = []

        for i in range(points_prev.shape[0]):
            for j in range(points_curr.shape[0]):
                if np.all(points_prev[i]==points_curr[j]):
                    same_points.append([i, j])
        #print(len(same_points))

        if len(same_points)<=1:
            return 1
        
        # find pairs
        sample_points = 13
        for t in range(sample_points):
            idx1, idx2 = np.random.choice(len(same_points), 2, replace=False)
            idx1_prev, idx1_curr = same_points[idx1][0], same_points[idx1][1]
            idx2_prev, idx2_curr = same_points[idx2][0], same_points[idx2][1]

            norm_curr = np.linalg.norm(points3d_curr[idx1_curr]-points3d_curr[idx2_curr])
            norm_prev = np.linalg.norm(points3d_prev[idx1_prev]-points3d_prev[idx2_prev])
            scale_factor.append(norm_curr/(norm_prev+0.000001))
        
        return np.median(scale_factor)



    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width = 1000, height=1000)  
        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()
        
        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    #print(R,t)
                    camera = self.create_camera(R, t)
                    #print("-------")
                    vis.add_geometry(camera)

                    pass
            except: pass
            
            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()


    def process_frames(self, queue):
        start = time.time()

        # initialize
        curr_pose = np.eye(4, dtype=np.float64)
        # frame 0 path
        frame_path_prev = self.frame_paths[0]

        # 先建立訓練字典圖片的數量
        N = 20 #初始訓練字典的數量
        K = 5  #K類
        L = 4  #字典數L層
        T = 0.77 #相似度閥值

        image_descriptors = orb_features(N) #提取特徵
        FEATS = []

        for feats in image_descriptors:
            FEATS += [np.array(fv, dtype='float32') for fv in feats]

        FEATS = np.vstack(FEATS) #將特徵轉成np陣列
        treeArray = constructTree(K, L, np.vstack(FEATS)) #建立字典樹
        tree = Tree(K, L, treeArray)
        matcher = Matcher(N, image_descriptors, tree)


        for idx, frame_path in enumerate(self.frame_paths[1:],1):
            print("Processing  Frmae {}".format(idx))
            img0 = cv.imread(frame_path_prev)
            img1 = cv.imread(frame_path)

            #TODO: compute camera pose here
            # -----------------Step2 Extract and match feature between Img_k+1 and Img_k -----------------
            kp0, kp1, points0, points1 = self.get_correspond_points("orb", img0,img1)            
            # draw_match(img0, img1, points0, points1)

            # -----------------Step3 Estimate the essentail matrix -----------------
            E, inlier = cv.findEssentialMat(points0,points1,self.K,cv.RANSAC,0.999,1.0)
            points0, points1 = self.sort_inliers(points0,points1,inlier)
            # draw_match(img0, img1, points0, points1)

            # -----------------Step4 Decompose the E_k,k+1 into  R_k+1 and t_k+1 -----------------
            rerval, R, t, inlier, triangularpoints = cv.recoverPose(E, points0, points1, self.K, distanceThresh=1000)
            points0, points1, points3d = self.sort_inlier3d(points0, points1, triangularpoints, inlier)
            # draw_match(img0, img1, points0, points1)
            
            # -----------------Step5 Compute Scale factor  -----------------
            if idx==1:
                scale_factor = 1
            else:
                scale_factor = self.get_scale_factor(points1_prev, points3d_prev, points0, points3d)
                #print(scale_factor)
                if scale_factor > 2:
                    scale_factor = 2
                t = scale_factor * t
            # -----------------Step6 Calculate the pose of camera k+1 relative to the first camera -----------------
            frame_pose = np.concatenate([R,t],-1)
            frame_pose = np.concatenate([frame_pose,np.zeros((1,4))],0)
            frame_pose[-1,-1] = 1.0

            curr_pose = curr_pose @ frame_pose
            R = curr_pose[:3,:3]
            t = curr_pose[:3,3]

            # -----------------Loop Detection -----------------
            des = update_image(idx)
            tree.update_tree(idx, des)
            # print("compute {}.jpg cosine similarity:".format(idx))
            res = {}
            for j in range(tree.N-1):
                # print('Image {} vs Image {}: {}'.format(i, j, matcher.cos_sim(tree.transform(i), tree.transform(j))))
                if matcher.cos_sim(tree.transform(idx), tree.transform(j)) >= T:
                    res[j] = matcher.cos_sim(tree.transform(idx), tree.transform(j))
            if res:
                r = max(res.items(), key=lambda x:x[1])[0]
                # 400 幀內不會有loop
                if (idx-r>400):
                    print("loop detection {}, {}.jpg".format(idx, r)) 

            
            #print(R,t)
            scale_factor_prev  = scale_factor
            points1_prev = points1
            points3d_prev = points3d
            frame_path_prev = frame_path

            #Send R,t to visualization process
            queue.put((R, t))
            kp_img = cv.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
            cv.imshow('frame', kp_img)
            if cv.waitKey(30) == 27: break
        
        # timer
        end = time.time()
        print("VO time: {}s".format(end-start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
