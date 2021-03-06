import numpy as np
import cv2
from tqdm import tqdm
import argparse
class seam_carving:
    def __init__(self, path, method, ratio) :
        self.img_path = path
        self.img = cv2.imread(path)
        self.img_copy = np.copy(self.img)
        self.method = method
        self.ratio = ratio
        print(f"Image size is: {np.shape(self.img)[1]} x {np.shape(self.img)[0]}.")

    
    def get_energy(self, method='default'):
        method=self.method
        energy_map = np.zeros_like(self.img)
        x_filter = np.array([[-1,0,1],
                    [-1,0,1],
                    [-1,0,1]])
        y_filter = np.transpose(x_filter)
        grad_x = cv2.filter2D(self.img,-1,x_filter)
        grad_y = cv2.filter2D(self.img,-1,y_filter)
        energy_map = np.sum(np.abs(grad_x),axis=2) + np.sum(np.abs(grad_y),axis=2)

        L_filter = np.array([[0,1,0],
                    [-1,0,0],
                    [0,0,0]])

        R_filter = np.array([[0,1,0],
                    [0,0,-1],
                    [0,0,0]])
        
        if method == "improved":
            L = cv2.filter2D(self.img,-1,L_filter)
            R = cv2.filter2D(self.img,-1,R_filter)

            C_L = np.sum(np.abs(grad_x),axis=2) + np.sum(np.abs(L),axis=2)
            C_R = np.sum(np.abs(grad_x),axis=2) + np.sum(np.abs(R),axis=2)
            C_U = np.sum(np.abs(grad_x),axis=2)/3
            return energy_map, C_L, C_U, C_R
        
        return energy_map

    def get_seam(self,method='default'):
        method = self.method
        h,w,_ = np.shape(self.img)


        pbar = tqdm(total = (h-1)*w)
        pbar.set_description("Calculating M")
        if method == 'default':
            energy_map = self.get_energy()
            M = energy_map.copy()
            path = np.zeros_like(M)
            for i in range(1,h):
                for j in range(w):
                    if j == 0:
                        idx = np.argmin(M[i-1,j:j+2])
                        M[i,j] = energy_map[i,j] + M[i-1,idx+j]
                        path[i,j] = idx+j
                    else:
                        idx = np.argmin(M[i-1,j-1:j+2])
                        M[i,j] = energy_map[i,j] + M[i-1,idx+j-1]
                        path[i,j] = idx+j-1
                    pbar.update(1)
        elif method == 'improved':
            energy_map, C_L, C_U, C_R = self.get_energy(method='improved')
            M = energy_map.copy()
            path = np.zeros_like(M)
            for i in range(1,h):
                for j in range(w):
                    if j == 0:
                        C = np.array([C_U[i,j], C_R[i,j+1]]) + M[i-1,j:j+2]
                        idx = np.argmin(C)
                        M[i,j] = energy_map[i,j] + M[i-1,idx+j]
                        path[i,j] = idx+j
                    elif j == w-1:
                        C = np.array([C_L[i,j-1], C_U[i,j]]) + M[i-1,j-1:j+1]
                        idx = np.argmin(C)
                        M[i,j] = energy_map[i,j] + M[i-1,idx+j-1]
                        path[i,j] = idx+j-1
                    else:
                        C = np.array([C_L[i,j-1], C_U[i,j], C_R[i,j+1]]) + M[i-1,j-1:j+2]
                        idx = np.argmin(C)
                        M[i,j] = energy_map[i,j] + M[i-1,idx+j-1]
                        path[i,j] = idx+j-1
                    pbar.update(1)
        pbar.close()
        return M, path
    
    def get_path(self):
        method = self.method
        M, path = self.get_seam(method=method)
        print(f"Image size is: {np.shape(self.img)[1]} x {np.shape(self.img)[0]}.")
        h,w,_ = np.shape(self.img)
        idx = np.argmin(M[-1,:])
        delete_mask = np.zeros((h,w), dtype=np.uint8)
        zeros = delete_mask.copy()
        for i in tqdm(range(h-1,-1,-1)):
            delete_mask[i,idx] = 1
            idx = path[i,idx]
        
        if self.flag == 0:
            cv2.imshow("Seam Carving",cv2.addWeighted(self.img,1,cv2.merge([zeros,zeros,delete_mask*255]),1,0))
            cv2.waitKey(1)
        else:
            cv2.imshow("Seam Carving",cv2.addWeighted(self.img,1,cv2.merge([zeros,zeros,delete_mask*255]),1,0).T)
            cv2.waitKey(1)
        delete_mask = cv2.merge([delete_mask, delete_mask, delete_mask])

        self.img = self.img[delete_mask<1].reshape(h,w-1,3)
        return self.img
        
    def opt(self):
        self.flag = 0
        h,w,_ = np.shape(self.img)

        ratio0 = w/h
        
        if h>=w:
            self.flag = 1
        
        if self.flag == 0: # h < w
            w_new = int(self.ratio*h)
            print(f"h<w object shape {h}x{w_new}")
            print(w, w_new)
            for i in range(w-w_new):
                sc.get_path()


        if self.flag == 1: # w < h
            h_new = int(w/self.ratio)
            self.img = self.img.T
            print(f"w<h object shape {h_new}x{w}")
            for i in range(h-h_new):
                sc.get_path()
            self.img = self.img.T


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Implementation of Seam Carving and its improvements')
    parser.add_argument("-m",'--method',choices=['default','improved'])
    parser.add_argument("-d",'--dir',default='./data/img4.jpg')
    parser.add_argument("-r",'--ratio',help='ratio=width/height')
    args = parser.parse_args()

    method = args.method
    ratio = float(args.ratio)
    path = args.dir

    print(f"Image Path = {path}, method = {method}, ratio = {ratio}")
    sc = seam_carving(path,method,ratio)

    sc.opt()

    cv2.imshow("Result", sc.img)
    cv2.imwrite("result.jpg",sc.img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()