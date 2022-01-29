import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy

class seam_carving:
    def __init__(self, path) :
        self.img_path = path
        self.img = cv2.imread(path)
        self.img_copy = np.copy(self.img)
        print(f"Image size is: {np.shape(self.img)[1]} x {np.shape(self.img)[0]}.")

    
    def get_energy(self, method='default'):
        energy_map = np.zeros_like(self.img)
        x_filter = np.array([[-1,0,1],
                    [-1,0,1],
                    [-1,0,1]])
        y_filter = np.transpose(x_filter)
        grad_x = cv2.filter2D(self.img,-1,x_filter)
        grad_y = cv2.filter2D(self.img,-1,y_filter)
        energy_map = np.sum(np.abs(grad_x),axis=2) + np.sum(np.abs(grad_y),axis=2)

        return energy_map

    def get_seam(self):
        energy_map = self.get_energy()
        M = energy_map.copy()
        h,w,_ = np.shape(self.img)
        path = np.zeros_like(M)

        pbar = tqdm(total = (h-1)*w)
        pbar.set_description("Calculating M")
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
            
        pbar.close()
        return M, path
    
    def get_path(self):
        M, path = self.get_seam()
        print(f"Image size is: {np.shape(self.img)[1]} x {np.shape(self.img)[0]}.")
        h,w,_ = np.shape(self.img)
        idx = np.argmin(M[-1,:])
        delete_mask = np.ones((h,w), dtype=np.uint8)

        for i in tqdm(range(h-1,-1,-1)):
            delete_mask[i,idx] = 0
            idx = path[i,idx]
        
        delete_mask = cv2.merge([delete_mask, delete_mask, delete_mask])

        # cv2.imshow("mask", delete_mask)
        self.img = self.img[delete_mask>=1].reshape(h,w-1,3)
        # cv2.imshow("removed_img",self.img)
        # cv2.waitKey(0)
        return self.img
        

sc = seam_carving("./data/img1.jpg")

for i in range(10):
    sc.get_path()

cv2.imshow("res", sc.img)
cv2.waitKey(0)
cv2.destroyAllWindows()