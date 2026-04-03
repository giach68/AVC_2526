import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

all_images = []
for fname in sorted(glob.glob(f"{"./NewSynthRTI/christ/images/single/material_1/TRAIN/"}/*.png")):
    img = cv2.imread(fname,cv2.IMREAD_COLOR)
    r = img[:,:,0].flatten()
    g = img[:,:,1].flatten()   
    b = img[:,:,2].flatten()
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    all_images.append(gray.astype(np.float32) / 255.0)
    
img = cv2.imread(fname,cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)

L=[]
    
with open('dome49.lp', 'r') as file:
    nimg = int(file.readline().strip() )  
    for _ in range(nimg):
        line = file.readline().strip()
        parts = line.strip().split()   #
        L.append(parts[1:])
       
#L = np.loadtxt("dome49.lp")

L = np.array(L).astype(np.float32)
L = L / np.linalg.norm(L, ord=2, axis=1, keepdims=True)

x=L[:,0]
y=L[:,1]


points = L[:, :2]    
images = np.stack(all_images)
# Create scatter plot
#plt.scatter(x, y, color='blue', marker='o')
tri = Delaunay(points)
# Plotting
plt.figure(figsize=(6, 6))
plt.triplot(points[:, 0], points[:, 1], tri.simplices, color='gray')
plt.plot(points[:, 0], points[:, 1], 'o', color='red')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Delaunay Triangulation of 2D Point Cloud')
plt.grid(True)
plt.axis('equal')
plt.show()

print(tri.simplices)


norms=np.linalg.lstsq(L, images, rcond=None)[0].T
print(norms.shape)
magnitudes = np.linalg.norm(norms, axis=1, keepdims=True)
norms /= magnitudes
norms = norms.reshape(img.shape)

plt.imshow(norms)
plt.show()



