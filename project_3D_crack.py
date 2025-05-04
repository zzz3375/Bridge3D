import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

crack_segment = "DJI_20241123142426_0013_V_obj1"

image_name = crack_segment[:-5] # image_name = "DJI_20241123142249_0002_V"
obj = crack_segment[26:] # obj = "obj1"

bbox  = np.load(r"yolo_cropped_images_undistorted_v8x-sea\{}_{}.jpg.npy".format(image_name, obj))
print(bbox.shape)  # (4,)
x1, y1, x2, y2 = map(int, bbox)

depth_map = np.load(r"full_depth_maps\{}.npy".format(image_name))
full_image = cv2.imread(r"dense\images\{}.JPG".format(image_name))

print(depth_map.shape)  
print(full_image.shape)  

mask = cv2.imdecode(np.fromfile(r"C:\Users\13694\Bridge3DReconstruction\yolo_cropped_images_undistorted_v8x-sea\{}_{}.png".format(image_name, obj), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

full_image[y1:y2, x1:x2] = mask[:,:,None].repeat(3,-1)

plt.subplot(121)
plt.imshow(depth_map, cmap='plasma')
plt.title('Depth Map')
plt.subplot(122)
plt.imshow(full_image)
plt.title('Colored Image')
plt.show()

cv2.imwrite(r"full_image.jpg", full_image)