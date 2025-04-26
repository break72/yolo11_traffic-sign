#颜色量化
import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
#加载图片
img = cv2.imread(r"F:\project\tsdr\TrafficSignDetection\TestFiles\46257.jpg")

def color_quantize(img, k):

    #将图像数据转成float32，将图像平为2D阵列，其中每行表示一个像素，每列表示B、G和R颜色通道
    data = np.float32(img).reshape(-1, 3)

    #定义K-Means聚类标准
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    #应用K-Means聚类
    ret, labels, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[labels.flatten()]
    res = res.reshape(img.shape)
    return res

color_ = color_quantize(img,3)
#cv2.imwrite("../images_deal/1.jpg", color_)
#resize_color = cv2.resize(color_,(960,720))
text_img = cv2.putText(
    color_,  # 图像
    f"K = {3}",  # 文本
    (10, 30),  # 文本左下角坐标
    cv2.FONT_HERSHEY_SIMPLEX,  # 字体类型
    1,  # 字体缩放因子
    (0, 0, 0),  # 文本颜色 (B, G, R)
    2  # 文本粗细
)
image_rgb = cv2.cvtColor(text_img,cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis('off')  # 去除坐标轴
plt.show()
# cv2.imshow("color",color_)
# cv2.waitKey(0)
