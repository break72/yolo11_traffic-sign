#coding:utf-8
import cv2
from ultralytics import YOLO

# 所需加载的模型目录
path = 'models/best.pt'
# 需要检测的图片地址
video_path = "TestFiles/1.mp4"
# 保存处理后的视频文件名
output_video_path = "output_video.mp4"

# Load the YOLO11 model
model = YOLO(path)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 获取视频的宽、高、帧率等信息
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 定义视频编解码器和输出视频文件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4格式
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Loop through the video frames
while cap.isOpened():
    # 读取视频帧
    success, frame = cap.read()

    if success:
        # 运行YOLO11检测
        results = model(frame)

        # 将检测结果绘制到帧上
        annotated_frame = results[0].plot()

        # 将处理后的帧写入输出视频文件
        out.write(annotated_frame)

        # 显示处理后的帧
        cv2.imshow("YOLO11 Inference", annotated_frame)

        # 按下'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 视频结束时退出循环
        break

# 释放资源
cap.release()
out.release()  # 释放VideoWriter对象
cv2.destroyAllWindows()
