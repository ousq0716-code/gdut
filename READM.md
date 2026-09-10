基于C++ 、 OpenCV和YOLOv8实现的多目标跟踪系统。

项目简介：
本项目基于 YOLOv8 实现目标检测，并结合 IoU、目标中心距离和 Kalman Filter 实现多目标跟踪。
系统可以对摄像头中的多个目标进行检测、匹配和持续跟踪，并为每个目标分配独立 ID。

项目主要功能：

* YOLOv8 目标检测
* Person 目标筛选
* 多目标 ID 分配
* IoU 匹配
* 中心距离匹配
* IoU + 中心距离联合匹配
* Kalman Filter 目标预测
* 目标丢失处理
* 目标运动方向判断
* 目标像素速度计算
* 目标运动轨迹显示
* Person 数量统计
* Track 数量统计
* CSV 数据记录
* CUDA GPU 加速

代码流程：
Camera
↓
YOLO Detection
↓
Person Filtering
↓
Kalman Prediction
↓
IoU + Center Distance Matching
↓
ID Management
↓
Motion Analysis
↓
Trajectory Visualization
↓
CSV Data Logging

使用的技术

* C++
* OpenCV
* YOLOv8
* ONNX Runtime
* CUDA
* Kalman Filter

运行结果图片：
运行结果的示意图保存在result
第一张图展示了人物识别和追踪效果 都用绿色的框进行标记
第二张图是当其中一个人物消失 黄色的框继续预测人物可能运行轨迹
具体效果可运行代码查看

输出数据

系统会生成 tracking\_data.csv，记录：

* Frame
* ID
* X
* Y
* Speed
* Direction



代码运行环境

* Windows
* Visual Studio
* OpenCV 4.14
* ONNX Runtime
* CUDA

所需环境自行配置



后续打算

* 更复杂场景下的目标匹配优化
* 更稳定的目标轨迹预测
* 更高效的推理优化
* 进一步扩展到三维目标跟踪



刚上研一 如有错误请多见谅

