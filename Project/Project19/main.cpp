// 
/*
整体流程：
            程序开始
            ↓
            打开摄像头
            ↓
            初始化 YOLO
            ↓
            创建 tracks 容器
            ↓
            初始化 next_id、frame_count
            ↓
            进入 while 循环
            ↓
            读取一帧图像
            ↓
            YOLO 检测
            ↓
            筛选 person
            ↓
            获取当前帧所有 person 检测框
            ↓
            【已有 Track】
            ↓
            Kalman predict()
            ↓
            预测已有 Track 的位置
            ↓
            更新 Track 预测框和中心点
            ↓
            计算每个 Track 与每个 person 的：
            ├─ IoU
            └─ 中心距离
            ↓
            计算综合匹配 Score
            ↓
            选择 Score 最高的 person
            ↓
            ┌──────────────────────┬─────────────────────┐
            │ 匹配成功             │ 匹配失败            │
            │                      │                     │
            │ 更新检测框           │ lose++              │
            │ 更新中心点           │ 保留预测位置        │
            │ Kalman correct()     │                     │
            │ 更新速度             │                     │
            │ 保存轨迹点           │                     │
            │ lose = 0             │                     │
            └──────────┬───────────┴──────────┬──────────┘
                    ↓                      ↓
                    └──────────┬───────────┘
                                ↓
                    删除 lose 超过阈值的 Track
                                ↓
                    检查是否存在未匹配 person
                                ↓
                        有未匹配 person？
                        ↓             ↓
                        是             否
                        ↓             ↓
                    创建新的 Track      继续
                    分配新的 ID
                    初始化 Kalman
                        ↓
                        └───────┬───────┘
                                ↓
                        计算每个 Track：
                        ├─ 速度
                        └─ 运动方向
                                ↓
                        绘制跟踪结果
                        ├─ 检测框
                        ├─ ID
                        ├─ Direction
                        ├─ Speed
                        └─ 运动轨迹
                                ↓
                        统计 Persons / Tracks
                                ↓
                        保存跟踪数据到 CSV
                                ↓
                            显示画面
                                ↓
                        waitKey 检查是否退出
                            ↓           ↓
                            是           否
                            ↓           ↓
                        程序结束     回到下一帧
*/


#include "iostream"
#include "string"
#include "cmath"
#include "opencv2/opencv.hpp"
#include "vector"
#include "yolo_cli.hpp"
#include "yolo_render.hpp"
#include "yolo_show.hpp"
#include "inference.h"
#include "algorithm"
#include "fstream"

// 定义结构体存储每个追踪物体的信息
struct Track
{
    cv::Rect box; // 物体框信息
    cv::Point2f center; // 物体框中心
    cv::Point2f velocity; // 速度
    cv::KalmanFilter KF; // 滤波预测
    int id; // 给人分配的ID
    int lose = 0; // 物体在检测中未出现的帧数
    std::vector<cv::Point2f> trajectory;
};

// 定义函数计算这一帧与下一帧框的交并比
float CalculateIoU(cv::Rect box1, cv::Rect box2)
{
    cv::Rect intersection = box1 & box2;

    if (intersection.area() <= 0)
    {
        return 0.0f;
    }

    float interArea = intersection.area();
    float unionArea = box1.area() + box2.area() - interArea;

    return interArea / unionArea;
}


int main(int argc, char** argv)
{
    cv::VideoCapture cap(0); // 打开摄像头

    // 创建文件存储检测信息
    //std::ofstream file("tracking_data.csv"); 
    //if (!file.is_open())
    //{
    //    std::cout << "CSV文件打开失败" << std::endl;
    //    return -1;
    //}

    //file << "frame, ID, X, Y, Speed, Direction\n";


    if (!cap.isOpened())
    {
        std::cout << "摄像头未启动" << std::endl;
        return -1;
    }

    yolo::Config config; // 创建yolo模型的变量信息

   
    config.model_path = yolo::ArgValue(
        argc,
        argv,
        "--model",
        "E:/pythontrain/yolov8n.onnx" // 模型路径改成自己的模型路径
    );

    // 定义模型参数
    config.conf = std::stof(
        yolo::ArgValue(argc, argv, "--conf", "0.3")
    );

    config.iou = std::stof(
        yolo::ArgValue(argc, argv, "--iou", "0.45")
    );

    config.cuda = true; // 默认开启显卡加速检测 

    auto predictor = yolo::Predictor(config);
    const std::vector<std::string>& names = predictor.names();

    std::vector<Track> tracks; // 定义容量体

    int next_id = 0; // 用于分配ID
    int frame_count = 0; // 用于计数存储数据

    while (true)
    {

        cv::Mat frame; // 这帧图像
        
        if (!cap.read(frame))
        {
            break;
        }

        frame_count++;

        cv::Mat semantic; // 未使用 占位
        auto results = predictor.predict(frame, semantic);

        std::vector<cv::Rect> persons; // 存储检测目标box信息

        for (auto& result : results)
        {
            std::string label = names[result.class_id];

            if (label == "person") // 检测为人时存储信息
            {
                persons.push_back(result.box);
            }
        }

        // 让 Kalman 滤波器预测这个 Track 下一帧应该出现在哪里，然后把 Track 的框和中心点移动到预测位置。
        for (auto& track : tracks)
        {
            cv::Mat prediction = track.KF.predict();
            float prediction_x = prediction.at<float>(0);
            float prediction_y = prediction.at<float>(1);

            float dx = prediction_x - track.center.x;
            float dy = prediction_y - track.center.y;

            track.box.x += cvRound(dx);
            track.box.y += cvRound(dy);

            track.center.x = prediction_x;
            track.center.y = prediction_y;

        }

        // 记录 persons 中的每一个人有没有已经被某个 Track 匹配过
        std::vector<bool> matched(persons.size(), false);

        // 追踪：遍历通过得分找出与上一帧最匹配的ID 
        for (auto& track : tracks)
        {
            float bestscore = 0.0f;
            int bestIndex = -1;
            for (int i = 0; i < persons.size(); i++)
            {
                if (matched[i])
                {
                    continue;
                }

                float iou = CalculateIoU(track.box, persons[i]); // 计算交变比

                // 计算 Track 预测位置和 YOLO 检测位置的距离，并把距离转换成可以和 IoU 一起比较的匹配分数
                float personCenterX = persons[i].x + persons[i].width / 2.0f;
                float personCenterY = persons[i].y + persons[i].height / 2.0f;
                float dx = track.center.x - personCenterX;
                float dy = track.center.y - personCenterY;
                float distance = std::sqrt(dx * dx + dy * dy);
                float maxDistance = frame.cols * 0.3f;
                float distanceScore = 1.0f - distance / maxDistance;

                if (distanceScore < 0)
                {
                    distanceScore = 0;
                }

                float score = 0.6f * iou + 0.4f * distanceScore; // 6：4比分

                if (score > bestscore)
                {
                    bestscore = score;
                    bestIndex = i;
                }
            }

            // 如果得分小于这个指标 认为与上一帧的所有检测的人都不匹配 认为是一个新的目标
            if (bestIndex != -1 && bestscore > 0.5f) // 0.5f可改 
            {
                cv::Point2f previousCenter = track.center;
                auto temp_x = track.center.x;
                auto temp_y = track.center.y;

                track.box = persons[bestIndex];

                track.center.x =track.box.x + track.box.width / 2.0f;

                track.center.y =track.box.y + track.box.height / 2.0f;
                track.trajectory.push_back(track.center);
                if (track.trajectory.size() > 50)
                {
                    track.trajectory.erase(track.trajectory.begin());
                }
                cv::Mat measurement(2, 1, CV_32F);
                
                measurement.at<float>(0) = track.center.x;
                measurement.at<float>(1) = track.center.y;

                track.KF.correct(measurement);
                float dx = track.center.x - previousCenter.x;
                float dy = track.center.y - previousCenter.y;

                track.velocity.x =0.7f * track.velocity.x + 0.3f * dx;

                track.velocity.y = 0.7f * track.velocity.y + 0.3f * dy;

                track.lose = 0;
                matched[bestIndex] = true;
            }
            else
            {
                track.lose++; // 检测原来出现过后面未匹配上的ID的帧数
            }
        }

        // 删去一定阈值未出现的目标的Track的信息
        tracks.erase(
            std::remove_if(
                tracks.begin(),
                tracks.end(),
                [](const Track& track)
                {
                    return track.lose > 150; // 可改 目前是5s
                }
            ),
            tracks.end()
        );

        // 创建新的track追踪目标
        for (int i = 0; i < persons.size(); i++)
        {
            if (!matched[i])
            {
                Track newTrack;

                newTrack.box = persons[i];
                newTrack.id = next_id++;
                newTrack.lose = 0;
                newTrack.KF = cv::KalmanFilter(4, 2, 0);

                newTrack.center.x =
                    newTrack.box.x + newTrack.box.width / 2.0f;

                newTrack.center.y =
                    newTrack.box.y + newTrack.box.height / 2.0f;

                newTrack.velocity.x = 0;
                newTrack.velocity.y = 0;

                newTrack.KF = cv::KalmanFilter(4, 2, 0);

                newTrack.KF.transitionMatrix = cv::Mat::eye(4, 4, CV_32F);

                newTrack.KF.transitionMatrix.at<float>(0, 2) = 1.0f;
                newTrack.KF.transitionMatrix.at<float>(1, 3) = 1.0f;

                newTrack.KF.measurementMatrix = cv::Mat::zeros(2, 4, CV_32F);

                newTrack.KF.measurementMatrix.at<float>(0, 0) = 1.0f;
                newTrack.KF.measurementMatrix.at<float>(1, 1) = 1.0f;

                newTrack.KF.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 0.03f;

                newTrack.KF.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * 0.5f;

                newTrack.KF.errorCovPost = cv::Mat::eye(4, 4, CV_32F);

                newTrack.KF.statePost.at<float>(0) = newTrack.center.x;

                newTrack.KF.statePost.at<float>(1) = newTrack.center.y;

                newTrack.KF.statePost.at<float>(2) = 0;
                newTrack.KF.statePost.at<float>(3) = 0;

                tracks.push_back(newTrack);
            }
        }

        // 创建新的图用于标记输出
        cv::Mat canvas = frame.clone();
        auto person_number = persons.size();
        auto Tracks_number = tracks.size();
        
        for (auto& track : tracks)
        {
            if (track.lose > 0)
            {
                cv::rectangle(
                    canvas,
                    track.box,
                    cv::Scalar(0, 255, 255),
                    2
                );

                continue;
            }

            cv::rectangle(
                canvas,
                track.box,
                cv::Scalar(0, 255, 0),
                2
            );

            std::string direction;
            float speed;

            speed = std::sqrt(track.velocity.x * track.velocity.x + track.velocity.y * track.velocity.y);

            // 判断人物移动轨迹是什么
            if (std::abs(track.velocity.x) < 2 && std::abs(track.velocity.y) < 2)
            {
                direction = "Stop";
            }
            else if (std::abs(track.velocity.x) > std::abs(track.velocity.y))
            {
                if (track.velocity.x > 0)
                {
                    direction = "Right";
                }
                else
                {
                    direction = "Left";
                }
            }
            else
            {
                if (track.velocity.y > 0)
                {
                    direction = "Down";
                }
                else
                {
                    direction = "Up";
                }
            }
            //file << frame_count << ","
            //    << track.id << ","
            //    << track.center.x << ","
            //    << track.center.y << ","
            //    << speed << ","
            //    << direction << "\n";

            cv::putText(
                canvas,
                "ID: " + std::to_string(track.id),
                cv::Point(track.box.x, track.box.y - 55),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 255, 0),
                2
            );


            cv::putText(
                canvas,
                "Direction: " + direction,
                cv::Point(track.box.x, track.box.y - 35),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(0, 255, 0),
                2
            );

            cv::putText(
                canvas,
                "Speed: " + std::to_string(speed),
                cv::Point(track.box.x, track.box.y - 15),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(0, 255, 0),
                2
            );

            //for (int i = 1; i < track.trajectory.size(); i++)
            //{
            //    cv::line(
            //        canvas,
            //        track.trajectory[i - 1],
            //        track.trajectory[i],
            //        cv::Scalar(255, 0, 0),
            //        2
            //    );
            //}
        }

        cv::putText(
            canvas,
            "Persons: " + std::to_string(person_number),
            cv::Point(20, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 255, 0),
            2
        );

        cv::putText(
            canvas,
            "Tracks: " + std::to_string(Tracks_number),
            cv::Point(20, 60),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 255, 0),
            2
        );
        cv::imshow("YOLO Track", canvas);

        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    return 0;
}