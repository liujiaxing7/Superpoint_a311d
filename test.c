#include <opencv2/opencv.hpp>
#include "vnn_superpointv1.h"

int main() {
    // 加载SuperPoint模型
    SuperPoint superpoint;
    superpoint.loadModel("superpoint.pb");

    // 加载输入图像
    cv::Mat image = cv::imread("test.png");
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::resize(image, image, cv::Size(640, 480));

    // 运行SuperPoint模型
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    superpoint(image, keypoints, descriptors);

    // 输出结果
    std::cout << "Number of keypoints detected: " << keypoints.size() << std::endl;

    return 0;
}
