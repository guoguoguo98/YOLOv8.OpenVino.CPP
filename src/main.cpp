#include <iostream>

#include "cli.hpp"
#include "yolov8.hpp"

#include <chrono>

std::map<int, cv::Scalar> colorMap;

cv::Scalar getRandomColor() {
    int blue = rand() % 256;
    int green = rand() % 256;
    int red = rand() % 256;
    return cv::Scalar(blue, green, red);
}

void drawDetections(cv::Mat &image, const std::vector<Detection> &detections, std::map<int, cv::Scalar> &colorMap) {
    for (const Detection &detection : detections) {
        auto colorMapIt = colorMap.find(detection.class_id);
        if (colorMapIt == colorMap.end()) {
            colorMapIt = colorMap.insert({detection.class_id, getRandomColor()}).first;
        }

        cv::Scalar color = colorMapIt->second;
        cv::rectangle(image, detection.box, color, 2);

        std::string label =
            "Class: " + std::to_string(detection.class_id) + " Score: " + std::to_string(detection.confidence);
        cv::Point textPosition(detection.box.x, detection.box.y - 10);
        cv::putText(image, label, textPosition, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}

int predictImage(YoloV8 model, Args args) {
    cv::Mat img = cv::imread(args.source);
    std::vector<Detection> detections = model.predict(img);

    drawDetections(img, detections, colorMap);

    cv::imshow(args.source, img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

int predictVideo(YoloV8 model, Args args) {
    cv::VideoCapture cap(args.source);
    float latency;

    while (cap.isOpened()) {
        cv::Mat frame;
        cap >> frame;

        if (!frame.empty()) {
            auto begin = std::chrono::steady_clock::now();
            std::vector<Detection> detections = model.predict(frame);

            if (detections.empty()) {
                std::cout << "No detections" << std::endl;
            }

            drawDetections(frame, detections, colorMap);
            auto end = std::chrono::steady_clock::now();

            cv::imshow(args.source, frame);

            latency = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
            std::cout << "Latency = " << latency << "ms\t";
            std::cout << "FPS = " << 1000.0 / latency << std::endl;
        }

        if (cv::waitKey(30) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

int main(int argc, char **argv) {
    Args args = parseArgs(argc, argv);

    YoloV8 model(args.modelPath, args.imgSize, args.gpu, args.scoreThresh, args.iouThresh);

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    if (args.type == IMAGE) {
        predictImage(model, args);
        return 0;
    }

    if (args.type == VIDEO) {
        predictVideo(model, args);
        return 0;
    }

    return 0;
}
