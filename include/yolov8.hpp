#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

class YoloV8 {
private:
    float score_threshold_;
    float iou_threshold_;
    std::vector<int> img_size_;

    ov::Shape model_input_shape_;
    ov::element::Type model_input_type_;
    ov::Shape ppp_input_shape_;
    ov::element::Type ppp_input_type_;
    ov::Shape model_output_shape_;
    ov::element::Type model_output_type_;

    std::shared_ptr<ov::CompiledModel> compiled_model_;
    std::shared_ptr<ov::InferRequest> infer_request_;

    void letterbox(cv::Mat &source, cv::Mat &dst, std::vector<float> &ratios);
    std::vector<Detection> nms(const ov::Tensor &output_tensor,
                               const ov::Shape &output_shape,
                               std::vector<float> &ratios);

    void loadAndCompileModel(const std::string &model_path, bool use_gpu);

public:
    YoloV8(std::string model_path, std::vector<int> imgsz, bool use_cuda, float score_thresh, float iou_thresh);
    ~YoloV8();  // Destructor to release resources

    std::vector<Detection> predict(cv::Mat &img);
};