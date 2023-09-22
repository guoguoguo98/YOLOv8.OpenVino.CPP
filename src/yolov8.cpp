#include "yolov8.hpp"
#include "utils.hpp"

YoloV8::YoloV8(std::string model_path, std::vector<int> imgsz, bool use_gpu, float score, float iou)
    : score_threshold_(score), iou_threshold_(iou), img_size_(std::move(imgsz)) {
    loadAndCompileModel(model_path, use_gpu);
}

YoloV8::~YoloV8() {
    compiled_model_.reset();
    infer_request_.reset();
}

void YoloV8::loadAndCompileModel(const std::string &model_path, bool use_gpu) {
    static ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    core.set_property(ov::cache_dir(".cache"));

    // Get model info
    model_input_shape_ = model->input().get_shape();
    model_input_type_ = model->input().get_element_type();
    model_output_shape_ = model->output().get_shape();
    model_output_type_ = model->output().get_element_type();

    // Preprocessing for the model
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    ppp.input()
        .tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    ppp.input()
        .preprocess()
        .convert_element_type(model_input_type_)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale({255., 255., 255.});
    ppp.input().model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(model_output_type_);

    // Embed above steps in the graph
    model = ppp.build();

    try {
        compiled_model_ = std::make_shared<ov::CompiledModel>(use_gpu ? core.compile_model(model, "GPU")
                                                                      : core.compile_model(model, "CPU"));
    } catch (const std::runtime_error &err) {
        std::cerr << LogWarning("Failed to use GPU. Using CPU instead...", err.what()) << std::endl;
        compiled_model_ = std::make_shared<ov::CompiledModel>(core.compile_model(model, "CPU"));
    }

    // Get ppp info
    ppp_input_shape_ = compiled_model_->input().get_shape();
    ppp_input_type_ = compiled_model_->input().get_element_type();

    infer_request_ = std::make_shared<ov::InferRequest>(compiled_model_->create_infer_request());
}

void YoloV8::letterbox(cv::Mat &source, cv::Mat &dst, std::vector<float> &ratios) {
    // Padding image to [n x n] dimension
    int max_size = std::max(source.cols, source.rows);
    int x_pad = max_size - source.cols;
    int y_pad = max_size - source.rows;
    float x_ratio = static_cast<float>(max_size) / static_cast<float>(model_input_shape_[3]);
    float y_ratio = static_cast<float>(max_size) / static_cast<float>(model_input_shape_[2]);

    cv::copyMakeBorder(source, dst, 0, y_pad, 0, x_pad, cv::BORDER_CONSTANT);  // Padding black
    cv::resize(dst, dst, cv::Size(model_input_shape_[3], model_input_shape_[2]), 0, 0, cv::INTER_NEAREST);

    ratios.push_back(x_ratio);
    ratios.push_back(y_ratio);
}

std::vector<Detection> YoloV8::nms(const ov::Tensor &output_tensor,
                                   const ov::Shape &output_shape,
                                   std::vector<float> &ratios) {
    float *detections = output_tensor.data<float>();

    std::vector<Detection> output;
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;

    int out_rows = output_shape[1];
    int out_cols = output_shape[2];
    const cv::Mat det_output(out_rows, out_cols, CV_32F, detections);

    for (int i = 0; i < det_output.cols; ++i) {
        const cv::Mat classes_scores = det_output.col(i).rowRange(4, 84);
        double score;
        cv::Point class_id_point;
        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > this->score_threshold_) {
            const float cx = det_output.at<float>(0, i) * ratios[0];
            const float cy = det_output.at<float>(1, i) * ratios[1];
            const float ow = det_output.at<float>(2, i) * ratios[0];
            const float oh = det_output.at<float>(3, i) * ratios[1];
            cv::Rect box(static_cast<int>((cx - 0.5 * ow)), static_cast<int>((cy - 0.5 * oh)), static_cast<int>(ow),
                         static_cast<int>(oh));

            boxes.push_back(box);
            class_ids.push_back(class_id_point.y);
            confidences.push_back(score);
        }
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, this->score_threshold_, this->iou_threshold_, nms_result);

    for (int i : nms_result) {
        Detection result;
        result.class_id = class_ids[i];
        result.confidence = confidences[i];
        result.box = boxes[i];
        output.push_back(result);
    }

    return output;
}

std::vector<Detection> YoloV8::predict(cv::Mat &img) {
    // Resize the image using letterbox
    cv::Mat img_input;
    std::vector<float> ratios;
    this->letterbox(img, img_input, ratios);

    // Create tensor from image
    ov::Tensor input_tensor(ppp_input_type_, ppp_input_shape_, img_input.data);

    // Create an infer request for model inference
    infer_request_->set_input_tensor(input_tensor);
    infer_request_->infer();

    // Retrieve inference results
    const ov::Tensor &output_tensor = infer_request_->get_output_tensor(0);
    const ov::Shape &output_shape = output_tensor.get_shape();
    std::vector<Detection> results = this->nms(output_tensor, output_shape, ratios);
    return results;
}
