# YOLOv8.OpenVino.cpp

YOLOv8 is a state-of-the-art object detector by [ultralytics](https://github.com/ultralytics/ultralytics). This project implements the YOLOv8 object detector in C++ with an OpenVINO backend to speed up inference performance.

## Features

- Supports both **image** and **video** inference.
- Provides **faster** inference speeds.

## Prerequisites

Tested on Macbook Pro M1.

- **CMake v3.8+** - Download from [https://cmake.org/](https://cmake.org/).
- **OpenVINO Toolkit 2022.1+** - Tested on version 2023.1.0. Download from [here](https://storage.openvinotoolkit.org/repositories/openvino/packages/).
- **OpenCV v4.6+** - Tested on version v4.8.0_7. Download from [here](https://github.com/opencv/opencv/releases/).

## Getting Started

1. Set the `OpenCV_DIR` environment variable to point to your `../../opencv/build` directory.
2. Set the `OpenVINO_DIR` environment variable to point to your `../../openvino/runtime/cmake` directory.
3. Run the following build commands:

   **For Mac with VS Developer Command Prompt:**

   ```bash
   cd <git-directory>

   # Install dependencies
   brew install openvino
   brew install opencv

   # Build
   cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release
   cd build
   make
   ```

## Inference

1. Export the ONNX file:

   ```bash
   pip install ultralytics
   yolo export model=yolov8s.pt format=onnx  # Export official model
   ```

2. To run the inference, execute the following command:

   ```bash
   yolo --model <MODEL_PATH> [-i <IMAGE_PATH> | -v <VIDEO_PATH>] [--imgsz IMAGE_SIZE] [--gpu] [--iou-thresh IOU_THRESHOLD] [--score-thresh CONFIDENCE_THRESHOLD]

   # Example:
   yolo --model yolov8x.onnx -i images/zidane.jpg
   ```

## License

This project is licensed under the [MIT License](LICENSE). See the [LICENSE](LICENSE) file for details.

## References

- YOLOv8 by ultralytics: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- YOLO-NAS-OpenVino-cpp by Y-T-G: [https://github.com/Y-T-G/YOLO-NAS-OpenVino-cpp]
- OpenVINO: [https://github.com/openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)
- OpenCV: [https://github.com/opencv/opencv](https://github.com/opencv/opencv)
