#include <Kinect.h>
#include <opencv2/opencv.hpp>

#include <direct.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <ctime>

template<class Interface>
inline void SafeRelease(Interface*& p) {
    if (p) {
        p->Release();
        p = nullptr;
    }
}

void CreateDirectoryRecursive(const std::string& path) {
    std::string current;
    for (size_t i = 0; i < path.size(); ++i) {
        char ch = path[i];
        current += ch;

        if (ch == '/' || ch == '\\') {
            _mkdir(current.c_str());
        }
    }
    _mkdir(current.c_str());
}

bool FileExists(const std::string& path) {
    FILE* file = nullptr;
    errno_t err = fopen_s(&file, path.c_str(), "rb");
    (void)err;
    if (file) {
        fclose(file);
        return true;
    }
    return false;
}

std::string GetNextImageName(const std::string& folder) {
    int index = 1;
    while (true) {
        std::ostringstream oss;
        oss << folder << "\\port_"
            << std::setw(4) << std::setfill('0') << index
            << ".jpg";

        if (!FileExists(oss.str())) {
            return oss.str();
        }
        ++index;
    }
}

int main() {
    HRESULT hr = S_OK;

    IKinectSensor* pSensor = nullptr;
    IColorFrameSource* pColorSource = nullptr;
    IColorFrameReader* pColorReader = nullptr;

    hr = GetDefaultKinectSensor(&pSensor);
    if (FAILED(hr) || !pSensor) {
        std::cout << "Failed to get Kinect sensor." << std::endl;
        return -1;
    }

    hr = pSensor->Open();
    if (FAILED(hr)) {
        std::cout << "Failed to open Kinect sensor." << std::endl;
        SafeRelease(pSensor);
        return -1;
    }

    hr = pSensor->get_ColorFrameSource(&pColorSource);
    if (FAILED(hr) || !pColorSource) {
        std::cout << "Failed to get color frame source." << std::endl;
        SafeRelease(pSensor);
        return -1;
    }

    hr = pColorSource->OpenReader(&pColorReader);
    if (FAILED(hr) || !pColorReader) {
        std::cout << "Failed to open color frame reader." << std::endl;
        SafeRelease(pColorSource);
        SafeRelease(pSensor);
        return -1;
    }

    const int colorWidth = 1920;
    const int colorHeight = 1080;
    const int saveIntervalMs = 1500;  // 每1.5秒自动保存一张

    cv::Mat colorMat(colorHeight, colorWidth, CV_8UC4);
    cv::Mat displayMat;
    cv::Mat bgrMat;

    // 这里改成你要保存训练图的目录
    const std::string saveFolder = R"(C:\Users\chang\Desktop\doc\Vision\yolo_port\images\train)";
    CreateDirectoryRecursive(saveFolder);

    cv::namedWindow("Kinect Color", cv::WINDOW_NORMAL);
    cv::resizeWindow("Kinect Color", 1280, 720);

    std::cout << "Auto capture started." << std::endl;
    std::cout << "Images will be saved to: " << saveFolder << std::endl;
    std::cout << "Press ESC to quit." << std::endl;

    int64 lastSaveTime = cv::getTickCount();
    double freq = cv::getTickFrequency();
    int savedCount = 0;

    while (true) {
        IColorFrame* pColorFrame = nullptr;
        hr = pColorReader->AcquireLatestFrame(&pColorFrame);

        if (SUCCEEDED(hr) && pColorFrame) {
            hr = pColorFrame->CopyConvertedFrameDataToArray(
                colorWidth * colorHeight * 4,
                colorMat.data,
                ColorImageFormat_Bgra
            );

            if (SUCCEEDED(hr)) {
                cv::resize(colorMat, displayMat, cv::Size(1280, 720));

                int64 now = cv::getTickCount();
                double elapsedMs = (now - lastSaveTime) * 1000.0 / freq;

                if (elapsedMs >= saveIntervalMs) {
                    cv::cvtColor(colorMat, bgrMat, cv::COLOR_BGRA2BGR);

                    std::string savePath = GetNextImageName(saveFolder);
                    bool ok = cv::imwrite(savePath, bgrMat);

                    if (ok) {
                        ++savedCount;
                        std::cout << "Saved: " << savePath << std::endl;
                    } else {
                        std::cout << "Failed to save image." << std::endl;
                    }

                    lastSaveTime = now;
                }

                std::ostringstream text1, text2;
                text1 << "Auto saving every " << saveIntervalMs << " ms";
                text2 << "Saved count: " << savedCount;

                cv::putText(displayMat, text1.str(), cv::Point(30, 50),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0,
                            cv::Scalar(0, 255, 0, 255), 2);

                cv::putText(displayMat, text2.str(), cv::Point(30, 100),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0,
                            cv::Scalar(0, 255, 0, 255), 2);

                cv::imshow("Kinect Color", displayMat);
            }
        }

        SafeRelease(pColorFrame);

        int key = cv::waitKey(1);
        if (key == 27) { // ESC
            break;
        }
    }

    cv::destroyAllWindows();

    SafeRelease(pColorReader);
    SafeRelease(pColorSource);

    if (pSensor) {
        pSensor->Close();
    }
    SafeRelease(pSensor);

    return 0;
}
