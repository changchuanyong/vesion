#include <Kinect.h>
#include <opencv2/opencv.hpp>

#include <direct.h>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

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

    cv::Mat colorMat(colorHeight, colorWidth, CV_8UC4);
    cv::Mat displayMat;
    cv::Mat bgrMat;

    std::string saveFolder = "C:\\Users\\chang\\Desktop\\doc\\vesion\\dataset\\live";
    std::string tempPath = saveFolder + "\\latest_tmp.jpg";
    std::string finalPath = saveFolder + "\\latest.jpg";

    CreateDirectoryRecursive(saveFolder);

    cv::namedWindow("Kinect Color", cv::WINDOW_NORMAL);
    cv::resizeWindow("Kinect Color", 1280, 720);

    std::cout << "Realtime capture started." << std::endl;
    std::cout << "Saving image to: " << finalPath << std::endl;
    std::cout << "Press ESC to quit." << std::endl;

    const int saveIntervalMs = 200;   // 先别太快，200ms 比较稳
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

                    bool ok = cv::imwrite(tempPath, bgrMat);
                    if (ok) {
                        std::remove(finalPath.c_str());
                        if (std::rename(tempPath.c_str(), finalPath.c_str()) == 0) {
                            ++savedCount;
                        }
                    }

                    lastSaveTime = now;
                }

                std::ostringstream text1, text2, text3;
                text1 << "Realtime saving every " << saveIntervalMs << " ms";
                text2 << "Saved count: " << savedCount;
                text3 << "Output: latest.jpg";

                cv::putText(displayMat, text1.str(), cv::Point(30, 50),
                            cv::FONT_HERSHEY_SIMPLEX, 0.9,
                            cv::Scalar(0, 255, 0, 255), 2);

                cv::putText(displayMat, text2.str(), cv::Point(30, 95),
                            cv::FONT_HERSHEY_SIMPLEX, 0.9,
                            cv::Scalar(0, 255, 0, 255), 2);

                cv::putText(displayMat, text3.str(), cv::Point(30, 140),
                            cv::FONT_HERSHEY_SIMPLEX, 0.9,
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