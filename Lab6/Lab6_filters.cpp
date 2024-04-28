#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main(){

    const std::string  IMG_DIR = "/home/den/CV_labs/Lab6/img/original/";
    const std::string  RES_DIR = "/home/den/CV_labs/Lab6/img/outputs/";

    cv::Mat img, img_gray, img_dilate, img_erode;
    img = cv::imread(IMG_DIR + "1.png");

    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::threshold(img_gray, img_gray, 100, 255, cv::THRESH_BINARY);

    cv::Mat B = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
    cv::morphologyEx(img, img_erode, cv::MORPH_ERODE, B, cv::Point(-1, -1), 1);
    cv::morphologyEx(img_erode, img_dilate, cv::MORPH_DILATE, B, cv::Point(-1, -1), 1);
    
    cv::imwrite(RES_DIR + "erode_dilate.png", img_dilate);
    return 0;
}
