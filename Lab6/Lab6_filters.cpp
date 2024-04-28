#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main(){

    const std::string  IMG_DIR = "/home/den/CV_labs/Lab6/img/original/";
    const std::string  RES_DIR = "/home/den/CV_labs/Lab6/img/outputs/";

    cv::Mat img, img_gray, img_open, img_close;
    img = cv::imread(IMG_DIR + "2.png");

    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::threshold(img_gray, img_gray, 220, 255, cv::THRESH_BINARY);

    // filtering website link in the right bottom corner
    cv::Mat B = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(12, 14));
    cv::morphologyEx(img_gray, img_open, cv::MORPH_OPEN, B, cv::Point(-1, -1), 1);
    cv::morphologyEx(img_open, img_close, cv::MORPH_CLOSE, B, cv::Point(-1, -1), 2);
    // apply mask on the grayscale img
    cv::Mat img_filtered, mask_rocket, img_filtered2;
    cv::medianBlur(img_filtered, img_filtered, 5);
    cv::cvtColor(img, img_filtered, cv::COLOR_BGR2GRAY);
    img_filtered.copyTo(img_filtered2);
    cv::bitwise_or(img_filtered, img_close, img_filtered);
    // cv::bitwise_and(img_filtered,~img_close, img_filtered);
    // cv::bitwise_and(img_filtered2, img_close, mask_rocket);
    // cv::bitwise_and(img_close, mask_rocket, img_close);
    // cv::bitwise_and(img_filtered, ~img_close, img_filtered);
    cv::imwrite(RES_DIR + "spacex2_only_close.png", img_filtered);
    return 0;
}
