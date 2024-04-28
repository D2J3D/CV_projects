#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat create_mask(cv::Mat img);

int main()
{

    const std::string IMG_DIR = "/home/den/CV_labs/Lab/img/original/";
    const std::string RES_DIR = "/home/den/CV_labs/Lab/img/outputs/";

    cv::Mat img, mask, img_separated;
    img = cv::imread(IMG_DIR + "m.png", cv::IMREAD_GRAYSCALE);
    mask = create_mask(img);
    cv::bitwise_and(img, mask, img_separated);
    cv::imwrite(RES_DIR + "balls_obj_test5.jpg", img_separated);

    return 0;
}

cv::Mat create_mask(cv::Mat img)
{
    cv::Mat img_copy;
    // cv::cvtColor(img, img_copy, cv::COLOR_BGR2GRAY);
    img.copyTo(img_copy);
    cv::Mat mask;
    cv::medianBlur(img_copy, img_copy, 5);
    cv::threshold(img_copy, mask, 240, 255, cv::THRESH_BINARY_INV);
    // return mask;
    cv::Mat B = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));

    // Erosion
    cv::Mat BW2;
    cv::morphologyEx(mask, BW2, cv::MORPH_ERODE, B, cv::Point(-1, -1), 5, cv::BORDER_CONSTANT, cv::Scalar(0));
    // return BW2;
    // Dilation
    cv::Mat D, C, S;
    cv::Mat T = cv::Mat::zeros(mask.size(), mask.type());
    int pix_num = mask.rows * mask.cols;
    while (cv::countNonZero(BW2) < pix_num)
    {
        cv::morphologyEx(BW2, D, cv::MORPH_DILATE, B, cv::Point(-1, -1), 4, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::morphologyEx(D, C, cv::MORPH_CLOSE, B, cv::Point(-1, -1), 11, cv::BORDER_CONSTANT, cv::Scalar(0));
        S = C - D;
        cv::bitwise_or(S, T, T);
        BW2 = D;
    }

    // Closing for border
    cv::morphologyEx(T, T, cv::MORPH_CLOSE, B, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(255));
    cv::bitwise_and(~T, mask, mask);
    return mask;
}