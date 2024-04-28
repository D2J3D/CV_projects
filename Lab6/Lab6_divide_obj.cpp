#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat create_mask(cv::Mat img);

cv::Mat find_borders(cv::Mat img);

int main()
{

    const std::string IMG_DIR = "/home/den/CV_labs/Lab6/img/original/";
    const std::string RES_DIR = "/home/den/CV_labs/Lab6/img/outputs/";

    cv::Mat img, mask, img_separated, border, img_copy;
    img.copyTo(img_copy);
    img = cv::imread(IMG_DIR + "3.png", cv::IMREAD_COLOR);
    border = find_borders(img_copy); //find initial borders
    // divide objects and calc a new border
    mask = create_mask(img);
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::bitwise_and(img_gray, mask, img_separated);
    cv::imwrite(RES_DIR + "img_separated.jpg", (img_separated));
    cv::cvtColor(img_separated, img_separated, cv::COLOR_GRAY2BGR);
    cv::imwrite(RES_DIR + "img_separated_borders.jpg", find_borders(img_separated));
    return 0;
}

cv::Mat create_mask(cv::Mat img)
{
    cv::Mat img_gray, img_thresholded;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::threshold(img_gray, img_thresholded, 84, 255, cv::THRESH_BINARY_INV);
    cv::Mat B = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));

    // erode
    cv::Mat BW2;
    cv::morphologyEx(img_thresholded, BW2, cv::MORPH_ERODE, 
    B, cv::Point(-1, -1), 14, cv::BORDER_CONSTANT, cv::Scalar(0));
    // dilate 
    cv::Mat D, C, S;
    cv::Mat T = cv::Mat::zeros(img_thresholded.size(), img_thresholded.type());
    int pix_num = img_thresholded.rows * img_thresholded.cols;
    while (cv::countNonZero(BW2) < pix_num)
    {

        cv::morphologyEx(BW2, D, cv::MORPH_DILATE, 
        B, cv::Point(-1, -1), 1,
        cv::BORDER_CONSTANT, cv::Scalar(0));

        cv::morphologyEx(D, C, cv::MORPH_CLOSE,
        B, cv::Point(-1, -1), 1,
        cv::BORDER_CONSTANT, cv::Scalar(0));

        S = C - D;
        cv::bitwise_or(S, T, T);
        BW2 = D;
    }

    // close for borders
    cv::morphologyEx(T, T, cv::MORPH_CLOSE, B, cv::Point(-1, -1), 14, cv::BORDER_CONSTANT, cv::Scalar(255));
    cv::bitwise_and(~T, img_thresholded, img_thresholded);

    return img_thresholded;

}


cv::Mat find_borders(cv::Mat img)
{
    // prepare img for calculation
    cv::Mat img_gray, img_thresholded;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::threshold(img_gray, img_thresholded, 80, 255, cv::THRESH_BINARY_INV);
    
    // find border
    cv::Mat img_erode;
    cv::Mat B = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::erode(img_thresholded, img_erode, B, cv::Point(-1, -1), 1);
    cv::Mat border;
    border = img_thresholded - img_erode;
    return border;
}
