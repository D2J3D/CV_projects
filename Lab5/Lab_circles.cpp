#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

std::vector<cv::Vec3f> findCircles(cv::Mat img, int minR, int maxR);

std::vector<cv::Vec3f> findCirclesWithR(cv::Mat img, int R);

cv::Mat drawCircles(cv::Mat img, std::vector<cv::Vec3f> lines);

int main()
{
    std::string IMG_DIR = "/home/den/CV_labs/Lab5/img/original/";
    std::string RES_DIR = "/home/den/CV_labs/Lab5/img/outputs/";

    cv::Mat img = cv::imread(IMG_DIR + "car.png");
    cv::Mat out;
    std::vector<cv::Vec3f> circles = findCircles(img, 60, 63);
    out = drawCircles(img, circles);
    return 0;
}

std::vector<cv::Vec3f> findCirclesWithR(cv::Mat img, int R)
{
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, gray, 9);
    cv::Canny(gray, gray, 50, 200, 3);
    std::vector<cv::Vec3f> circles;
    // setting min and max radiuses to be the same, cause we want method to find
    //a circle with a particular R radius
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
                     gray.rows / 16, 
                     100, 30, R, R                                 
    );

    return circles;
}

std::vector<cv::Vec3f> findCircles(cv::Mat img, int minR, int maxR)
{
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, gray, 5);
    // cv::Canny(gray, gray, 50, 200, 3);
    // cv::imwrite("/home/den/CV_labs/Lab5/img/outputs/car_canny.png", gray);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
                     gray.rows / 16, 
                     100, 30, minR, maxR 
    );

    return circles;
}

cv::Mat drawCircles(cv::Mat img, std::vector<cv::Vec3f> circles)
{
    cv::Mat out;
    out = img.clone();
    for (size_t i = 0; i < circles.size(); i++)
    {
        cv::Vec3f c = circles[i];
        cv::Point center = cv::Point(c[0], c[1]);
        // circle center
        circle(out, center, 1, cv::Scalar(0, 100, 100), 3, cv::LINE_AA);
        // circle outline
        int radius = c[2];
        circle(out, center, radius, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
    }

    return out;
}
