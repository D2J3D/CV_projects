#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

std::vector<cv::Vec4i> findLines(cv::Mat img, int threshold);

cv::Mat drawLines(cv::Mat img, std::vector<cv::Vec4i> lines);

double norm(cv::Vec2i p1, cv::Vec2i p2);

void countLines(std::vector<cv::Vec4i> lines);

cv::Mat img_centered_rotate(cv::Mat, double phi);

int main()
{
    std::string IMG_DIR = "/home/den/CV_labs/Lab5/img/original/";
    std::string RES_DIR = "/home/den/CV_labs/Lab5/img/outputs/";

    cv::Mat img = cv::imread(IMG_DIR + "pattern.png");
    cv::Mat img_help;
    img.copyTo(img_help);
    std::vector<cv::Vec4i> lines = findLines(img_help, 70);
    cv::Mat out;
    out = drawLines(img_help, lines);
    countLines(lines);
    return 0;
}

std::vector<cv::Vec4i> findLines(cv::Mat img, int threshold)
{
    cv::Mat dst;
    cv::Mat hsv;
    cv::Canny(img, dst, 50, 200, 3);
    std::vector<cv::Vec4i> linesP;    // will hold the results of the detection
    cv::HoughLinesP(dst, linesP, 1, CV_PI / 180, threshold, 50, 10); // runs the actual detection
    return linesP;
}

cv::Mat drawLines(cv::Mat img, std::vector<cv::Vec4i> lines)
{
    cv::Mat out;
    out = img.clone();
    for (size_t i = 0; i < lines.size(); i++)
    {
        cv::Vec4i l = lines[i];
        cv::Point start = cv::Point(l[0], l[1]);
        cv::Point end = cv::Point(l[2], l[3]);
        // draw line and start-end points (blue for start, green for end)
        cv::line(out, start, end, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
        cv::line(out, start, start, cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
        cv::line(out, end, end, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
    }
    return out;
}

// for finding the longest and the shortest line
// we will use the standart Euclidean metrics
double norm(cv::Vec2i p1, cv::Vec2i p2)
{
    return std::sqrt(std::pow(p1[0] - p2[0], 2) + std::pow(p1[1] - p2[1], 2));
}

// print info about the lines (max, min, their amount)
void countLines(std::vector<cv::Vec4i> lines)
{
    std::cout << "Amount of lines " << lines.size() << std::endl;
    int x0, y0, x1, y1, len;
    std::vector<double> lengths;
    for (size_t i = 0; i < lines.size(); i++)
    {
        cv::Vec4i l = lines[i];
        x0 = l[0], y0 = l[1], x1 = l[2], y1 = l[3];
        cv::Vec2i p{x0, y0};
        cv::Vec2i p1{x1, y1};
        lengths.push_back(norm(p, p1));
    }
    std::sort(lengths.begin(), lengths.end());
    std::cout << "Maximum line len " << lengths[lengths.size() - 1] << std::endl;
    std::cout << "Minimum line len " << lengths[0] << std::endl;
}

cv::Mat img_centered_rotate(cv::Mat img, double phi)
{
    double r_phi = phi * M_PI / 180.0;
    cv::Mat T2 = (cv::Mat_<double>(3, 3) << std::cos(r_phi), -std::sin(r_phi), 0, std::sin(r_phi), std::cos(r_phi), 0, 0, 0, 1);
    cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1, 0, -(img.rows - 1) / 2.0, 0, 1, -(img.cols - 1) / 2.0, 0, 0, 1);
    cv::Mat T3 = (cv::Mat_<double>(3, 3) << 1, 0, (img.rows - 1) / 2.0, 0, 1, (img.cols - 1) / 2.0, 0, 0, 1);

    cv::Mat T = cv::Mat(T3 * T2 * T1, cv::Rect(0, 0, 3, 2));
    cv::Mat img_rotated;
    cv::warpAffine(img, img_rotated, T, cv::Size(img.cols, img.rows));
    return img_rotated;
}