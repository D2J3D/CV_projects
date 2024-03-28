#include<iostream>
#include<cmath>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>

cv::Mat barrel_distortion(cv::Mat img);

cv::Mat pincushion_distortion(cv::Mat img);

int main() {
    cv::Mat img_with_bd = cv::imread("/home/den/CV_labs/Lab2/images/original/bd.jpg", cv::IMREAD_COLOR);
    cv::imwrite("/home/den/CV_labs/Lab2/images/results/after_pd_applied_img.jpg", pincushion_distortion(img_with_bd));
    
    cv::Mat img_with_pd = cv::imread("/home/den/CV_labs/Lab2/images/original/pd.jpg", cv::IMREAD_COLOR);
    cv::imwrite("/home/den/CV_labs/Lab2/images/results/after_bd_applied_img.jpg", barrel_distortion(img_with_pd));
}

cv::Mat barrel_distortion(cv::Mat img) {
    cv::Mat xi, yi;
    std::vector<float> t_x, t_y;
    for (int i = 0; i < img.cols; i++) {
        t_x.push_back(float(i));
    }
    for (int i = 0; i < img.rows; i++) {
        t_y.push_back(float(i));
    }

    cv::repeat(cv::Mat(t_x).reshape(1, 1), img.rows, 1, xi);
    cv::repeat(cv::Mat(t_y).reshape(1, 1).t(), 1, img.cols, yi);

    double xmid = xi.cols / 2.0;
    double ymid = xi.rows / 2.0;
    xi -= xmid;
    xi /= xmid;
    yi -= ymid;
    yi /= ymid;

    cv::Mat r, theta;
    cv::cartToPolar(xi, yi, r, theta);
    double F3(0.07), F5(0.12);
    cv::Mat r3, r5;

    cv::pow(r, 3, r3);
    cv::pow(r, 5, r5);
    r += r3 * F3;
    r += r5 * F5;

    cv::Mat u, v;
    cv::polarToCart(r, theta, u, v);
    u *= xmid;
    u += xmid;
    v *= ymid;
    v += ymid;

    cv::Mat img_barrel;
    cv::remap(img, img_barrel, u, v, cv::INTER_LINEAR);
    return img_barrel;
}

cv::Mat pincushion_distortion(cv::Mat img) {
    cv::Mat xi, yi;
    std::vector<float> t_x, t_y;
    for (int i = 0; i < img.cols; i++) {
        t_x.push_back(float(i));
    }
    for (int i = 0; i < img.rows; i++) {
        t_y.push_back(float(i));
    }

    cv::repeat(cv::Mat(t_x).reshape(1, 1), img.rows, 1, xi);
    cv::repeat(cv::Mat(t_y).reshape(1, 1).t(), 1, img.cols, yi);

    double xmid = xi.cols / 2.0;
    double ymid = xi.rows / 2.0;
    xi -= xmid;
    xi /= xmid;
    yi -= ymid;
    yi /= ymid;

    cv::Mat r, theta;
    cv::cartToPolar(xi, yi, r, theta);
    double F3(-0.2);
    cv::Mat r2;

    cv::pow(r, 2, r2);
    r += r2 * F3;

    cv::Mat u, v;
    cv::polarToCart(r, theta, u, v);
    u *= xmid;
    u += xmid;
    v *= ymid;
    v += ymid;

    cv::Mat img_pincushion;
    cv::remap(img, img_pincushion, u, v, cv::INTER_LINEAR);
    return img_pincushion;
}