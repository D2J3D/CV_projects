#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include<cmath>
#include<sciplot/sciplot.hpp>

void plot_proj_y(cv::Mat projection);

void plot_proj_x(cv::Mat projection);

cv::Mat find_x_projection(cv::Mat img);

cv::Mat find_y_projection(cv::Mat img);


int main() {
    const std::string PATH = "C:/Users/gridd/Downloads/text_ph.jpg";
    cv::Mat img = cv::imread(PATH, cv::IMREAD_COLOR);
    cv::imshow("img", img);
    cv::waitKey(0);
    find_x_projection(img);
    find_y_projection(img);
}

cv::Mat find_y_projection(cv::Mat img) {
    cv::Mat proj_y = cv::Mat(img.rows, 1, CV_32F);
    bool bw = (img.channels() == 1);
    for (int i = 0; i < img.rows; i++) {
        double sum = 0;
        for (int j = 0; j < img.cols; j++) {
            if (bw) {
                sum += img.at<uchar>(i, j);
            }
            else {
                const cv::Vec3b &pix = img.at<cv::Vec3b>(i, j);
                sum += pix[0] + pix[1] + pix[2];
            }
        }
        proj_y.at<float>(i) = sum;
    }
    for (int i = 0; i < img.rows; i++) {
            proj_y.at<float>(i) /= 256 * img.channels();
            std::cout << "LOG " << proj_y.at<float>(i);
    }
    plot_proj_y(proj_y);
    return proj_y;
}

void plot_proj_y(cv::Mat projection) {
    sciplot::Vec x = sciplot::linspace(0, projection.rows, projection.rows);
    std::vector<double> y;
    for (int i = 0; i < projection.rows; i++) {
        y.push_back(projection.at<float>(i));
    }
    sciplot::Plot2D plot;
    plot.xlabel("row");
    plot.ylabel("intensity");

    plot.legend().hide();

    plot.drawCurve(y, x).lineWidth(1.01).lineColor("blue");
    sciplot::Figure fig = { {plot} };
    sciplot::Canvas canvas = { {fig} };
    canvas.save("y_projection.pdf");
    canvas.show();
}


cv::Mat find_x_projection(cv::Mat img) {
    cv::Mat proj_x = cv::Mat(1, img.cols, CV_32F);
    bool bw = (img.channels() == 1);
    for (int j = 0; j < img.cols; j++) {
        double sum = 0;
        for (int i = 0; i < img.rows; i++) {
            if (bw) {
                sum += img.at<uchar>(i, j);
            }
            else {
                const cv::Vec3b& pix = img.at<cv::Vec3b>(i, j);
                sum += pix[0] + pix[1] + pix[2];
            }
        }
        proj_x.at<float>(j) = sum;
    }
    for (int j = 0; j < img.cols; j++) {
        proj_x.at<float>(j) /= 256 * img.channels();
        std::cout << "LOG " << proj_x.at<float>(j);
    }
    plot_proj_x(proj_x);
    return proj_x;
}

void plot_proj_x(cv::Mat projection) {
    sciplot::Vec x = sciplot::linspace(0, projection.cols, projection.cols);
    std::vector<double> y;
    for (int i = 0; i < projection.cols; i++) {
        y.push_back(projection.at<float>(i));
    }
    sciplot::Plot2D plot;
    plot.xlabel("col");
    plot.ylabel("intensity");

    plot.legend().hide();

    plot.drawCurve(x, y).lineWidth(1.01).lineColor("blue");
    sciplot::Figure fig = { {plot} };
    sciplot::Canvas canvas = { {fig} };
    canvas.save("x_projection.pdf");
    canvas.show();
}