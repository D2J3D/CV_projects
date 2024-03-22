#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include<cmath>
#include<sciplot/sciplot.hpp>

void plot_profile(cv::Mat img_profile) {
    sciplot::Vec x = sciplot::linspace(0.0, img_profile.cols, img_profile.cols);
    std::vector<int> y;
    uchar* ptr = img_profile.ptr();
    for (int i = 0; i < img_profile.cols; i++) {
        y.push_back(ptr[i]);
    }
    sciplot::Plot2D plot;
    plot.xlabel("x");
    plot.ylabel("y");

    plot.xrange(0.0, img_profile.cols);
    plot.yrange(0.0, 256);
    plot.legend().hide();

    plot.drawCurve(x,y).lineWidth(1.01).lineColor("blue");
    sciplot::Figure fig = { {plot} };
    sciplot::Canvas canvas = { {fig} };

    canvas.show();
    canvas.save("barcode_profile.pdf");
}

int main() {
    const std::string PATH = "C:/Users/gridd/Downloads/ph2.jpeg";
    cv::Mat img = cv::imread(PATH, cv::IMREAD_GRAYSCALE);
    cv::Mat profile = img.row(img.rows / 2);
    plot_profile(profile);
    return 0;
}
