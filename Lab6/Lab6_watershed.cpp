#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

void bwareaopen(const cv::Mat &A, cv::Mat &C, int dim, int conn);

cv::Mat segmentation_watershed(cv::Mat img);

int main()
{

    const std::string IMG_DIR = "/home/den/CV_labs/Lab/img/original/";
    const std::string RES_DIR = "/home/den/CV_labs/Lab/img/outputs/segmentation_watershed/";

    cv::Mat img;
    img = cv::imread(IMG_DIR + "balls.png");

    cv::Mat markers = segmentation_watershed(img);
    cv::imwrite(RES_DIR + "balls.png", markers);

    return 0;
}

cv::Mat segmentation_watershed(cv::Mat img)
{
    cv::Mat img_gray, img_bw;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::threshold(img_gray, img_bw, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    bwareaopen(img_bw, img_bw, 10, 4);

    cv::Mat B = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
    cv::morphologyEx(img_bw, img_bw, cv::MORPH_CLOSE, B);

    cv::Mat img_fg;
    double img_fg_min, img_fg_max;
    cv::distanceTransform(img_bw, img_fg, cv::DIST_L2, 5);
    cv::minMaxLoc(img_fg, &img_fg_min, &img_fg_max);
    cv::threshold(img_fg, img_fg, 0.6 * img_fg_max, 255, 0);
    img_fg.convertTo(img_fg, CV_8U, 255.0 / img_fg_max);  
    cv::Mat markers;
    int num = cv::connectedComponents(img_fg, markers);

    cv::Mat img_bg=cv::Mat::zeros(img_bw.size(), img_bw.type());
    cv::Mat markers_bg=markers.clone();
    cv::watershed(img, markers_bg);
    img_bg.setTo(cv::Scalar(255), markers_bg==-1);

    cv::Mat img_unk;
    cv::bitwise_not(img_bg, img_unk);
    cv::subtract(img_unk, img_fg, img_unk);

    markers += 1;
    markers.setTo(cv::Scalar(0), img_unk == 255);
    cv::watershed(img, markers);

    cv::Mat markers_jet;
    markers.convertTo(markers_jet, CV_8U, 255.0/(num+1));
    cv::applyColorMap(markers_jet, markers_jet, cv::COLORMAP_JET);

    img.setTo(cv::Scalar(255, 255, 0), markers==-1);
    return img;
}

void bwareaopen(const cv::Mat &A, cv::Mat &C, int dim, int conn)
{
    if (A.channels() != 1 && A.type() != A.type() != CV_8U && A.type() != CV_32F)
    {
        return;
    }

    cv::Mat labels, stats, centers;
    int num = cv::connectedComponentsWithStats(A, labels, stats, centers, conn);

    C = A.clone();
    std::vector<int> td;
    for (int i = 0; i < num; i++)
    {
        if (stats.at<int>(i, cv::CC_STAT_AREA) < dim)
        {
            td.push_back(i);
        }
    }
    if (td.size() > 0)
    {
        if (A.type() == CV_8U)
        {
            for (int i = 0; i < C.rows; i++)
            {
                for (int j = 0; j < C.cols; j++)
                {
                    for (int k = 0; k < td.size(); ++k)
                    {
                        if (labels.at<int>(i, j) == td[k])
                        {
                            C.at<uchar>(i, j) = 0;
                            continue;
                        }
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < C.rows; i++)
            {
                for (int j = 0; j < C.cols; j++)
                {
                    for (int k = 0; k < td.size(); k++)
                    {
                        if (labels.at<int>(i, j) == td[k])
                        {
                            C.at<float>(i, j) = 0;
                            continue;
                        }
                    }
                }
            }
        }
    }
}
