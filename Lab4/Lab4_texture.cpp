#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

void bwareaopen(const cv::Mat &A, cv::Mat &C, int dim, int conn = 8);

cv::Mat calc_hist(cv::Mat img);

std::vector<double> calc_params(cv::Mat hist);

int main()
{

    return 0;
}

cv::Mat calc_hist(cv::Mat img)
{
    int histSize = 256;
    float range [] = {0, 256};
    const float * histRange[] = {range};
    cv::Mat hist;
    cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, histRange);
    return hist;
}

std::vector<double> calc_params(cv::Mat img, cv::Mat hist)
{
    double m = 0;
    double m = 0;
    for (int x = 0; x < img.cols; ++x){
        for (int y = 0; y < img.rows; ++y){
            int intensity = img.at<uchar>(y, x);
            m += intensity * hist.at<float>(intensity) / (img.rows * img.cols);
        }
    }
    // st
    // std::cout << "Среднее значение p(z): " << m << std::endl;
}
void bwareaopen(const cv::Mat &A, cv::Mat &C, int dim, int conn = 8)
{
    if (A.channels() != 1 && A.type() != CV_8U && A.type() != CV_32F)
        return;

    // Find all connected components
    cv::Mat labels, stats, centers;
    int num = cv::connectedComponentsWithStats(A, labels, stats, centers, conn);

    // Clone image
    C = A.clone();

    // Check size of all connected components
    std::vector<int> td;
    for (int i = 0; i < num; ++i)
    {
        if (stats.at<int>(i, cv::CC_STAT_AREA) < dim)
        {
            td.push_back(i);
        }
    }

    // Remove small areas
    if (td.size() > 0)
    {
        if (A.type() == CV_8U)
        {
            for (int i = 0; i < C.rows; ++i)
            {
                for (int j = 0; j < C.cols; ++j)
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
    }
    else
    {
        for (int i = 0; i < C.rows; ++i)
        {
            for (int j = 0; j < C.cols; ++j)
            {
                for (int k = 0; k < td.size(); ++k)
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
