#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

cv::Mat binary_threshold(cv::Mat img, double threshold);

cv::Mat double_binary_threshold(cv::Mat img, double t1, double t2);

double calc_median_threshold(cv::Mat img);

double calc_gradient_threshold(cv::Mat img);

int *calc_hist(cv::Mat img, double min, double max);

int *calc_hist_classic(cv::Mat img);

std::vector<double> calc_prob_dist(int *hist, int img_size);

double calc_prob(std::vector<double> prob_dist, int from, int to);

double calc_mat_exp(std::vector<double> prob_dist, int from, int to);

double calc_threshold_otsu(cv::Mat img);

int main()
{
    std::string ORIGINAL_DIR = "/home/den/CV_labs/Lab4/img/original/";
    std::string RES_DIR = "/home/den/CV_labs/Lab4/img/outputs/binarization/";
    cv::Mat img = cv::imread(ORIGINAL_DIR + "2.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    cv::Mat img_copy;
    img.copyTo(img_copy);
    cv::threshold(img, img_copy, 120, 200, cv::THRESH_BINARY);
    cv::imwrite(RES_DIR + "cv_bin/double_binary_treshold_cv.png", img_copy);

    img.copyTo(img_copy);
    cv::threshold(img, img_copy, 0, 255, cv::THRESH_OTSU);
    cv::imwrite(RES_DIR + "cv_bin/otsu_treshold_cv.png", img_copy);

    img.copyTo(img_copy);
    cv::imwrite(RES_DIR + "own_bin/double_binary_treshold.png", double_binary_threshold(img_copy, 127, 220));

    img.copyTo(img_copy);
    cv::imwrite(RES_DIR + "own_bin/binary_treshold.png", binary_threshold(img_copy, 120));

    img.copyTo(img_copy);
    double median_t = calc_median_threshold(img_copy);
    std::cout << "MEDIAN THRESHOLD " << median_t << std::endl;
    cv::imwrite(RES_DIR + "own_bin/median_treshold.png", binary_threshold(img_copy, median_t));

    img.copyTo(img_copy);
    double grad_t = calc_gradient_threshold(img_copy);
    std::cout << "GRADIENT THRESHOLD " << grad_t << std::endl;
    cv::imwrite(RES_DIR + "own_bin/gradient_treshold.png", binary_threshold(img_copy, grad_t));

    img.copyTo(img_copy);
    double otsu_threshold = calc_threshold_otsu(img_copy);
    std::cout << "OTSU THRESHOLD " << otsu_threshold << std::endl;
    cv::imwrite(RES_DIR + "own_bin/otsu_treshold.png", binary_threshold(img_copy, otsu_threshold));
    return 0;
}

cv::Mat float_binary_threshold(cv::Mat img, double threshold)
{
    auto depth = img.depth();
    if (depth == CV_8U)
    {
        img.convertTo(img, CV_32F, 1.0 / 255.0);
    }

    for (int i = 0; i < img.rows; i++)
    {
        float *row_ptr = img.ptr<float>(i);
        for (int j = 0; j < img.cols; j++)
        {
            if (row_ptr[j] <= threshold)
            {
                row_ptr[j] = 0;
            }
            else
            {
                row_ptr[j] = 1;
            }
        }
    }
    if (depth == CV_8U)
    {
        img.convertTo(img, CV_8U, 255);
    }

    return img;
}

cv::Mat binary_threshold(cv::Mat img, double threshold)
{
    double t = threshold / 255.0; // converting threshold and image to double from 0 to 1 for easier binarization process
    auto depth = img.depth();
    if (depth == CV_8U)
    {
        img.convertTo(img, CV_32F, 1.0 / 255.0);
    }

    for (int i = 0; i < img.rows; i++)
    {
        float *row_ptr = img.ptr<float>(i);
        for (int j = 0; j < img.cols; j++)
        {
            if (row_ptr[j] <= t)
            {
                row_ptr[j] = 0;
            }
            else
            {
                row_ptr[j] = 1;
            }
        }
    }
    if (depth == CV_8U)
    {
        img.convertTo(img, CV_8U, 255);
    }

    return img;
}

cv::Mat double_binary_threshold(cv::Mat img, double t1, double t2)
{
    double threshold1 = t1 / 255.0, threshold2 = t2 / 255.0; // неудобно задавать границы сразу в формате чисел от 0 до 1, поэтому выполняетс преобразование
    auto depth = img.depth();
    if (depth == CV_8U)
    {
        img.convertTo(img, CV_32F, 1.0 / 255.0);
    }
    for (int i = 0; i < img.rows; i++)
    {
        float *row_ptr = img.ptr<float>(i);
        for (int j = 0; j < img.cols; j++)
        {
            if ((threshold1 < row_ptr[j] <= threshold2))
            {
                row_ptr[j] = 0;
            }
            else
            {
                row_ptr[j] = 1;
            }
        }
    }
    if (depth == CV_8U)
    {
        img.convertTo(img, CV_8U, 255);
    }
    return img;
}

double calc_median_threshold(cv::Mat img)
{
    auto depth = img.depth();
    if (depth == CV_8U)
    {
        img.convertTo(img, CV_32F, 1.0 / 255.0);
    }
    double min, max;
    cv::minMaxLoc(img, &min, &max);
    double threshold = (max - min) / 2;
    threshold *= 255; // converting to [0, 255] for the implemented threshlding methods
    return threshold;
}

double calc_gradient_threshold(cv::Mat img)
{
    double t = 0;
    double grad = 0;

    auto depth = img.depth();
    if (depth == CV_8U)
    {
        img.convertTo(img, CV_32F, 1.0 / 255.0);
    }

    for (int i = 1; i < img.rows - 1; i++)
    {
        for (int j = 1; j < img.cols + 1; j++)
        {
            double x_gradient = std::fabs(img.at<float>(i, j + 1) - img.at<float>(i, j - 1));
            double y_gradient = std::fabs(img.at<float>(i + 1, j) - img.at<float>(i - 1, j));
            double g = std::max(x_gradient, y_gradient);
            t += g * img.at<float>(i, j);
            grad += g;
        }
    }
    t /= grad;
    t *= 255; // converting to [0, 255] for the implemented binariztion methods
    return t;
}

int *calc_hist(cv::Mat img, double min, double max)
{
    // calc hist, bins from [img_min, img_max]
    int histSize = max - min + 1;
    int *hist = new int[histSize];
    for (int t = 0; t < histSize; t++)
    {
        hist[t] = 0;
    }
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            uchar intensity = img.at<uchar>(i, j);
            hist[int(intensity - min)]++;
        }
    }
    return hist;
}

int *calc_hist_classic(cv::Mat img)
{
    // calc hist, bins from [0, 256]
    int histSize = 256;
    int *hist = new int[histSize];
    for (int t = 0; t < histSize; t++)
    {
        hist[t] = 0;
    }
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            uchar intensity = img.at<uchar>(i, j);
            hist[int(intensity)]++;
        }
    }
    return hist;
}

std::vector<double> calc_prob_dist(int *hist, int img_size)
{
    std::vector<double> prob_dist;
    int hist_size = sizeof(hist) / sizeof(int);
    for (int i = 0; i < 256; i++)
    {
        prob_dist.push_back(hist[i] * 1.0 / img_size);
    }

    return prob_dist;
}

double calc_prob(std::vector<double> prob_dist, int from, int to)
{
    double prob = 0;
    for (int i = from; i <= to; i++)
    {
        prob += prob_dist[i];
    }

    return prob;
}

double calc_mat_exp(std::vector<double> prob_dist, int from, int to)
{
    double mat_exp = 0;
    for (int i = from; i <= to; i++)
    {
        mat_exp += i * prob_dist[i];
    }

    return mat_exp;
}

double calc_threshold_otsu(cv::Mat img)
{
    double min, L, t = 0;
    int img_size = img.cols * img.rows;
    cv::minMaxLoc(img, &min, &L);

    int *hist = calc_hist_classic(img);
    std::vector<double> prob_dist = calc_prob_dist(hist, img_size);
    double threshold = 0, maxSigma = -1;
    double w1, w2, mu1, mu2, disp;
    for (int k = 1; k < L; k++)
    {
        w1 = 0, w2 = 0, mu1 = 0, mu2 = 0, disp = 0;
        w1 = calc_prob(prob_dist, 0, k);
        w2 = 1 - w1;
        mu1 = calc_mat_exp(prob_dist, 0, k) / w1;
        mu2 = calc_mat_exp(prob_dist, k + 1, L) / w2;

        disp = w1 * w2 * (mu1 - mu2) * (mu1 - mu2);

        if (disp > maxSigma)
        {
            maxSigma = disp;
            threshold = k;
        }
    }

    return threshold;
}
