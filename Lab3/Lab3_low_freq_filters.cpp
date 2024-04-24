#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

cv::Mat arithmetic_averaging_filtration(cv::Mat img);

cv::Mat gauss_filtration(cv::Mat img, cv::Size kernel_size);

double localize(cv::Mat img, double y, double x, int n, int m, int q);

cv::Mat contrharmonic_filter(cv::Mat img, int q);

void apply_and_save_pack_contrharmonic_filter(std::vector<cv::Mat> images_to_filter, std::vector<std::string> filenames, std::vector<int> coeffs_q);

int main()
{
    const std::string DIR = "/home/den/CV_labs/Lab3/images/outputs/noises/";
    std::vector<cv::Mat> noise_imgs{cv::imread(DIR + "img_with_additive_noise.png", cv::IMREAD_GRAYSCALE), cv::imread(DIR + "img_with_gauss_noise.png", cv::IMREAD_GRAYSCALE), cv::imread(DIR + "img_with_impluse_noise_0sp.png", cv::IMREAD_GRAYSCALE), cv::imread(DIR + "img_with_impluse_noise_1sp.png", cv::IMREAD_GRAYSCALE), cv::imread(DIR + "img_with_impluse_noise_05sp.png", cv::IMREAD_GRAYSCALE), cv::imread(DIR + "img_with_multiplicative_noise.png", cv::IMREAD_GRAYSCALE), cv::imread(DIR + "img_with_quant_noise.png", cv::IMREAD_GRAYSCALE)};
    std::vector<int> coeffs{0, 3, 2, -22, -5, -5, -3};
    std::vector<std::string> filenames = {"img_with_additive_noise.png", "img_with_gauss_noise.png", "img_with_impulse_noise_0sp.png", "img_with_impluse_noise_1sp.png", "img_with_impluse_noise_05sp.png", "img_with_multiplicative_noise.png", "img_with_quant_noise.png"};
    apply_and_save_pack_contrharmonic_filter(noise_imgs, filenames, coeffs);

    for (int i = 0; i < noise_imgs.size(); i++)
    {
        cv::Mat img_copy;
        noise_imgs[i].copyTo(img_copy);
        cv::imwrite("/home/den/CV_labs/Lab3/images/outputs/low_freq_filters/arithmetic_average_" + filenames[i], arithmetic_averaging_filtration(img_copy));
    }

    for (int i = 0; i < noise_imgs.size(); i++)
    {
        cv::Mat img_copy;
        noise_imgs[i].copyTo(img_copy);
        cv::imwrite("/home/den/CV_labs/Lab3/images/outputs/low_freq_filters/gauss_" + filenames[i], gauss_filtration(img_copy, cv::Size(7, 7)));
    }
}

double localize(cv::Mat img, double y, double x, int n, int m, int q)
{
    double sum = 0;
    for (int j = -n / 2; j < n / 2; j++)
    {
        for (int i = -m / 2; i < m / 2; i++)
        {
            int current_coord_y = y + j;
            int current_coord_x = x + i;
            if ((current_coord_x >= 0) && (current_coord_x < img.cols) && (current_coord_y >= 0) && (current_coord_y < img.rows))
            {
                sum += std::pow(img.at<float>(current_coord_y, current_coord_x), q);
            }
        }
    }
    return sum;
}

cv::Mat contrharmonic_filter(cv::Mat img, int Q)
{
    cv::Mat img_copy = cv::Mat::zeros(img.size(), CV_32F);
    auto depth = img.depth();
    if (depth == CV_8U)
    {
        img.convertTo(img, CV_32F, 1.0 / 255);
    }

    for (int y = 0; y < img.rows; y++)
    {
        float *old_ptr = img.ptr<float>(y);
        float *new_ptr = img_copy.ptr<float>(y);
        for (int x = 0; x < img.cols; x++)
        {
            int pad_size = 3;
            new_ptr[x] = localize(img, y, x, pad_size, pad_size, Q + 1) / localize(img, y, x, pad_size, pad_size, Q);
        }
    }

    img_copy.convertTo(img_copy, CV_8U, 255);
    return img_copy;
}

cv::Mat arithmetic_averaging_filtration(cv::Mat img)
{
    cv::Mat out;
    cv::blur(img, out, cv::Size(3, 3));
    return out;
}

cv::Mat gauss_filtration(cv::Mat img, cv::Size kernel_size)
{
    cv::Mat img_out;
    cv::GaussianBlur(img, img_out, kernel_size, 0, 0);
    return img_out;
}

void apply_and_save_pack_contrharmonic_filter(std::vector<cv::Mat> images_to_filter, std::vector<std::string> filenames, std::vector<int> coeffs_q)
{

    for (int i = 0; i < images_to_filter.size(); i++)
    {
        cv::Mat img_copy;
        images_to_filter[i].copyTo(img_copy);
        cv::imwrite("/home/den/CV_labs/Lab3/images/outputs/low_freq_filters/contrharmonic3_" + filenames[i], contrharmonic_filter(img_copy, coeffs_q[i]));
    }
}
