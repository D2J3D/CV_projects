#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<random>
#include<cmath>
#include<iomanip>

cv::Mat impulse_noise(cv::Mat img, double d, double s_vs_p);

cv::Mat additive_noise(cv::Mat img, double var);

cv::Mat multiplicative_noise(cv::Mat img, double var);

cv::Mat gauss_noise(cv::Mat img, double mean, double var);

std::vector<float> unique(const cv::Mat& img, bool sort);

cv::Mat q_noise(cv::Mat img);

int main(){
    const std::string PATH = "/home/den/CV_labs/Lab3/images/original/RQC.png";
    cv::Mat img = cv::imread(PATH, cv::IMREAD_GRAYSCALE);
    cv::Mat img_copy;

    img.copyTo(img_copy);
    cv::imwrite("/home/den/CV_labs/Lab3/images/outputs/noises/img_with_impluse_noise_05sp.png", impulse_noise(img_copy, 0.07, 0.5));

    img.copyTo(img_copy);
    cv::imwrite("/home/den/CV_labs/Lab3/images/outputs/noises/img_with_impluse_noise_0sp.png", impulse_noise(img_copy, 0.07, 0));

    img.copyTo(img_copy);
    cv::imwrite("/home/den/CV_labs/Lab3/images/outputs/noises/img_with_impluse_noise_1sp.png", impulse_noise(img_copy, 0.07, 1));

    img.copyTo(img_copy);
    cv::imwrite("/home/den/CV_labs/Lab3/images/outputs/noises/img_with_additive_noise.png", additive_noise(img_copy, 0.07));


    img.copyTo(img_copy);
    cv::imwrite("/home/den/CV_labs/Lab3/images/outputs/noises/img_with_multiplicative_noise.png", multiplicative_noise(img_copy, 0.07));

    img.copyTo(img_copy);
    cv::imwrite("/home/den/CV_labs/Lab3/images/outputs/noises/img_with_gauss_noise.png", gauss_noise(img_copy, 0, 0.07));

    img.copyTo(img_copy);
    cv::imwrite("/home/den/CV_labs/Lab3/images/outputs/noises/img_with_quant_noise.png", q_noise(img_copy));

    return 0;
}

cv::Mat impulse_noise(cv::Mat img, double d, double s_vs_p){
    cv::Mat img_new;
    // Converting img to floating points
    if (img.depth() == CV_8U){
        img.convertTo(img_new, CV_32F, 1.0 / 255);
    }
    else{
        img.copyTo(img_new);
    }

    // Divide into 3 BGR channels
    std::vector<cv::Mat> img_BGR;
    cv::split(img_new, img_BGR);

    // Creating salt and pepper noise (with s_vs_p distribution) for each channel  
    for (int channel = 0; channel < img_BGR.size(); channel++){
        img_new = cv::Mat(img_BGR[channel].size(), CV_32F);
        cv::randu(img_new, cv::Scalar(0), cv::Scalar(1));
        if(img_BGR[channel].depth() == CV_8U){
            img_BGR[channel].setTo(cv::Scalar(255), img_new < d * s_vs_p);
        }
        else{
            img_BGR[channel].setTo(cv::Scalar(1), img_new < d * s_vs_p);
        }
        img_BGR[channel].setTo(cv::Scalar(0), (img_new >= d * s_vs_p) & (img_new < d));
    }

    // Merging back to single img
    cv::Mat img_out;
    cv::merge(img_BGR, img_out);
    if (img.depth() == CV_8U){
        img_out.convertTo(img_out, CV_8U, 255);
    }
    return img_out;
}

cv::Mat multiplicative_noise(cv::Mat img, double var){
    cv::Mat img_new;
    // Converting img to floating points
    if (img.depth() == CV_8U){
        img.convertTo(img_new, CV_32F, 1.0 / 255);
    }
    else{
        img.copyTo(img_new);
    }

    // Divide into 3 BGR channels
    std::vector<cv::Mat> img_BGR;
    cv::split(img_new, img_BGR);

    for (int channel = 0; channel < img_BGR.size(); channel++){
        img_new = cv::Mat(img_BGR[channel].size(), CV_32F);
        cv::Mat img_new_t;
        img_new.convertTo(img_new_t, CV_8U, 255);
        cv::imwrite("real_test_img_new.jpg", img_new_t);
        cv::randn(img_new, cv::Scalar(0), cv::Scalar(std::sqrt(var)));
        if (img_BGR[channel].depth() == CV_8U){
            cv::Mat img_BGR_32f = cv::Mat(img_BGR[channel], CV_32F);
            img_BGR[channel].convertTo(img_BGR_32f, CV_32F);
            img_BGR_32f += img_BGR_32f.mul(img_new);
        }
        else{
            img_BGR[channel] += img_BGR[channel].mul(img_new);
        }
    }
    cv::Mat img_out;
    cv::merge(img_BGR, img_out);
    if (img.depth() == CV_8U){
        img_out.convertTo(img_out, CV_8U, 255);
    }

    return img_out;
}

cv::Mat additive_noise(cv::Mat img, double var){
    cv::Mat img_new;
    // Converting img to floating points
    if (img.depth() == CV_8U){
        img.convertTo(img_new, CV_32F, 1.0 / 255);
    }
    else{
        img.copyTo(img_new);
    }

    // Divide into 3 BGR channels
    std::vector<cv::Mat> img_BGR;
    cv::split(img_new, img_BGR);

    for (int channel = 0; channel < img_BGR.size(); channel++){
        img_new = cv::Mat(img_BGR[channel].size(), CV_32F);
        cv::Mat img_new_t;
        img_new.convertTo(img_new_t, CV_8U, 255);
        cv::imwrite("real_test_img_new.jpg", img_new_t);
        cv::randn(img_new, cv::Scalar(0), cv::Scalar(std::sqrt(var)));
        if (img_BGR[channel].depth() == CV_8U){
            cv::Mat img_BGR_32f = cv::Mat(img_BGR[channel], CV_32F);
            img_BGR[channel].convertTo(img_BGR_32f, CV_32F);
            img_BGR_32f += img_new;
        }
        else{
            img_BGR[channel] += img_new;
        }
    }
    cv::Mat img_out;
    cv::merge(img_BGR, img_out);
    if (img.depth() == CV_8U){
        img_out.convertTo(img_out, CV_8U, 255);
    }

    return img_out;
}

cv::Mat gauss_noise(cv::Mat img, double mean, double var){
    cv::Mat img_new;
    // Converting img to floating points
    if (img.depth() == CV_8U){
        img.convertTo(img_new, CV_32F, 1.0 / 255);
    }
    else{
        img.copyTo(img_new);
    }

    // Divide into 3 BGR channels
    std::vector<cv::Mat> img_BGR;
    cv::split(img_new, img_BGR);

    for (int channel = 0; channel < img_BGR.size(); channel++){
        img_new = cv::Mat(img_BGR[channel].size(), CV_32F);
        cv::Mat img_new_t;
        img_new.convertTo(img_new_t, CV_8U, 255);
        cv::imwrite("real_test_img_new.jpg", img_new_t);
        cv::randn(img_new, cv::Scalar(mean), cv::Scalar(std::sqrt(var)));
        if (img_BGR[channel].depth() == CV_8U){
            cv::Mat img_BGR_32f = cv::Mat(img_BGR[channel], CV_32F);
            img_BGR[channel].convertTo(img_BGR_32f, CV_32F);
            img_BGR_32f += img_new * 255;
        }
        else{
            img_BGR[channel] += img_new;
        }
    }
    cv::Mat img_out;
    cv::merge(img_BGR, img_out);
    if (img.depth() == CV_8U){
        img_out.convertTo(img_out, CV_8U, 255);
    }

    return img_out;
}

std::vector<float> unique(const cv::Mat& img, bool sort){
    if (img.depth() != CV_32F){
        return std::vector<float>();
    }

    std::vector<float> out;
    int rows = img.rows;
    int cols  = img.cols * img.channels();

    if (img.isContinuous()){
        cols *= rows;
        rows = 1;
    }
    for (int y = 0; y < rows; y++){
        const float* ptr = img.ptr<float>(y);
        for (int x = 0; x < cols; x++){
            float value = ptr[x];
            if (std::find(out.begin(), out.end(), value) == out.end()){
                out.push_back(value);
            }
        }
    }

    if (sort){
        std::sort(out.begin(), out.end());
    }
    return out;
}

cv::Mat q_noise(cv::Mat img){
    cv::Mat img_new;
    // Converting img to floating points
    if (img.depth() == CV_8U){
        img.convertTo(img_new, CV_32F, 1.0 / 255);
    }
    else{
        img_new = img.clone();
    }

    size_t vals = unique(img_new, false).size();
    vals = (size_t)std::pow(2, std::ceil(std::log2(vals)));
    int rows = img_new.rows;
    int cols = img_new.cols * img_new.channels();
    if (img_new.isContinuous()){
        cols *= rows;
        rows = 1;
    }
    using param_t = std::poisson_distribution<int>::param_type;
    std::default_random_engine engine;
    std::poisson_distribution<> poisson;
    for (int i = 0; i < rows; i++){
        float* ptr = img_new.ptr<float>(i);
        for (int j = 0; j < cols; j++){
            ptr[j] = float(poisson(engine, param_t({ ptr[j] * vals }))) / vals;
        }
    }
    if(img.depth() == CV_8U){
        img_new.convertTo(img_new, CV_8U, 255);
    }
    return img_new;
}