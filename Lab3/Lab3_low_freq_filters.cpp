#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<cmath>

double localize_pix(cv::Mat img, int pix_coords[2], int w_size[2], int q);

cv::Mat arithmetic_averaging_filtration(cv::Mat img);

cv::Mat contrharmonic_filtration(cv::Mat img, int q);

cv::Mat gauss_filtration(cv::Mat img, cv::Size kernel_size);

void apply_and_save_pack_contrharmonic_filter(std::vector<cv::Mat> images_to_filter, std::vector<std::string> filenames, std::vector<int> coeffs_q);

int main(){
    const std::string DIR = "/home/den/CV_labs/Lab3/images/outputs/noises/";
    std::vector<cv::Mat> noise_imgs{cv::imread(DIR + "img_with_additive_noise.png"), cv::imread(DIR + "img_with_gauss_noise.png"), cv::imread(DIR + "img_with_impluse_noise_0sp.png"), cv::imread(DIR + "img_with_impluse_noise_1sp.png"), cv::imread(DIR + "img_with_impluse_noise_05sp.png"), cv::imread(DIR + "img_with_multiplicative_noise.png"), cv::imread(DIR + "img_with_quant_noise.png")};
    std::vector<int> coeffs{-1, -1, 2, 5, -2, -1, -1};
    std::vector<std::string> filenames = {"img_with_additive_noise.png", "img_with_gauss_noise.png", "img_with_impulse_noise_0sp.png", "img_with_impulse_noise_05sp.png", "img_with_impulse_noise_1sp.png", "img_with_multiplicative_noise.png","img_with_quant_noise.png"};
    std::cout << "INFORMATION";
    apply_and_save_pack_contrharmonic_filter(noise_imgs, filenames, coeffs);
}

double localize_pix(cv::Mat img, int pix_coords[2], int w_size[2], int q){
    int n = w_size[0], m = w_size[1];
    double sum = 0;
    for (int j = -n/2; j < n/2; j++){
        for (int i = -m/2; i < m/2; i++){
            int current_coord_y = pix_coords[0] + j;
            int current_coord_x = pix_coords[1] + i; 
            if ((current_coord_x >= 0) && (current_coord_x < img.cols) && (current_coord_y >= 0) && (current_coord_y < img.rows)){
                sum += std::pow(img.at<float>(current_coord_y, current_coord_x), q);
            }
        }
    }
    return sum;
}

cv::Mat arithmetic_averaging_filtration(cv::Mat img){
    cv::Mat out;
    cv::blur(img, out, cv::Size(3, 3));
    return out;
}

cv::Mat contrharmonic_filtration(cv::Mat img, int q){
    cv::Mat new_img;
    auto depth = img.depth();
    if (depth == CV_8U){
        img.convertTo(new_img, CV_32F, 1.0 / 255);
    }
    else{
        img.copyTo(new_img);
    }
    if (depth == CV_8U){
        new_img.convertTo(new_img, CV_8U, 255);
    }
    cv::imwrite("/home/den/CV_labs/Lab3/images/1.png", new_img);
    for (int y = 0; y < new_img.rows; y++){
        float* new_row_ptr = new_img.ptr<float>(y);
        for (int x = 0; x < new_img.cols; x++){
            int pix_cord[2]{y, x};
            int w_size[2]{3, 3};
            new_row_ptr[x] = (localize_pix(img, pix_cord, w_size, q+1)) / (localize_pix(img, pix_cord, w_size, q));
        }
    }

    if (depth == CV_8U){
        new_img.convertTo(new_img, CV_8U, 255);
    }
    return new_img;
}

void apply_and_save_pack_contrharmonic_filter(std::vector<cv::Mat> images_to_filter, std::vector<std::string> filenames, std::vector<int> coeffs_q){
    cv::Mat img_copy;
    for (int i = 0; i < images_to_filter.size(); i++){
        images_to_filter[i].copyTo(img_copy);
        cv::imwrite("/home/den/CV_labs/Lab3/images/outputs/low_freq_filters/contrharmonic_" + filenames[i], contrharmonic_filtration(img_copy, coeffs_q[i]));
    }
}

