#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<cmath>

cv::Mat Roberts_filter(cv::Mat img);

cv::Mat Previt_filter(cv::Mat img);

cv::Mat Sobel_filter(cv::Mat img);

cv::Mat Laplace_filter(cv::Mat img);

cv::Mat Canny_filter(cv::Mat img);

void apply_and_save_pack_high_filter(cv::Mat(*filter)(cv::Mat), std::string filtername, std::vector<cv::Mat> images_to_filter, std::vector<std::string> filenames);

// int main(){
//     cv::Mat img = cv::imread("/home/den/CV_labs/Lab3/images/original/RQC.png", cv::IMREAD_GRAYSCALE);
//     cv::Mat img_copy;
//     img.copyTo(img_copy);
//     cv::imwrite("/home/den/CV_labs/Lab3/images/outputs/high_freq_filters/RQC_Roberts_filtered.png", Roberts_filter(img_copy));

//     img.copyTo(img_copy);
//     cv::imwrite("/home/den/CV_labs/Lab3/images/outputs/high_freq_filters/RQC_Previt_filtered.png", Previt_filter(img_copy));

//     img.copyTo(img_copy);
//     cv::imwrite("/home/den/CV_labs/Lab3/images/outputs/high_freq_filters/RQC_Sobel_filtered.png", Sobel_filter(img_copy));

//     img.copyTo(img_copy);
//     cv::imwrite("/home/den/CV_labs/Lab3/images/outputs/high_freq_filters/RQC_Laplace_filtered.png", Laplace_filter(img_copy));

//     img.copyTo(img_copy);
//     cv::imwrite("/home/den/CV_labs/Lab3/images/outputs/high_freq_filters/RQC_Canny_filtered.png", Canny_filter(img_copy));

//     return 0;
// }

cv::Mat Roberts_filter(cv::Mat img){
    if (img.depth() == CV_8U){
        img.convertTo(img, CV_32F, 1.0 / 255);
    }
    cv::Mat G_x = (cv::Mat_<double>(2, 2) << -1, 1, 0, 0);
    cv::Mat G_y = (cv::Mat_<double>(2, 2) << 1, 0, -1, 0);
    cv::Mat img_x, img_y, img_out;
    cv::filter2D(img, img_x, -1, G_x);
    cv::filter2D(img, img_y, -1, G_y);
    cv::magnitude(img_x, img_y, img);

    if(img.depth() == CV_32F){
        img.convertTo(img, CV_8U, 255);
    }
    return img;

}

cv::Mat Previt_filter(cv::Mat img){
    if (img.depth() == CV_8U){
        img.convertTo(img, CV_32F, 1.0 / 255);
    }
    cv::Mat G_x = (cv::Mat_<double>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    cv::Mat G_y = (cv::Mat_<double>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
    cv::Mat img_x, img_y, img_out;
    cv::filter2D(img, img_x, -1, G_x);
    cv::filter2D(img, img_y, -1, G_y);
    cv::magnitude(img_x, img_y, img);

    if(img.depth() == CV_32F){
        img.convertTo(img, CV_8U, 255);
    }
    return img;
}


cv::Mat Sobel_filter(cv::Mat img){
    if (img.depth() == CV_8U){
        img.convertTo(img, CV_32F, 1.0 / 255);
    }
    cv::Mat G_x = (cv::Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat G_y = (cv::Mat_<double>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    cv::Mat img_x, img_y, img_out;
    cv::filter2D(img, img_x, -1, G_x);
    cv::filter2D(img, img_y, -1, G_y);
    cv::magnitude(img_x, img_y, img);

    if(img.depth() == CV_32F){
        img.convertTo(img, CV_8U, 255);
    }
    return img;
}


cv::Mat Laplace_filter(cv::Mat img){
    if (img.depth() == CV_8U){
        img.convertTo(img, CV_32F, 1.0 / 255);
    }
    cv::Mat w = (cv::Mat_<double>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
    cv::filter2D(img, img, -1, w);

    if(img.depth() == CV_32F){
        img.convertTo(img, CV_8U, 255);
    }
    return img;
}

cv::Mat Canny_filter(cv::Mat img){
    cv::Mat copy;
    if (img.depth() == CV_8U){
        img.convertTo(copy, CV_16F, 1.0 / 255);
    }
    else{
        img.copyTo(copy);
    }

    cv::Canny(img, copy, 73, 81);
    if(img.depth() == CV_8U){
        copy.convertTo(copy, CV_8U, 255);
    }
    return copy;
}


void apply_and_save_pack_high_filter(cv::Mat(*filter)(cv::Mat), std::string filtername, std::vector<cv::Mat> images_to_filter, std::vector<std::string> filenames){
    cv::Mat img_copy;
    for (int i = 0; i < images_to_filter.size(); i++){
        images_to_filter[i].copyTo(img_copy);
        cv::imwrite("/home/den/CV_labs/Lab3/images/results/" + filtername + "_" + filenames[i], filter(img_copy));
    }
}