#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<cmath>

cv::Mat skin_tone_segmentation(cv::Mat img);

int main(){
    std::string ORIGINAL_DIR = "/home/den/CV_labs/Lab4/img/original/";
    std::string RES_DIR = "/home/den/CV_labs/Lab4/img/outputs/segmentation/";
    cv::Mat img = cv::imread(ORIGINAL_DIR + "face2.png");
   
    cv::Mat img_copy;
    img.copyTo(img_copy);
    cv::imwrite(RES_DIR + "skin_tone_segmentation.png", skin_tone_segmentation(img_copy));
    return 0;
}


cv::Mat skin_tone_segmentation(cv::Mat img){
    for (int row = 0; row < img.rows; row++){
        cv::Vec3b* ptr = img.ptr<cv::Vec3b>(row);
        for (int col = 0; col < img.cols; col++){
            uchar B = ptr[col][0], G = ptr[col][1], R = ptr[col][2];
            double r = 1.0 * R/(R+G+B), g = 1.0 * G/(R+G+B), b = 1.0 *B/(R+G+B);
            if (((r / g) > 1.185) && ((r * b)/(std::pow((r + g + b), 2)) > 0.107) && ((r * g)/(std::pow((r + g + b), 2)) > 0.112)){
                ptr[col] = cv::Vec3b(0, 255, 0);
            }
        }
    }
    return img;
}

