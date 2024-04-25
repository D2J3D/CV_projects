#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<cmath>

void apply_and_save_pack_low_filter(cv::Mat(*low_filter)(cv::Mat), std::string filter, std::vector<cv::Mat> images_to_filter, std::vector<std::string> filenames);

cv::Mat weighted_median_filtration(cv::Mat img);

cv::Mat rank_filtration(cv::Mat img);

cv::Mat wiener_filtration(cv::Mat img);

cv::Mat median_filtration(cv::Mat img);

double window_resize(cv::Mat img, int row, int col, int kernel_size, int max_s);

cv::Mat adapt_filtration(cv::Mat img);

int main(){
    const std::string DIR = "/home/den/CV_labs/Lab3/images/outputs/noises/";
    std::vector<cv::Mat> noise_imgs{cv::imread(DIR + "img_with_additive_noise.png", cv::IMREAD_GRAYSCALE), cv::imread(DIR + "img_with_gauss_noise.png", cv::IMREAD_GRAYSCALE), cv::imread(DIR + "img_with_impulse_noise_0sp.png", cv::IMREAD_GRAYSCALE), cv::imread(DIR + "img_with_impulse_noise_1sp.png", cv::IMREAD_GRAYSCALE), cv::imread(DIR + "img_with_impulse_noise_05sp.png", cv::IMREAD_GRAYSCALE), cv::imread(DIR + "img_with_multiplicative_noise.png", cv::IMREAD_GRAYSCALE), cv::imread(DIR + "img_with_quant_noise.png", cv::IMREAD_GRAYSCALE)};
    std::vector<int> coeffs{0, 3, 2, -22, -5, -5, -3};
    std::vector<std::string> filenames = {"img_with_additive_noise.png", "img_with_gauss_noise.png", "img_with_impulse_noise_0sp.png", "img_with_impulse_noise_1sp.png", "img_with_impulse_noise_05sp.png", "img_with_multiplicative_noise.png", "img_with_quant_noise.png"};
    
    apply_and_save_pack_low_filter(rank_filtration, "rank", noise_imgs, filenames);
    apply_and_save_pack_low_filter(rank_filtration, "rank_", noise_imgs, filenames);
    apply_and_save_pack_low_filter(wiener_filtration, "wiener_filter_", noise_imgs, filenames);
    apply_and_save_pack_low_filter(median_filtration, "median_", noise_imgs, filenames);
    return 0;
}

void apply_and_save_pack_low_filter(cv::Mat(*low_filter)(cv::Mat), std::string filter, std::vector<cv::Mat> images_to_filter, std::vector<std::string> filenames){
    cv::Mat img_copy;
    for (int i = 0; i < images_to_filter.size(); i++){
        images_to_filter[i].copyTo(img_copy);
        cv::imwrite("/home/den/CV_labs/Lab3/images/outputs/nonlinear_filters/" + filter + "_" + filenames[i], low_filter(img_copy));
    }
}

cv::Mat gauss_filtration(cv::Mat img, cv::Size kernel_size){
    cv::Mat out;
    cv::GaussianBlur(img, out, kernel_size, 0);
    return out;
}

cv::Mat weighted_median_filtration(cv::Mat img){
    int k_size[] = {3, 3}; 
    cv::Mat kernel = (cv::Mat_<double>(3, 3) << 2, 1, 3, 2, 1, 1, 3, 1, 2);
    int rank = 4;
    cv::Mat img_copy;
    if (img.depth() == CV_8U){
        img.convertTo(img_copy, CV_32F, 1.0 / 255);
    }
    else{
        img.copyTo(img_copy);
    }

    cv::copyMakeBorder(img_copy, img_copy, int((k_size[0] - 1) / 2), int((k_size[0] / 2)), int((k_size[1] - 1) / 2 ), int(k_size[1] / 2), cv::BORDER_REPLICATE);
    
    std::vector<cv::Mat> img_BGR;
    cv::split(img_copy, img_BGR);

    for (int channel = 0; channel < img_BGR.size(); channel++){
        cv::Mat img_tmp = cv::Mat::zeros(img_BGR[channel].size(), img_BGR[channel].type());
        std::vector<double> c;
        c.reserve(k_size[0] * k_size[1]);
        for (int i = 0; i < img_BGR[channel].rows; i++){
            for (int j = 0; j < img_BGR[channel].cols; j++){
                c.clear();
                for (int a = 0; a < k_size[0]; a++){
                    for (int b = 0; b < k_size[1]; b++){
                        for (int k = 0; k < kernel.at<double>(a,b); k++){
                            c.push_back(img_BGR[channel].at<float>(i + a, j + b));
                        }
                    } 
                }
                std::sort(c.begin(), c.end());
                img_tmp.at<float>(i, j) = float(c[rank]);
            }
        }
        img_BGR[channel] = img_tmp;
    }

    cv::Mat img_out;
    cv::merge(img_BGR, img_out);

    if (img.depth() == CV_8U){
        img_out.convertTo(img_out, CV_8U, 255);
    }

    return img_out;
}

cv::Mat rank_filtration(cv::Mat img){
    int k_size[] = {3, 3}; 
    cv::Mat kernel = cv::Mat::ones(cv::Size(k_size[0], k_size[1]), CV_64F);
    int rank = 9;
    cv::Mat img_copy;
    if (img.depth() == CV_8U){
        img.convertTo(img_copy, CV_32F, 1.0 / 255);
    }
    else{
        img.copyTo(img_copy);
    }

    cv::copyMakeBorder(img_copy, img_copy, int((k_size[0] - 1) / 2), int((k_size[0] / 2)), int((k_size[1] - 1) / 2 ), int(k_size[1] / 2), cv::BORDER_REPLICATE);
    
    std::vector<cv::Mat> img_BGR;
    cv::split(img_copy, img_BGR);

    for (int channel = 0; channel < img_BGR.size(); channel++){
        cv::Mat img_tmp = cv::Mat::zeros(img_BGR[channel].size(), img_BGR[channel].type());
        std::vector<double> c;
        c.reserve(k_size[0] * k_size[1]);
        for (int i = 0; i < img_BGR[channel].rows; i++){
            for (int j = 0; j < img_BGR[channel].cols; j++){
                c.clear();
                for (int a = 0; a < k_size[0]; a++){
                    for (int b = 0; b < k_size[1]; b++){
                        c.push_back(img_BGR[channel].at<float>(i + a, j + b));
                    } 
                }
                std::sort(c.begin(), c.end());
                img_tmp.at<float>(i, j) = float(c[rank - 1]);
            }
        }
        img_BGR[channel] = img_tmp;
    }

    cv::Mat img_out;
    cv::merge(img_BGR, img_out);

    if (img.depth() == CV_8U){
        img_out.convertTo(img_out, CV_8U, 255);
    }

    return img_out;
}


cv::Mat wiener_filtration(cv::Mat img){
    int k_size[] = {5, 5};
    cv::Mat kernel = cv::Mat::ones(k_size[0], k_size[1], CV_64F);
    double k_sum = cv::sum(kernel)[0];
    cv::Mat img_copy;
    if (img.depth() == CV_8U){
        img.convertTo(img_copy, CV_32F, 1.0 / 255);
    }
    else{
        img.copyTo(img_copy);
    }
       cv::copyMakeBorder(img_copy, img_copy, int((k_size[0] - 1) / 2), int((k_size[0] / 2)), int((k_size[1] - 1) / 2 ), int(k_size[1] / 2), cv::BORDER_REPLICATE);
       std::vector<cv::Mat> img_BGR;
       cv::split(img_copy, img_BGR);
       for (int channel = 0; channel < img_BGR.size(); channel++){
        cv::Mat img_tmp = cv::Mat::zeros(img.size(), img_BGR[channel].type());
        double v(0);
        for (int i = 0; i < img.rows; i++){
            for (int j =0; j < img.cols; j++){
                double m(0), q(0);
                for (int a = 0; a  < k_size[0]; a++){
                    for (int b = 0; b < k_size[1]; b++){
                        double t = img_BGR[channel].at<float>(i + a, j + b) * kernel.at<double>(a, b);
                        m += t;
                        q += t * t;

                    }
                }
                m /= k_sum;
                q /= k_sum;
                q -= m * m;
                v += q;
            }
        }
        v /= img.cols * img.rows;
        for (int i = 0; i < img.rows; i++){
            for (int j = 0; j < img.cols; j++){
                double m(0), q(0);
                for (int a = 0; a < k_size[0]; a++){
                    for (int b = 0; b < k_size[1]; b++){
                        double t = img_BGR[channel].at<float>(i + a, j + b) * kernel.at<double>(a, b);
                        m += t;
                        q += t * t;
                    }
                }
                m /= k_sum;
                q /= k_sum;
                q -= m * m;
                double im  = img_BGR[channel].at<float>(i + (k_size[0] - 1) / 2, j + (k_size[1] - 1)/2);
                if (q < v){
                    img_tmp.at<float>(i, j) = float(m);

                }
                else{
                    img_tmp.at<float>(i, j) = float((im - m) * (1 - v / q) + m);
                }
            }
        }
        img_BGR[channel] = img_tmp;
    }
    cv::Mat img_out;
    cv::merge(img_BGR, img_out);
    if (img.depth() == CV_8U){
        img_out.convertTo(img_out, CV_8U, 255);
    }
    return img_out;
}

cv::Mat median_filtration(cv::Mat img){
    cv::Mat img_out;
    cv::medianBlur(img, img_out, 3);
    return img_out;
}

double median_intensity_calc(cv::Mat img){
    std::vector<double> img_intesities; 
    for (int i = 0; i < img.rows; i++){
        for (int j = 0; j < img.cols; j++){
            img_intesities.push_back(img.at<float>(i, j));
        }
    }
    std::sort(img_intesities.begin(), img_intesities.end());
    double z_med;
    if (img_intesities.size() % 2 == 0) {
        z_med = (img_intesities[img_intesities.size() / 2 - 1] + img_intesities[img_intesities.size() / 2]) / 2;
    } 
    else {
        z_med = img_intesities[img_intesities.size() / 2];
    }
    return z_med;
}

std::vector<double> calc_parameters(cv::Mat img, int pix_coords[2], int s){
    double z_max = 0, z_min = 0, z_med = 0;
    double sum = 0;
    double intensity = 0;
    std::vector<double> img_intesities; 
    for (int j = -s/2; j < s/2; j++){
        for (int i = -s/2; i < s/2; i++){
            int current_coord_y = pix_coords[0] + j;
            int current_coord_x = pix_coords[1] + i; 
            if ((current_coord_x >= 0) && (current_coord_x < img.cols) && (current_coord_y >= 0) && (current_coord_y < img.rows)){
                intensity = img.at<float>(current_coord_y, current_coord_x);
                if (intensity > z_max){
                    z_max = intensity;
                }
                if (intensity <= z_min){
                    z_min = intensity;
                }
                img_intesities.push_back(intensity);
            }
        }
    }
    if (img_intesities.size() % 2 == 0) {
        z_med = (img_intesities[img_intesities.size() / 2 - 1] + img_intesities[img_intesities.size() / 2]) / 2;
    } 
    else {
        z_med = img_intesities[img_intesities.size() / 2];
    }
    std::vector<double> parameters = {z_max, z_min, z_med}; 
    return parameters;
}

double window_resize(cv::Mat img, int row, int col, int kernel_size, int max_size){
    std::vector<float> img_intesities; 
    int wsize = kernel_size / 2;
    for (int i = -wsize; i < wsize; i++){
        for (int j = -wsize; j < wsize ; j++){
            int current_row = row + i;
            int current_col = col + j; 
            img_intesities.push_back(img.at<float>(current_row, current_col));
        }
    }
    std::sort(img_intesities.begin(), img_intesities.end());
    float z = img.at<float>(row, col);
    float z_min = img_intesities[0];
    float z_max = img_intesities[img_intesities.size()-1];
    float z_med = img_intesities[(int) (img_intesities.size() / 2)];
    double A1, A2, B1, B2;
    A1 = z_med - z_min;
    A2 = z_med - z_max;

    if (A1 > 0 && A2 < 0){
        B1 = z - z_min;
        B2 = z - z_max;
        if (B1 > 0 && B2 < 0){
            return z;
        }
        else{
            return z_med;
        }
    }
    else{
        if (wsize < max_size){
            return window_resize(img, row, col, kernel_size + 2, max_size);
        }
        else{
            return z_med;
        }
    }


}

cv::Mat adapt_filtration(cv::Mat img){
    int kernel_size = 3, max_size = 8;
    cv::Mat img_copy;
    if (img.depth() == CV_8U){
        img.convertTo(img_copy, CV_32F, 1.0 / 255);
    }
    else{
        img.copyTo(img_copy);
    }
    std::vector<cv::Mat> img_BGR;
    cv::split(img_copy, img_BGR);
    for (int channel = 0; channel < img_BGR.size(); channel++){
        for (int i = max_size; i < img.rows - max_size; i++){
            float* row_ptr = img.ptr<float>(i);
            for (int j = max_size; j < img.cols - max_size; j++){
                row_ptr[j] = window_resize(img_BGR[channel], i, j, kernel_size, max_size);
            }
        }
    }
    cv::Mat img_out;
    cv::merge(img_BGR, img_out);
    if (img.depth() == CV_8U){
        img_out.convertTo(img_out, CV_8U, 255);
    }
    return img_out;
}
