#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include<cmath>

std::vector<cv::Mat> calc_hists(cv::Mat img);

cv::Mat draw_hist(std::vector<cv::Mat> hist_data, int histSize);

cv::Mat calc_chist(cv::Mat img_channel);

cv::Mat hist_replace(cv::Mat img, int dist);

cv::Mat nonlinear_stretch_lut(double img_min, double img_max, double alfa);

cv::Mat nonlinear_stretch_transformation(cv::Mat img, double alfa);

cv::Mat uniform_lut(cv::Mat c_hist, double img_min, double img_max);

cv::Mat uniform_transformation(cv::Mat img, double alfa);

cv::Mat exp_lut(cv::Mat img, cv::Mat hist, double img_min, double img_max, double alfa);

cv::Mat exp_transformation(cv::Mat img, double alfa);

cv::Mat rayleigh_lut(cv::Mat c_hist, double img_min, double img_max, double alfa);

cv::Mat rayleigh_transformation(cv::Mat img, double alfa);

cv::Mat two_thirds_lut(cv::Mat c_hist); 

cv::Mat two_thirds_transformation(cv::Mat img);

cv::Mat hyperbolic_lut(cv::Mat c_hist, double alfa);

cv::Mat hyperbolic_transformation(cv::Mat, double alfa);

cv::Mat equalizeHist(cv::Mat img);

cv::Mat CLAHE_transform(cv::Mat img);

void apply_alfa_and_save(cv::Mat(*)(cv::Mat, double), cv::Mat img, double alfa, std::string filename, bool make_hist, bool make_equalization, bool make_stretch);

void apply_and_save(cv::Mat(*)(cv::Mat), cv::Mat img, std::string filename, bool make_hist, bool make_equalization, bool make_stretch);



int main() {
    const std::string PATH = "C:/Users/gridd/Downloads/asoka.jpg";
    cv::Mat img = cv::imread(PATH, cv::IMREAD_COLOR);
    cv::Mat new_img = hist_replace(img, 15);
    cv::imshow("img_shifted", new_img);
    cv::imwrite("D:/cv/lab1/img_shifted_hists.jpg", draw_hist(calc_hists(new_img), 256));
    cv::imwrite("D:/cv/lab1/img_hist_replacement.jpg", new_img);
    apply_alfa_and_save(nonlinear_stretch_transformation, img, 0.5, "nonlinear_stretch_transformation.jpg", false, false, false);
    apply_alfa_and_save(exp_transformation, img, 0.03, "exp_transformation.jpg", false, false, false);
    apply_alfa_and_save(uniform_transformation, img, 0.5, "uniform_transformation.jpg", false, false, false);
    apply_alfa_and_save(rayleigh_transformation, img, 8, "rayleigh_transformation.jpg", false, true, false);
    apply_and_save(two_thirds_transformation, img, "two_thirds_transformation.jpg", false, false, false);
    apply_alfa_and_save(hyperbolic_transformation, img, 5, "hyperbolic_transformation_with_hist_stretched011.jpg", false, true, true);
    apply_and_save(equalizeHist, img, "equalizedHist.jpg", false, false, false);
    apply_and_save(CLAHE_transform, img, "CLAHE_transformed.jpg", false, false, false);

    cv::waitKey(0);

}

std::vector<cv::Mat> calc_hists(cv::Mat img) {
    std::vector<cv::Mat> bgr_planes;
    split(img, bgr_planes);
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
    cv::Mat b_hist, g_hist, r_hist;
    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate);

    return std::vector<cv::Mat> {b_hist, g_hist, r_hist};

}

cv::Mat calc_chist(cv::Mat img_channel) {
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
    cv::Mat chist;
    cv::calcHist(&img_channel, 1, 0, cv::Mat(), chist, 1, &histSize, histRange, uniform, accumulate); 

  

    for (int i = 1; i < chist.size[0]; ++i) {
        chist.at<float>(i) += chist.at<float>(i - 1);
    }

    for (int i = 0; i < chist.size[0]; ++i) {
        chist.at<float>(i) /= img_channel.size[0] * img_channel.size[1];
    }

    return chist;

}

cv::Mat draw_hist(std::vector<cv::Mat> hist_data, int histSize) {
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat b_hist = hist_data[0], g_hist = hist_data[1], r_hist = hist_data[2];
    normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    for (int i = 1; i < histSize; i++)
    {
        line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            cv::Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
            cv::Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
            cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    return histImage;
}

cv::Mat hist_replace(cv::Mat img, int dist) {
    for (int row = 0; row < img.rows; row++) {
        cv::Vec3b* ptr = img.ptr<cv::Vec3b>(row);
        for (int col = 0; col < img.cols; col++) {
            ptr[col] = cv::Vec3b(ptr[col][0] + dist, ptr[col][1] + dist, ptr[col][2] + dist);
        }
    }
    return img;
}

cv::Mat nonlinear_stretch_lut(double img_min, double img_max, double alfa) {
    cv::Mat lut = cv::Mat(1, 256, CV_8U);

    uchar* lut_ptr = lut.ptr();

    for (int i = 0; i < 256; i++) {
        double var = (i - img_min) / (img_max - img_min);
        if (var < 0) {
            lut_ptr[i] = 0;
        }
        else {
            lut_ptr[i] = cv::saturate_cast<uchar>(255 * std::pow(var, alfa));
        }
    }
    return lut;
}

cv::Mat nonlinear_stretch_transformation(cv::Mat img, double alfa) {
    cv::Mat img_new;
    std::vector<cv::Mat> img_BGR;
    cv::split(img, img_BGR);

    for (int channel = 0; channel < img_BGR.size(); channel++) {
        img_new = cv::Mat(img_BGR[channel].size(), img_BGR[channel].type());
        double img_max, img_min;
        cv::minMaxLoc(img_BGR[channel], &img_min, &img_max);
        cv::LUT(img_BGR[channel], nonlinear_stretch_lut(img_min, img_max, alfa), img_new);
        img_BGR[channel] = img_new;
    }

    cv::merge(img_BGR, img_new);
    return img_new;
}

cv::Mat uniform_lut(cv::Mat c_hist, double img_min, double img_max) {
    cv::Mat lut = cv::Mat(1, 256, CV_8U);
    uchar* lut_ptr = lut.ptr();

    for (int i = 0; i < 256; i++) {
        double var = (img_max - img_min) * c_hist.at<float>(i) + img_min;
        if (var < 0) {
            lut_ptr[i] = 0;
        }
        else {
            lut_ptr[i] = cv::saturate_cast<uchar>(var);
        }
    }

    return lut;
}

cv::Mat uniform_transformation(cv::Mat img, double alfa) {
    cv::Mat img_new;
    std::vector<cv::Mat> img_BGR;

    cv::split(img, img_BGR);

    for (int channel = 0; channel < img_BGR.size(); channel++) {
        img_new = cv::Mat(img_BGR[channel].size(), img_BGR[channel].type());
        cv::Mat c_hist = calc_chist(img_BGR[channel]);
        double img_min, img_max;
        cv::minMaxLoc(img_BGR[channel], &img_min, &img_max);
        cv::LUT(img_BGR[channel], uniform_lut(c_hist, img_min, img_max), img_new);
        img_BGR[channel] = img_new;
    }

    cv::merge(img_BGR, img_new);

    return img_new;
}

cv::Mat exp_lut(cv::Mat img, cv::Mat hist, double img_min, double img_max, double alfa) {
    cv::Mat lut = cv::Mat(1, 256, CV_8U);
    uchar* lut_ptr = lut.ptr();

    for (int i = 0; i < 256; i++) {
        double var = img_min - (1.0 / alfa) * std::log(1 - hist.at<float>(i));
        if (var < 0) {
            lut_ptr[i] = 0;
        }
        else {
            lut_ptr[i] = cv::saturate_cast<uchar>(var);
        }

    }

    return lut;
}

cv::Mat exp_transformation(cv::Mat img, double alfa) {
    cv::Mat img_new;
    std::vector<cv::Mat> img_BGR;
    cv::split(img, img_BGR);
    for (int channel = 0; channel < img_BGR.size(); channel++) {
        cv::Mat chist = calc_chist(img_BGR[channel]);
        double img_min, img_max;
        cv::minMaxLoc(img_BGR[channel], &img_min, &img_max);
        cv::Mat img_new = cv::Mat(img_BGR[channel].rows, img_BGR[channel].cols, img_BGR[channel].depth());
        cv::LUT(img_BGR[channel], exp_lut(img_BGR[channel], chist, img_min, img_max, alfa), img_new);
        img_BGR[channel] = img_new;
    }

    cv::merge(img_BGR, img_new);
    return img_new;
}

cv::Mat rayleigh_lut(cv::Mat c_hist, double img_min, double img_max, double alfa) {
    cv::Mat lut = cv::Mat(1, 256, CV_8U);
    uchar* lut_ptr = lut.ptr();

    for (int i = 0; i < 256; i++) {
        double var = img_min + std::sqrt(2 * alfa * alfa * std::log(1.0/(1 - c_hist.at<float>(i))));
        if (var < 0) {
            lut_ptr[i] = 0;
        }
        else {
            lut_ptr[i] = cv::saturate_cast<uchar>(var);
        }
    }
    return lut;
}

cv::Mat rayleigh_transformation(cv::Mat img, double alfa) {
    cv::Mat img_new;
    std::vector<cv::Mat> img_BGR;
    cv::split(img, img_BGR);
    for (int channel = 0; channel < img_BGR.size(); channel++) {
        img_new = cv::Mat(img_BGR[channel].rows, img_BGR[channel].cols, img_BGR[channel].depth());
        cv::Mat chist = calc_chist(img_BGR[channel]);
        double img_min, img_max;

        cv::minMaxLoc(img_BGR[channel], &img_min, &img_max);
        cv::LUT(img_BGR[channel], rayleigh_lut(chist, img_min, img_max, alfa), img_new);
        img_BGR[channel] = img_new;
    }

    cv::merge(img_BGR, img_new);
    return img_new;
}

cv::Mat two_thirds_lut(cv::Mat c_hist) {
    cv::Mat lut = cv::Mat(1, 256, CV_8U);
    uchar* lut_ptr = lut.ptr();

    for (int i = 0; i < 256; i++) {
        double var = std::pow(c_hist.at<float>(i), 2.0 / 3.0);
        //std::cout << "LOG - " << var << "\n";
        
        if (var == 0) {
            lut_ptr[i] = 0;
        }
        else {
            lut_ptr[i] = cv::saturate_cast<uchar>(255 * var);
        }
    }

    return lut;
}

cv::Mat two_thirds_transformation(cv::Mat img) {
    std::vector<cv::Mat> img_BGR;
    cv::Mat img_new;

    cv::split(img, img_BGR);

    for (int channel = 0; channel < img_BGR.size(); channel++) {
        img_new = cv::Mat(img_BGR[channel].size(), img_BGR[channel].type());
        cv::Mat c_hist = calc_chist(img_BGR[channel]);
        cv::LUT(img_BGR[channel], two_thirds_lut(c_hist), img_new);
        img_BGR[channel] = img_new;
    }

    cv::merge(img_BGR, img_new);
    return img_new;
}

cv::Mat hyperbolic_lut(cv::Mat c_hist, double alfa) {
    cv::Mat lut = cv::Mat(1, 256, CV_8U);
    uchar* lut_ptr = lut.ptr();

    for (int i = 0; i < 256; i++) {
        double var = std::pow(alfa, c_hist.at<float>(i));
        if (var == 0) {
            lut_ptr[i] = 0;
        }
        else {
            lut_ptr[i] = cv::saturate_cast<uchar>(var);
        }
    }
    return lut;
}

cv::Mat hyperbolic_transformation(cv::Mat img, double alfa) {
    cv::Mat img_new;
    std::vector<cv::Mat> img_BGR;

    cv::split(img, img_BGR);

    for (int channel = 0; channel < img_BGR.size(); channel++) {
        img_new = cv::Mat(img_BGR[channel].size(), img_BGR[channel].type());
        cv::Mat c_hist = calc_chist(img_BGR[channel]);
        cv::LUT(img_BGR[channel], hyperbolic_lut(c_hist, alfa), img_new);
        img_BGR[channel] = img_new;
    }

    cv::merge(img_BGR, img_new);
    return img_new;
}

cv::Mat equalizeHist(cv::Mat img) {
    std::vector<cv::Mat> img_BGR;

    cv::split(img, img_BGR);
    for (int channel = 0; channel < img_BGR.size(); channel++) {
        cv::Mat img_new; //= cv::Mat(img_BGR[channel].size(), img_BGR[channel].type());
        cv::equalizeHist(img_BGR[channel], img_new);
        img_BGR[channel] = img_new;
    }

    cv::Mat img_new;
    cv::merge(img_BGR, img_new);

    return img_new;
}

cv::Mat CLAHE_transform(cv::Mat img) {
    std::vector<cv::Mat> img_BGR;
    cv::split(img, img_BGR);
    for (int channel = 0; channel < img_BGR.size(); channel++) {
        cv::Mat img_new;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2);
        clahe -> apply(img_BGR[channel], img_new);
        img_BGR[channel] = img_new;
    }

    cv::Mat img_new;
    cv::merge(img_BGR, img_new);
    return img_new;
}

void apply_alfa_and_save(cv::Mat(*filter)(cv::Mat, double), cv::Mat img, double alfa, std::string filename, bool make_hist, bool make_equalization, bool make_stretch) {
    const std::string DIR = "D:/cv/lab1/";
    cv::Mat filtered_img = filter(img, alfa);
    std::vector<cv::Mat> hists = calc_hists(filtered_img);
    cv::Mat hist_image = draw_hist(hists, 256);
    if (make_equalization) {
        filtered_img = equalizeHist(filtered_img);
    }
    if (make_hist) {
        filtered_img = hist_replace(filtered_img, 80);
    }
    if (make_stretch) {
        filtered_img = nonlinear_stretch_transformation(filtered_img, 2);
    }

    std::cout << '\n'<< DIR + filename << ".jpg" << "\n";
    cv::imwrite(DIR + filename, filtered_img);
    cv::imwrite(DIR + "hists_" + filename, hist_image);

}

void apply_and_save(cv::Mat(*filter)(cv::Mat), cv::Mat img, std::string filename, bool make_hist, bool make_equalization, bool make_stretch) {
    const std::string DIR = "D:/cv/lab1/";
    cv::Mat filtered_img = filter(img);
    cv::Mat final_image;
    std::vector<cv::Mat> hists = calc_hists(filtered_img);
    cv::Mat hist_image = draw_hist(hists, 256);
    if (make_equalization) {
        filtered_img = equalizeHist(filtered_img);
    }
    if (make_hist) {
        filtered_img = hist_replace(filtered_img, 180);
    }
    if (make_stretch) {
        filtered_img = nonlinear_stretch_transformation(filtered_img, 0.5);
    }
    cv::imwrite(DIR + filename, filtered_img);
    cv::imwrite(DIR + "hists_" + filename, hist_image);
   
}