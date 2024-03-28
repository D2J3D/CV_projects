#include<iostream>
#include<cmath>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>


cv::Mat img_shift(cv::Mat img, int x_hist, int y_hist);

cv::Mat img_reflection(cv::Mat img);

cv::Mat uniform_resize(cv::Mat img, double coeff);

cv::Mat img_rotate(cv::Mat, double phi);

cv::Mat img_centered_rotate(cv::Mat, double phi);

cv::Mat affine_resize(cv::Mat img);

cv::Mat img_bevel(cv::Mat img, double s);

cv::Mat uniform_ROI_transform(cv::Mat img, double s);

cv::Mat projective_transform(cv::Mat img, double params[9]);

cv::Mat polinomial_transformation(cv::Mat img, double T[2][6]);

cv::Mat sin_transformation(cv::Mat img, double s);

int main() {
    const std::string PATH = "/home/den/CV_labs/Lab2/images/original/g_disney_talesofthejedi_870_05_47cd6d1c.jpeg";
    cv::Mat img = cv::imread(PATH, cv::IMREAD_COLOR);
    cv::Mat img_copy;
    // Сдвиг изображения
    img.copyTo(img_copy);
    cv::imwrite("/home/den/CV_labs/Lab2/images/results/shift_img.jpg", img_shift(img_copy, 50, 100));

    //Отражение изображения
    img.copyTo(img_copy);
    cv::imwrite("/home/den/CV_labs/Lab2/images/results/img_reflection.jpg", img_reflection(img_copy));
    
    //Однородное масштабирование изображения
    img.copyTo(img_copy);
    cv::imwrite("/home/den/CV_labs/Lab2/images/results/img_uniform_resized.jpg", uniform_resize(img_copy, 20));
    
    // Поворот изображения
    img.copyTo(img_copy);
    cv::imwrite("/home/den/CV_labs/Lab2/images/results/img_rotation.jpg", img_rotate(img_copy, 14));

    // Поворот изображения вокруг его центра
    img.copyTo(img_copy);
    cv::imwrite("/home/den/CV_labs/Lab2/images/results/img_center_rotation.jpg", img_centered_rotate(img_copy, 30));

    // Аффинное отображение
    img.copyTo(img_copy);
    cv::imwrite("/home/den/CV_labs/Lab2/images/results/affine_resize.jpg", affine_resize(img_copy));

    // Скос изображения
    img.copyTo(img_copy);
    cv::imwrite("/home/den/CV_labs/Lab2/images/results/img_bevel.jpg", img_bevel(img_copy, 0.3));

    // Кусочно-линейные преобразование
    img.copyTo(img_copy);
    cv::imwrite("/home/den/CV_labs/Lab2/images/results/roi_transformed_img.jpg", uniform_ROI_transform(img_copy, 2));

    // Проекционное отображение
    img.copyTo(img_copy);
    double projective_coeffs[9] = {1.1, 0.35, 0, 0, 0.2, 1.1, 0.00075, 0.0005, 1};
    cv::imwrite("/home/den/CV_labs/Lab2/images/results/projective_transformed_image.jpg", projective_transform(img_copy, projective_coeffs));

    // Полиномиальное отображение
    img.copyTo(img_copy);
    double polinomial_coeffs[2][6] = {{0, 1, 0, 0.00001, 0.002, 0.002}, {0, 0, 1, 0, 0, 0}};
    cv::imwrite("/home/den/CV_labs/Lab2/images/results/polinomial_transformed_image.jpg", polinomial_transformation(img_copy, polinomial_coeffs));

    // Синусоидальное искажение
    img.copyTo(img_copy);
    cv::imwrite("/home/den/CV_labs/Lab2/images/results/sinusoidal_distortion_img.jpg", sin_transformation(img_copy, 9));

}

cv::Mat img_shift(cv::Mat img, int x_hist, int y_hist) {
    cv::Mat T = (cv::Mat_ <double>(2, 3) << 1, 0, x_hist, 0, 1, y_hist);
    cv::Mat img_shift;
    cv::warpAffine(img, img_shift, T, cv::Size(img.cols, img.rows));
    return img_shift;
}

cv::Mat img_reflection(cv::Mat img) {
    cv::Mat T = (cv::Mat_ <double>(2, 3) << 1, 0, 0, 0, -1, img.rows - 1);
    cv::Mat img_reflected;
    cv::warpAffine(img, img_reflected, T, cv::Size(img.cols, img.rows));
    return img_reflected;
}

cv::Mat uniform_resize(cv::Mat img, double coeff) {
    cv::Mat T = (cv::Mat_ <double>(2, 3) << coeff, 0, 0, 0, coeff, 0);
    cv::Mat img_resized;
    cv::warpAffine(img, img_resized, T, cv::Size(int(img.cols * coeff), int(img.rows * coeff)));
    return img_resized;
}

cv::Mat img_rotate(cv::Mat img, double phi) {
    double r_phi = phi * M_PI / 180.0;
    cv::Mat T = (cv::Mat_ <double>(2, 3) << std::cos(r_phi), -std::sin(r_phi), 0, std::sin(r_phi), std::cos(r_phi), 0);
    cv::Mat img_rotated;
    cv::warpAffine(img, img_rotated, T, cv::Size(img.cols, img.rows));

    return img_rotated;
}

cv::Mat img_centered_rotate(cv::Mat img, double phi) {
    double r_phi = phi * M_PI / 180.0;
    cv::Mat T2 = (cv::Mat_ <double>(3, 3) << std::cos(r_phi), -std::sin(r_phi), 0, std::sin(r_phi), std::cos(r_phi), 0, 0, 0, 1);
    cv::Mat T1 = (cv::Mat_ <double>(3, 3) << 1, 0, -(img.rows - 1) / 2.0, 0, 1, -(img.cols - 1) / 2.0, 0, 0, 1);
    cv::Mat T3 = (cv::Mat_ <double>(3, 3) << 1, 0, (img.rows - 1) / 2.0, 0, 1, (img.cols - 1) / 2.0, 0, 0, 1);

    cv::Mat T = cv::Mat(T3 * T2 * T1, cv::Rect(0, 0, 3, 2));
    cv::Mat img_rotated;
    cv::warpAffine(img, img_rotated, T, cv::Size(img.cols, img.rows));
    return img_rotated;
}

cv::Mat affine_resize(cv::Mat img) {
    std::vector <cv::Point2f> ptr_src = { {50, 100}, {350, 100}, {50, 400} };
    std::vector <cv::Point2f> ptr_dst = { {50, 150},{200, 150},{100, 400} };

    cv::Mat T = cv::getAffineTransform(ptr_src, ptr_dst);
    cv::Mat img_resized;
    cv::warpAffine(img, img_resized, T, cv::Size(img.cols, img.rows));

    return img_resized;
}

cv::Mat img_bevel(cv::Mat img, double s) {
    cv::Mat T = (cv::Mat_ <double>(2, 3) << 1, s, 0, 0, 1, 0);
    cv::Mat img_bevel;
    cv::warpAffine(img, img_bevel, T, cv::Size(img.cols, img.rows));
    return img_bevel;
}



cv::Mat uniform_ROI_transform(cv::Mat img, double s) {
    cv::Mat T = (cv::Mat_ <double>(2, 3) << s, 0, 0, 0, 1, 0);
    cv::Mat img_piece_linear = img.clone();
    cv::Mat img_right = img_piece_linear(cv::Rect(int(img.cols / 2.0), 0, img.cols - int(img.cols / 2.0), img.rows));
    cv::warpAffine(img_right, img_right, T, cv::Size(img.cols - int(img.cols / 2.0), img.rows));
    return img_piece_linear;
}

cv::Mat projective_transform(cv::Mat img, double params[9]) {
    // params[9] = A, B, C, D, E, F, G, I
    cv::Mat T = (cv::Mat_ <double>(3, 3) << params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]);
    cv::Mat img_projective;
    cv::warpPerspective(img, img_projective, T, cv::Size(img.cols, img.rows));
    return img_projective;
}

cv::Mat polinomial_transformation(cv::Mat img, double T[2][6]) {
    cv::Mat img_polinomial;
    if (img.depth() == CV_8U) {
        img.convertTo(img_polinomial, CV_32F, 1.0 / 255);
    }
    else {
        img.copyTo(img_polinomial);
    }
    std::vector<cv::Mat> img_BGR;
    cv::split(img_polinomial, img_BGR);

    for (int channel = 0; channel < img_BGR.size(); channel++) {
        img_polinomial = cv::Mat::zeros(img_BGR[channel].rows, img_BGR[channel].cols, img_BGR[channel].type());
        for (int x = 0; x < img_BGR[channel].cols; x++) {
            for (int y = 0; y < img_BGR[channel].rows; y++) {
                int xnew = int(round(T[0][0] + x * T[0][1] + y * T[0][2] + x * x * T[0][3] + x * y * T[0][4] + y * y * T[0][5]));
                int ynew = int(round(T[1][0] + x * T[1][1] + y * T[1][2] + x * x * T[1][3] + x * y * T[1][4] + y * y * T[1][5]));
                if ((xnew >= 0) && (xnew < img_BGR[channel].cols) && (ynew >= 0) && (ynew < img_BGR[channel].rows)) {
                    img_polinomial.at<float>(ynew, xnew) = img_BGR[channel].at<float>(y, x);
                }
           
            }
        }
        img_BGR[channel] = img_polinomial;
    }

    cv::merge(img_BGR, img_polinomial);
    if (img.depth() == CV_8U) {
        img_polinomial.convertTo(img_polinomial, CV_8U, 255);
    }

    return img_polinomial;
}

cv::Mat sin_transformation(cv::Mat img, double s) {
    cv::Mat u = cv::Mat::zeros(img.size(), CV_32F);
    cv::Mat v = cv::Mat::zeros(img.size(), CV_32F);
    for (int x = 0; x < img.cols; x++) {
        for (int y = 0; y < img.rows; y++) {
            u.at<float>(y, x) = float(x + s * std::sin(2 * M_PI * y / 90.0));
            v.at<float>(y, x) = float(y);
        }
    }
    cv::Mat sinusoid_image;
    cv::remap(img, sinusoid_image, u, v, cv::INTER_LINEAR);
    return sinusoid_image;
}
