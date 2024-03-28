#include<iostream>
#include<cmath>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/stitching.hpp>

cv::Mat glue_images(cv::Mat img1, cv::Mat img2);

cv::Mat auto_glue_images(cv::Mat img1, cv::Mat img2);

int main() {
	cv::Mat left_part = cv::imread("/home/den/CV_labs/Lab2/images/original/glue_part_1.jpeg", cv::IMREAD_COLOR);
	cv::Mat right_part = cv::imread("/home/den/CV_labs/Lab2/images/original/glue_part_2.jpeg", cv::IMREAD_COLOR);

	cv::Mat final_img = glue_images(left_part, right_part);
    cv::imwrite("/home/den/CV_labs/Lab2/images/results/manually_glued_img.jpeg", final_img);
	return 0;
}

cv::Mat glue_images(cv::Mat left_part, cv::Mat right_part) {
	int templ_size = 20;
	cv::Mat templ = left_part(cv::Rect(left_part.cols - templ_size - 1, 0, templ_size, left_part.rows));
	
	cv::Mat res;
	cv::matchTemplate(right_part, templ, res, cv::TM_CCOEFF);
	double min_val, max_val;
	cv::Point2i min_loc, max_loc;
	cv::minMaxLoc(res, &min_val, &max_val, &min_loc, &max_loc);
	cv::Mat final_img = cv::Mat::zeros(left_part.rows, left_part.cols + right_part.cols - max_loc.x - templ_size, left_part.type());
	left_part.copyTo(final_img(cv::Rect(0, 0, left_part.cols, left_part.rows)));
	cv::Mat right_part_for_glue = right_part(cv::Rect(max_loc.x + templ_size, 0, right_part.cols - max_loc.x - templ_size, right_part.rows));
	right_part_for_glue.copyTo(final_img(cv::Rect(left_part.cols, 0, right_part.cols - max_loc.x - templ_size, right_part.rows)));
	return final_img;
}
  
cv::Mat auto_glue_images(cv::Mat left_part, cv::Mat right_part) {
	cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
	std::vector<cv::Mat> imgs;
	imgs.push_back(left_part);
	imgs.push_back(right_part);
	cv::Mat img_stitched;
	cv::Stitcher::Status status = stitcher->stitch(imgs, img_stitched);
	return img_stitched;
}
