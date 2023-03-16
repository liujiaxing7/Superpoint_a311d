#pragma once
#include <cmath>
#include "opencv2/opencv.hpp"

class SpRun
{

protected:
	double nn_thresh;
	double nms_dist;
	long long loc_channel;
	long long desc_channel;
	long long pixnum;
	long long rsize_h;
	long long rsize_w;
	int pad = 4;
	int border = 4;
	long long h;
	long long w;
	int cell = 8;
	double conf_thresh = 0.015;

	long long** pts_save;
	double* score_save;
	double** desc_save;
	
public:
	SpRun();
	SpRun(int loc_c, int desc_c, int pixn, int rsize_h, int rsize_w);
	~SpRun();
	long long count;
	void set_count(long long value);
	long long get_count();
	void get_sp_result(long long** pts_result, double* score_result, double** desc_result);
	void grid_sample(double*** coarse_desc, double** samp_pts, long long count, double** desc);
	void calc(float*** semi, float*** coarse_desc_, cv::Mat img);

};