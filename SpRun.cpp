#pragma once
#include "SpRun.h"
#include <iostream>

SpRun::SpRun()
{
}


SpRun::SpRun(int loc_c, int desc_c, int pixn, int rh, int rw) 
{
	loc_channel = loc_c;
	desc_channel = desc_c;
	pixnum = pixn;
	rsize_h = rh;
	rsize_w = rw;
	h = int(floor(rh / 8));
	w = int(floor(rw / 8));
}
SpRun::~SpRun()
{

}

void SpRun::set_count(long long value) {
	count = value;
}

long long SpRun::get_count() {
	//nms 적용하고, 이미지 범위 내에 있는 좌표들의 개수 return
	return count;
}

void SpRun::get_sp_result(long long** pts_result, double* score_result, double** desc_result) {

	//결과값 복사
	memcpy(pts_result[0], pts_save[0], sizeof(long long) * count);
	memcpy(pts_result[1], pts_save[1], sizeof(long long) * count);

	memcpy(score_result, score_save, sizeof(double) * count);

	for (size_t chanIdx = 0; chanIdx < desc_channel; ++chanIdx) {
		memcpy(desc_result[chanIdx], desc_save[chanIdx], sizeof(double) * count);
	}


	//delete
	for (size_t i = 0; i < 2; i++) {
		delete[] pts_save[i];
	}
	delete[] pts_save;
	delete[] score_save;
	delete[] desc_save;

	pts_save = nullptr;
	score_save = nullptr;
	desc_save = nullptr;
}

void SpRun::grid_sample(double*** coarse_desc, double** samp_pts, long long count, double** desc) {
	// grid_sample 알고리즘 참고 링크
	// https://stackoverflow.com/questions/73300183/understanding-the-torch-nn-functional-grid-sample-op-by-concrete-example

	double** result = new double* [desc_channel]; //(desc_channel,count)
	double* norm = new double[count];

	for (size_t i = 0; i < desc_channel; i++) {
		double* result_2 = new double[count];
		result[i] = result_2;
	}

	for (size_t channel = 0; channel < desc_channel; channel++) {
		for (size_t p_cnt = 0; p_cnt < count; p_cnt++) {
			double cvt_x = ((samp_pts[p_cnt][0] + 1.) * w - 1.) / 2.;
			double cvt_y = ((samp_pts[p_cnt][1] + 1.) * h - 1.) / 2.;

			double grid_a = (cvt_x)-floor(cvt_x);
			double grid_b = (cvt_y)-floor(cvt_y);
			double grid_x = coarse_desc[channel][int(floor(cvt_y))][int(floor(cvt_x))];
			double grid_y = coarse_desc[channel][int(floor(cvt_y))][int(ceil(cvt_x))];
			double grid_w = coarse_desc[channel][int(ceil(cvt_y))][int(floor(cvt_x))];
			double grid_z = coarse_desc[channel][int(ceil(cvt_y))][int(ceil(cvt_x))];

			double out = (grid_a * grid_b * grid_z) + ((1 - grid_a) * (1 - grid_b) * grid_x) + ((1 - grid_a) * grid_b * grid_w) + ((1 - grid_b) * grid_a * grid_y);
			result[channel][p_cnt] = out;
		}
	}

	//norm 계산 (count축 기준)
	for (size_t p_cnt = 0 ; p_cnt < count ; p_cnt++){
		double norm_acc= 0;
		for (size_t channel = 0; channel < desc_channel; channel++) {
			norm_acc += pow(result[channel][p_cnt],2);
			if (channel == desc_channel - 1) {
				norm[p_cnt] = sqrt(norm_acc);
			}
		}
	}

	//계산한 norm으로 나눠줌
	for (size_t channel = 0; channel < desc_channel; channel++) {
		for (size_t p_cnt = 0; p_cnt < count; p_cnt++) {
			result[channel][p_cnt] = result[channel][p_cnt] / norm[p_cnt];
		}
	}

	for (size_t chanIdx = 0; chanIdx < desc_channel; ++chanIdx) {
		memcpy(desc[chanIdx], result[chanIdx], sizeof(double) * count);
	}


	//delete
	for (size_t i = 0; i < desc_channel; ++i) {
		delete[] result[i];
	}
	delete[] result;
	delete[] norm;

	result = nullptr;
	norm = nullptr;
}

void SpRun::calc(float*** semi, float*** coarse_desc_, cv::Mat img) {
	// semi : (1,loc_channel, outpixNum)
	// coarse_desc : (1, desc_channel, pixnum)


	//point location pixel-wise softmax


	// nodust : (loc_channel-1 ,pixnum) -> dustbin 차원을 제거
	double** nodust = new double* [loc_channel - 1]; 

	 // dense_p : (loc_channel, pixnum) -> semi에 exp() 취한 값 저장
	double** dense_p = new double* [loc_channel];

	// expSum : (pixnum) ,pixel wise expsum -> dense_p에서, 한픽셀 위치의 모든 채널값을 다 더함.
	double* expSum = new double[pixnum]; 

	for (size_t chanIdx = 0; chanIdx < loc_channel-1; chanIdx++) {
		double* nodust_c = new double[pixnum]();
		nodust[chanIdx] = nodust_c;
	}

	for (size_t chanIdx = 0; chanIdx < loc_channel; chanIdx++) {
		double* dense_c = new double[pixnum];

		//한 픽셀 위치의 채널값 더해주는 과정
		for (size_t pixIdx = 0; pixIdx < pixnum; pixIdx++) {
			double val0 = exp(double(semi[0][chanIdx][pixIdx]));

			if (chanIdx == 0) {
				expSum[pixIdx] = val0;
			}
			else {
				expSum[pixIdx] = expSum[pixIdx] + val0;
			}
			dense_c[pixIdx] = val0; //exp취한 값은 dense_p로 저장
		}
		dense_p[chanIdx] = dense_c;
	}

	// dense_p / expSun 한 결과에서, loc_channel의 가장 마지막은 빼주면서 dustbin을 제거함.
	for (size_t chanIdx = 0; chanIdx < loc_channel - 1; chanIdx++) {
		for (size_t pixIdx = 0; pixIdx < pixnum; pixIdx++) {
			nodust[chanIdx][pixIdx] = dense_p[chanIdx][pixIdx] / (expSum[pixIdx]);
		}
	}

	// transpose, reshape 과정
	// nodust : (loc_channel -1 , pixnum ) -> nodust_t : (h, w, loc_channel-1) 
	// pixnum = h*w

	double*** nodust_t = new double** [h];
	for (size_t i = 0; i < h; i++) {
		double** single_h = new double* [w];
		nodust_t[i] = single_h;
		for (size_t j = 0; j < w; j++) {
			double* single_c = new double[loc_channel - 1];
			nodust_t[i][j] = single_c;
		}
	}

	for (size_t nc = 0; nc < loc_channel - 1; nc++) {
		for (size_t pn = 0; pn < pixnum; pn++) {
			int nh = pn / h;
			int nw = pn % h;
			nodust_t[nh][nw][nc] = nodust[nc][pn]; //(62,62,64)
		}
	}

	//heatmap : (h,cell,w,cell)
	double**** heatmap = new double*** [h];
	for (size_t i = 0; i < h; i++) {
		double*** heatmap1d = new double** [cell];
		heatmap[i] = heatmap1d;
		for (size_t j = 0; j < cell; j++) {
			double** heatmap2d = new double* [w];
			heatmap[i][j] = heatmap2d;
			for (size_t k = 0; k < w; k++) {
				double* heatmap3d = new double[cell];
				heatmap[i][j][k] = heatmap3d;
			}
		}
	}


	//reshape nodust_t : (h,w,loc_channel-1) -> (h,w,√loc_channel,√loc_channel)
	// ex ) (62,62,64)-> (62,62,8,8)
	// 
	//transpose(0,2,1,3) -- (62,62,8,8) -> heatmap :(62,8,62,8)
	for (size_t nh = 0; nh < h; nh++) {
		for (size_t nw = 0; nw < w; nw++) {
			for (size_t nc = 0; nc < loc_channel - 1; nc++) {
				int cell1 = nc / cell;
				int cell2 = nc % cell;
				heatmap[nh][cell1][nw][cell2] = nodust_t[nh][nw][nc]; //(62,8,62,8);
			}
		}
	}


	//reshape heatmap:(h,√loc_channel,w,√loc_channel) -> heatmap_r:((h*√loc_channel),(w*√loc_channel))
	// ex) (62,8,62,8) -> (496*496)
	double* heatmap_r = new double[h * w * cell * cell];

	long long cnt = 0; //heatmap에서 conf_threshold 이상인 좌표 개수 !!
	for (size_t nh = 0; nh < h; nh++) {
		for (size_t c1 = 0; c1 < cell; c1++) {
			for (size_t nw = 0; nw < w; nw++) {
				for (size_t c2 = 0; c2 < cell; c2++) {
					heatmap_r[(nh * w * cell * cell) + (c1 * w * cell) + (nw * cell) + (c2)] = heatmap[nh][c1][nw][c2];
					if (heatmap[nh][c1][nw][c2] >= conf_thresh)
						cnt += 1;
				}
			}
		}
	}

	//여기가 python의 superpoint.run에서 pts.
	long long** pts_xy = new long long* [2]; //(2, cnt)
	long long* pt_y = new long long[cnt]; //pts[0]
	long long* pt_x = new long long[cnt]; //pts[1]
	double* pt_score = new double[cnt]; //pts[2]

	long long cnt_idx = 0;

	// heatmap_r 에서 conf_thresh값 이상인 픽셀의 x,y좌표와 score(heatmap 값)을 pts_xy에 저장한다.
	for (size_t pixIdx = 0; pixIdx < h * w * cell * cell; pixIdx++) {
		if (heatmap_r[pixIdx] >= conf_thresh) {
			long long idx_y = pixIdx / (h * cell);
			long long idx_x = pixIdx % (w * cell);

			pt_x[cnt_idx] = idx_x;
			pt_y[cnt_idx] = idx_y;
			pt_score[cnt_idx] = heatmap_r[pixIdx];
			cnt_idx++;
		}
	}
	pts_xy[0] = pt_y;
	pts_xy[1] = pt_x;


	//nms 시작


	//score 내림차순 정렬
	long long* sorted_idx = new long long[cnt]; //score 내림차순 정렬한 인덱스 -> python : inds1
	for (size_t i = 0; i < cnt; i++)
		sorted_idx[i] = i;

	for (size_t i = 0; i < cnt; i++) {
		for (size_t j = 0; j < cnt - 1; j++) {
			if (pt_score[sorted_idx[j]] < pt_score[sorted_idx[j + 1]]) {
				int temp = sorted_idx[j];
				sorted_idx[j] = sorted_idx[j + 1];
				sorted_idx[j + 1] = temp;
			}
		}
	}

	// 내림차순 정렬한 index로 x,y도 정렬 
	long long** sorted_xy = new long long* [2]; //(2,cnt)
	long long* sorted_x = new long long[cnt]; //내림차순 정렬된 x좌표
	long long* sorted_y = new long long[cnt]; //내림차순 정렬된 y좌표
	double* sorted_score = new double[cnt]; //내림차순 정렬된 score
	for (size_t i = 0; i < cnt; i++) {
		sorted_x[i] = pts_xy[0][sorted_idx[i]];
		sorted_y[i] = pts_xy[1][sorted_idx[i]];
		sorted_score[i] = pt_score[sorted_idx[i]];
	}

	sorted_xy[0] = sorted_x;
	sorted_xy[1] = sorted_y;

	long long** grid = new long long* [rsize_h + 2 * pad]; // Track NMS data ,zero padding 
	long long** inds = new long long* [rsize_h]; // sorted 적용한 좌표의 score 순위를 픽셀 위치에 저장 

	for (size_t i = 0; i < rsize_h + 2 * pad; i++) {
		long long* grid_w = new long long[rsize_w + 2 * pad]();
		grid[i] = grid_w;
	}

	for (size_t i = 0; i < rsize_h; i++) {
		long long* inds_w = new long long[rsize_w]();
		inds[i] = inds_w;
	}

	//정렬한 좌표순서대로 grid엔 1값을, inds엔 score 순위(해당좌표에 score 순위가 있음)
	for (size_t i = 0; i < cnt; i++) {
		int x = sorted_xy[0][i];
		int y = sorted_xy[1][i];
		grid[x+pad][y + pad] = 1; //padding 고려
		inds[x][y] = i;
	}

	//nms해서 주변 포인트 지워줌
	// score 큰 순으로 돌아가면서, grid에서 1인 좌표의 중앙값은 -1로 남기고, 나머지 주변 1은 지워줌
	long long nms_count = 0;
	for (size_t i = 0; i < cnt; i++) {
		int pt_y = sorted_xy[0][i] + pad;
		int pt_x = sorted_xy[1][i] + pad;
		if (grid[pt_y][pt_x] == 1) {
			
			for (size_t ph = 0; ph < 2*pad+1; ph++) {
				for (size_t pw = 0; pw < 2*pad+1 ; pw++) {
					grid[pt_y-pad+ph][pt_x-pad+ pw] = 0;
				}
			}
			grid[pt_y][pt_x] = -1;
			nms_count += 1;
		}
	}

	//최종적으로 nms해서 남은 좌표값만 count
	set_count(nms_count);

	//nms해서 남은 애들(grid = -1)의 좌표만 구해줌
	long long* keepx = new long long[count];
	long long* keepy = new long long[count];

	long long keepcnt = 0;
	for (size_t i = 4; i < rsize_h + pad; i++) {//
		for (size_t j = 4; j < rsize_w + pad; j++) {
			if (grid[i][j] == -1) {
				keepx[keepcnt] = j - pad;
				keepy[keepcnt] = i - pad;
				keepcnt++;
			}
		}
	}

	//좌표에 score 순위가 매겨졌던 inds에서도, nms해서 남은 좌표들의 순위만 inds_keep에 저장
	long long* inds_keep = new long long[count]; //nms 적용 후 남은 좌표의 score 순위
	for (size_t i = 0; i < count; i++) {
		int y = keepy[i];
		int x = keepx[i];
		inds_keep[i] = inds[y][x];
	}

	//nms로 남은 score들
	double* keep_score = new double[count];
	for (size_t i = 0; i < count; i++) {
		int inds_k = inds_keep[i];
		keep_score[i] = sorted_score[inds_k];
	}

	//nms로 남은 score의 내림차순 인덱스 구하기 
	long long* sorted_keep_idx = new long long[count];
	for (size_t i = 0; i < count; i++)
		sorted_keep_idx[i] = i;

	for (size_t i = 0; i < count; i++) {
		for (size_t j = 0; j < count - 1; j++) {
			if (keep_score[sorted_keep_idx[j]] < keep_score[sorted_keep_idx[j + 1]]) {
				int temp = sorted_keep_idx[j];
				sorted_keep_idx[j] = sorted_keep_idx[j + 1];
				sorted_keep_idx[j + 1] = temp;
			}
		}
	}

	//nms적용후 score별로 내림차순된 x,y,score 
	long long ** nms_sorted_xy = new long long* [2];
	long long* nms_sorted_x = new long long[count];
	long long* nms_sorted_y = new long long[count];
	double* nms_sorted_score = new double[count];

	for (size_t i = 0; i < count; i++) {
		nms_sorted_x[i] = keepx[sorted_keep_idx[i]];
		nms_sorted_y[i] = keepy[sorted_keep_idx[i]];
		nms_sorted_score[i] = keep_score[sorted_keep_idx[i]];
	}

	//머지..여기 xy바뀐거같은데 확인해봐약ㄷㄷ..
	nms_sorted_xy[1] = nms_sorted_y;
	nms_sorted_xy[0] = nms_sorted_x;

	//nms 끝//


	//이미지를 벗어나는 좌표 제거
	bool* toremove = new bool[count];
	long long tm_count = 0;
	for (size_t i = 0; i < count; i++) {
		if (((nms_sorted_xy[0][i] < border) || (nms_sorted_xy[0][i] >= (rsize_w - border))) || ((nms_sorted_xy[1][i] < border) || (nms_sorted_xy[1][i] >= (rsize_h - border))))
			toremove[i] = false;
		else {
			toremove[i] = true;
			tm_count++;
		}
	}
	//이미지 벗어나는 좌표 제거한 후의 좌표 개수 count 설정 
	set_count(tm_count);

	//최종 calc 결과 저장
	pts_save = new long long* [2]; // pts_save : (2, count)
	long long* pts_save_x = new long long[count];
	long long* pts_save_y = new long long[count];
	score_save = new double[count]; // score_save : (count)

	pts_save[1] = pts_save_y;
	pts_save[0] = pts_save_x;

	long long idx = 0;
	for (size_t cnt = 0; cnt < nms_count; cnt++) {
		if (toremove[cnt] == true) {
			pts_save[0][idx] = nms_sorted_xy[0][idx];
			pts_save[1][idx] = nms_sorted_xy[1][idx];
			score_save[idx] = nms_sorted_score[idx];
			idx++;
		}
	}

	//feature point 확인용 출력
	//cv::Mat imgmat(cv::Size(500, 500), CV_8UC3);
	//cv::cvtColor(img, imgmat, cv::COLOR_GRAY2BGR);
	//cv::Mat imgcp;
	//imgmat.copyTo(imgcp);
	//for (int i = 0; i < count; i++) {
	//	cv::circle(imgcp, cv::Point(nms_sorted_xy[1][i], nms_sorted_xy[0][i]),1,cv::Scalar(0,0,255),1,8,0);
	//}
	//cv::imwrite("C:/Users/jsyoon/PycharmProjects/Pytorch/venv/SuperPoint/cppresult/point.bmp", imgcp);


	//grid_sample 적용하기 전 전처리 

	double** samp_pts= new double* [count]; //(count,2);
	for (size_t i = 0; i < count; i++) {
		double* samp_xy = new double[2];

		//nms하고 내림차순으로 정렬한 x,y좌표값을 (-1~1)사이 값으로 정규화
		samp_xy[0] = ((nms_sorted_xy[0][i]) / (rsize_w / 2.)) - 1.; //normalized(-1~1)
		samp_xy[1] = ((nms_sorted_xy[1][i]) / (rsize_h / 2.)) - 1.; //normalized(-1~1)
		samp_pts[i] = samp_xy;
	}

	//grid_sample한 결과 descriptor 저장할 곳
	desc_save = new double* [desc_channel]; //(desc_channel,count)
	for (size_t i = 0; i < desc_channel; i++) {
		double* desc_2d = new double[count];
		desc_save[i] = desc_2d;
	}

	//맨 처음에 입력받은 coarse_desc를 grid_sample 입력에 맞게 reshape
	double*** coarse_desc = new double** [desc_channel]; //(desc_channel, h,w )
	
	for (size_t c = 0; c < desc_channel; c++) {
		double** desc_h = new double* [h];
		for (size_t dh = 0; dh < h; dh++) {
			double* desc_w = new double[w];
			for (size_t dw = 0; dw < w; dw++) {
				desc_w[dw] = double(coarse_desc_[0][c][(dh * w) + dw]);
			}
			desc_h[dh] = desc_w;
		}
		coarse_desc[c] = desc_h;
	}
	//grid_sample 수행
	grid_sample(coarse_desc, samp_pts, count, desc_save);


	//delete

	for (size_t i = 0; i < loc_channel-1; i++) {
		delete[] nodust[i];
	}
	for (size_t i = 0; i < loc_channel; i++) {
		delete[] dense_p[i];
	}
	delete[] nodust;
	delete[] dense_p;
	delete[] expSum;


	for (size_t i = 0; i < h; i++) {
		for (size_t j = 0; j < w; j++) {
			delete[] nodust_t[i][j];
		}
		delete[] nodust_t[i];
	}
	delete[] nodust_t;
	
	for (size_t i = 0; i < h; i++) {
		for (size_t j = 0; j < cell; j++) {
			for (size_t k = 0; k < w; k++) {
				delete[] heatmap[i][j][k];
			}
			delete[] heatmap[i][j];
		}
		delete[] heatmap[i];
	}
	delete[] heatmap;
	delete[] heatmap_r;
	

	for (size_t i = 0; i < 2; i++) {
		delete[] pts_xy[i];
	}
	delete[] pts_xy;
	delete[] pt_score;

	for (size_t i = 0; i < 2; i++) {
		delete[] sorted_xy[i];
	}
	delete[] sorted_xy;
	delete[] sorted_idx;
	delete[] sorted_score;

	for (size_t i = 0; i < rsize_h + 2 * pad; i++) {
		delete[] grid[i];
	}
	for (size_t i = 0; i < rsize_h; i++) {
		delete[] inds[i];
	}
	delete[] grid;
	delete[] inds;
	delete[] keepx;
	delete[] keepy;
	delete[] inds_keep;
	delete[] keep_score;
	delete[] sorted_keep_idx;

	for (size_t i = 0; i < 2; i++) {
		delete[] nms_sorted_xy[i];
	}
	delete[] nms_sorted_xy;
	delete[] nms_sorted_score;
	delete[] toremove;

	for (size_t i = 0; i < count; i++) {
		delete[] samp_pts[i];
	}
	delete[] samp_pts;

	for (size_t i = 0; i < desc_channel; i++) {
		for (size_t j = 0; j < h; j++) {
			delete[] coarse_desc[i][j];
		}
		delete[] coarse_desc[i];
	}
	delete[] coarse_desc;

	nodust = nullptr;
	dense_p = nullptr;
	expSum = nullptr;
	nodust_t = nullptr;
	heatmap = nullptr;
	pts_xy = nullptr;
	pt_score = nullptr;
	sorted_xy = nullptr;
	sorted_idx = nullptr;
	sorted_score = nullptr;
	grid = nullptr;
	inds = nullptr;
	keepx = nullptr;
	keepy = nullptr;
	inds_keep = nullptr;
	keep_score = nullptr;
	sorted_keep_idx = nullptr;
	nms_sorted_xy = nullptr;
	nms_sorted_score = nullptr;
	toremove = nullptr;
	samp_pts = nullptr;
	coarse_desc = nullptr;
}