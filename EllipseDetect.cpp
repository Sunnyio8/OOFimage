#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
#include "lsd.h"
#include"lsd.c"  
#include "opencv2/core/core.hpp" 
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2\opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace cv;


typedef long int integer;
typedef double doublereal;

struct point2d
{
	double x, y;
};

struct  point5d
{
	double x, y;
	double a, b;
	double phi;
};


typedef struct PairGroup_s
{
	point pairGroupInd;
	point2d center;  //(x0,y0)
	point2d axis;    //(a,b)
	double  phi;     //angle of orientation  
}PairGroup;

typedef struct  PairGroupList_s
{
	int length;
	PairGroup* pairGroup;
}PairGroupList;


typedef struct PairGroupNode_s
{
	point pairGroupInd;
	point2d center;  //(x0,y0)
	point2d axis;    //(a,b)
	double  phi;     //angle of orientation  
	PairGroupNode_s* next;
}PairGroupNode;






//对线段按照凸性和距离进行分组
//lines: 输入的lines_num条线段，每条线段8个值，存着x1,y1,x2,y2,dx,dy,length,polarity
//lines_num:
//输出分组groups. 每个组是一个vector<int>
//注意：切记用完region,需要在函数外面手动释放region
void groupLSs(double* lines, int line_num, int* region, int imgx, int imgy, vector<vector<int>>* groups)
{
	if (line_num == 0)
	{
		groups = NULL;
		return;
	}
	unsigned char isEnd = 0;//是否还可以继续搜寻
	int currentLine; //当前线段
	char* label = (char*)calloc(line_num, sizeof(char));
	memset(label, 0, sizeof(char) * line_num); //init the label all to be zero
	int* group_up = (int*)malloc(sizeof(int) * line_num);//申请足够内存，存储延线段方向得到的分组的线段
	int* group_down = (int*)malloc(sizeof(int) * line_num);//存储线段反方向分组的线段
	int group_up_cnt, group_down_cnt;
	//coorlist * head,*tail;
	vector<int> group_temp;
	point2d dir_vec1, dir_vec2;
	point* votebin = (point*)calloc(line_num, sizeof(point));//申请足够内存，用来投票. x记录线段索引，y记录票数
	int bincnt = 0;
	int xx, yy, temp;
	double start_angle, end_angle, angle_delta;
	for (int i = 0; i < line_num; i++)
	{
		if (label[i] == 0)//未被分组过
		{
			group_up_cnt = group_down_cnt = 0;//每开始寻找一组，需要置零
			//先从第i条线段的头部开始搜索，进行分组,结果存在group_up里面
			group_up[group_up_cnt++] = i;//记录线段i,注意线段是0~line_num-1
			isEnd = 0;//置零，表示还可以从当前线段开始搜索，还未结束
			currentLine = i;
			while (isEnd == 0)
			{
				label[currentLine] = 1; //标记该线段已经被分组
				//head = tail = NULL;
				bincnt = 0;
				dir_vec1.x = lines[currentLine * 8 + 4];
				dir_vec1.y = lines[currentLine * 8 + 5];
				if (lines[currentLine * 8 + 7] == 1)//极性为正
				{
					//将dir_vec1逆时针旋转45°
					dir_vec2.x = (dir_vec1.x + dir_vec1.y) * 0.707106781186548; // sqrt(2)/2 = 0.707106781186548
					dir_vec2.y = (-dir_vec1.x + dir_vec1.y) * 0.707106781186548;
				}
				else
				{
					//将dir_vec1顺时针旋转45°
					dir_vec2.x = (dir_vec1.x - dir_vec1.y) * 0.707106781186548; // sqrt(2)/2 = 0.707106781186548
					dir_vec2.y = (dir_vec1.x + dir_vec1.y) * 0.707106781186548;
				}
				for (int j = 1; j <= 4; j++)
					for (int k = 1; k <= 4; k++)//在4x4邻域内搜索
					{
						xx = (int)(lines[currentLine * 8 + 2] * 0.8 + j * dir_vec1.x + k * dir_vec2.x);
						yy = (int)(lines[currentLine * 8 + 3] * 0.8 + j * dir_vec1.y + k * dir_vec2.y);
						if (xx < 0 || xx >= imgx || yy < 0 || yy >= imgy)//越界
							continue;
						temp = region[yy * imgx + xx];
						if (temp > 0)//表示有线段的支持区域，在1~line_num
						{
							region[yy * imgx + xx] = -temp;//取负数标记
							for (xx = 0; xx < bincnt; xx++)
							{
								if (votebin[xx].x == temp - 1)//如果以前投票过，直接在相应的bin的票数上加1
								{
									votebin[xx].y++;
									break;
								}
							}
							if (xx == bincnt)//如果以前没有投票过，增加该线段，并记录票数为1
							{
								if (bincnt == line_num)
									error("group ls error1");
								votebin[bincnt].x = temp - 1;
								votebin[bincnt].y = 1;
								bincnt++; //bin的总数加1
							}
						}
					}
				//寻找投票最多的线段，并且需要满足数量大于一定值
				temp = 0;
				for (int j = 0; j < bincnt; j++)
				{
					if (votebin[j].y > temp)
					{
						temp = votebin[j].y;
						xx = votebin[j].x;//借用xx变量
					}
				}
				if (temp >= 5 && label[xx] == 0 && lines[8 * xx + 7] == lines[8 * i + 7])//待实验调整参数值
				{
					if (group_up_cnt == line_num)
						error("group ls error2");
					yy = group_up_cnt - 1;//借用yy变量
					start_angle = atan2(lines[8 * group_up[yy] + 5], lines[8 * group_up[yy] + 4]);
					end_angle = atan2(lines[8 * xx + 5], lines[8 * xx + 4]);
					angle_delta = rotateAngle(start_angle, end_angle, (int)lines[8 * i + 7]);
					if (angle_delta <= M_3_8_PI)//相邻两线段的旋转夹角也需要满足在pi/4内
					{
						group_up[group_up_cnt++] = xx;//压入线段
						currentLine = xx; //更新当前搜索线段
					}
					else
						isEnd = 1;
				}
				else
					isEnd = 1;//结束，已经找不到可以分组的线段了
			}
			//先从第i条线段的尾部开始搜索，进行分组,结果存在group_down里面。记住，第i条线段在group_up和group_down中的0索引处都储存了
			group_down[group_down_cnt++] = i;
			isEnd = 0;//置零，表示还可以从当前线段开始搜索，还未结束
			currentLine = i;
			while (isEnd == 0)
			{
				label[currentLine] = 1; //标记该线段已经被分组
				//head = tail = NULL;
				bincnt = 0;
				dir_vec1.x = -lines[currentLine * 8 + 4];
				dir_vec1.y = -lines[currentLine * 8 + 5];
				if (lines[currentLine * 8 + 7] == 1)//极性相同
				{
					//将dir_vec1顺时针旋转45°
					dir_vec2.x = (dir_vec1.x - dir_vec1.y) * 0.707106781186548; // sqrt(2)/2 = 0.707106781186548
					dir_vec2.y = (dir_vec1.x + dir_vec1.y) * 0.707106781186548;
				}
				else
				{
					//将dir_vec1顺时针旋转45°
					dir_vec2.x = (dir_vec1.x + dir_vec1.y) * 0.707106781186548; // sqrt(2)/2 = 0.707106781186548
					dir_vec2.y = (-dir_vec1.x + dir_vec1.y) * 0.707106781186548;
				}
				for (int j = 1; j <= 4; j++)
					for (int k = 1; k <= 4; k++)//在4x4邻域内搜索
					{
						xx = (int)(lines[currentLine * 8 + 0] * 0.8 + j * dir_vec1.x + k * dir_vec2.x);
						yy = (int)(lines[currentLine * 8 + 1] * 0.8 + j * dir_vec1.y + k * dir_vec2.y);
						if (xx < 0 || xx >= imgx || yy < 0 || yy >= imgy)//越界
							continue;
						temp = region[yy * imgx + xx];
						if (temp > 0)//表示有线段的支持区域，在1~line_num
						{
							region[yy * imgx + xx] = -temp;//取负数标记
							for (xx = 0; xx < bincnt; xx++)
							{
								if (votebin[xx].x == temp - 1)//如果以前投票过，直接在相应的bin的票数上加1
								{
									votebin[xx].y++;
									break;
								}
							}
							if (xx == bincnt)//如果以前没有投票过，增加该线段，并记录票数为1
							{
								if (bincnt == line_num)
									error("group ls error3");
								votebin[bincnt].x = temp - 1;
								votebin[bincnt].y = 1;
								bincnt++; //bin的总数加1
							}
						}
					}
				//寻找投票最多的线段，并且需要满足数量大于一定值
				temp = 0;
				for (int j = 0; j < bincnt; j++)
				{
					if (votebin[j].y > temp)
					{
						temp = votebin[j].y;
						xx = votebin[j].x;//借用xx变量
					}
				}
				if (temp >= 5 && label[xx] == 0 && lines[8 * xx + 7] == lines[8 * i + 7])//待实验调整参数值
				{
					if (group_down_cnt == line_num)
						error("group ls error2");
					yy = group_down_cnt - 1;//借用yy变量
					start_angle = atan2(lines[8 * group_down[yy] + 5], lines[8 * group_down[yy] + 4]);
					end_angle = atan2(lines[8 * xx + 5], lines[8 * xx + 4]);
					angle_delta = rotateAngle(end_angle, start_angle, (int)lines[8 * i + 7]);//注意此时需要调换一下，因为是从尾部开始搜索
					if (angle_delta < M_3_8_PI)//相邻两线段的旋转夹角也需要满足在pi/4内,pi*3/8 = 66.5°
					{
						group_down[group_down_cnt++] = xx; //压入线段
						currentLine = xx; //更新当前搜索线段
					}
					else
						isEnd = 1;
				}
				else
					isEnd = 1;//结束，已经找不到可以分组的线段了
			}
			(*groups).push_back(group_temp); //添加线段分组
			temp = (*groups).size() - 1;
			for (int j = group_down_cnt - 1; j >= 0; j--)
			{
				(*groups)[temp].push_back(group_down[j]);
			}
			for (int j = 1; j < group_up_cnt; j++)//由于第i条线段在group_up和group_down都储存了，所以就从索引1开始
			{
				(*groups)[temp].push_back(group_up[j]);
			}
		}
	}
	free(label);
	free(group_up);
	free(group_down);
	free(votebin);
}


//计算groups中每个组的跨度
//输入：
//lines: 输入的lines_num条线段，每条线段8个值，存着x1,y1,x2,y2,dx,dy,length,polarity
//lines_num:
//groups: 分组，每个分组都存着线段的索引
//输出:
//coverages: 每个组的跨度，当组内线段只有1条时，跨度为0. coverages的长度等于组的数量 = groups.size()
//注意，coverages用前不需要申请内存，coverages用完后，需要在函数外手动释放内存，长度等于分组数量
void calcuGroupCoverage(double* lines, int line_num, vector<vector<int>> groups, double*& coverages)
{
	int groups_num = groups.size();
	int temp;
	double start_angle, end_angle;
	coverages = (double*)malloc(sizeof(double) * groups_num);
	for (int i = 0; i < groups_num; i++)
	{
		temp = groups[i].size() - 1;
		if (groups[i].size() == 0)//第i个分组只有1条线段，则跨度为0
		{
			coverages[i] = 0;
		}
		else
		{
			start_angle = atan2(lines[8 * groups[i][0] + 5], lines[8 * groups[i][0] + 4]);
			end_angle = atan2(lines[8 * groups[i][temp] + 5], lines[8 * groups[i][temp] + 4]);
			coverages[i] = rotateAngle(start_angle, end_angle, (int)lines[8 * groups[i][0] + 7]);
		}
	}
}


void calculateGradient2(double* img_in, unsigned int imgx, unsigned int imgy, image_double* angles)
{
	if (img_in == NULL || imgx == 0 || imgy == 0)
		error("calculateGradient error!");
	image_double mod = new_image_double(imgx, imgy);
	(*angles) = new_image_double(imgx, imgy);
	unsigned int x, y, adr;
	double com1, com2;
	double gx, gy;
	double norm, norm_square;
	double threshold;
	double sum = 0;
	double value;
	//double max_grad = 0.0;
	//边界初始为NOTDEF
	for (x = 0; x < imgx; x++)
	{
		(*angles)->data[x] = NOTDEF;
		(*angles)->data[(imgy - 1) * imgx + x] = NOTDEF;
		(mod)->data[x] = NOTDEF;
		(mod)->data[(imgy - 1) * imgx + x] = NOTDEF;
	}
	for (y = 0; y < imgy; y++)
	{
		(*angles)->data[y * imgx] = NOTDEF;
		(*angles)->data[y * imgx + imgx - 1] = NOTDEF;
		(mod)->data[y * imgx] = NOTDEF;
		(mod)->data[y * imgx + imgx - 1] = NOTDEF;
	}
	/* compute gradient on the remaining pixels */
	for (x = 1; x < imgx - 1; x++)
		for (y = 1; y < imgy - 1; y++)
		{
			adr = y * imgx + x;
			/*
			   Norm 2 computation using 2x2 pixel window:
				 A B C
				 D E F
				 G H I
			   and
				 com1 = C-G,  com2 = I-A.
			   Then
				 gx = C+2F+I - (A+2D+G)=com1+com2+2(F-D)   horizontal difference
				 gy = G+2H+I - (A+2B+C)=-com1+com2+2(H-B)   vertical difference
			   com1 and com2 are just to avoid 2 additions.
			 */
			com1 = img_in[adr - imgx + 1] - img_in[adr + imgx - 1];
			com2 = img_in[adr + imgx + 1] - img_in[adr - imgx - 1];

			gx = (com1 + com2 + 2 * (img_in[adr + 1] - img_in[adr - 1])) / (8.0 * 255); /* gradient x component */
			gy = (-com1 + com2 + 2 * (img_in[adr + imgx] - img_in[adr - imgx])) / (8.0 * 255); /* gradient y component */
			norm_square = gx * gx + gy * gy;
			sum += norm_square;

			norm = sqrt(norm_square); /* gradient norm */

			(mod)->data[adr] = norm; /* store gradient norm */
			 /* gradient angle computation */
			(*angles)->data[adr] = atan2(gy, gx);
		}
	threshold = 2 * sqrt(sum / (imgx * imgy));//自动阈值
	//non maximum suppression
	for (x = 1; x < imgx - 1; x++)
		for (y = 1; y < imgy - 1; y++)
		{
			adr = y * imgx + x;
			value = (*angles)->data[adr];
			if ((mod)->data[adr] < threshold)
			{
				(*angles)->data[adr] = NOTDEF;
				continue;
			}
			if ((value > -M_1_8_PI && value <= M_1_8_PI) || (value <= -M_7_8_PI) || (value > M_7_8_PI))
			{
				if ((mod)->data[adr] <= (mod)->data[adr + 1] || (mod)->data[adr] <= (mod)->data[adr - 1])
					(*angles)->data[adr] = NOTDEF;
			}
			else if ((value > M_1_8_PI && value <= M_3_8_PI) || (value > -M_7_8_PI && value <= -M_5_8_PI))
			{
				if ((mod)->data[adr] <= (mod)->data[adr - imgx - 1] || (mod)->data[adr] <= (mod)->data[adr + imgx + 1])
					(*angles)->data[adr] = NOTDEF;
			}
			else if ((value > M_3_8_PI && value <= M_5_8_PI) || (value > -M_5_8_PI && value <= -M_3_8_PI))
			{
				if ((mod)->data[adr] <= (mod)->data[adr - imgx] || (mod)->data[adr] <= (mod)->data[adr + imgx])
					(*angles)->data[adr] = NOTDEF;
			}
			else
			{
				if ((mod)->data[adr] <= (mod)->data[adr - imgx + 1] || (mod)->data[adr] <= (mod)->data[adr + imgx - 1])
					(*angles)->data[adr] = NOTDEF;
			}
		}
	//也标记到mod图上面
	//for(x=1;x<imgx-1;x++)
	//	for(y=1;y<imgy-1;y++)
	//	{
	//		if((*angles)->data[y*imgx+x] == NOTDEF)
	//			(mod)->data[y*imgx+x] = NOTDEF;
	//	}
	free_image_double(mod);
}


//input: dataxy为数据点(xi,yi),总共有datanum个
//output: 拟合矩阵S. 注意：S需要事先申请内存，double S[36].
inline void calcuFitMatrix(point2d* dataxy, int datanum, double* S)
{
	double* D = (double*)malloc(datanum * 6 * sizeof(double));
	memset(D, 0, sizeof(double) * datanum);
	for (int i = 0; i < datanum; i++)
	{
		D[i * 6] = dataxy[i].x * dataxy[i].x;
		D[i * 6 + 1] = dataxy[i].x * dataxy[i].y;
		D[i * 6 + 2] = dataxy[i].y * dataxy[i].y;
		D[i * 6 + 3] = dataxy[i].x;
		D[i * 6 + 4] = dataxy[i].y;
		D[i * 6 + 5] = 1;
	}
	for (int i = 0; i < 6; i++)
	{
		for (int j = i; j < 6; j++)
		{
			//S[i*6+j]
			for (int k = 0; k < datanum; k++)
				S[i * 6 + j] += D[k * 6 + i] * D[k * 6 + j];
		}
	}
	free(D);//释放内存
	//对称矩阵赋值
	for (int i = 0; i < 6; i++)
		for (int j = 0; j < i; j++)
			S[i * 6 + j] = S[j * 6 + i];
}

//input: fit matrixes S1,S2. length is 36.
//output: fit matrix S_out. S_out = S1 + S2.
//S_out事先需要申请内存
inline void addFitMatrix(double* S1, double* S2, double* S_out)
{
	int ind;
	for (int i = 0; i < 6; i++)
		for (int j = i; j < 6; j++)
		{
			ind = i * 6 + j;
			S_out[ind] = S1[ind] + S2[ind];
		}
	//对称矩阵赋值
	for (int i = 0; i < 6; i++)
		for (int j = 0; j < i; j++)
			S_out[i * 6 + j] = S_out[j * 6 + i];
}


int ellipse2Param(double* p, double param[])
{
	// ax^2 + bxy + cy^2 + dx + ey + f = 0 
	double a, b, c, d, e, f;
	double thetarad, cost, sint, cos_squared, sin_squared, cos_sin, Ao, Au, Av, Auu, Avv, tuCentre, tvCentre, wCentre, uCentre, vCentre, Ru, Rv;
	a = p[0];
	b = p[1];
	c = p[2];
	d = p[3];
	e = p[4];
	f = p[5];

	thetarad = 0.5 * atan2(b, a - c);
	cost = cos(thetarad);
	sint = sin(thetarad);
	sin_squared = sint * sint;
	cos_squared = cost * cost;
	cos_sin = sint * cost;
	Ao = f;
	Au = d * cost + e * sint;
	Av = -d * sint + e * cost;
	Auu = a * cos_squared + c * sin_squared + b * cos_sin;
	Avv = a * sin_squared + c * cos_squared - b * cos_sin;

	if (Auu == 0 || Avv == 0) { param[0] = 0; param[1] = 0; param[2] = 0; param[3] = 0; param[4] = 0; return 0; }
	else
	{
		tuCentre = -Au / (2. * Auu);
		tvCentre = -Av / (2. * Avv);
		wCentre = Ao - Auu * tuCentre * tuCentre - Avv * tvCentre * tvCentre;
		uCentre = tuCentre * cost - tvCentre * sint;
		vCentre = tuCentre * sint + tvCentre * cost;
		Ru = -wCentre / Auu;
		Rv = -wCentre / Avv;
		//     if (Ru>0) Ru=pow(Ru,0.5);
		//     else Ru=-pow(-Ru,0.5);
		//     if (Rv>0) Rv=pow(Rv,0.5);
		//     else Rv=-pow(-Rv,0.5);
		if (Ru <= 0 || Rv <= 0)//长短轴小于0的情况？？？
			return 0;
		Ru = sqrt(Ru);
		Rv = sqrt(Rv);
		param[0] = uCentre; param[1] = vCentre;
		param[2] = Ru; param[3] = Rv; param[4] = thetarad;
		//会出现Ru < Rv情况，对调一下
		if (Ru < Rv)
		{
			param[2] = Rv;
			param[3] = Ru;
			if (thetarad < 0)//调换长短轴，使得第三个参数为长轴，第四个为短轴
				param[4] += M_1_2_PI;
			else
				param[4] -= M_1_2_PI;
			if (thetarad < -M_1_2_PI)//长轴倾角限定在-pi/2 ~ pi/2，具备唯一性
				param[4] += M_PI;
			if (thetarad > M_1_2_PI)
				param[4] -= M_PI;
		}
	}
	return 1;
}


int fitEllipse(point2d* dataxy, int datanum, double* ellipara)
{
	double* D = (double*)malloc(datanum * 6 * sizeof(double));
	double S[36];
	memset(D, 0, sizeof(double) * datanum);
	memset(S, 0, sizeof(double) * 36);
	double beta[6];
	for (int i = 0; i < datanum; i++)
	{
		D[i * 6] = dataxy[i].x * dataxy[i].x;
		D[i * 6 + 1] = dataxy[i].x * dataxy[i].y;
		D[i * 6 + 2] = dataxy[i].y * dataxy[i].y;
		D[i * 6 + 3] = dataxy[i].x;
		D[i * 6 + 4] = dataxy[i].y;
		D[i * 6 + 5] = 1;
	}
	for (int i = 0; i < 6; i++)
		for (int j = i; j < 6; j++)
		{
			//S[i*6+j]
			for (int k = 0; k < datanum; k++)
				S[i * 6 + j] += D[k * 6 + i] * D[k * 6 + j];
		}
	free(D);//释放内存
	//对称矩阵赋值
	for (int i = 0; i < 6; i++)
		for (int j = 0; j < i; j++)
			S[i * 6 + j] = S[j * 6 + i];
	Eigen::Matrix<double, 6, 6> C;
	C << 0, 0, 2, 0, 0, 0,
		0, -1, 0, 0, 0, 0,
		2, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0;
	Eigen::Matrix<double, 6, 6> matrix_S;
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			matrix_S(i, j) = S[i * 6 + j];
		}
	}
	Eigen::Matrix<double, 6, 6> inv_S;
	inv_S = matrix_S.inverse();
	Eigen::Matrix<double, 6, 6> S_C;
	S_C = inv_S * C;
	Eigen::EigenSolver<Eigen::Matrix<double, 6, 6>> eig(S_C);
	Eigen::MatrixXd eig_value = eig.pseudoEigenvalueMatrix();
	Eigen::MatrixXd eig_vector = eig.pseudoEigenvectors();
	//cout << "matrix values = \n"<< eig_value << endl;
	//cout << "matrix vectors = \n" << eig_vector << endl;
	double new_value[6];
	for (int i = 0; i < 6; i++)
	{
		new_value[i] = eig_value(i, i);
	}
	for (int i = 0; i < 6; i++) {
		int index = -1;
		if (new_value[i] > 0)
		{
			index = i;
		}
		if (index != -1)
		{
			beta[0] = eig_vector(index + 0, index);
			beta[1] = eig_vector(index + 1, index);
			beta[2] = eig_vector(index + 2, index);
			beta[3] = eig_vector(index + 3, index);
			beta[4] = eig_vector(index + 4, index);
			beta[5] = eig_vector(index + 5, index);
			ellipse2Param(beta, ellipara);//ax^2 + bxy + cy^2 + dx + ey + f = 0, transform to (x0,y0,a,b,phi)
			return 1;
		}
	}


	return 0;
}




int fitEllipse2(double* S, double* ellicoeff)
{
	Eigen::Matrix<double, 6, 6> C;
	C << 0, 0, 2, 0, 0, 0,
		0, -1, 0, 0, 0, 0,
		2, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0;
	// eig(S,C) eig(inv(S)*C)
	Eigen::Matrix<double, 6, 6> matrix_S;
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			matrix_S(i, j) = S[i * 6 + j];
		}
	}
	//cout << "S = \n" << matrix_S << endl;
	Eigen::Matrix<double, 6, 6> inv_S;
	inv_S = matrix_S.inverse();
	//cout << "inv_S = \n" << inv_S << endl;
	Eigen::Matrix<double, 6, 6> S_C;
	S_C = inv_S * C;
	//cout << "S_C = \n" << S_C << endl;
	Eigen::EigenSolver<Eigen::Matrix<double, 6, 6>> eig(S_C);
	Eigen::MatrixXd eig_value = eig.pseudoEigenvalueMatrix();
	Eigen::MatrixXd eig_vector = eig.pseudoEigenvectors();
	//cout << "matrix values = \n"<< eig_value << endl;
	//cout << "matrix vectors = \n" << eig_vector << endl;

	double new_value[6];
	for (int i = 0; i < 6; i++)
	{
		new_value[i] = eig_value(i, i);
	}

	for (int i = 0; i < 6; i++) {
		int index = -1;
		if (new_value[i] > 0)
		{
			index = i;
		}
		if (index != -1)
		{
			if (eig_vector(index + 0, index) < 0)
			{
				ellicoeff[0] = -eig_vector(index + 0, index); 
				ellicoeff[1] = -eig_vector(index + 1, index);
				ellicoeff[2] = -eig_vector(index + 2, index);
				ellicoeff[3] = -eig_vector(index + 3, index);
				ellicoeff[4] = -eig_vector(index + 4, index);
				ellicoeff[5] = -eig_vector(index + 5, index);
			}
			else
			{
				ellicoeff[0] = eig_vector(index + 0, index); //[6*0+index];
				ellicoeff[1] = eig_vector(index + 1, index); //[6*1+index];
				ellicoeff[2] = eig_vector(index + 2, index); //[6*2+index];
				ellicoeff[3] = eig_vector(index + 3, index); //[6*3+index];
				ellicoeff[4] = eig_vector(index + 4, index); //[6*4+index];
				ellicoeff[5] = eig_vector(index + 5, index); //[6*5+index];
			}
			return 1;
		}

	}
	return 0;
}




void drawEllipse(Mat img, double* ellipara)
{
	Point peliicenter(ellipara[0], ellipara[1]);
	Size  saxis(ellipara[2], ellipara[3]);
	//Mat ellimat = Mat::zeros(img.rows,img.cols,CV_8UC3);
	//ellimat.setTo(255);
	static int ccc = 0;
	static unsigned int cnt = 0;
	if (cnt % 2 == 0)
		ccc = 0;
	else
	{
		ccc = 255;
		cout << cnt / 2 << '\t' << ellipara[0] << '\t' << ellipara[1] << "\t" << ellipara[2] << '\t' << ellipara[3] << '\t' << ellipara[4] << endl;
	}
	cnt++;

	Mat imgtemp = img.clone();
	ellipse(imgtemp, peliicenter, saxis, ellipara[4] * 180 / M_PI, 0, 360, (Scalar(0, 255, ccc)), 2);
	namedWindow("w1");
	imshow("w1", imgtemp);
	//waitKey(0);
}


inline bool isEllipseEqual(double* ellipse1, double* ellipse2, double centers_distance_threshold, double semimajor_errorratio, double semiminor_errorratio, double angle_errorratio, double iscircle_ratio)
{
	bool con1 = (abs(ellipse1[0] - ellipse2[0]) < centers_distance_threshold && abs(ellipse1[1] - ellipse2[1]) < centers_distance_threshold &&
		abs(ellipse1[2] - ellipse2[2]) / MAX(ellipse1[2], ellipse2[2]) < semimajor_errorratio && abs(ellipse1[3] - ellipse2[3]) / MIN(ellipse1[3], ellipse2[3]) < semiminor_errorratio);
	bool con2 = (ellipse1[3] / ellipse1[2] >= iscircle_ratio);//0.9 0.85
	bool con3 = (ellipse2[3] / ellipse2[2] >= iscircle_ratio);
	bool con4 = ((con2 && con3) || (con2 == false && con3 == false && abs(ellipse1[4] - ellipse2[4]) <= angle_errorratio * M_PI));
	return (con1 && con4);
}


//输入
//lsd算法检测得到的线段集合lines的数量line_num，return的返回值是line_nums条线段，为一维double型数组lines，长度为8*n，每8个为一组
//存着x1,y1,x2,y2,dx,dy,length,polarity
//groups: 线段分组，每个组存按照几何分布顺序顺时针或者逆时针存储着线段索引，线段索引范围是0~line_num-1. 这里由于是指针，使用时要注意(*group)
//first_group_ind、second_group_ind是匹配组队的索引，当提取salient hypothesis时，second_group_ind = -1, fit_matrix2 = NULL.
//fit_matrix1, fit_matrix2, 分别是组队的对应的拟合矩阵
//angles, 是边缘点图+梯度方向。 无边缘点时是NODEF
//distance_tolerance:
//group_inliers_num:记录着各个组的支持内点数量的数组，实时更新，初始时为0
//输出
//ellipara

bool calcEllipseParametersAndValidate(double* lines, int line_num, vector<vector<int>>* groups, int first_group_ind, int second_group_ind, double* fit_matrix1, double* fit_matrix2, image_double angles, double distance_tolerance, unsigned int* group_inliers_num, point5d* ellipara)
{
	double S[36]; //拟合矩阵S
	double Coefficients[6] = { 0,0,0,0,0,0 };// ax^2 + bxy + cy^2 + dx + ey + f = 0 
	double param[5], param2[5];
	int info, addr;
	rect rec;
	rect_iter* iter;
	int rec_support_cnt, rec_inliers_cnt;
	bool flag1 = TRUE, flag2 = TRUE;
	double point_normalx, point_normaly, point_normal, temp;
	vector<point> first_group_inliers, second_group_inliers;
	point pixel_temp;
	double semimajor_errorratio, semiminor_errorratio, iscircle_ratio;
	if (fit_matrix2 == NULL || second_group_ind == -1)//只对一个覆盖度较大的组进行拟合
	{
		for (int i = 0; i < 36; i++)
			S[i] = fit_matrix1[i];
	}
	else
	{
		addFitMatrix(fit_matrix1, fit_matrix2, S);//对组对进行拟合， S = fit_matrix1 + fit_matrix2
	}
	info = fitEllipse2(S, Coefficients);// ax^2 + bxy + cy^2 + dx + ey + f = 0, a > 0
	if (info == 0)//拟合失败
	{
		ellipara = NULL;
		return FALSE;
	}
	ellipse2Param(Coefficients, param);// (x0,y0,a,b,phi)

	if (min(param[2], param[3]) < 3 * distance_tolerance || max(param[2], param[3]) > min(angles->xsize, angles->ysize) || param[0] < 0 || param[0] > angles->xsize || param[1] < 0 || param[1] > angles->ysize)
	{
		ellipara = NULL;
		return FALSE;
	}
	
	//Mat img;
	//img = imread("C:/Users/65406/source/repos/EllipseDetect/EllipseDetect/ce1.png");
	//drawEllipse(img, param);
	//cout << "first param:" << param[0] << " " << param[1] << " " << param[2] << " " << param[3] << " " << param[4] << " " << endl;
	//组队中的 first group先进行内点准则验证，并且更新组的支持内点数量
	for (unsigned int i = 0; i < (*groups)[first_group_ind].size(); i++)
	{
		addr = (*groups)[first_group_ind][i] * 8; //第first_group_ind分组的第i条线段索引*8
		rec.x1 = lines[addr];
		rec.y1 = lines[addr + 1];
		rec.x2 = lines[addr + 2];
		rec.y2 = lines[addr + 3];
		rec.x = (rec.x1 + rec.x2) / 2;
		rec.y = (rec.y1 + rec.y2) / 2;
		rec.dx = lines[addr + 4];
		rec.dy = lines[addr + 5];
		rec.width = 3 * distance_tolerance;
		rec_support_cnt = rec_inliers_cnt = 0;//清零很重要
		if (lines[addr + 7] == 1) //极性一致
		{
			for (iter = ri_ini(&rec); !ri_end(iter); ri_inc(iter))//线段1
			{
				if (iter->x >= 0 && iter->y >= 0 && iter->x < angles->xsize && iter->y < angles->ysize)
				{
					temp = angles->data[iter->y * angles->xsize + iter->x];//内点的梯度方向
					if (temp != NOTDEF)
					{
						//test point's normal is (ax0+by0/2+d/2, cy0+bx0/2+e/2)
						point_normalx = Coefficients[0] * iter->x + (Coefficients[1] * iter->y + Coefficients[3]) / 2;
						point_normaly = Coefficients[2] * iter->y + (Coefficients[1] * iter->x + Coefficients[4]) / 2;
						point_normal = atan2(-point_normaly, -point_normalx); //边缘点的法线方向,指向椭圆内侧
						rec_inliers_cnt++;
						if (angle_diff(point_normal, temp) <= M_1_8_PI) //+- 22.5°内 且 || d - r || < 3 dis_t
						{
							rec_support_cnt++;
							pixel_temp.x = iter->x; pixel_temp.y = iter->y;
							first_group_inliers.push_back(pixel_temp);//添加该线段对应的内点
						}
					}
				}
			}
		}
		else
		{
			for (iter = ri_ini(&rec); !ri_end(iter); ri_inc(iter))//线段1
			{
				//外接矩形可能会越界
				if (iter->x >= 0 && iter->y >= 0 && iter->x < angles->xsize && iter->y < angles->ysize)
				{
					temp = angles->data[iter->y * angles->xsize + iter->x];//内点的梯度方向
					if (temp != NOTDEF)
					{
						//test point's normal is (ax0+by0/2+d/2, cy0+bx0/2+e/2)
						point_normalx = Coefficients[0] * iter->x + (Coefficients[1] * iter->y + Coefficients[3]) / 2;
						point_normaly = Coefficients[2] * iter->y + (Coefficients[1] * iter->x + Coefficients[4]) / 2;
						point_normal = atan2(point_normaly, point_normalx); //边缘点的法线方向,指向椭圆外侧
						rec_inliers_cnt++;
						if (angle_diff(point_normal, temp) <= M_1_8_PI) //+- 22.5°内 且 || d - r || < 3 dis_t
						{
							rec_support_cnt++;
							pixel_temp.x = iter->x; pixel_temp.y = iter->y;
							first_group_inliers.push_back(pixel_temp);//添加该线段对应的内点
						}
					}
				}
			}
		}
		if (!(rec_support_cnt > 0 && (rec_support_cnt >= 0.8 * lines[addr + 6] || rec_support_cnt * 1.0 / rec_inliers_cnt >= 0.6)))
		{
			flag1 = FALSE; //flag1 初始化时为TRUE, 一旦组内有一条线段不满足要求，直接false, 内点准则验证不通过
			break;
		}

	}
	if (flag1 == TRUE && first_group_inliers.size() >= 0.8 * group_inliers_num[first_group_ind])//靠近最大统计过的内点,通过验证
	{
		if (first_group_inliers.size() >= group_inliers_num[first_group_ind])//更新组出现过的最大内点数
			group_inliers_num[first_group_ind] = first_group_inliers.size();
	}
	else
		flag1 = FALSE;
	if (second_group_ind == -1 || fit_matrix2 == NULL)//只对一个覆盖度较大的组进行拟合
	{
		ellipara->x = param[0];//因为无论如何，都需要返回显著性强的椭圆
		ellipara->y = param[1];
		ellipara->a = param[2];
		ellipara->b = param[3];
		ellipara->phi = param[4];
		if (flag1 == TRUE)//通过内点再次拟合，提高质量
		{
			point2d* dataxy = (point2d*)malloc(sizeof(point2d) * first_group_inliers.size());
			for (unsigned int i = 0; i < first_group_inliers.size(); i++)
			{
				dataxy[i].x = first_group_inliers[i].x;
				dataxy[i].y = first_group_inliers[i].y;
			}
			info = fitEllipse(dataxy, first_group_inliers.size(), param2);
			free(dataxy); //释放内存
			cout << first_group_ind << endl;
			cout << "param:" << param[0] << " " << param[1] << " " << param[2] << " " << param[3] << " " << param[4] << " " << endl;
			cout << "param2:" << param2[0] << " " << param2[1] << " " << param2[2] << " " << param2[3] << " " << param2[4] << " " << endl;
			
			if (info == 1 && isEllipseEqual(param2, param, 3 * distance_tolerance, 0.1, 0.1, 0.1, 0.9))
			{
				
				ellipara->x = param2[0];//更新椭圆，提高品质
				ellipara->y = param2[1];
				ellipara->a = param2[2];
				ellipara->b = param2[3];
				ellipara->phi = param2[4];

				Mat img;
				img = imread("C:/Users/65406/source/repos/EllipseDetect/EllipseDetect/ce1.png");
				drawEllipse(img, param2);
			}
		}
		return TRUE;//对于只有一个组的提取椭圆，此时直接返回
	}

	return FALSE;
}




//输入
//lsd算法检测得到的线段集合lines的数量line_num，return的返回值是line_nums条线段，为一维double型数组lines，长度为8*n，每8个为一组
//存着x1,y1,x2,y2,dx,dy,length,polarity
//groups: 线段分组，每个组存按照几何分布顺序顺时针或者逆时针存储着线段索引，线段索引范围是0~line_num-1
//coverages: 每个分组的角度覆盖范围0~2pi，如果组里只有1条线段，覆盖角度为0。数组长度等于分组的数量。
//angles 存边缘点的梯度方向gradient direction, 无边缘点位NOTDEF
//返回值 PairedGroupList* list 返回的是初始椭圆集合的数组，长度list->length. 
//切记，该内存在函数内申请，用完该函数记得释放内存，调用函数freePairedSegmentList()进行释放
PairGroupList* getValidInitialEllipseSet(double* lines, int line_num, vector<vector<int>>* groups, double* coverages, image_double angles, double distance_tolerance, int specified_polarity)
{
	PairGroupList* pairGroupList = NULL;
	PairGroupNode* head, * tail;
	int pairlength = 0;
	point2d pointG1s, pointG1e, pointG2s, pointG2e, g1s_ls_dir, g1e_ls_dir, g2s_ls_dir, g2e_ls_dir;
	double polarity;
	point5d ellipara;
	int groupsNum = (*groups).size();//组的数量
	double* fitMatrixes = (double*)malloc(sizeof(double) * groupsNum * 36);//定义拟合矩阵S_{6 x 6}. 每个组都有一个拟合矩阵
	unsigned int* supportInliersNum = (unsigned int*)malloc(sizeof(int) * groupsNum);//用于存储每个组曾经最大出现的支持内点数量
	memset(fitMatrixes, 0, sizeof(double) * groupsNum * 36);
	memset(supportInliersNum, 0, sizeof(unsigned int) * groupsNum);//初始化为0.
	int i, j;
	int cnt_temp, ind_start, ind_end;
	bool info;

	//实例化拟合矩阵Si
	point2d* dataxy = (point2d*)malloc(sizeof(point2d) * line_num * 2);//申请足够大内存, line_num条线段，共有2line_num个端点
	for (i = 0; i < groupsNum; i++)
	{
		cnt_temp = 0;//千万注意要清0
		for (j = 0; j < (*groups)[i].size(); j++)
		{
			//每一条线段有2个端点
			dataxy[cnt_temp].x = lines[(*groups)[i][j] * 8];
			dataxy[cnt_temp++].y = lines[(*groups)[i][j] * 8 + 1];
			dataxy[cnt_temp].x = lines[(*groups)[i][j] * 8 + 2];
			dataxy[cnt_temp++].y = lines[(*groups)[i][j] * 8 + 3];
		}
		calcuFitMatrix(dataxy, cnt_temp, fitMatrixes + i * 36);
	}
	free(dataxy);//释放内存
	head = tail = NULL;//将初始椭圆集合存储到链表中
	int salient_num = 0;
	//selection of salient elliptic hypothesis
	for (i = 0; i < groupsNum; i++)
	{
		if (coverages[i] >= M_4_9_PI)//当组的覆盖角度>= 4pi/9 = 80°, 我们认为具有很大的显著性，可直接拟合提取
		{
			if (specified_polarity == 0 || (lines[(*groups)[i][0] * 8 + 7] == specified_polarity))
			{
				info = calcEllipseParametersAndValidate(lines, line_num, groups, i, -1, (fitMatrixes + i * 36), NULL, angles, distance_tolerance, supportInliersNum, &ellipara);
				salient_num++;

			}
		}

	}
	cout << "salient num = " << salient_num << endl;
	return pairGroupList;
}





int main()
{
    /*double* image;*/
    //int X = 128;  /* x image size */
    //int Y = 128;  /* y image size */

    /* create a simple image: left half black, right half gray */
    /*image = (double*)malloc(X * Y * sizeof(double));*/
    /*if (image == NULL)
    {
        fprintf(stderr, "error: not enough memory\n");
        exit(EXIT_FAILURE);
    }*/
    //for (x = 0; x < X; x++)
    //    for (y = 0; y < Y; y++)
    //        image[x + y * X] = x < X / 2 ? 0.0 : 64.0; /* image(x,y) */

    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
    Mat img;
    Mat grayimg;
    img = imread("C:/Users/65406/source/repos/EllipseDetect/EllipseDetect/ce2.png");  
    cvtColor(img, grayimg, COLOR_BGR2GRAY);//转为灰度图
    int imgx, imgy;
    imgy = (int)img.cols;
    imgx = (int)img.rows;
    double *image = (double*)malloc(imgx * imgy * sizeof(double));//将输入矩阵中的图像数据转存到一维数组中
    if (image == NULL)
    {
        fprintf(stderr, "error: not enough memory\n");
        exit(EXIT_FAILURE);
    }
    for (int x = 0; x < imgx; x++)
        for (int y = 0; y < imgy; y++)
            image[x + y * imgx] = grayimg.data[y + x * imgy];
    
    int n;//线段数量
    vector<vector<int>> groups;
	double* coverages;
    int* reg;
    int reg_x;
    int reg_y;
    /* LSD call */
    double* out = mylsd(&n, image, imgx, imgy, &reg, &reg_x, &reg_y);
	// out为一维double型数组,长度为8*n，每8个为一组，格式为（x1,y1,x2,y2,dx,dy,width,polarity)
    groupLSs(out, n, reg, reg_x, reg_y, &groups); 
	/*for (int i = 0; i < groups.size(); i++) {
		for( int j = 0; j < groups[i].size(); j++)
		{
			cout << groups[i][j] << "  ";
		}
		cout << endl;
		
	}*/
		

    cout <<"The number of output arc-support line segments: " << n << endl;
	cout << "The number of arc-support groups: " << groups.size() << endl;
	free(reg);
	calcuGroupCoverage(out, n, groups, coverages);//计算每个组的覆盖角度
	/*for (int i = 0; i < groups.size(); i++)
	{
		cout << coverages[i] << endl;
	}*/

	//int groups_t = 0;
	/*for (int i = 0; i<groups.size(); i++)
	{
		groups_t+= groups[i].size();
	}
	printf("Groups' total ls num:%i\n",groups_t);*/

	image_double angles = nullptr;
	int edge_process_select = 1; //version2, sobel; version 3 canny
	if (edge_process_select == 1)
	{
		calculateGradient2(image, imgx, imgy, &angles);
	}
	
	PairGroupList* pairGroupList;
	double distance_tolerance = 2;
	double* candidates; //候选椭圆
	double* candidates_out;//输出候选椭圆指针
	int  candidates_num = 0;//候选椭圆数量
	int specified_polarity = 0;//1,指定检测的椭圆极性要为正; -1指定极性为负; 0表示两种极性椭圆都检测
	pairGroupList = getValidInitialEllipseSet(out, n, &groups, coverages, angles, distance_tolerance, specified_polarity);





    //for (int i = 0; i < n; i++)
    //{
    //    for (int j = 0; j < 7; j++)
    //        printf("%f ", out[7 * i + j]);
    //    printf("\n");
    //}






    //输出线段检测的图像
    Mat ls_mat = Mat::zeros(imgy, imgx, CV_8UC1);
    for (int i = 0; i < n; i++)//draw lines
    {
        Point2d p1(out[8 * i], out[8 * i + 1]), p2(out[8 * i + 2], out[8 * i + 3]);
        line(ls_mat, p1, p2, Scalar(255, 0, 0));
    } 
    imshow("img1", ls_mat);
    //imwrite("img.jpg", ls_mat);
    waitKey();




    /* free memory */
    free((void*)image);
    free((void*)out);

    return EXIT_SUCCESS;

}


// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
