#include <cstdio>
#include <iostream>
#include <omp.h>
#include <time.h>
#include <cstdlib>
#include <cstring>
#include <assert.h>
#include <math.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <unordered_set>

#include <mmintrin.h> 	// MMX
#include <xmmintrin.h>	// SSE
#include <emmintrin.h>	// SSE2
#include <immintrin.h>	// AVX

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

template<typename T> T * ArrayAlloc(const size_t n0)
{
	T * ptr __attribute__((aligned(16))) = (T*)malloc(sizeof(T) * n0);
	memset(ptr, 0, sizeof(T) * n0);
	return ptr;
}

#define ArrayFree3D(ptr) \
{ \
	free(**(ptr)); \
	free(*(ptr)); \
	free(ptr); \
}

#define ArrayFree2D(ptr) \
{ \
	free(*(ptr)); \
	free(ptr); \
}

/* @brief: template function for 2d array allocation
 * @param: n0, n1, size of dimensions
 * @retval: buffer, buffer for array
 */
template<typename T> void ArrayAlloc2D(T ***buffer, const int n0, const int n1)
{
	*buffer = (T**)malloc(sizeof(T*) * n0);
	(*buffer)[0] = (T*)malloc(sizeof(T) * n0 * n1);
	for(int i0 = 1; i0 < n0; i0++) (*buffer)[i0] = (*buffer)[0] + i0 * n1;
	return;
}

/* @brief: template function for 3d array allocation
 * @param: n0, n1, n2 size of dimensions
 * @retval: buffer, buffer for array
 */
template<typename T> void ArrayAlloc3D(T ****buffer, const int n0, const int n1, const int n2)
{
	*buffer = (T***)malloc(sizeof(T**) * n0);
	(*buffer)[0] = (T**)malloc(sizeof(T*) * n0 * n1);
	for(int i0 = 1; i0 < n0; i0++) (*buffer)[i0] = (*buffer)[0] + i0 * n1;
	(*buffer)[0][0] = (T*)malloc(sizeof(T) * n0 * n1 * n2);
	for(int i1 = 1; i1 < n1; i1++) (*buffer)[0][i1] = (*buffer)[0][0] + i1 * n2;
	for(int i0 = 1; i0 < n0; i0++) {
		for(int i1 = 0; i1 < n1; i1++) {
			(*buffer)[i0][i1] = (*buffer)[0][0] + i0 * n1 * n2 + i1 * n2;
		}
	}
	return;
}


struct LineCount {
	int count;
	Point2i start;
	Point2i end;
	LineCount() : count(0), start(Point2i(0, 0)), end(Point2i(0, 0)) { };
	LineCount(int count_, const Point2i& start_, const Point2i& end_) : count(count_), start(start_), end(end_) { };	
};

void houghLine(const Mat& edge, vector<Vec4i>& line, int count, double minTheta = -89.0, double maxTheta = 89.0, double minB = 4.0, double maxB = 4.0)
{
	int h = edge.rows; int w = edge.cols;
	minB = minB * h; maxB = maxB * h;
	const int nSamples = 201;
	const double dTheta = (maxTheta - minTheta) / (nSamples - 1.0);
	const double db = (maxB - minB) / (nSamples - 1.0);
	LineCount** countBuff;
	ArrayAlloc2D<LineCount>(&countBuff, nSamples, nSamples);
	memset(countBuff[0], 0, sizeof(LineCount) * nSamples * nSamples);
	double * k = ArrayAlloc<double>(nSamples);
	for(int sample = 0; sample < nSamples; sample++) {
		double theta = minTheta + sample * dTheta;
		double slope = tan(theta);	
		k[sample] = slope;
	}


	LineCount * ct_ptr = countBuff[0];
	for(int icount = 0; icount < nSamples * nSamples; icount++, ct_ptr++) {
		ct_ptr[0].start = Point2i(w - 1, 0);
		ct_ptr[0].end = Point2i(0, 0); 
	}
	for(int y = 0; y < h; y++) {
		uchar * ptr = edge.ptr<uchar>(y);
		for(int x = 0; x < w; x++, ptr++) {
			if(*ptr == 0xff) {
				for(int sample = 0; sample < nSamples; sample++) {
					double slope = k[sample];
					double b = - slope * x + y;
					int sample_b = static_cast<int>((b - minB) / db);
					if(sample_b >= 0 && sample_b <= nSamples - 1) {
						LineCount& lct = countBuff[sample][sample_b];
						lct.count++;
						int x1 = lct.start.x;
						int x2 = lct.end.x;
						if(x < x1) {
							lct.start = Point2i(x, y);
						}
						else if(x > x2) {
							lct.end = Point2i(x, y);
						}
					}
				}
			}
		}
	}

	ct_ptr = countBuff[0];
	for(int sample = 0; sample < nSamples; sample++, ct_ptr += nSamples) {
		for(int sample_b = 0; sample_b < nSamples; sample_b++) {
			LineCount& lct = ct_ptr[sample_b];
			if(lct.count >= count) {
				Point2i start = lct.start;
				Point2i end = lct.end;		
				line.push_back(Vec4i(start.x, start.y, end.x, end.y));
			}		
		}
	}		
}
