#include <cstdio>
#include <iostream>
#include <omp.h>
#include <time.h>
#include <cstdlib>
#include <cstring>
#include <assert.h>
#include <math.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <unordered_set>
#include <queue>

#include <mmintrin.h> 	// MMX
#include <xmmintrin.h>	// SSE
#include <emmintrin.h>	// SSE2
#include <immintrin.h>	// AVX

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

template<typename SRC_TYPE, typename DST_TYPE, int channels> void gaussianBlur(const Mat& src, Mat& dst, Size ksize, double sigmaX, double sigmaY)
{
	// get gaussian kernel
	int kx = 2 * static_cast<int>(3.0 * sigmaX) + 1;
	int ky = 2 * static_cast<int>(3.0 * sigmaY) + 1;
	kx = min(kx, ksize.width);
	ky = min(ky, ksize.height);
	assert(kx % 2 == 1 && ky % 2 == 1);
	int radiusX = kx / 2;
	int radiusY = ky / 2;
	kx = radiusX + 1;
	ky = radiusY + 1;
	float kerY[ky];
	float kerX[kx];
	assert(sigmaX > 0 && sigmaY > 0);
	double weightX = 1.0 / (sqrt(2.0 * M_PI) * sigmaX);
	double weightY = 1.0 / (sqrt(2.0 * M_PI) * sigmaY);
	double invSqrSigmaX = 1.0 / (2.0 * sigmaX * sigmaX);
	double invSqrSigmaY = 1.0 / (2.0 * sigmaY * sigmaY);
	double sumY = 0.0;
	for(int y = 0; y < ky; y++) {
		kerY[y] = weightY * exp( - y * y * invSqrSigmaY );
		if(y > 0) sumY += 2.0 * kerY[y];
		else sumY += kerY[y];
	}
	for(int y = 0; y < ky; y++) kerY[y] /= sumY;
	double sumX = 0.0; 
	for(int x = 0; x < kx; x++) {
		kerX[x] = weightX * exp( - x * x * invSqrSigmaX );
		if(x > 0) sumX += 2.0 * kerX[x];
		else sumX += kerX[x];
	}
	for(int x = 0; x < kx; x++) kerX[x] /= sumX;
	
	int h = src.rows;
	int w = src.cols;
	int w_ = w;
#ifdef _SSE2
	w = w % 16 == 0 ? w : 16 * (w / 16 + 1);
#endif
	int ww = w + 2 * radiusX;
	int ww_ = w_ + 2 * radiusX;
	int hh = h + 2 * radiusY;
	assert(src.rows == dst.rows && src.cols == dst.cols);
	Mat rowBuff;
	Mat colBuff;
	if(channels == 1) {
		rowBuff = Mat(h, ww, CV_8UC1);
		colBuff = Mat(hh, w, CV_32FC1);
	} 
	else if(channels == 2) {
		rowBuff = Mat(h, ww, CV_8UC2);
		colBuff = Mat(hh, w, CV_32FC2);
	}
	else if(channels == 3) {
		rowBuff = Mat(h, ww, CV_8UC3);
		colBuff = Mat(hh, w, CV_32FC3);
	}
	const int threadNum = 4;
	for(int y = 0; y < h; y++) {
		SRC_TYPE * rowPtr = rowBuff.ptr<SRC_TYPE>(y) + radiusX;
		const SRC_TYPE * srcPtr = src.ptr<SRC_TYPE>(y);
		memcpy(rowPtr, srcPtr, sizeof(SRC_TYPE) * w_);
	}
	
	// left & right
	for(int y = 0; y < h; y++) {
		SRC_TYPE * rowPtr = rowBuff.ptr<SRC_TYPE>(y);
		for(int x = 0; x < radiusX; x++) {
			rowPtr[x] = rowPtr[radiusX];
			rowPtr[x + ww_ - radiusX] = rowPtr[ww_ - radiusX - 1];
		}
	}

#ifdef _SSE2
	int cn = channels;
	assert(w % 16 == 0);
	__m128i z = _mm_setzero_si128();
	// apply gaussian filter
	#ifdef _OPENMP
	#pragma omp parallel num_threads(threadNum) shared(rowBuff, colBuff, dst, kerX, kerY)
	#endif
	{
	// row filter
	#ifdef _OPENMP
	#pragma omp for schedule(static, h / threadNum)
	#endif
	for(int y = 0; y < h; y++) {
		SRC_TYPE * rowPtr = rowBuff.ptr<SRC_TYPE>(y) + radiusX;
		DST_TYPE * colPtr = colBuff.ptr<DST_TYPE>(y + radiusY);
		uchar * srcRaw = reinterpret_cast<uchar*>(rowPtr);
		float * dstRaw = reinterpret_cast<float*>(colPtr);
		int x = 0;
		for( ; x < w * cn; x += 16, srcRaw += 16) {
			__m128 f = _mm_load_ss(kerX);
			f = _mm_shuffle_ps(f, f, 0);
			__m128i x0 = _mm_loadu_si128((__m128i*)(srcRaw));
			__m128i x1, x2, x3, x4, y0;
			x1 = _mm_unpackhi_epi8(x0, z);
			x2 = _mm_unpacklo_epi8(x0, z);
			x3 = _mm_unpackhi_epi16(x2, z);
			x4 = _mm_unpacklo_epi16(x2, z);
			x2 = _mm_unpacklo_epi16(x1, z);
			x1 = _mm_unpackhi_epi16(x1, z);
			__m128 s1, s2, s3, s4;
			s1 = _mm_mul_ps(f, _mm_cvtepi32_ps(x1));
			s2 = _mm_mul_ps(f, _mm_cvtepi32_ps(x2));
			s3 = _mm_mul_ps(f, _mm_cvtepi32_ps(x3));
			s4 = _mm_mul_ps(f, _mm_cvtepi32_ps(x4));
			for(int k = 1; k < kx; k++) {
				f = _mm_load_ss(kerX + k);
				f = _mm_shuffle_ps(f, f, 0);
				uchar * shi = srcRaw + k * cn;
				uchar * slo = srcRaw - k * cn;
				x0 = _mm_loadu_si128((__m128i*)(shi));
				y0 = _mm_loadu_si128((__m128i*)(slo));
				x1 = _mm_unpackhi_epi8(x0, z);
				x2 = _mm_unpacklo_epi8(x0, z);
				x3 = _mm_unpackhi_epi8(y0, z);
				x4 = _mm_unpacklo_epi8(y0, z);
				x1 = _mm_add_epi16(x1, x3);
				x2 = _mm_add_epi16(x2, x4);
			
				x3 = _mm_unpackhi_epi16(x2, z);
				x4 = _mm_unpacklo_epi16(x2, z);
				x2 = _mm_unpacklo_epi16(x1, z);
				x1 = _mm_unpackhi_epi16(x1, z);
				s1 = _mm_add_ps(s1, _mm_mul_ps(f, _mm_cvtepi32_ps(x1)));
				s2 = _mm_add_ps(s2, _mm_mul_ps(f, _mm_cvtepi32_ps(x2)));
				s3 = _mm_add_ps(s3, _mm_mul_ps(f, _mm_cvtepi32_ps(x3)));
				s4 = _mm_add_ps(s4, _mm_mul_ps(f, _mm_cvtepi32_ps(x4)));	
			}
			_mm_storeu_ps(dstRaw + x, s4);
			_mm_storeu_ps(dstRaw + x + 4, s3);
			_mm_storeu_ps(dstRaw + x + 8, s2);
			_mm_storeu_ps(dstRaw + x + 12, s1);
		}
	}
	
	if(omp_get_thread_num() == 0) {
		DST_TYPE * topPtr = colBuff.ptr<DST_TYPE>(0);
		DST_TYPE * botPtr = colBuff.ptr<DST_TYPE>(hh - radiusY);
		DST_TYPE * topLin = colBuff.ptr<DST_TYPE>(radiusY);
		DST_TYPE * botLin = colBuff.ptr<DST_TYPE>(hh - radiusY - 1);	
		for(int y = 0; y < radiusY; y++, topPtr += w, botPtr += w) {
			memcpy(topPtr, topLin, sizeof(DST_TYPE) * w);
			memcpy(botPtr, botLin, sizeof(DST_TYPE) * w);
		}
	}

	// column filter
	#ifdef _OPENMP
	#pragma omp for schedule(static, h / threadNum)
	#endif
	for(int y = 0; y < h; y++) {
		DST_TYPE * srcPtr = colBuff.ptr<DST_TYPE>(y + radiusY);
		SRC_TYPE * dstPtr = dst.ptr<SRC_TYPE>(y);
		float * srcRaw = reinterpret_cast<float*>(srcPtr);
		uchar * dstRaw = reinterpret_cast<uchar*>(dstPtr);
		int x = 0;
		for( ; x < w * cn; x += 16, srcRaw += 16) {
			__m128 f = _mm_load_ss(kerY);
			f = _mm_shuffle_ps(f, f, 0);
			__m128 s1, s2, s3, s4;
			__m128 s0;
			s1 = _mm_loadu_ps(srcRaw);
			s2 = _mm_loadu_ps(srcRaw + 4);
			s3 = _mm_loadu_ps(srcRaw + 8);
			s4 = _mm_loadu_ps(srcRaw + 12);
			s1 = _mm_mul_ps(s1, f);
			s2 = _mm_mul_ps(s2, f);
			s3 = _mm_mul_ps(s3, f);
			s4 = _mm_mul_ps(s4, f);
			for(int k = 1; k < ky; k++) {
				f = _mm_load_ss(kerY + k);
				f = _mm_shuffle_ps(f, f, 0);
				s0 = _mm_add_ps(_mm_loadu_ps(srcRaw + k * w * cn), _mm_loadu_ps(srcRaw - k * w * cn));
				s1 = _mm_add_ps(s1, _mm_mul_ps(f, s0));
				s0 = _mm_add_ps(_mm_loadu_ps(srcRaw + 4 + k * w * cn), _mm_loadu_ps(srcRaw + 4 - k * w * cn));
				s2 = _mm_add_ps(s2, _mm_mul_ps(f, s0));
				s0 = _mm_add_ps(_mm_loadu_ps(srcRaw + 8 + k * w * cn), _mm_loadu_ps(srcRaw + 8 - k * w * cn));
				s3 = _mm_add_ps(s3, _mm_mul_ps(f, s0));
				s0 = _mm_add_ps(_mm_loadu_ps(srcRaw + 12 + k * w * cn), _mm_loadu_ps(srcRaw + 12 - k * w * cn));
				s4 = _mm_add_ps(s4, _mm_mul_ps(f, s0));
			}
			__m128i x1 = _mm_cvttps_epi32(s1);
			__m128i x2 = _mm_cvttps_epi32(s2);
			__m128i x3 = _mm_cvttps_epi32(s3);
			__m128i x4 = _mm_cvttps_epi32(s4);
			x1 = _mm_packs_epi32(x1, x2);
			x2 = _mm_packs_epi32(x3, x4);
			x1 = _mm_packus_epi16(x1, x2);
			uchar buff[16] __attribute__((aligned(16)));
			int len = min(16, w_ * cn - x);
			_mm_store_si128((__m128i*)buff, x1);
			if(len > 0) memcpy(dstRaw + x, buff, len);
		}
	}
	}
#else
	// apply gaussian filter
	#ifdef _OPENMP
	#pragma omp parallel num_threads(threadNum) shared(rowBuff, colBuff, dst, kerX, kerY)
	#endif
	{
	// row filter
	#ifdef _OPENMP
	#pragma omp for schedule(static, h / threadNum)
	#endif
	for(int y = 0; y < hh; y++) {
		SRC_TYPE * rowPtr = rowBuff.ptr<SRC_TYPE>(y) + radiusX;
		DST_TYPE * colPtr = colBuff.ptr<DST_TYPE>(y);
		for(int x = 0; x < w; x++) {
			DST_TYPE vec = kerX[0] * rowPtr[x];
			for(int xx = 1; xx < kx; xx++) {
				vec += kerX[xx] * (rowPtr[x + xx] + rowPtr[x - xx]);
			}
			colPtr[x] = vec;
		}
	}

	if(omp_get_thread_num() == 0) {
		DST_TYPE * topPtr = colBuff.ptr<DST_TYPE>(0);
		DST_TYPE * botPtr = colBuff.ptr<DST_TYPE>(hh - radiusY);
		DST_TYPE * topLin = colBuff.ptr<DST_TYPE>(radiusY);
		DST_TYPE * botLin = colBuff.ptr<DST_TYPE>(hh - radiusY - 1);	
		for(int y = 0; y < radiusY; y++, topPtr += w, botPtr += w) {
			memcpy(topPtr, topLin, sizeof(DST_TYPE) * w);
			memcpy(botPtr, botLin, sizeof(DST_TYPE) * w);
		}
	}	

	// column filter
	#ifdef _OPENMP
	#pragma omp for schedule(static, h / threadNum)
	#endif
	for(int y = 0; y < h; y++) {
		DST_TYPE * srcPtr = colBuff.ptr<DST_TYPE>(y + radiusY);
		SRC_TYPE * dstPtr = dst.ptr<SRC_TYPE>(y);
		for(int x = 0; x < w; x++) {
			DST_TYPE vec = kerY[0] * srcPtr[x];		
			for(int yy = 1; yy < ky; yy++) {
				vec += kerY[yy] * (*(srcPtr + yy * w + x) + *(srcPtr - yy * w + x));
			}
			vec = vec < 0 ? 0.0 : vec; vec = vec > 0xff ? 0xff : vec;
			dstPtr[x] = SRC_TYPE(vec);
		}
	}
	}
#endif



}

void myGaussianBlur(const Mat& src, Mat& dst, Size ksize, double sigmaX, double sigmaY)
{
	int channels = src.channels();
	if(channels == 3) {
		gaussianBlur<Vec3b, Vec3f, 3>(src, dst, ksize, sigmaX, sigmaY);
	}
	else if(channels == 2) {
		gaussianBlur<Vec2b, Vec2f, 2>(src, dst, ksize, sigmaX, sigmaY);
	}
	else if(channels == 1) {
		gaussianBlur<uchar, float, 1>(src, dst, ksize, sigmaX, sigmaY);
	}
}

template <typename SRC_TYPE, typename DST_TYPE>
void gaussGradMagOri(const Mat& gauss, Mat& mag, Mat& ori)
{
	int h = gauss.rows; int w = gauss.cols;
	const int templateBorder = 1;	
	const double FIXPT_SCALE = 1.0 / 255;

	for(int y = templateBorder; y < h - templateBorder; y++) {
		const SRC_TYPE * g_ptr = gauss.ptr<SRC_TYPE>(y);
		DST_TYPE * m_ptr = mag.ptr<DST_TYPE>(y);
		DST_TYPE * o_ptr = ori.ptr<DST_TYPE>(y);
		for(int x = templateBorder; x < w - templateBorder; x++) {
			double gx = (g_ptr[x + 1] - g_ptr[x - 1]) * 2.0 
				  + (g_ptr[x + w + 1] - g_ptr[x + w - 1])
				  + (g_ptr[x - w + 1] - g_ptr[x - w - 1]);
			double gy = (g_ptr[x + w] - g_ptr[x - w]) * 2.0
				  + (g_ptr[x + w + 1] - g_ptr[x - w + 1])
				  + (g_ptr[x + w - 1] - g_ptr[x - w - 1]);
			gx *= FIXPT_SCALE * 0.25;
			gy *= FIXPT_SCALE * 0.25;
			double m = sqrt(0.5 * gx * gx + 0.5 * gy * gy);
			double o = atan2(gy, gx);
		//	o = o < 0 ? - o : o;
			o = o < 0 ? o + M_PI : o;
			o = o / M_PI * 180.0;
			m_ptr[x] = m;
			o_ptr[x] = o;
		}
	}
}


template <typename SRC_TYPE>
void edgeExtrator(const Mat& mag, const Mat& ori, Mat& edge, vector<Vec2i>& stPix, vector<Vec2i>& wkPix, double stVal, double wkVal)
{	
	int h = mag.rows; int w = mag.cols;
	assert(h == ori.rows && w == ori.cols);
	const int templateBorder = 1;	

	for(int y = templateBorder; y < h - templateBorder; y++) {
		const SRC_TYPE * m_ptr = mag.ptr<SRC_TYPE>(y);
		const SRC_TYPE * o_ptr = ori.ptr<SRC_TYPE>(y);
		uchar * e_ptr = edge.ptr<uchar>(y);
		for(int x = templateBorder; x < w - templateBorder; x++) {
			double m = m_ptr[x];
			double o = o_ptr[x];
			double m_left(0.0), m_right(0.0);
			if(o <= 22.5 || o > 157.5) {
				m_left = m_ptr[x - 1];
				m_right = m_ptr[x + 1];
			} else if(o > 22.5 && o <= 67.5) {
				m_left = m_ptr[x - w - 1];
				m_right = m_ptr[x + w + 1];
			} else if(o > 67.5 && o <= 112.5) {
				m_left = m_ptr[x - w];
				m_right = m_ptr[x + w];
			} else if(o > 112.5 && o <= 157.5) {
				m_left = m_ptr[x - w + 1];
				m_right = m_ptr[x + w - 1];
			}
			if(m > m_left && m > m_right) {
				if(m > stVal) {
					stPix.push_back(Vec2i(x, y));
					e_ptr[x] = 0xff;
				}
				else if(m <= stVal && m > wkVal) {
					wkPix.push_back(Vec2i(x, y));
					e_ptr[x] = 0xf;
				}
				else if(m < wkVal) {
					e_ptr[x] = 0x0;
				}	
			} else {
				e_ptr[x] = 0x0;
			}
		}
	}
}

void blobAnalysis(Mat& edge, vector<Vec2i>& stPix, vector<Vec2i>& wkPix)
{
	int h = edge.rows; int w = edge.cols;

	while(!stPix.empty()) {
		Vec2i& pix = stPix.back();
		int x = pix[0];
		int y = pix[1];
		stPix.pop_back();
//		assert(x > 0 && x < w - 1 && y > 0 && y < h - 1);
		if(x < 1 || x > w - 2 || y < 1 || y > h - 2) {
			continue;
		}
		uchar * ptr = edge.ptr<uchar>(y) + x;
	//	if(ptr[1] == 0xff || ptr[-1] == 0xff
	//	|| ptr[w] == 0xff || ptr[-w] == 0xff
	//	|| ptr[w + 1] == 0xff || ptr[w - 1] == 0xff 
	//	|| ptr[-w + 1] == 0xff || ptr[-w - 1] == 0xff) {
	//		stPix.pop_back();
	//	}
//		else {
			for(int j = - 1; j <= 1; j++) {
				for(int i = - 1; i <= 1; i++) {
					if(i == 0 && j == 0) continue;
					else if(ptr[j * w + i] == 0xf) {
						stPix.push_back(Vec2i(x + i, y + j));
						ptr[j * w + i] = 0xff;			
					}
				}
			}
//		}	
	}
	for(size_t i = 0; i < wkPix.size(); i++) {
		Vec2i pix = wkPix[i];
		int x = pix[0]; int y = pix[1];
		uchar* val = edge.ptr<uchar>(y) + x;
		if(*val < 0xff) *val = 0x0;
	}
//	for(size_t i = 0; i < wkPix.size(); i++) {
//		Vec2i pix = wkPix[i];
//		int x = pix[0];
//		int y = pix[1];
////		cout << x << "\t" << y << "\n";
//		assert(x > 0 && x < w - 1 && y > 0 && y < h - 1);
//		uchar * ptr = edge.ptr<uchar>(y) + x;
//		if(ptr[1] == 0xff || ptr[-1] == 0xff
//		|| ptr[w] == 0xff || ptr[-w] == 0xff
//		|| ptr[w + 1] == 0xff || ptr[w - 1] == 0xff 
//		|| ptr[-w + 1] == 0xff || ptr[-w - 1] == 0xff) {
//			ptr[0] = 0xff;
//		}
//	}
}

bool canny(const Mat& src, Mat& edge, double stVal, double wkVal)
{
	assert(stVal > wkVal && wkVal > 0.0 && stVal <= 1.0);	
	int cn = src.channels();
	if(cn != 1) {
		fprintf(stderr, "ERROR: canny algorithm asserts number of channels of image is 1, but argument is %d\n", cn);
		return false;
	}

	assert(cn == 1);
	
	Mat g(src.size(), src.type());
	Mat mag(src.size(), CV_32FC1);
	Mat ori(src.size(), CV_32FC1);

	myGaussianBlur(src, g, Size(5, 5), 1.4, 1.4);
	gaussGradMagOri<uchar, float>(g, mag, ori);
//	imwrite("mag.jpg", mag);
//	imwrite("ori.jpg", ori);
	vector<Vec2i> stPix;
	vector<Vec2i> wkPix;
	edgeExtrator<float>(mag, ori, edge, stPix, wkPix, stVal, wkVal);
	blobAnalysis(edge, stPix, wkPix);
	return true;
}

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
	LineCount(const LineCount& lct) {
		count = lct.count;
		start = lct.start;
		end = lct.end;
	}	
};

inline bool operator<(const LineCount& __left, const LineCount& __right)
{
	return __left.count > __right.count;
}

struct comp {
	inline bool operator()(const LineCount& __left, const LineCount& __right)
	{
		return __left.count > __right.count;
	}
};

void houghLine(const Mat& edge, vector<Vec4i>& line, int threshold, int numLines, double dTheta = 1 / 180.0 * M_PI, double dRho = 1)
{
	int h = edge.rows; int w = edge.cols;
	
#define ROUND(x) (static_cast<int>(x + 0.5))
	int numTheta = ROUND(M_PI / dTheta);
	double maxRho = sqrt(h * h + w * w);
	int numRho = ROUND(2.0 * maxRho / dRho);
	
	vector<vector<LineCount> > countBuff(numTheta + 2, vector<LineCount>(numRho + 2, LineCount(0, Point2i(w - 1, 0), Point2i(0, 0))));
	vector<LineCount> sortBuff(numLines, LineCount(threshold, Point2i(0, 0), Point2i(0, 0)));
	priority_queue<LineCount> sortBuff_(sortBuff.begin(), sortBuff.end());
	vector<LineCount>().swap(sortBuff);
	double * cosTab = ArrayAlloc<double>(numTheta);
	double * sinTab = ArrayAlloc<double>(numTheta);
	double theta = 0.0;
	for(int n = 0; n < numTheta; n++, theta += dTheta) {
		cosTab[n] = cos(static_cast<double>(theta)) / dRho;
		sinTab[n] = sin(static_cast<double>(theta)) / dRho;
	}

	
	for(int y = 0; y < h; y++) {
		const uchar * ptr = edge.ptr<uchar>(y);
		for(int x = 0; x < w; x++, ptr++) {
			if(*ptr == 0xff) {
				for(int iTheta = 0; iTheta < numTheta; iTheta++) {
					double rho = x * cosTab[iTheta] + y * sinTab[iTheta];
					int iRho = static_cast<int>(rho + (maxRho) / dRho);
		//			if(iRho >= numRho || iRho < 0) {
		//				cout << iRho << "\n";
		//				return;
		//			}
					LineCount& lct = countBuff[iTheta + 1][1 + iRho];
					lct.count++;
				//	cout << iRho << "\t" << lct.count << "\n";
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

	for(int iTheta = 1; iTheta < numTheta + 1; iTheta++) {
		for(int iRho = 1; iRho < numRho + 1; iRho++) {
			LineCount& lct = countBuff[iTheta][iRho];
			if(lct.count >= threshold
			&& lct.count >= countBuff[iTheta + 1][iRho].count && lct.count >= countBuff[iTheta - 1][iRho].count
			&& lct.count >= countBuff[iTheta][iRho + 1].count && lct.count >= countBuff[iTheta][iRho - 1].count) {
				const LineCount& pqTop = sortBuff_.top();
				if(lct.count >= pqTop.count) {
					sortBuff_.pop();
					sortBuff_.push(lct);
				}
				
			}
		}
	}

	while(!sortBuff_.empty()) {
		const LineCount& pqEle = sortBuff_.top();
//		cout << pqEle.count << "\n";
		const Point2i& start = pqEle.start;
		const Point2i& end = pqEle.end;
		line.push_back(Vec4i(start.x, start.y, end.x, end.y));
		sortBuff_.pop();
	}
	free(sinTab); free(cosTab);
}

int main(int argc, char * argv[])
{
	if(argc < 5) {
		fprintf(stdout, "Usage: inputfile edgefile strongval weakval\n");
		return -1;
	}
	const char * inputimage = argv[1];
	const char * edgeimage = argv[2];
	double strongval = atof(argv[3]);
	double weakval = atof(argv[4]);
	Mat src = imread(inputimage);
	Mat srcGray;
	cvtColor(src, srcGray, CV_BGR2GRAY);
	Mat edge(src.size(), CV_8U);
	MatIterator_<uchar> it = edge.begin<uchar>(), it_end = edge.end<uchar>();
	for(; it != it_end; it++) *it = 0x0;
	bool ret = canny(srcGray, edge, strongval, weakval);
	if(ret) fprintf(stdout, "INFO: canny edge extractor executed successfully!\n");
	imwrite(edgeimage, edge);
	vector<Vec4i> line_;
	houghLine(edge, line_, 100, 4);
	Scalar color(0, 255, 0);
	for(unsigned int i = 0; i < line_.size(); i++) {
		int x1 = line_[i][0];
		int y1 = line_[i][1];
		int x2 = line_[i][2];
		int y2 = line_[i][3];
		line(src, Vec2i(x1, y1), Vec2i(x2, y2), color, 3);
	}
	imshow("image", src);
	waitKey();
	return 0;	
}
