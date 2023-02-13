
#include <iostream>
#include <stdio.h>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>




using namespace cv;
using namespace std;



Mat origin = imread("original.png", 0);

float getPSNR(Mat noise)
{

    double MSE = 0;
    long double sum = 0;
    for (auto i = 0; i < origin.rows; i++) {
        for (auto j = 0; j < origin.cols; j++) {
            sum += pow(((origin.at<uchar>(i, j)) - (noise.at<uchar>(i, j))), 2);
        }
    }
    double size = origin.cols * origin.rows;
    MSE = sum / size;
    double PSNR = 10 * log10((255 * 255) / MSE);
    return PSNR;
}

void MyHarmonicMeanFilter(InputArray input, OutputArray output, int windowSize) {
    const Mat& img = input.getMat();
    output.create(img.size(), img.type());
    Mat dst = output.getMat();
    int range = (windowSize - 1) / 2;
    float sum = 0;
    uchar temp;

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            sum = 0;
            for (int t = -range; t <= range; t++)    //y축
                for (int s = -range; s <= range; s++) { //x축

                    temp = img.at<uchar>(min(img.rows - 1, max(y + t, 0)), min(img.cols - 1, (max(x + s, 0))));
                    sum += pow(temp, -1);

                }
            dst.at<uchar>(y, x) = (windowSize * windowSize) / sum;
        }
    }
}

void MyContraharmonicMeanFilter(InputArray input, OutputArray output, int windowSize, float Q) {
    const Mat& img = input.getMat();
    output.create(img.size(), img.type());
    Mat dst = output.getMat();
    int range = (windowSize - 1) / 2;
    double sum1 = 0;
    double sum2 = 0;
    double temp = 0;

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            sum1 = 0;
            sum2 = 0;
            temp = 0;
            for (int t = -range; t <= range; t++)    //y축
                for (int s = -range; s <= range; s++) { //x축
                    temp = img.at<uchar>(min(img.rows - 1, max(y + t, 0)), min(img.cols - 1, (max(x + s, 0))));
                    sum1 += pow(temp, Q + 1);
                    sum2 += pow(temp, Q);
                }
            dst.at<uchar>(y, x) = sum1 / sum2;
        }
    }
}



void MyAlpha_trimmedMeanFilter(InputArray input, OutputArray output, int windowSize, float d) {
    const Mat& img = input.getMat();
    output.create(img.size(), img.type());
    Mat dst = output.getMat();
    int range = (windowSize - 1) / 2;
    std::vector<uchar> get_median;
    int half = d / 2;
    int sum = 0;

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            get_median.clear();
            sum = 0;
            for (int t = -range; t <= range; t++)    //y축
                for (int s = -range; s <= range; s++) { //x축
                    get_median.push_back(img.at<uchar>(min(img.rows - 1, max(y + t, 0)), min(img.cols - 1, (max(x + s, 0)))));
                }
            std::sort(get_median.begin(), get_median.end());
            for (int i = half; i < get_median.size() - half; i++) sum += get_median[i];
            dst.at<uchar>(y, x) = sum / ((windowSize * windowSize) - d);
        }
    }
}
//void autoMedianFilter(InputArray input, OutputArray output, int windowSize, int maxSize) {
//    const Mat& img = input.getMat();
//    output.create(img.size(), img.type());
//    Mat dst = output.getMat();
//    int range = (windowSize - 1) / 2;
//    std::vector<uchar> get_median;
//
//    for (int y = 0; y < img.rows; y++) {
//        for (int x = 0; x < img.cols; x++) {
//            get_median.clear();
//            for (int t = -range; t <= range; t++)    //y축
//                for (int s = -range; s <= range; s++) { //x축
//                    get_median.push_back(img.at<uchar>(min(img.rows - 1, max(y + t, 0)), min(img.cols - 1, (max(x + s, 0)))));
//                }
//            std::sort(get_median.begin(), get_median.end());
//            if (get_median[get_median.size() / 2] - get_median[0] > 0 && get_median[get_median.size() / 2] - get_median[get_median.size() - 1]<0)
//            {
//                if (img.at<uchar>(y,x) - get_median[0] > 0 && img.at<uchar>(y, x) - get_median[get_median.size() - 1] < 0) 
//                {
//                    dst.at<uchar>(y, x) = img.at<uchar>(y, x);
//                }
//                else
//                {
//                    dst.at<uchar>(y, x) = get_median[get_median.size() / 2];
//                }
//            }
//            else {
//
//               if (range <= maxSize) 
//               {
//
//               }
//            }
//        }
//    }
//}



int main(int argc, const char* argv[])
{
    imshow("origin", origin);//original

    const int Ksize = 5;

    Mat Gauss_10 = imread("NoisyImage/uniform_0.15.png", 0);
    imshow("noisy", Gauss_10);
    double difference = getPSNR(Gauss_10);
    cout << difference << "\n\n";
    Mat result1, result2, result3, result4, result5, result6, result7,result8;
    Mat avg_kernel = Mat::ones(5, 5, CV_32F) / 25;

    TickMeter tm;

    tm.start();
    GaussianBlur(Gauss_10, result1, Size(Ksize, Ksize), 11);
    tm.stop();
    cout << "GaussianBlur : " << tm.getTimeMilli() << " msec" << endl;

    tm.reset();
    tm.start();
    medianBlur(Gauss_10, result2, Ksize);
    tm.stop();
    cout << "medianBlur : " << tm.getTimeMilli() << " msec" << endl;

    tm.reset();
    tm.start();
    filter2D(Gauss_10, result3, -1, avg_kernel, Point(-1, -1), (0, 0), 4);
    tm.stop();
    cout << "filter2D : " << tm.getTimeMilli() << " msec" << endl;

    tm.reset();
    tm.start();
    MyHarmonicMeanFilter(Gauss_10, result4, Ksize);
    tm.stop();
    cout << "harmonic : " << tm.getTimeMilli() << " msec" << endl;


    tm.reset();
    tm.start();
    MyAlpha_trimmedMeanFilter(Gauss_10, result6, Ksize, 6);
    tm.stop();
    cout << "Alpha-trimmed Mean : " << tm.getTimeMilli() << " msec" << endl;


    tm.reset();
    tm.start();
    MyContraharmonicMeanFilter(Gauss_10, result7, Ksize, -2);
    tm.stop();
    cout << "ContraharmonicMeanFilter(-2) : " << tm.getTimeMilli() << " msec" << endl;

    tm.reset();
    tm.start();
    MyContraharmonicMeanFilter(Gauss_10, result8, Ksize, 1);
    tm.stop();
    cout << "ContraharmonicMeanFilter(1) : " << tm.getTimeMilli() << " msec" << endl;


    imshow("GaussianBlur", result1);
    imshow("medianBlur", result2);
    imshow("filter2D", result3);
    imshow("harmonic", result4);
    imshow("Alpha-trimmed Mean", result6);
    imshow("ContraharmonicMeanFilter(-2)", result7);
    imshow("ContraharmonicMeanFilter(1)", result8);
    double difference1 = getPSNR(result1);
    double difference2 = getPSNR(result2);
    double difference3 = getPSNR(result3);
    double difference4 = getPSNR(result4);
    double difference6 = getPSNR(result6);
    double difference7 = getPSNR(result7);
    double difference8 = getPSNR(result8);

    if (difference1 == -1) cout << "result1 is same origin" << endl;
    else cout << "GaussianBlur's PSNR : " << difference1 << endl;
    if (difference2 == -1) cout << "result1 is same origin" << endl;
    else cout << "medianBlur's PSNR : " << difference2 << endl;
    if (difference3 == -1) cout << "result1 is same origin" << endl;
    else cout << "filter2D's PSNR : " << difference3 << endl;
    if (difference4 == -1) cout << "result1 is same origin" << endl;
    else cout << "harmonic's PSNR : " << difference4 << endl;
    if (difference6 == -1) cout << "result1 is same origin" << endl;
    else cout << "Alpha-trimmed Mean's PSNR : " << difference6 << endl;
    if (difference7 == -1) cout << "result1 is same origin" << endl;
    else cout << "ContraharmonicMeanFilter(-2)'s PSNR : " << difference7 << endl;
    if (difference8 == -1) cout << "result1 is same origin" << endl;
    else cout << "ContraharmonicMeanFilter(1)'s PSNR : " << difference8 << endl;
    waitKey();


}