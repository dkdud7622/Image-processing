#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void myMedianFilter(InputArray input, OutputArray output, int windowSize) {
    const Mat& img = input.getMat();
    output.create(img.size(), img.type());
    Mat dst = output.getMat();
    int range = (windowSize-1) / 2;
    std::vector<uchar> get_median;

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            get_median.clear();
            for (int t = -range; t <= range; t++)    //y축
                for (int s = -range; s <= range; s++) { //x축
                    get_median.push_back(img.at<uchar>(min(img.rows-1,max(y+t,0)),min(img.cols-1,(max(x+s,0)))));
                }
            std::sort(get_median.begin(), get_median.end());
            dst.at<uchar>(y, x) = get_median[get_median.size() / 2];
        } 
    }
}

int main()
{
    Mat img = imread("MedianFilterInput.png",0);
    Mat input, out1, out2;
    resize(img, input, Size(img.cols/2,img.rows/2));

    myMedianFilter(input, out1, 5);
    //medianBlur(input, out2, 5);

    imshow("my median", out1);
    //imshow("median", out2);

    waitKey();
}