
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
#include <windows.h>





using namespace cv;
using namespace std;

Mat output;
Mat complement;
Mat copyImg;
Mat temp;
Point startPosition;
bool isClick = false;


void doDilation()
{
    int morph_size = 4;
    Mat element = getStructuringElement(MORPH_CROSS, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
    dilate(temp, temp, element, Point(-1, -1), 1);

    bitwise_and(complement, temp, temp);

}
void makeBoundary() {
    int morph_size = 5;
    Mat element = getStructuringElement(MORPH_CROSS, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
    erode(copyImg, output, element);
    subtract(copyImg, output, output);
    bitwise_not(output, complement);

}

void onMouse(int event, int x, int y, int flags, void* param)
{


    switch (event)
    {
    case EVENT_LBUTTONDOWN:
        cout << x << ", " << y << endl;
        startPosition = Point(x, y);
        if (flags & EVENT_LBUTTONDOWN)
            circle(temp, startPosition, 3, (255));
        isClick = true;
        break;

        // left button down -> reset.
    case EVENT_RBUTTONDOWN:
        makeBoundary();
        temp = Scalar(0);
        isClick = false;
    }
}

int checkEnd(Mat prev, Mat now)
{

    int sum = 0;
    for (auto i = 0; i < prev.rows; i++) {
        for (auto j = 0; j < prev.cols; j++) {
            sum += now.at<uchar>(i, j) - prev.at<uchar>(i, j);
            if (sum != 0)
                return 0;
        }
    }
        return 1;
}


int main(int argc, const char* argv[])
{
    int count = 0;

    Mat img = imread("Profile.png", 0);
    img.copyTo(copyImg);

    Mat checkEnd1, checkEnd2;
    img.copyTo(checkEnd1);
    bitwise_not(img, checkEnd2);


    Mat prev;
    output.copyTo(prev);
    prev = Scalar(0);

    makeBoundary();

    //img's complement
    /*imshow("img", img);
    imshow("complement", complement);*/

    output.copyTo(temp);
    temp = Scalar(0);
    /*imshow("temp", temp);*/
    int isEnd = 0;
    while (true)
    {
        imshow("result", output);
        setMouseCallback("result", onMouse, 0);
        
        if (isClick == true)
        {
            doDilation();
            bitwise_or(output, temp, output); //temp + output
            if (count > 1)
            //If the difference between prev and the current output is zero, the end.
                isEnd = checkEnd(prev, output); 
            output.copyTo(prev);
            count++;
        }

        if (isEnd == 1)
        {
            isClick = false;
            cout << "the end" << endl;
            isEnd = 0;
            count = 0;
        }
        waitKey(27);
    }

}