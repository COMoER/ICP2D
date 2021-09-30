//
// Created by comoer on 2021/9/30.
//
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc,char** argv)
{
    Mat img = imread(argv[1]);
    namedWindow("show",WINDOW_NORMAL);
    resizeWindow("show",1000,600);
    Rect rect = selectROI("show",img);
    imwrite("../data/R0_crop.jpg",img(rect));
}