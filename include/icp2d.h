//
// Created by comoer on 2021/9/25.
//

#ifndef ICP2D_ICP2D_H
#define ICP2D_ICP2D_H
#include <iostream>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/problem.h>
#include <ceres/rotation.h>

using namespace std;
using namespace cv;

class AffineICP
{
private:
    cv::flann::Index* kdtree;
    vector<KeyPoint> features_origin;
    Mat descriptor_origin;
    vector<Point2f> points_origin; // the point set in the origin image
    Ptr<ORB> orb;
    Mat cache_origin;
public:
    AffineICP(const Mat& origin);
    void solve(const Mat& view,Mat &H,double release_radio = 4.,int max_iter = 10,double thres = 0.1);
    ~AffineICP();

};

#endif //ICP2D_ICP2D_H
