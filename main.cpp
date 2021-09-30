#include <iostream>
#include "icp2d.h"

int main(int argc,char** argv) {
    if(argc < 3)
    {
        printf("usage:\n icp2d img_origin img_view");
        return 1;
    }
    Mat origin = imread(argv[1]);
    Mat view = imread(argv[2]);

    AffineICP icp(origin);
    Mat H(3,3,CV_32F);
    icp.solve(view,H,4,300,0.1);

    //from the example of opencv
    //! [estimate-homography]
    cout << "H:\n" << H << endl;
    //! [estimate-homography]

    //! [pose-from-homography]
    // Normalization to ensure that ||c1|| = 1
    double norm = sqrt(H.at<double>(0,0)*H.at<double>(0,0) +
                       H.at<double>(1,0)*H.at<double>(1,0) +
                       H.at<double>(2,0)*H.at<double>(2,0));

    H /= norm;
    Mat c1  = H.col(0);
    Mat c2  = H.col(1);
    Mat c3 = c1.cross(c2);

    Mat tvec = H.col(2);
    Mat R(3, 3, CV_64F);

    for (int i = 0; i < 3; i++)
    {
        R.at<double>(i,0) = c1.at<double>(i,0);
        R.at<double>(i,1) = c2.at<double>(i,0);
        R.at<double>(i,2) = c3.at<double>(i,0);
    }
    //! [pose-from-homography]

    //! [polar-decomposition-of-the-rotation-matrix]
    cout << "R (before polar decomposition):\n" << R << "\ndet(R): " << determinant(R) << endl;
    Mat W, U, Vt;
    SVDecomp(R, W, U, Vt);
    R = U*Vt;
    cout << "R (after polar decomposition):\n" << R << "\ndet(R): " << determinant(R) << endl;
    //! [polar-decomposition-of-the-rotation-matrix]
    cout<<"tvec"<<tvec<<endl;
    return 0;
}
