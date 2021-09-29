//
// Created by comoer on 2021/9/25.
//

#include "icp2d.h"

AffineICP::AffineICP(const Mat &origin)
/**@brief the class to get high accuracy 2D texture contour matching
 *
 * @param origin cv Mat, image as the reference
 */
{
    //get gray image
    Mat gray;
    cvtColor(origin, gray, COLOR_BGR2GRAY);
    //initialize orb detector
    orb->create(); // default arguments
    //to detect the origin contour and feature keypoints
    orb->detect(gray, features_origin);
    orb->compute(gray, features_origin, descriptor_origin);
    // find contours on the origin image
    Mat binary;
    adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 5, 0);
    vector<Vec4i> hierarchy;
    vector<vector<Point2f>> contours_origin;
    findContours(binary, contours_origin, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    //TODO: contours filter
    for (const vector<Point2f> &contour:contours_origin)
        for (const Point2f &point:contour)points_origin.push_back(point);

    //initialize kdtree
    Mat pointMat = Mat(points_origin).reshape(1);
    flann::KDTreeIndexParams param(2);
    kdtree = new flann::Index(pointMat, param);


}

void AffineICP::solve(const Mat &view, Mat &H,double release_radio,int max_iter,double thres) {
    //maybe take long time,check it
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    /***** main process ******/
    /***** preprocessing ******/
    Mat gray;
    cvtColor(view, gray, COLOR_BGR2GRAY);
    //to detect the view contour and feature keypoints
    vector<KeyPoint> features_view;
    Mat descriptor_view;
    orb->detect(gray, features_view);
    orb->compute(gray, features_view, descriptor_view);

    /**** get initialization using keypoints matching ****/
    vector<DMatch> match;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptor_origin,descriptor_view,match);

    double min_dist = 10000,max_dist = 0;
    for(const DMatch& m:match)
    {
        double dist = m.distance;
        if(dist<min_dist) min_dist = dist;
        if(dist>max_dist) max_dist = dist;
    }
    vector<DMatch> goods;
    vector<Point2f> pointset1;
    vector<Point2f> pointset2;

    for(const DMatch& m:match)
    {
        if(m.distance < max(30.,2*min_dist))
        {
            pointset1.push_back(features_origin[m.queryIdx].pt);
            pointset2.push_back(features_view[m.trainIdx].pt);
        }
    }
    findHomography(pointset1,pointset2,RANSAC,3,H);
    vector<Point2f> point_t;
    perspectiveTransform(pointset1,point_t,H);
    double max_reproject_dist = 0;
    double error = 0;
    //calc the error and the max distance
    for(int i=0;i<point_t.size();++i)
    {
        double dist = norm(point_t[i]-pointset2[i]);
        if(dist>max_reproject_dist)max_reproject_dist = dist;
        error += dist;
    }
    max_reproject_dist *= release_radio; //constant:the release radio
    printf("init error is %.3f",error);

    // find contours on the view image
    Mat binary;
    adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 5, 0);
    vector<Vec4i> hierarchy;
    vector<vector<Point2f>> contours_view;

    findContours(binary, contours_view, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<Point2f> points_view; // the point set in the view image
    for (const vector<Point2f> &contour:contours_view)
        for (const Point2f &point:contour)
            points_view.push_back(point);

    //loop find the matching
    double iter_error = 10000;
    for(int iter = 0;iter<max_iter&iter_error>thres;++iter) {

        vector<Point2f> loop_pointset1;
        vector<Point2f> loop_pointset2;
        for(const Point2f& p:points_view)
        {
            int queryNum = 1;//KNN
            vector<float> vecQuery(2);//searcher point
            vector<int> vecIndex(queryNum);
            vector<float> vecDist(queryNum);
            cv::flann::SearchParams params(32);

            vecQuery = {p.x,p.y};
            kdtree->knnSearch(vecQuery,vecIndex,vecDist,queryNum,params);
            if(vecDist[0]<max_reproject_dist)
            {
                loop_pointset2.push_back(p);
                loop_pointset1.push_back(points_origin[vecIndex[0]]);
            }
        }
        printf("find %zu match.",loop_pointset1.size());

        if(!loop_pointset1.empty())
        {
            iter_error = 0;
            findHomography(loop_pointset1,loop_pointset2,RANSAC,3,H);
            vector<Point2f> loop_point_t;
            perspectiveTransform(pointset1,loop_point_t,H);
            //calc the error
            for(int i=0;i<loop_point_t.size();++i)
            {
                double dist = norm(point_t[i]-loop_pointset2[i]);
                iter_error += dist;
            }
        }
        printf("iter %d/%d finished, error is %.3f",iter+1,max_iter,iter_error);
    }


    //examine
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "cost time:" << time_used.count() << endl;

}

AffineICP::~AffineICP() {
    delete kdtree;
}
