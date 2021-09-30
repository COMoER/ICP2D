//
// Created by comoer on 2021/9/25.
//

#include "icp2d.h"

AffineICP::AffineICP(const Mat &origin): features_origin(),
descriptor_origin(),
points_origin()// the point set in the origin image
/**@brief the class to get high accuracy 2D texture contour matching
 *
 * @param origin cv Mat, image as the reference
 */
{
    //get gray image
    cache_origin = origin.clone();
    Mat gray;
    cvtColor(origin, gray, COLOR_BGR2GRAY);
    //initialize orb detector
    orb = ORB::create(); // default arguments
    //to detect the origin contour and feature keypoints
    orb->detect(gray, features_origin);
    orb->compute(gray, features_origin, descriptor_origin);
    // find contours on the origin image
    Mat binary;
//    adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 5, 0);
    Canny(gray,binary,70,130);

//    vector<Vec4i> hierarchy;
    vector<vector<Point>> contours_origin;
    findContours(binary, contours_origin, noArray(),RETR_EXTERNAL, CHAIN_APPROX_NONE);
    //TODO: contours filter
    for (const vector<Point> &contour:contours_origin)
        for (const Point &point:contour)points_origin.emplace_back(double(point.x),double(point.y));

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
        if(m.distance < 2*min_dist)
        {
            pointset1.push_back(features_origin[m.queryIdx].pt);
            pointset2.push_back(features_view[m.trainIdx].pt);
            goods.push_back(m);
        }

    }
    vector<int> mask;
    H = findHomography(pointset1,pointset2,RANSAC,3,mask);

    //show the rough matching result

    namedWindow("rough result",WINDOW_NORMAL);
    resizeWindow("rough result",1000,600);
    Mat rough_r;
    drawMatches(cache_origin,features_origin,view,features_view,goods,rough_r);
    imshow("rough result",rough_r);
    waitKey(0);
    rough_r = cache_origin.clone();


    vector<Point2f> point_t;
    perspectiveTransform(pointset2,point_t,H.inv());
    double max_reproject_dist = 0;
    double error = 0;
    //calc the error and the max distance
    for(int i=0;i<point_t.size();++i)
    {
        double dist = norm(point_t[i]-pointset1[i]);
        if(dist>max_reproject_dist)max_reproject_dist = dist;
        error += dist;
        circle(rough_r,Point2i(int(point_t[i].x),int(point_t[i].y)),5,Scalar(0,255,0),-1);
        circle(rough_r,Point2i(int(pointset1[i].x),int(pointset1[i].y)),3,Scalar(255,0,0),-1);
    }

    //    drawMatches(cache_origin,features_origin,view,features_view,goods,rough_r);
//    warpPerspective(cache_origin,rough_r,H,Size(view.cols,view.rows));
//    imshow("rough result",rough_r);
//    waitKey(0);

//    destroyWindow("rough result");
    //////////////

    max_reproject_dist *= release_radio; //constant:the release radio
    printf("init error is %.3f\n",error/point_t.size());

    // find contours on the view image
    Mat binary;
//    adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 3, 1);
    Canny(gray,binary,70,130);
    imshow("rough result",binary);
    waitKey();

    vector<vector<Point>> contours_view;

    findContours(binary, contours_view, noArray(),RETR_EXTERNAL, CHAIN_APPROX_NONE);

    binary = view.clone();
    drawContours(binary,contours_view,-1,Scalar(0,255,0));
    imshow("rough result",binary);
    waitKey();

    vector<Point2f> points_view; // the point set in the view image

    //TODO: contours filter
    for (const vector<Point> &contour:contours_view)
        for (const Point &point:contour)points_view.emplace_back(double(point.x),double(point.y));

    //loop find the matching
    double iter_error = 10000;
    vector<Point2f> loop_point_t;
    bool init_flag = false;
    for(int iter = 0;iter<max_iter&iter_error>thres;++iter) {
        //using pre H
        perspectiveTransform(points_view,loop_point_t,H.inv());

        vector<Point2f> loop_pointset1;
        vector<Point2f> loop_pointset2;
        if(!init_flag) rough_r = cache_origin.clone();
        int k = 0;

        for(const Point2f& p:loop_point_t)
        {
            int queryNum = 1;//KNN
            vector<float> vecQuery(2);//searcher point
            vector<int> vecIndex(queryNum);
            vector<float> vecDist(queryNum);
            cv::flann::SearchParams params(32);

            vecQuery = {p.x,p.y};
            kdtree->knnSearch(vecQuery,vecIndex,vecDist,queryNum,params);


            if(vecDist[0]<min(30.,max_reproject_dist))
            {
                loop_pointset2.push_back(points_view[k]);
                loop_pointset1.push_back(points_origin[vecIndex[0]]);
            }
            ++k;
        }

        if(!init_flag)
        {
            double max_iter_dist = 0.;
            perspectiveTransform(loop_pointset1,loop_point_t,H);
            //calc the error
            rough_r = view.clone();
            for(int i=0;i<loop_point_t.size();++i)
            {
                double dist = norm(loop_point_t[i]-loop_pointset2[i]);
                iter_error += dist;
                if(dist>max_iter_dist) max_iter_dist = dist;

                circle(rough_r,Point2i(int(loop_point_t[i].x),int(loop_point_t[i].y)),1,Scalar(0,255,0),-1);
                circle(rough_r,Point2i(int(loop_pointset2[i].x),int(loop_pointset2[i].y)),1,Scalar(255,0,0),-1);
            }
            max_reproject_dist = max_iter_dist;
            putText(rough_r,"init",Point(20,20),FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1);
            imshow("rough result",rough_r);
            waitKey(0);
            init_flag = true;
        }

        printf("find %zu match.\n",loop_pointset1.size());

        if(!loop_pointset1.empty())
        {
            iter_error = 0;
            H = findHomography(loop_pointset1,loop_pointset2,RANSAC,3);
            cout<<H<<endl;
            perspectiveTransform(loop_pointset1,loop_point_t,H);
            //calc the error
            rough_r = view.clone();
            for(int i=0;i<loop_point_t.size();++i)
            {
                double dist = norm(loop_point_t[i]-loop_pointset2[i]);
                iter_error += dist;
                circle(rough_r,Point2i(int(loop_point_t[i].x),int(loop_point_t[i].y)),1,Scalar(0,255,0),-1);
                circle(rough_r,Point2i(int(loop_pointset2[i].x),int(loop_pointset2[i].y)),1,Scalar(255,0,0),-1);
            }
            char s[20];
            iter_error = iter_error/loop_pointset1.size();
            sprintf(s,"error %.3f",iter_error);
            putText(rough_r,s,Point(20,20),FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1);
            imshow("rough result",rough_r);
            waitKey(30);
        }
        printf("iter %d/%d finished, error is %.3f\n",iter+1,max_iter,iter_error);
    }

    waitKey();
    //examine
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "cost time:" << time_used.count() << endl;

}

AffineICP::~AffineICP() {
    delete kdtree;
}
