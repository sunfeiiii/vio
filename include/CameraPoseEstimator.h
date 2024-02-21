#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>

using namespace std;
using namespace cv;

class CameraPoseEstimator {
public:

    int processedFramesCount;//计数器
    Ptr<ORB> orb;
    Ptr<DescriptorMatcher> matcher;
    Mat cameraMatrix;
    Mat distCoeffs;

    vector<KeyPoint> lastKeypoints;
    vector<KeyPoint> currentKeypoints;
    Mat lastDescriptors;
    Mat currentDescriptors;
    vector<DMatch> allMatchedDescriptors;
    vector<Point3d> allPoints3D;
    pcl::visualization::CloudViewer viewer{"点云查看器"};
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud{new pcl::PointCloud<pcl::PointXYZ>};

    //构造函数，初始化
    CameraPoseEstimator() : processedFramesCount(0) {
        orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::FAST_SCORE, 31, 20);//特征提取器
        matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);//描述符匹配器
        cameraMatrix = (Mat_<double>(3, 3) << 391.57645976863694, 0, 326.83301528066227, 
                                              0, 392.2173924045597, 235.30947818084246, 
                                              0, 0, 1);//相机内参
        distCoeffs = (Mat_<double>(4, 1) << 0.007532405272341989, -0.003198723534231893, 
                                           -0.00015249992792258453, 0.001638891018727039);//相机畸变
    };

    //特征提取和匹配
    bool detectAndMatch(const Mat& currentImage, vector<KeyPoint>& keypoints, vector<Point3d>& points3D) {
        orb->detectAndCompute(currentImage, noArray(), currentKeypoints, currentDescriptors);
        //若特征提取次数超过10次，则开始匹配描述符
        if (processedFramesCount >= 10) {
            //确保有足够描述符，否则会匹配失败报错
            if (lastDescriptors.rows >= 10  && currentDescriptors.rows >= 10) {
                matcher->match(lastDescriptors, currentDescriptors, allMatchedDescriptors);
            }
        }
        //若有匹配的描述符
        if (!allMatchedDescriptors.empty()) {
            vector<Point2f> matchedPoints1, matchedPoints2;
            vector<KeyPoint> matchedKeypoints1, matchedKeypoints2;
            for (const auto& matchedDescriptor : allMatchedDescriptors) {
                //获取匹配的特征点索引
                int queryIdx = matchedDescriptor.queryIdx;
                int trainIdx = matchedDescriptor.trainIdx;

                Point2f point1 = lastKeypoints[queryIdx].pt;
                Point2f point2 = currentKeypoints[trainIdx].pt;
                matchedPoints1.push_back(point1);
                matchedPoints2.push_back(point2);
                matchedKeypoints1.push_back(lastKeypoints[queryIdx]);
                matchedKeypoints2.push_back(currentKeypoints[trainIdx]);
            }
            keypoints = matchedKeypoints2;
            //若特征点足够，则进行三角化，构建地图点
            if (matchedPoints1.size() >= 20) {
                //使用匹配上的足够的特征点计算本质矩阵
                Mat E = findEssentialMat(matchedPoints1, matchedPoints2, cameraMatrix, RANSAC);
                Mat R, t;
                //恢复位姿，获得R，t
                recoverPose(E, matchedPoints1, matchedPoints2, cameraMatrix, R, t);
                triangulatePoints(R, t, matchedPoints1, matchedPoints2, points3D);
                allPoints3D.insert(allPoints3D.end(), points3D.begin(), points3D.end());
                // 更新点云对象
                updatePointCloud();
                drawPointCloud();
                cout<<allPoints3D.size()<<endl;
            }
        }
        //更新特征点
        lastKeypoints = currentKeypoints;
        lastDescriptors = currentDescriptors;
        //计数器+1
        processedFramesCount++;

        return true;
    };

    Point2f pixelToNormalized(const Point2f& pixel, const Mat& cameraMatrix) {
        Mat distortedPixel = (Mat_<double>(2, 1) << pixel.x, pixel.y);
        Mat undistortedPixel;
        undistortPoints(distortedPixel, undistortedPixel, cameraMatrix, distCoeffs);
        Point2f normalizedPoint(undistortedPixel.at<double>(0), undistortedPixel.at<double>(1));
        return normalizedPoint;
    };

    void triangulatePoints(const Mat& R, const Mat& t, const vector<Point2f>& points1, const vector<Point2f> points2, vector<Point3d>& points3D) {
        Mat T1 = Mat::eye(3, 4, CV_64F);
        Mat T2 = Mat::eye(3, 4, CV_64F);
        R.copyTo(T2.colRange(0,3));
        t.copyTo(T2.col(3));

        vector<Point2f> normalizedPoints1, normalizedPoints2;
        for (const auto& pt : points1) {
            normalizedPoints1.push_back(pixelToNormalized(pt, cameraMatrix));
        }
        for (const auto& pt : points2) {
            normalizedPoints2.push_back(pixelToNormalized(pt, cameraMatrix));
        }
        Mat points4D;
        cv::triangulatePoints(T1, T2, normalizedPoints1, normalizedPoints2, points4D);
        points3D.clear();
        for (int i = 0; i < points4D.cols; i++) {
            points3D.push_back(Point3d(
                points4D.at<double>(0, i) / points4D.at<double>(3, i),
                points4D.at<double>(1, i) / points4D.at<double>(3, i),
                points4D.at<double>(2, i) / points4D.at<double>(3, i)
            ));
        }
    }; 
    // 添加用于绘制三维点云的函数
    void drawPointCloud() {
        // points3D是相机位姿下的三维点坐标
        for (const auto& point3D : allPoints3D) {
            cloud->push_back(pcl::PointXYZ(point3D.x, point3D.y, point3D.z));
        }
        // 设置点云
        viewer.showCloud(cloud);
        while (!viewer.wasStopped()) {
            pcl_sleep(0.1);  // 指定等待时间（以秒为单位）
        }       
    };

private:
    // 用于更新点云对象的函数
    void updatePointCloud() {
        cloud->clear();
        for (const auto& point3D : allPoints3D) {
            cloud->push_back(pcl::PointXYZ(point3D.x, point3D.y, point3D.z));
        }
    };
};