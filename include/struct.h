#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

struct  FrameData {
    double timestamp;
    Mat image;
    Eigen::Vector3d accel;
    Eigen::Vector3d gyro;
};

struct FeaturePoint {
    Point2f imagePoint; 
    Point3d worldPoint; 
};