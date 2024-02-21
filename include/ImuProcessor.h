#include "struct.h"

using namespace Eigen;

class ImuProcessor {
public:

    Quaterniond q;//四元数
    Vector3d v;
    Vector3d p;
    //构造函数，初始化
    ImuProcessor() :q(Quaterniond::Identity()), v(Vector3d::Zero()), p(Vector3d::Zero()){
    };

    void processImuData(const FrameData& prevFrameData, const FrameData& currentFrameData) {
        //联合标定矩阵
        Matrix4d T_cam_imu;
        T_cam_imu << 0.99995, -0.00851610, 0.003972445, 0.043946,
                     0.008539021, 0.9999467, -0.005786767, 0.01996812,
                     -0.0039229529, 0.005820432, 0.999975, 0.01636738,
                     0.0, 0.0, 0.0, 1.0;
        double dt = currentFrameData.timestamp - prevFrameData.timestamp;
        //取前后两帧加速度，角速度均值
        Vector3d avgAccel = 0.5 * (prevFrameData.accel + currentFrameData.accel);
        Vector3d avgGyro = 0.5 * (prevFrameData.gyro + currentFrameData.gyro);
        //将imu的平均角速度和平均加速度转化到相机坐标系下
        Vector3d imu_acc_cam = T_cam_imu.block<3, 3>(0, 0) * avgAccel;
        Vector3d imu_gyro_cam = T_cam_imu.block<3, 3>(0, 0) * avgGyro;

        //使用imu的角速度更新四元数
        Quaterniond dq(1, imu_gyro_cam.x() * dt / 2, imu_gyro_cam.y() * dt / 2, imu_gyro_cam.z() * dt / 2);
        q = (q * dq).normalized();
        
        //使用四元数将加速度转化到世界坐标系下
        Vector3d un_acc = q * imu_acc_cam;

        //更新速度和位置
        v += un_acc *dt;
        p += v * dt;
    };
};