#include "VisualInertialOdometry.h"
#include <librealsense2/rs.hpp>

using namespace std;
using namespace cv;

int main() {
    //查找realsense
    rs2::context ctx;
    auto list = ctx.query_devices();
    if (list.size() == 0) {
        throw std::runtime_error("No realsense !!!");
    }
    
    //创建realsense对象，建立数据流
    rs2::device dev = list.front();
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
    pipe.start(cfg);

    //创建视觉里程计对象并初始化
    VisualInertialOdometry vio;

    //循环
    while(true) {
        
        //创建帧数据对象
        FrameData frameData;
        //获取数据流
        auto frames = pipe.wait_for_frames();
        rs2::motion_frame accel_frame = frames.first(RS2_STREAM_ACCEL);
        rs2::motion_frame gyro_frame = frames.first(RS2_STREAM_GYRO);
        rs2::frame color_frame = frames.get_color_frame();
        //帧数据存储当前帧，imu数据，添加时间戳
        frameData.image = Mat(Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
        frameData.timestamp = color_frame.get_timestamp() / 1000;
        cvtColor(frameData.image, frameData.image, COLOR_BGR2GRAY);
        Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
        clahe->apply(frameData.image, frameData.image);
        frameData.accel = Vector3d(accel_frame.get_motion_data().x, accel_frame.get_motion_data().y, accel_frame.get_motion_data().z);
        frameData.gyro = Vector3d(gyro_frame.get_motion_data().x, gyro_frame.get_motion_data().y, gyro_frame.get_motion_data().z);
        //视觉里程计处理帧
        vio.processFrame(frameData);
        //定义需要查询深度的像素，并查询深度
        Point2f Pixel(320, 240);
        double depth = vio.queryDepth(Pixel);
        //cout<<"Depth at (" <<Pixel.x<<","<<Pixel.y<<") :"<< depth<<endl;


        //显示帧数据
        //addText(frameData.image, frameData.accel, Point(20,20), FONT_HERSHEY_SIMPLEX);
        //addText(frameData.image, frameData.gyro, Point(20,20), FONT_HERSHEY_SIMPLEX);
        //imshow("Frame", frameData.image);
        //waitKey(1);
        //存储当前帧为上一帧
    }
}