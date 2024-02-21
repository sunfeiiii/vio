#include "CameraPoseEstimator.h"
#include "ImuProcessor.h"
#include <g2o/core/linear_solver.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/se3quat.h>

using namespace std;
using namespace g2o;

class   VisualInertialOdometry {
public:

    unique_ptr<CameraPoseEstimator> cameraPoseEstimator;
    unique_ptr<ImuProcessor> imuProcessor;
    g2o::SparseOptimizer optimizer;
    FrameData lastFrameData;
    deque<FrameData> frameWindow;
    vector<FeaturePoint> featurePoints;
    int maxVertexId = 1;

    //构造函数，初始化
    VisualInertialOdometry() {
        cameraPoseEstimator = make_unique<CameraPoseEstimator>();//相机位姿估计
        imuProcessor = make_unique<ImuProcessor>();//imu处理器
        auto linearSolver = make_unique<LinearSolverEigen<BlockSolverX::PoseMatrixType>>(); //线性求解器
        auto blockSolver = make_unique<BlockSolverX>(move(linearSolver));//块求解器
        optimizer.setAlgorithm(new OptimizationAlgorithmLevenberg(move(blockSolver)));//优化器
    };

    //处理帧数据
    void processFrame(const FrameData& currentFrameData) {
        //若不为第一帧，则进行特征提取与匹配
        if (!lastFrameData.image.empty()) {
            vector<KeyPoint> keypoints;
            vector<Point3d> points3D;
            //特征提取匹配得到特征点和地图点
            if (cameraPoseEstimator->detectAndMatch(currentFrameData.image, keypoints, points3D)) {
                double dt = currentFrameData.timestamp - lastFrameData.timestamp;//时间
                //imuProcessor->processImuData(lastFrameData, currentFrameData);//预积分
                //addDataToOptimizer(currentFrameData, keypoints, points3D);//优化
                updateFeaturePoints(keypoints, points3D);//更新特征点和地图点
            }
        }
        //更新上一帧
        lastFrameData = currentFrameData;
    };

    //g2o图优化过程
    void addDataToOptimizer(const FrameData& frameData, const vector<KeyPoint>& keypoints, vector<Point3d>& points3D) {
        //将imu位姿作为优化初始化位姿
        g2o::VertexSE3Expmap* imuPoseVertex = new g2o::VertexSE3Expmap();
        imuPoseVertex->setId(1);
        imuPoseVertex->setFixed(true);
        imuPoseVertex->setEstimate(g2o::SE3Quat(imuProcessor->q, imuProcessor->p));
        
        //优化器添加点和边
        for (size_t i = 0; i < points3D.size(); i++) {
            g2o::VertexPointXYZ* point = new g2o::VertexPointXYZ();
            point->setId(maxVertexId++);
            point->setMarginalized(true);
            point->setEstimate(Vector3d(points3D[i].x, points3D[i].y, points3D[i].z));
            optimizer.addVertex(point);

            g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
            edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(point));
            edge->setVertex(1, imuPoseVertex);
            Vector2d measurement(keypoints[i].pt.x, keypoints[i].pt.y);
            edge->setMeasurement(measurement);
            edge->setInformation(Matrix2d::Identity());

            edge->setParameterId(0, 0);
            optimizer.addEdge(edge);
        }
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        
        //更新优化后的地图点
        for (size_t i = 0; i < points3D.size(); i++) {
            g2o::VertexPointXYZ* vertex = dynamic_cast<g2o::VertexPointXYZ*>(optimizer.vertex(i + 1));
            if (vertex) {
                Vector3d optimizedPoint = vertex->estimate();
                points3D[i] = Point3d(optimizedPoint.x(), optimizedPoint.y(), optimizedPoint.z());
            }
        }
    };
    
    //将当前帧匹配的特征点和地图点更新进featurepoints
    void updateFeaturePoints(const vector<KeyPoint>& keypoints, vector<Point3d> points3D) {
        featurePoints.clear();
        for (size_t i = 0; i < keypoints.size(); i++) {
            featurePoints.push_back({keypoints[i].pt, points3D[i]});
        }
    };

    //查询深度
    double queryDepth(const Point2f& pixel) {
        double minDistance = std::numeric_limits<double>::max();
        double depth = -1.0;
        for (const auto& featurePoint : featurePoints) {
            double distance = cv::norm(featurePoint.imagePoint - pixel);
            if (distance < minDistance) {
                minDistance = distance;
                depth = featurePoint.worldPoint.z;
            }
        }
        return depth;
    };
};