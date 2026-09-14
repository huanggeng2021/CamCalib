

#include "calibration/CustomCalibrator.h"

namespace camcalib{


// 对点集进行归一化
std::pair<std::vector<cv::Point2d>, Eigen::Matrix3d> CustomCalibrator::normalizeObjectCoordinates(
        const std::vector<cv::Point2d>& coordinates
) const {

    if (coordinates.empty()) {
        throw std::invalid_argument("输入点集不能为空");
    }

    const int pointCount = coordinates.size();

    // 计算质心

    double centerX = 0.0;
    double centerY = 0.0;

    for(const auto& point : coordinates){      

        centerX += point.x;
        centerY += point.y;
    }

    centerX /= static_cast<double>(pointCount);
    centerY /= static_cast<double>(pointCount);

    // 计算平面内的欧式距离

    double meanDistance = 0.0;

    for (const auto& point : coordinates){

        const double dx = point.x - centerX;
        const double dy = point.y - centerY;

        meanDistance += std::hypot(dx, dy);   // sqrt(dx^2, dy^2)
    }

    meanDistance /= static_cast<double>(pointCount);

    if (meanDistance < 1e-12) {
        throw std::runtime_error("物点集合发生退化");
    }

    // 缩放至平均距离

    const double scale = std::sqrt(2.0) / meanDistance;

    Eigen::Matrix3d transform;

    transform <<                        //  放缩矩阵
        scale, 0.0, -scale * centerX,
        0.0, scale, -scale * centerY,
        0.0, 0.0, 1.0;

    // 输出归一化后的坐标

    std::vector<cv::Point2d>  normalizedPoints;
    normalizedPoints.reserve(pointCount);

    for(const auto& point : coordinates){

        const double normalizedX = scale * (point.x - centerX);
        const double normalizedY = scale * (point.y - centerY);

        const cv::Point2d p (normalizedX, normalizedY);
        normalizedPoints.push_back(p);
    }

    return {std::move(normalizedPoints), transform};

}

// 数据类型转换
std::vector<cv::Point2d> CustomCalibrator::point3f2point2d(
        const std::vector<cv::Point3f>& worldPoints
    ) const{

    const int pointCount = static_cast<int>(worldPoints.size());

    std::vector<cv::Point2d> worldPoints2d;
    worldPoints2d.reserve(pointCount);

    for(const auto& point : worldPoints){

        worldPoints2d.push_back(
            cv::Point2d(static_cast<double>(point.x), static_cast<double>(point.y))
        );
    }

    return worldPoints2d;

}

// 计算所有位姿的Homo矩阵
std::vector<Eigen::Matrix3d> CustomCalibrator::estimateAllPoseHomography(
    const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2d>>& imagePoints
) const{

    if(objectPoints.size() != imagePoints.size()){

        return {};
    }

    const int matrixCount = imagePoints.size();

    std::vector<Eigen::Matrix3d> matrixVector;

    for(int i = 0; i < matrixCount; ++i){

        Eigen::Matrix3d homo =  estimateHomography(objectPoints.at(i), imagePoints.at(i));

        matrixVector.push_back(homo);

    }

    return matrixVector;

}


Eigen::Matrix3d CustomCalibrator::estimateHomography(
    const std::vector<cv::Point3f>& objectPoints,
    const std::vector<cv::Point2d>& imagePoints
) const{

    // 对objectPoints 与 imagePoints 进行归一化， 为什么要做归一化呢， 为什么平均距离为sqrt(2)呢
    // 归一化
    std::vector<cv::Point2d> objectPoints2d;
    objectPoints2d = point3f2point2d(objectPoints);

    auto [noramlizedObjectPoints2d, transformObject] = normalizeObjectCoordinates(objectPoints2d);
    auto [normalizeImagePoints, transformImgae] = normalizeObjectCoordinates(imagePoints);

    // 计算变换矩阵
    const int pointsNumb =  imagePoints.size();

    // step1: 使用点对构造方程组 设方程AX = 0
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * pointsNumb, 9);

    for(int i = 0; i < pointsNumb; ++i){
        const cv::Point2d& objPoint = noramlizedObjectPoints2d[i];
        const cv::Point2d& imgPoint = normalizeImagePoints[i];

        double X = static_cast<double>(objPoint.x); 
        double Y = static_cast<double>(objPoint.y);
        double u = imgPoint.x;
        double v = imgPoint.y;

        const int row0 = 2 * i;
        const int row1 = 2 * i + 1;
 
        A.row(row0) << X, Y, 1,    // Eigen::vector 可以使用使用这种方式
                   0.0, 0.0, 0.0,
                   -u * X, -u * Y, -u;

        A.row(row1) << 0.0, 0.0, 0.0,
                       X, Y, 1.0,
                       -v * X, -v * Y, -v;

    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    // v的最后一列对应最小奇异值   为什么呢
    Eigen::VectorXd h = svd.matrixV().col(8);

    Eigen::Matrix3d HNormalized;

    HNormalized << h(0), h(1), h(2),   // 归一化尺度下的H
                   h(3), h(4), h(5),
                   h(6), h(7), h(8);

    // 进行反归一化   反归一化是为了将H变换到真实的空间尺度中
    Eigen::Matrix3d H = transformImgae.inverse() * HNormalized * transformObject;

    // 固定 H 尺度   H的自由度只有8个  H = nH  
    if(std::abs(H(2, 2)) > 1e-12){   
        H /= H(2,2);
    }else{

        const double norm = H.norm(); // 防止出现H(2, 2) = 0的情况
    }

    return H;

}

// 该矩阵的构造详情见张氏标定推导
Eigen::Matrix<double, 6, 1> CustomCalibrator::makeV(
    const Eigen::Matrix3d& H,
    int i, int j 
) const{

    Eigen::Matrix<double, 6, 1> v;

    v << H(0, i) * H(0, j),
         H(0, i) * H(1, j) + H(1, j) * H(0, j),
         H(1, i) * H(1, j),
         H(2, i) * H(0, j) + H(0, i) * H(2, j),
         H(2, i) * H(1, j) + H(1, i) * H(2, j),
         H(2, i) * H(2, j);


    return v;
}

// 通过homo矩阵计算内参初值
cv::Mat CustomCalibrator::estimateIntrinsics(
    const std::vector<Eigen::Matrix3d>& homographies
) const{

    CV_Assert(homographies.size() >= 3);

    Eigen::MatrixXd V(     // size = homographies.size() *2  6列 
        static_cast<Eigen::Index>(homographies.size() *2), 6
    );

    // 构造矩阵V    方程为VB = 0    其中B为K^-t * K
    for(size_t k = 0; k < homographies.size(); ++k){

        const Eigen::Matrix3d& H = homographies[k];

        // h1^T B h2 = 0    旋转矩阵 列向量正交
        V.row(static_cast<Eigen::Index>(2 * k)) = makeV(H, 0, 1).transpose();

        // h1^T B h1 - h2^T B h2 = 0    ||r1|| = ||r2|| 单位向量
        V.row(static_cast<Eigen::Index>(2 * k + 1)) = (makeV(H, 0, 0) - makeV(H, 1, 1)).transpose();
    }

    // 求 Vb = 0 的最小二乘齐次解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(V, Eigen::ComputeFullV);

    Eigen::VectorXd b = svd.matrixV().col(5);    // b 只与内参有关

    // step3: 由于vb = 0, 是齐次方程， b 与-b 等价， 需要确定正确符号

    double B11 = b(0);
    double B12 = b(1);
    double B22 = b(2);
    double B13 = b(3);
    double B23 = b(4);
    double B33 = b(5);

    // B = K^(-T)K^(-1) 理论上为正定矩阵， 至少满足B11 > 0

    if(B11 < 0.0){
        b = -b;

        double B11 = b(0);
        double B12 = b(1);
        double B22 = b(2);
        double B13 = b(3);
        double B23 = b(4);
        double B33 = b(5);
    }


    // 计算内参
    const double denominator = B11 * B22 - B12 * B12;
    CV_Assert(std::abs(denominator) > 1e-12);
    CV_Assert(std::abs(B11) > 1e-12);

     /*
     * 主点纵坐标 v0
     *
     *      B12 B13 - B11 B23
     * v0 = -------------------
     *      B11 B22 - B12^2
     */

    const double v0 = (B12 * B13 - B11 * B23) / denominator;

    /*
     * lambda
     *
     *          B13^2 + v0(B12 B13 - B11 B23)
     * λ = B33 - --------------------------------
     *                         B11
     */

    const double lambde = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;

    /*
     * fx = alpha
     *
     * alpha = sqrt(lambda / B11)
     */

    const double alpha = std::sqrt(lambde / B11);

    /*
     * fy = beta
     *
     *             lambda B11
     * beta = sqrt(-------------)
     *             B11 B22-B12²
     */

     const double beta = std::sqrt((lambde * B11 )/ denominator);

    /*
     * skew = gamma
     *
     * gamma = -B12 * alpha² * beta / lambda
     */

     const double gamma = -B12 * alpha * alpha * beta / lambde;

     /*
     * 主点横坐标 u0
     *
     *      gamma*v0   B13*alpha²
     * u0 = -------- - -----------
     *        beta        lambda
     */

     const double u0 = gamma * v0 /beta - B13 * alpha * alpha / lambde;


    // ------------------------------------------------------------
    // Step 5：构造相机内参矩阵 K
    //
    //     [ fx   skew   cx ]
    // K = [ 0     fy    cy ]
    //     [ 0      0     1 ]
    // ------------------------------------------------------------

    cv::Mat K = (cv::Mat_<double>(3,3)<<
    alpha, gamma, u0,
    0.0, beta, v0,
    0.0, 0.0, 1.0);

    return K;

}


void CustomCalibrator::estimateExtrinsics(
    const cv::Mat& cameraMatrix,
    const std::vector<cv::Mat>& homographies,
    std::vector<cv::Mat>& rotationVectors,
    std::vector<cv::Mat>& translationVectors
    ) const{

    CV_Assert(!cameraMatrix.empty());
    CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3);
    CV_Assert(!homographies.empty());

    rotationVectors.clear();
    translationVectors.clear();

    rotationVectors.reserve(homographies.size());
    translationVectors.reserve(homographies.size());

    // 内参矩阵K
    cv::Mat K;
    cameraMatrix.convertTo(K, CV_64F);

    cv::Mat kinv = K.inv(); //  求逆

    for(size_t i = 0; i < homographies.size(); ++i){
        
        CV_Assert(homographies[i].rows == 3 && homographies[i].cols == 3);

        cv::Mat H;
        homographies[i].convertTo(H, CV_64F);

        // --------------------------------------------------------
        // H = lambda * K * [r1 r2 t]
        //
        // 因此：
        //
        // K^-1 H = lambda * [r1 r2 t]
        // --------------------------------------------------------

        cv::Mat h1 = H.col(0);
        cv::Mat h2 = H.col(1);
        cv::Mat h3 = H.col(2);

        cv::Mat kinv_h1 = kinv * h1;
        cv::Mat kinv_h2 = kinv * h2;
        cv::Mat kinv_h3 = kinv * h3;

        // --------------------------------------------------------
        // 计算尺度因子
        //
        // 理论上：
        //
        // lambda = 1 / ||K^-1 h1||
        //        = 1 / ||K^-1 h2||
        //
        // 实际有噪声，因此使用两者平均
        // --------------------------------------------------------

        const double norm1 = cv::norm(kinv_h1);
        const double norm2 = cv::norm(kinv_h2);

        CV_Assert(norm1 > 1e-12);
        CV_Assert(norm2 > 1e-12);

        const double lambda = 2.0 / (norm1 + norm2);

        // --------------------------------------------------------
        // 得到旋转矩阵前两列和平移向量
        // --------------------------------------------------------

        cv::Mat r1 = lambda * kinv_h1;
        cv::Mat r2 = lambda * kinv_h2;
        cv::Mat t = lambda * kinv_h3;

        // r3 = r1  r2
        cv::Mat r3 = r1.cross(r2);

        // --------------------------------------------------------
        // 构造初始旋转矩阵
        //
        // R_init = [r1 r2 r3]
        //
        // 由于单应矩阵存在噪声，
        // 此时 R_init 通常不严格满足：
        //
        // R^T R = I
        // det(R) = 1
        // --------------------------------------------------------

        cv::Mat Rinit(3, 3, CV_64F);
        r1.copyTo(Rinit.col(0));
        r2.copyTo(Rinit.col(1));
        r3.copyTo(Rinit.col(2));

        // --------------------------------------------------------
        // 使用 SVD 将 Rinit 投影到最近的旋转矩阵
        //
        // Rinit = U * W * V^T
        //
        // 最近的正交矩阵：
        //
        // R = U * V^T
        // --------------------------------------------------------

        cv::SVD svd(Rinit, cv::SVD::FULL_UV);

        cv::Mat U  = svd.u;
        cv::Mat Vt = svd.vt;

        cv::Mat R = U * Vt;

        // --------------------------------------------------------
        // 保证 R 属于 SO(3)
        //
        // 正确旋转矩阵要求：
        //
        // det(R) = +1
        //
        // 如果 det(R) = -1，
        // 当前结果包含镜像反射，需要修正
        // --------------------------------------------------------

        if(cv::determinant(R) < 0.0){
            U.col(2) *= -1.0;
            R = U * Vt;
        }

        // --------------------------------------------------------
        // 将旋转矩阵转换为 Rodrigues 旋转向量
        // --------------------------------------------------------

        cv::Mat rvec;
        cv::Rodrigues(R, rvec);

        rotationVectors.push_back(rvec.clone());
        translationVectors.push_back(t.clone());
    }

    
}


  
cv::Mat CustomCalibrator::initializeDistortion(
    const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2d>>& imagePoints,
    const cv::Mat& cameraMatrix,
    const std::vector<cv::Mat>& rotationVectors,
    const std::vector<cv::Mat>& translationVectors
    ) const{

    CV_Assert(!cameraMatrix.empty());
    CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3);

    CV_Assert(objectPoints.size() == imagePoints.size());
    CV_Assert(objectPoints.size() == rotationVectors.size());
    CV_Assert(objectPoints.size() == translationVectors.size());

    cv::Mat K;
    cameraMatrix.convertTo(K, CV_64F);

    const double fx = K.at<double>(0, 0);
    const double skew = K.at<double>(0, 1);
    const double cx = K.at<double>(0, 2);

    const double fy = K.at<double>(1, 1);
    const double cy = K.at<double>(1, 2);

    // ------------------------------------------------------------
    // 统计总点数
    //
    // 每个角点提供两条方程：
    // 一条来自 u
    // 一条来自 v
    // ------------------------------------------------------------

    size_t totalPoints = 0;

    for(const auto& pts : objectPoints){
        totalPoints += pts.size();
    }

    CV_Assert(totalPoints > 0);

    cv::Mat D (static_cast<int>(2 * totalPoints), 2, CV_64F);

    cv::Mat d (static_cast<int>(2 * totalPoints), 1, CV_64F);

    int row = 0;

    // ------------------------------------------------------------
    // 遍历每一个标定位姿
    // ------------------------------------------------------------

    for (size_t i = 0; i < objectPoints.size(); ++i){

        CV_Assert(objectPoints[i].size() == imagePoints[i].size());

        // 旋转向量 -> 旋转矩阵
        cv::Mat R;
        cv::Rodrigues(rotationVectors[i], R);

        R.convertTo(R, CV_64F);

        cv::Mat t;
        translationVectors[i].convertTo(t, CV_64F);

        CV_Assert(t.total() == 3);

        if (t.rows == 1)
        {
            t = t.t();
        }

        // --------------------------------------------------------
        // 遍历这一帧所有标定点
        // --------------------------------------------------------

        for(size_t j = 0; j < objectPoints[i].size(); ++j){

            const cv::Point3f& Pw = objectPoints[i][j];
            const cv::Point2d& observed = imagePoints[i][j];

            cv::Mat P = (cv::Mat_<double>(3, 1) <<
            static_cast<double>(Pw.x),
            static_cast<double>(Pw.y),
            static_cast<double>(Pw.z));

            // ----------------------------------------------------
            // 相机坐标：
            //
            // Pc = R * Pw + t
            // ----------------------------------------------------

            cv::Mat Pc = R * Pw + t;

            const double Xc = Pc.at<double>(0, 0);
            const double Yc = Pc.at<double>(1, 0);
            const double Zc = Pc.at<double>(2, 0);

            CV_Assert(std::abs(Zc) > 1e-12);

            // ----------------------------------------------------
            // 归一化相机坐标
            // ----------------------------------------------------

            const double x = Xc / Zc;
            const double y = Yc / Zc;

            const double r2 = x * x + y * y;
            const double r4 = r2 * r2;

            // ----------------------------------------------------
            // 当前模型下的无畸变理论像素坐标
            //
            // u = fx*x + skew*y + cx
            // v = fy*y + cy
            // ----------------------------------------------------

            const double u = fx * x + skew * y + cx;
            const double v = fy * y + cy;

            const double ud = observed.x;
            const double vd = observed.y;

            // ----------------------------------------------------
            // 构造：
            //
            // (ud-u) = (u-cx) * (k1*r² + k2*r⁴)
            //
            // (vd-v) = (v-cy) * (k1*r² + k2*r⁴)
            // ----------------------------------------------------

            D.at<double>(row, 0) = (u - cx) * r2;
            D.at<double>(row, 1) = (u - cx) * r4;

            d.at<double>(row, 0) = ud - u;
            
            ++row;


        }
    }

    // ------------------------------------------------------------
    // 最小二乘求：
    //
    // D * k = d
    //
    // k = [k1, k2]^T
    // ------------------------------------------------------------

    cv::Mat k;

    const bool success = cv::solve(D, d, K, cv::DECOMP_SVD);

    CV_Assert(success);
    CV_Assert(k.rows == 2 && k.cols == 1);

    const double k1 = k.at<double>(0, 0);
    const double k2 = k.at<double>(1, 0);

    // ------------------------------------------------------------
    // OpenCV 常用畸变参数格式：
    //
    // [k1, k2, p1, p2, k3]
    //
    // 当前这里只初始化 k1、k2
    // 其余先设为 0
    // ------------------------------------------------------------

    cv::Mat distortionCoefficients =
    (cv::Mat_<double>(1, 5) <<k1, k2, 0.0, 0.0, 0.0);

    return distortionCoefficients;

}



}

