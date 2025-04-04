#ifndef MAP_ACC_HPP
#define MAP_ACC_HPP
    #include <Eigen/Sparse>
    #include <Eigen/Dense>
    #include "kinematics.hpp"

    using Eigen::MatrixXd;
    using Eigen::ArrayXXd;
    using Eigen::Vector3d;
    using Eigen::VectorXd;

    typedef Eigen::Triplet<double> T;

    // Filter
    ArrayXXd map_acc(ArrayXXd gyro,ArrayXXd accel, ArrayXXd gyro2,ArrayXXd accel2, double freq, double priNoise, double gyrNoise, double accNoise, Vector3d rUA, Vector3d rFA, ArrayXXd q1_imu, ArrayXXd q2_imu, double tol, double lam, int max_iter);
    
    // add weights
    VectorXd addWeights(int n, double prior_noise,double gyro_noise, double acc_noise); 
    
    // build sparse matrix for levenberg-marquardt
    Eigen::SparseMatrix<double> buildSparseDiag(Eigen::SparseMatrix<double> h); 

    // build lambda matrix for levenberg-marquardt
    Eigen::SparseMatrix<double> buildLam(double lam, int n);

    // build loss error
    MatrixXd build_loss_e(ArrayXXd X, ArrayXXd X0, ArrayXXd X2, ArrayXXd X2_init, ArrayXXd gyr, ArrayXXd acc, MatrixXd KK1, ArrayXXd gyr2, ArrayXXd acc2, MatrixXd KK2, double dT, Vector3d rUA, Vector3d rFA, int n);


    // build loss jacobian    
    Eigen::SparseMatrix<double> build_loss_J2(ArrayXXd X, ArrayXXd X0, ArrayXXd X2, ArrayXXd X2_init, ArrayXXd gyr, ArrayXXd acc, MatrixXd KK1, ArrayXXd gyr2, ArrayXXd acc2, MatrixXd KK2, double dT, Vector3d rUA, Vector3d rFA, int n);

    // prior error
    MatrixXd e_n(ArrayXXd q,ArrayXXd q0); 

    // prior jacobian
    Matrix3d J_n(ArrayXXd q,ArrayXXd q0);

    // gyro error
    MatrixXd e_w(ArrayXXd X, ArrayXXd gyro, double dT);

    // gyro error jacobian
    MatrixXd J_w(ArrayXXd X, double dT); 

    // link error
    MatrixXd e_link(ArrayXXd X1, ArrayXXd acc1, MatrixXd KK1, Vector3d rUA, ArrayXXd X2, ArrayXXd acc2, MatrixXd KK2, Vector3d rFA); 

    // link jacobian
    MatrixXd J_link(ArrayXXd X1, ArrayXXd acc1, MatrixXd KK1, Vector3d rUA, ArrayXXd X2, ArrayXXd acc2, MatrixXd KK2, Vector3d rFA); 

    // accel error 
    MatrixXd e_a(ArrayXXd X,ArrayXXd acc); 

    // accel error jacobian
    MatrixXd J_a(ArrayXXd X); 

    //calculation of numerical derivative of gyroscope
    ArrayXXd omDot(ArrayXXd gyro, double dT);

    //calculation of numerical derivative of gyroscope
    MatrixXd K(ArrayXXd gyr, ArrayXXd gyr_dot); 

#endif
