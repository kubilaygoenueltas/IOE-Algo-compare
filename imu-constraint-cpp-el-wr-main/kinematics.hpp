#ifndef KINEMATICS_HPP
#define KINEMATICS_HPP
    #include <iostream>
    #include <math.h>
    #include <Eigen/Dense> 
    using Eigen::MatrixXd; 
    using Eigen::ArrayXXd;
    using Eigen::Array4d;
    using Eigen::Matrix3d;
    using Eigen::Vector3d; 

    Matrix3d skew(Vector3d x); 

    ArrayXXd quatConj(ArrayXXd q); 
    Matrix3d quatToDCM(ArrayXXd q); 

    ArrayXXd quatMultiply(ArrayXXd q1, ArrayXXd q2);
    ArrayXXd quatMultiply(ArrayXXd q1, ArrayXXd q2, ArrayXXd q3);

    ArrayXXd quatX(double theta); 
    ArrayXXd quatY(double theta); 
    ArrayXXd quatZ(double theta);
    ArrayXXd quatToEuler(ArrayXXd q); 
    ArrayXXd quatNormalize(ArrayXXd q); 

    ArrayXXd atan2(ArrayXXd y, ArrayXXd x); 


#endif