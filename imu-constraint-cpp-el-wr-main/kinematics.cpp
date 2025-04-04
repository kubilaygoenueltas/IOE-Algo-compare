#include "kinematics.hpp"
/*
Todo:

*/

Matrix3d skew(Vector3d x)
{
  Matrix3d out; 
  out << 0, -x(2), x(1),
        x(2), 0, -x(0),
        -x(1), x(0), 0; 
  return out; 
}

ArrayXXd quatConj(ArrayXXd q)
{
    ArrayXXd qOut(q.rows(),4); 
    qOut << q.col(0), -q.col(1), -q.col(2), -q.col(3);
    return qOut; 
}

Matrix3d quatToDCM(ArrayXXd q)
{
    Matrix3d R; 
    R << q(0,0)*q(0,0) + q(0,1)*q(0,1) - q(0,2)*q(0,2) - q(0,3)*q(0,3), 
        -2*q(0,0)*q(0,3) + 2*q(0,1)*q(0,2), 
         2*q(0,0)*q(0,2) + 2*q(0,1)*q(0,3),
         2*q(0,0)*q(0,3) + 2*q(0,1)*q(0,2),
         q(0,0)*q(0,0) - q(0,1)*q(0,1) + q(0,2)*q(0,2) - q(0,3)*q(0,3), 
        -2*q(0,0)*q(0,1)+2*q(0,2)*q(0,3),
        -2*q(0,0)*q(0,2)+2*q(0,1)*q(0,3),
         2*q(0,0)*q(0,1)+2*q(0,2)*q(0,3),
         q(0,0)*q(0,0) - q(0,1)*q(0,1) - q(0,2)*q(0,2) + q(0,3)*q(0,3);
    return R; 
}

ArrayXXd quatToEuler(ArrayXXd q)
{
  ArrayXXd qOut(q.rows(),3); 
  ArrayXXd a(q.rows(),1);
  ArrayXXd b(q.rows(),1);
  ArrayXXd c(q.rows(),1);
  ArrayXXd d(q.rows(),1);

  a << 2*(q.col(0)*q.col(3) + q.col(1) * q.col(2));
  b << q.col(0)*q.col(0) + q.col(1)*q.col(1) - q.col(2)*q.col(2) - q.col(3)*q.col(3); 
  c << 2*(q.col(0)*q.col(1) + q.col(2)*q.col(3)); 
  d << q.col(0)*q.col(0) - q.col(1)*q.col(1) - q.col(2)*q.col(2) + q.col(3)*q.col(3); 

  qOut << atan2(a,b), 
          Eigen::asin(2*(q.col(0) * q.col(2) - q.col(3) *q.col(1))),
          atan2(c,d); 

  return qOut; 
}

ArrayXXd quatMultiply(ArrayXXd q1, ArrayXXd q2)
{
  ArrayXXd qOut(q1.rows(),4); 
  qOut << q1.col(0) * q2.col(0) - q1.col(1) * q2.col(1) - q1.col(2) * q2.col(2) - q1.col(3) * q2.col(3),
          q1.col(0) * q2.col(1) + q1.col(1) * q2.col(0) + q1.col(2) * q2.col(3) - q1.col(3) * q2.col(2),
          q1.col(0) * q2.col(2) - q1.col(1) * q2.col(3) + q1.col(2) * q2.col(0) + q1.col(3) * q2.col(1),
          q1.col(0) * q2.col(3) + q1.col(1) * q2.col(2) - q1.col(2) * q2.col(1) + q1.col(3) * q2.col(0); 
  return qOut; 
}

ArrayXXd quatMultiply(ArrayXXd q1, ArrayXXd q2, ArrayXXd q3)
{
    return quatMultiply(quatMultiply(q1, q2), q3); 
}

ArrayXXd quatX(double theta)
{
  ArrayXXd a(1,4);
  a << cos(0.5*theta), sin(0.5*theta), 0, 0; 
  return a; 
}

ArrayXXd quatY(double theta)
{
  ArrayXXd a(1,4);
  a << cos(0.5*theta), 0, sin(0.5*theta), 0; 
  return a; 
}

ArrayXXd quatZ(double theta)
{
  ArrayXXd a(1,4);
  a << cos(0.5*theta), 0, 0, sin(0.5*theta); 
  return a; 
}

ArrayXXd quatNormalize(ArrayXXd q)
{
    ArrayXXd N(q.rows(),1); 
    N = Eigen::sqrt(q.col(0)*q.col(0) +  q.col(1)*q.col(1) +  q.col(2)*q.col(2) +  q.col(3)*q.col(3)); 

    ArrayXXd qOut(q.rows(),4);
    qOut << q.col(0)/N, q.col(1)/N, q.col(2)/N, q.col(3)/N; 
    return qOut; 
}
ArrayXXd atan2(ArrayXXd y, ArrayXXd x)
{
    // https://forum.kde.org/viewtopic.php?t=112142
    return y.binaryExpr(x, [] (double a, double b) { return std::atan2(a,b);} );

}
