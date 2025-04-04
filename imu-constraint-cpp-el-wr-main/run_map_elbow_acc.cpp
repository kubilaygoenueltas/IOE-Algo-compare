#include <iostream>
#include <fstream>
#include <stdlib.h>     /* atof */

#include "kinematics.hpp"
#include "map_elbow_acc.hpp"

// compile:
// g++ -O3 -I .\eigen-3.4.0 kinematics.cpp run_map_elbow_acc.cpp map_elbow_acc.cpp -o map_elbow_acc
// run:
// open git bash,  execute ./run_map_elbow_acc.sh 


ArrayXXd importFile(std::string fileName)
{
  std::ifstream myfile(fileName.c_str());
  double acc1[3]; 
  double gyr1[3];
  double acc2[3]; 
  double gyr2[3];

  std::vector<double> ax1, ay1, az1; 
  std::vector<double> gx1, gy1, gz1;
  std::vector<double> ax2, ay2, az2; 
  std::vector<double> gx2, gy2, gz2; 

  if (myfile.is_open())
  {
    while ( myfile >> acc1[0] >> acc1[1] >> acc1[2]
                   >> gyr1[0] >> gyr1[1] >> gyr1[2]
                   >> acc2[0] >> acc2[1] >> acc2[2]
                   >> gyr2[0] >> gyr2[1] >> gyr2[2])
    {
      ax1.push_back(acc1[0]); 
      ay1.push_back(acc1[1]); 
      az1.push_back(acc1[2]); 
      gx1.push_back(gyr1[0]); 
      gy1.push_back(gyr1[1]); 
      gz1.push_back(gyr1[2]); 
      
      ax2.push_back(acc2[0]); 
      ay2.push_back(acc2[1]); 
      az2.push_back(acc2[2]); 
      gx2.push_back(gyr2[0]); 
      gy2.push_back(gyr2[1]); 
      gz2.push_back(gyr2[2]);
    }

    std::cout << "Matrix Length: " << ax1.size() << std::endl; 
    Eigen::Map<VectorXd> v1(ax1.data(),ax1.size()); 
    Eigen::Map<VectorXd> v2(ay1.data(),ay1.size()); 
    Eigen::Map<VectorXd> v3(az1.data(),az1.size()); 
    Eigen::Map<VectorXd> v4(gx1.data(),gx1.size()); 
    Eigen::Map<VectorXd> v5(gy1.data(),gy1.size()); 
    Eigen::Map<VectorXd> v6(gz1.data(),gz1.size()); 
        
    Eigen::Map<VectorXd> v7(ax2.data(),ax2.size()); 
    Eigen::Map<VectorXd> v8(ay2.data(),ay2.size()); 
    Eigen::Map<VectorXd> v9(az2.data(),az2.size()); 
    Eigen::Map<VectorXd> v10(gx2.data(),gx2.size()); 
    Eigen::Map<VectorXd> v11(gy2.data(),gy2.size()); 
    Eigen::Map<VectorXd> v12(gz2.data(),gz2.size()); 


    ArrayXXd data(ax1.size(),12) ; 
    data << v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12; 
    return data;
  }

  else
  {
    std::cout << "Unable to open file";
    MatrixXd a(1,1); 
    a << -1; 
    return a; 
  }
}

int main(int argc, char **argv)
{
    std::cout << argc << std::endl; 
    std::string file = argv[1]; 

    std::cout << "Importing " << file << std::endl; 
    ArrayXXd data = importFile(file);
    int n = data.rows(); 
    ArrayXXd accel1 = data.block(0,0,n,3);
    ArrayXXd gyro1 = data.block(0,3,n,3);
    ArrayXXd accel2 = data.block(0,6,n,3);
    ArrayXXd gyro2 =  data.block(0,9,n,3);

    double freq = atof(argv[2]); 
    double priNoise = atof(argv[3]);
    double gyrNoise = atof(argv[4]); 
    double conNoise = atof(argv[5]); 
    double dofNoise = atof(argv[6]); 

    Vector3d rUA, rUA2, rFA; 
    ArrayXXd q1_imu(1,4), q2_imu(1,4); 
    rUA2 << atof(argv[7]), atof(argv[8]), atof(argv[9]); 
    rFA << atof(argv[10]), atof(argv[11]), atof(argv[12]); 
    q1_imu << atof(argv[13]), atof(argv[14]), atof(argv[15]), atof(argv[16]); 
    q2_imu << atof(argv[17]), atof(argv[18]), atof(argv[19]), atof(argv[20]); 
    double tol = atof(argv[21]); 
    double lam = atof(argv[22]); 
    int max_iter = atoi(argv[23]); 

    std:: cout << "freq: " << freq << std::endl; 
    std:: cout << "priNoise: " << priNoise << std::endl; 
    std:: cout << "gyrNoise: " << gyrNoise << std::endl; 
    std:: cout << "conNoise: " << conNoise << std::endl; 
    std:: cout << "dofNoise: " << dofNoise << std::endl; 
    std:: cout << "rUA2: " << rUA2.transpose() << std::endl; 
    std:: cout << "rFA: " << rFA.transpose() << std::endl; 
    std:: cout << "q1_imu: " << q1_imu << std::endl; 
    std:: cout << "q2_imu: " << q2_imu << std::endl; 
    std:: cout << "tol: " << tol << std::endl; 
    std:: cout << "lam: " << lam << std::endl; 
    std:: cout << "max_iter: " << max_iter << std::endl; 

    std::cout << "running filter... " << std::endl; 
    ArrayXXd out = map_elbow_acc(gyro1, accel1, gyro2, accel2, freq, priNoise, gyrNoise, conNoise, dofNoise, rUA2, rFA, q1_imu, q2_imu, tol, lam, max_iter); 

    std::ofstream myfile ("out.txt");
    if (myfile.is_open())
    {
         myfile << out;
    }

    return 0;
}

