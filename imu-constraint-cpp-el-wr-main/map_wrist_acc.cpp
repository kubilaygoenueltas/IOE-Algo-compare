#include "map_wrist_acc.hpp"

ArrayXXd map_wrist_acc(ArrayXXd gyro,ArrayXXd accel, ArrayXXd gyro2,ArrayXXd accel2, double freq, double priNoise, double gyrNoise, double conNoise, double dofNoise, Vector3d rUA, Vector3d rFA, ArrayXXd q1_imu, ArrayXXd q2_imu, double tol, double lam, int max_iter)
{
    double dT = 1.0/freq; 
    ArrayXXd gyr = quatToDCM(q1_imu).transpose()*gyro.matrix().transpose();
    ArrayXXd acc = quatToDCM(q1_imu).transpose()*accel.matrix().transpose();
    ArrayXXd gyr2 = quatToDCM(q2_imu).transpose()*gyro2.matrix().transpose(); 
    ArrayXXd acc2 = quatToDCM(q2_imu).transpose()*accel2.matrix().transpose(); 

    rUA = quatToDCM(q1_imu).transpose()*rUA;
    rFA = quatToDCM(q2_imu).transpose()*rFA;

    ArrayXXd gyr_dot1 = omDot(gyr.transpose(),dT); 
    ArrayXXd gyr_dot2 = omDot(gyr2.transpose(),dT); 
    MatrixXd KK1 = K(gyr.transpose(), gyr_dot1);
    MatrixXd KK2 = K(gyr2.transpose(), gyr_dot2);

    int n = gyr.cols(); 

    ArrayXXd X(4,n-4), X2(4,n-4);
    X << ArrayXXd::Ones(1,n-4), ArrayXXd::Zero(3,n-4);
    X2 << ArrayXXd::Ones(1,n-4), ArrayXXd::Zero(3,n-4);
    VectorXd x_prev = VectorXd::Zero(6*(n-4)); 
    VectorXd xx = VectorXd::Zero(6*(n-4)); 
    
    ArrayXXd X1_init(4,1), X2_init(4,1); 
    X1_init << 1, 0, 0, 0;
    X2_init << 1, 0, 0, 0;

    VectorXd WW = addWeights(n,priNoise,gyrNoise,conNoise,dofNoise);
    Eigen::DiagonalMatrix<double,Eigen::Dynamic> W(WW.rows());
    W.diagonal() << WW; 

    ArrayXXd out(n-4,12); 

    Eigen::SparseMatrix<double> L, h_diag; 
    L = buildLam(lam, 6*(n-4)); 

    double er=0, er_prev; 

    for (int iter = 1; iter <= max_iter; iter++)
    {
       std::cout << "iter: " << iter << std::endl; 
       MatrixXd ee = build_loss_e(X, X1_init, X2, X2_init, gyr.leftCols(n), acc.leftCols(n), KK1, gyr2.leftCols(n), acc2.leftCols(n), KK2, dT, rUA, rFA, n);

       Eigen::SparseMatrix<double> JJ = build_loss_J2(X, X1_init, X2, X2_init, gyr.leftCols(n), acc.leftCols(n), KK1, gyr2.leftCols(n), acc2.leftCols(n), KK2, dT, rUA, rFA, n);

       MatrixXd grad = JJ.transpose()*W*ee;

       Eigen::SparseMatrix<double> hess = JJ.transpose()*W*JJ;

       // Solve Sparse Matrix
       Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double> > solver;
       //Eigen::SparseLU<Eigen::SparseMatrix<double> > solver;

       er_prev = er;
       x_prev = xx; 
       er = (ee.transpose()*W*ee).value(); 

       std::cout << "  loss: " << er << std::endl;

       if (er < er_prev || iter == 1)
       {
            // Levemberg marquardt
            MatrixXd grad = JJ.transpose()*W*ee;
            Eigen::SparseMatrix<double> h = JJ.transpose()*W*JJ;
            h_diag= buildSparseDiag(h); 
            Eigen::SparseMatrix<double> hess = h + L*h_diag;  

            // Gauss Newton
            // MatrixXd grad = JJ.transpose()*W*ee;
            // Eigen::SparseMatrix<double> hess = JJ.transpose()*W*JJ;

            // solve
            solver.compute(hess);
            xx = solver.solve(-grad);

            int nn = n-4; 
            // Correct Quaternion
            ArrayXXd q1(nn,4);
            VectorXd x1 = xx.head(3*nn); 
            q1 << ArrayXXd::Ones(nn,1), 0.5*x1.reshaped(3,nn).transpose(); 
            X = quatNormalize(quatMultiply(X.transpose(),q1)).transpose();

            ArrayXXd q2(nn,4);
            VectorXd x2 = xx.tail(3*nn); 
            q2 << ArrayXXd::Ones(nn,1), 0.5*x2.reshaped(3,nn).transpose();
            X2 = quatNormalize(quatMultiply(X2.transpose(),q2)).transpose();
        }

        else
        {
            std::cout << "local minimum reached" << std::endl; 
            xx = x_prev; 
            break; 
        }

        VectorXd asdf = ((xx-x_prev).cwiseInverse().cwiseProduct(xx)).cwiseAbs(); 
        if (asdf.all() < tol && iter > 1)
        {
            std::cout << "tolerance criteria reached" << std::endl; 
            break; 
        }

        if (iter == max_iter)
        {
            std::cout << "Max Iter Reached" << std::endl; 
        }
    }
    std::cout <<"Done!" << std::endl; 
    out << quatMultiply(quatConj(X.transpose()), X2.transpose()), X.transpose(), X2.transpose(); 
    return out; 
}

VectorXd addWeights(int n, double prior_noise,double gyro_noise, double acc_noise, double dof_noise)
{
    n = n - 4; 
    double pri = (1/prior_noise)*(1/prior_noise); 
    double acc = (1/acc_noise)*(1/acc_noise); 
    double gyr = (1/gyro_noise)*(1/gyro_noise);
    double dof = (1/dof_noise)*(1/dof_noise);

    VectorXd W(n*10); 
    int i = 0;
    std::cout << "adding pri weight... " << std::endl; 
    W.segment(i,3) << pri, pri, pri; 
    i = i + 3; 

    std::cout << "adding gyr weight... " << std::endl; 
    for (int j=0; j< n-1; j++) 
    {
        W.segment(i,3) << gyr, gyr, gyr; 
        i = i+3; 
    }

    std::cout << "adding pri weight... " << std::endl; 
    W.segment(i,3) << pri, pri, pri; 
    i = i + 3; 

    std::cout << "adding gyr weight... " << std::endl; 
    for (int j=0; j< n-1; j++) 
    {
        W.segment(i,3) << gyr, gyr, gyr; 
        i = i+3; 
    }

    std::cout << "adding link weight... " <<  std::endl; 
    for (int j=0; j< n; j++) 
    {
        W.segment(i,3) << acc, acc,acc;
        i = i+3; 
    }

    std::cout << "adding dof weight... " << std::endl; 
    for (int j=0; j< n; j++) 
    {
        W.segment(i,1) << dof;
        i = i+1; 
    }

    return W; 
}

Eigen::SparseMatrix<double> buildSparseDiag(Eigen::SparseMatrix<double> h)
{
    int nn = h.rows(); 
    Eigen::SparseMatrix<double> L(nn,nn); 
    std::vector<T> tripletList;
    for (int i=0; i<nn; i++)
    {
        tripletList.push_back(T(i,i,h.coeff(i,i))); 
    }
    L.setFromTriplets(tripletList.begin(), tripletList.end());
    return L; 
}

Eigen::SparseMatrix<double> buildLam(double lam, int n)
{
    Eigen::SparseMatrix<double> L(n,n); 
    std::vector<T> tripletList;

    for (int i=0; i<n; i++)
    {
        tripletList.push_back(T(i,i,lam));
    }
    L.setFromTriplets(tripletList.begin(), tripletList.end());
    return L; 
}



MatrixXd build_loss_e(ArrayXXd X1, ArrayXXd X1_init, ArrayXXd X2, ArrayXXd X2_init,
                     ArrayXXd gyr1, ArrayXXd acc1, MatrixXd KK1,
                     ArrayXXd gyr2, ArrayXXd acc2, MatrixXd KK2, double dT, Vector3d rUA, Vector3d rFA, int n)

{
    n = n - 4; 
    MatrixXd ee(3,3*n); 
    MatrixXd ee2(10*n,1); 

    ee << e_n(X1,X1_init), e_w(X1,gyr1.block(0,2,3,n),dT), e_n(X2,X2_init), e_w(X2,gyr2.block(0,2,3,n),dT), e_link(X1,acc1,KK1,rUA,X2,acc2,KK2,rFA); 
    ee2 << ee.reshaped(), e_dof(X1,X2); 
    return ee2; 
}

Eigen::SparseMatrix<double> build_loss_J2(ArrayXXd X, ArrayXXd X0, ArrayXXd X2, ArrayXXd X2_init, ArrayXXd gyr, ArrayXXd acc, MatrixXd KK1, ArrayXXd gyr2, ArrayXXd acc2, MatrixXd KK2, double dT, Vector3d rUA, Vector3d rFA, int n)
{
    n = n - 4; 

    Matrix3d J1, J11; 
    MatrixXd J2(3,6), J3(1,6); 
    std::vector<T> tripletList;
    Eigen::SparseMatrix<double> JJ(n*10,n*6);
    
    int i = 0; 
    J1 = J_n(X,X0);
    for (int ii = 0; ii < 3; ii++)
    {
        for (int jj = 0; jj < 3; jj++)
        {
            tripletList.push_back(T(ii + i,jj,J1(ii,jj)));
        }
    }
    i = i + 3; 

    // gyro
    for (int j=0; j< n-1; j++)
    {   
        J2 = J_w(X.block(0,j,4,2),dT);
        for (int ii = 0; ii < 3; ii++)
        {
            for (int jj = 0; jj < 6; jj++)
            {
                tripletList.push_back(T(ii + i,jj + j*3,J2(ii,jj)));
            }
        }
        i = i + 3;
    }

    J1 = J_n(X2,X2_init);
    for (int ii = 0; ii < 3; ii++)
    {
        for (int jj = 0; jj < 3; jj++)
        {
            tripletList.push_back(T(ii + i,jj + 3*n,J1(ii,jj)));
        }
    }
    i = i + 3; 

    // gyro
    for (int j=0; j< n-1; j++)
    {   
        J2 = J_w(X2.block(0,j,4,2),dT);
        for (int ii = 0; ii < 3; ii++)
        {
            for (int jj = 0; jj < 6; jj++)
            {
                tripletList.push_back(T(ii + i,jj + j*3 +3*n,J2(ii,jj)));
            }
        }
        i = i + 3;
    }

    // acc
    for (int j=0; j<n; j++)
    {
        J2 = J_link(X.block(0,j,4,1), acc.block(0,j+2,3,1), KK1.block(j*3,0,3,3), rUA, X2.block(0,j,4,1), acc2.block(0,j+2,3,1), KK2.block(j*3,0,3,3), rFA);
        J1 = J2.leftCols(3);
        J11 = J2.rightCols(3); 

        for (int ii = 0; ii < 3; ii++)
        {
            for (int jj = 0; jj < 3; jj++)
            {   
                tripletList.push_back(T(ii + i,jj + j*3,J1(ii,jj)));
                tripletList.push_back(T(ii + i,jj + j*3 + 3*n,J11(ii,jj)));
            }
        }

        i = i + 3;
    }

    // dof
    for (int j=0; j<n; j++)
    {
        J3 = J_dof(X.block(0,j,4,1),X2.block(0,j,4,1)); 
        for (int jj = 0; jj < 3; jj++)
        {   
            tripletList.push_back(T(i,jj + j*3,J3(jj)));
            tripletList.push_back(T(i,jj + j*3 + 3*n,J3(jj+3)));
        }
        i = i + 1; 
    }

    JJ.setFromTriplets(tripletList.begin(), tripletList.end());
    // std::cout << JJ << std::endl; 
    return JJ; 
}

// prior error
MatrixXd e_n(ArrayXXd q,ArrayXXd q0)
{
    ArrayXXd e(q.cols(),4);
    e = 2*quatMultiply(quatConj(q0.col(0).transpose()),q.col(0).transpose());

    MatrixXd out(q0.cols(),3); 
    out << e.col(1), e.col(2), e.col(3); 
    return out.transpose(); 
}

// prior jacobian
Matrix3d J_n(ArrayXXd q,ArrayXXd q0)
{
    Eigen::ArrayXd qq; 
    MatrixXd D(4,3); 
    D << 0, 0, 0,
         1.0, 0, 0,
         0, 1.0, 0,
         0, 0, 1.0;

    qq = quatMultiply(quatConj(q0.transpose()),q.col(0).transpose()).transpose(); 
    MatrixXd Q(4,4); 
    Q <<  qq(0),  -qq(1), -qq(2), -qq(3),
          qq(1),   qq(0), -qq(3),  qq(2),
          qq(2),   qq(3),  qq(0), -qq(1),
          qq(3),  -qq(2),  qq(1),  qq(0); 
    return D.transpose()*Q*D; 
}

// gyro error
MatrixXd e_w(ArrayXXd X, ArrayXXd gyro, double dT)
{
    int n = X.cols(); 
    ArrayXXd q(X.rows()-1,4); 
    q = quatMultiply(quatConj(X.leftCols(n-1).transpose()),X.rightCols(n-1).transpose()); 
    MatrixXd e(3,n-1); 
    return (2/dT) * q.rightCols(3).transpose() - gyro.rightCols(n-1); 
}

// gyro error jacobian
MatrixXd J_w(ArrayXXd X, double dT)
{
    MatrixXd D(4,3); 
    D << 0, 0, 0,
         1.0, 0, 0,
         0, 1.0, 0,
         0, 0, 1.0;

    MatrixXd K(4,4); 
    K << 1.0,  0,  0,  0,
         0, -1.0,  0,  0,
         0,  0, -1.0,  0,
         0,  0,  0, -1.0;

    ArrayXXd XX = quatMultiply(quatConj(X.col(0).transpose()),X.col(1).transpose()); 
    MatrixXd qL(4,4); 
    qL <<  XX(0), -XX(1), -XX(2), -XX(3), 
           XX(1),  XX(0), -XX(3),  XX(2),
           XX(2),  XX(3),  XX(0), -XX(1),
           XX(3), -XX(2),  XX(1),  XX(0);
    
    MatrixXd qR(4,4); 
    qR <<  XX(0), -XX(1), -XX(2), -XX(3), 
           XX(1),  XX(0),  XX(3), -XX(2),
           XX(2), -XX(3),  XX(0),  XX(1),
           XX(3),  XX(2), -XX(1),  XX(0);

    MatrixXd J(3,6);     
    J << (1.0/dT)*D.transpose()*qR*K*D, (1.0/dT)*D.transpose()*qL*D;
    return J;
}

// link error
MatrixXd e_link(ArrayXXd X1, ArrayXXd acc1, MatrixXd KK1, Vector3d rUA, ArrayXXd X2, ArrayXXd acc2, MatrixXd KK2, Vector3d rFA)
{
    int n = acc1.cols() - 4; 
    MatrixXd e(3,n); 
    Matrix3d C1, C2; 
    for (int i=0; i < n; i++)
    { 
        C1 = quatToDCM(X1.col(i).transpose()); 
        C2 = quatToDCM(X2.col(i).transpose());
        e.col(i) = C1*(acc1.col(i+2).matrix() - KK1.block(i*3,0,3,3)*rUA) - C2*(acc2.col(i+2).matrix() - KK2.block(i*3,0,3,3)*rFA);
    }
    
    return e;
}

// link jacobian
MatrixXd J_link(ArrayXXd X1, ArrayXXd acc1, MatrixXd KK1, Vector3d rUA, ArrayXXd X2, ArrayXXd acc2, MatrixXd KK2, Vector3d rFA)
{
    MatrixXd C1 = quatToDCM(X1.transpose()); 
    Vector3d a1 = acc1.matrix()-KK1*rUA;
    Matrix3d H1 = -C1*skew(a1); 

    MatrixXd C2 = quatToDCM(X2.transpose());
    Vector3d a2 = acc2.matrix()-KK2*rFA;
    Matrix3d H2 = C2*skew(a2); 

    MatrixXd out(3,6);
    out << H1, H2;
    return out; 
}

// accel error 
MatrixXd e_a(ArrayXXd X,ArrayXXd acc)
{
    double g = 9.81; 
    int n = X.cols(); 
    MatrixXd G(n,4); 
    G << ArrayXXd::Zero(n,3), ArrayXXd::Ones(n,1)*g; 
    ArrayXXd q = quatMultiply(quatConj(X.transpose()), G, X.transpose()); 
    return acc - q.rightCols(3).transpose();
}

// accel error jacobian
MatrixXd J_a(ArrayXXd X)
{
    MatrixXd g(3,1);
    g << 0.0, 0.0, 9.81; 

    MatrixXd C = quatToDCM(X.transpose()); 
    return -skew(C.transpose()*g); 
}

// dof error
MatrixXd e_dof(ArrayXXd X1, ArrayXXd X2)
{
    ArrayXXd q = quatMultiply(quatConj(X1.transpose()),X2.transpose()); 

    //e = [0 1 0]*transpose(ua)*fa*[0;0;1]; 
    return 2*(q.col(2)*q.col(3) - q.col(0)*q.col(1)); 
}

// dof error jacobian
MatrixXd J_dof(ArrayXXd X1, ArrayXXd X2)
{
    Matrix3d R = quatToDCM(X1.transpose());
    Matrix3d RR = quatToDCM(X2.transpose());
    MatrixXd e1(3,1); 
    MatrixXd e2(3,1); 
    MatrixXd e3(3,1); 

    e1 << 1.0, 0.0, 0.0;
    e2 << 0.0, 1.0, 0.0;
    e3 << 0.0, 0.0, 1.0;

    Matrix3d J1 = R.transpose()*RR;
    Matrix3d J2 = RR.transpose()*R; 

    MatrixXd J(1,6); 
    J  << e3.transpose()*J1*e3, 0.0, -e1.transpose()*J1*e3, e2.transpose()*J2*e2, -e1.transpose()*J2*e2, 0.0; 
    return J;
}


ArrayXXd omDot(ArrayXXd gyro, double dT)
{
    int n = gyro.rows(); 
    ArrayXXd out(n-4,3); 
    for (int i=2; i<n-2; i++)
    {
        out.row(i-2) = (gyro.row(i-2) - 8*gyro.row(i-1) + 8*gyro.row(i+1) - gyro.row(i+2))/(12*dT); 
    }
    return out; 
}

MatrixXd K(ArrayXXd gyr, ArrayXXd gyr_dot)
{
    int n = gyr_dot.rows(); 
    MatrixXd out(3*n,3); 

    for (int i = 0; i < n; i++)
    {
        out.block(i*3,0,3,3) << -gyr(i+2,1)*gyr(i+2,1) - gyr(i+2,2)*gyr(i+2,2),
                                gyr(i+2,0)*gyr(i+2,1) - gyr_dot(i,2), 
                                gyr_dot(i,1) + gyr(i+2,0)*gyr(i+2,2),
                                gyr_dot(i,2) + gyr(i+2,0)*gyr(i+2,1), 
                                -gyr(i+2,0)*gyr(i+2,0) - gyr(i+2,2)*gyr(i+2,2), 
                                gyr(i+2,1)*gyr(i+2,2) - gyr_dot(i,0),
                                gyr(i+2,0)*gyr(i+2,2) - gyr_dot(i,1),
                                 gyr_dot(i,0)+gyr(i+2,1)*gyr(i+2,2),
                                -gyr(i+2,0)*gyr(i+2,0) - gyr(i+2,1)*gyr(i+2,1); 
    }
    return out; 
}

