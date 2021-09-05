#ifndef KERNELFUNCTIONS_H
#define KERNELFUNCTIONS_H

struct Kernel
{
public:
    static const int Linear = 0;
    static const int Poly = 1;
    static const int Gaussian = 2;

    Eigen::MatrixXd linear(Eigen::MatrixXd &m1, Eigen::MatrixXd &m2)
    {
        return m1 * m2.transpose();
    }

    Eigen::MatrixXd poly(Eigen::MatrixXd &m1, Eigen::MatrixXd &m2)
    {
        return ((m1 * m2.transpose()).array() - 1).pow(poly_n);
    }

    Eigen::MatrixXd gaussian(Eigen::MatrixXd &m1, Eigen::MatrixXd &m2)
    {
        double gamma = 1. / 13;
        Eigen::MatrixXd result(m1.rows(), m2.rows());
        for (int i = 0; i < m1.rows(); ++i)
            result.row(i) = ((m2.rowwise() - m1.row(i)).rowwise().lpNorm<1>().array().pow(2) * (-gamma)).exp();
        return result;
    }
    void setKernel(int _kernel_type)
    {
        kernel_type = _kernel_type;
        switch (kernel_type)
        {
        case 0:
            kernel_function = &Kernel::linear;
            break;
        case 1:
            kernel_function = &Kernel::poly;
            break;
        case 2:
            kernel_function = &Kernel::gaussian;
            break;
        default:
            kernel_function = &Kernel::linear;
            break;
        }
    }
    double K(double x1, double x2)
    {
        Eigen::MatrixXd X1(1, 1), X2(1, 1);
        X1(0, 0) = x1;
        X2(0, 0) = x2;
        return (this->*kernel_function)(X1, X2)(0, 0);
    }
    
protected:
    int kernel_type;
    int poly_n;
    Eigen::MatrixXd (Kernel::*kernel_function) (Eigen::MatrixXd &m1, Eigen::MatrixXd &m2);
};

#endif