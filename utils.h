#ifndef UTILS_H
#define UTILS_H

#include <memory>
#include <string>
#include <set>
#include <list>
// #include 
#include <iostream>
#include <ostream>
#include <istream>
#include <sstream>
#include <fstream>
#include <regex>
#define EIGEN_USE_BLAS
#include <Eigen/Core>

using std::shared_ptr;
using std::unique_ptr;
using std::make_shared;
using std::cout;
using std::cin;
using std::set;

namespace Scaler{
    const int STD = 0;
    const int MINMAX = 1;
    const int NORMALIZE = 2;
    const int COL = 0;
    const int ROW = 1;
    const int NORM_L2 = 0;
    const int NORM_L1 = 1;
    const int NORM_MAX = 2;

    template<typename T>
    void scale(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& feature, int flags[])
    {
        if (flags[1] == COL)
        {
            if (flags[0] == STD)
            {
                auto mean(feature.colwise().mean());
                feature.rowwise() -= mean;
                auto std_dev = feature.colwise().norm() / feature.rows();
                feature.array().rowwise() /= std_dev.array();
            }
            else if (flags[0] == MINMAX)
            {
                auto arr = feature.array();
                auto min = arr.colwise().minCoeff();
                auto range = arr.colwise().maxCoeff() - min;
                arr.rowwise() -= min;
                arr.rowwise() /= range;
            }
            else if(flags[0] == NORMALIZE)
            {
                auto norm = flags[2];
                auto arr = feature.array();
                if (norm==NORM_L2)
                    arr.rowwise() /= arr.colwise().norm();
                else if (norm==NORM_L1)
                    arr.rowwise() /= arr.abs().colwise().sum();
                else if (norm==NORM_MAX)
                    arr.rowwise() /= arr.abs().colwise().maxCoeff();
            }
        }
        else{
            flags[1] = COL;
            feature.transposeInPlace();
            scale(feature, flags);
            feature.transposeInPlace();
        }
    }

    template<typename T>
    void standardize(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& feature, int flag = COL)
    {
        int args[] = {STD, flag};
        scale(feature, args);
    }
    template<typename T>
    void minmax(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& feature, int flag = COL)
    {
        int args[] = {MINMAX, flag};
        scale(feature, args);
    }
    template<typename T>
    void normalize(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& feature, int flag = COL, int norm = NORM_L2)
    {
        int args[] = {NORMALIZE, flag, norm};
        scale(feature, args);
    }
}

template <typename T>
int loadMatrix(const std::string& path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &M, const std::string& pattern = " ")
{    
    std::ifstream str(path);
    std::string line, ele;
    std::regex r(pattern);
    
    if (str.is_open())
    {
        std::vector<std::vector<std::string>> str_matrix;
        while(std::getline(str, line))
        {
            str_matrix.push_back(std::vector<std::string>(std::sregex_token_iterator(line.begin(), line.end(), r, -1),
                                        std::sregex_token_iterator()));
        }
        size_t row = str_matrix.size();
        size_t col = str_matrix[0].size();
        if (row > 0 && col > 0)
        {
            M = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(row, col);
            for (size_t i = 0; i < row; ++i)
                for (size_t j = 0; j < col; ++j)
                    M(i, j) = std::stod(str_matrix[i][j]);
            return 1;
        }
    }
    return 0;
}

int loadMatrix_2(const std::string& path, Eigen::MatrixXd &M, const std::string& pattern = " ");

inline double calculate_error(const Eigen::MatrixXd& predict, const Eigen::MatrixXd& gts)
{
    return (predict.array() - gts.array()).abs().sum();
}

inline double hinge_loss(const Eigen::MatrixXd& predict, const Eigen::MatrixXd& gts)
{
    auto mask = (predict.array() * gts.array() > 0).select(Eigen::MatrixXd::Ones(gts.rows(), 1),
                                                            Eigen::MatrixXd::Zero(gts.rows(), 1));
    return ((predict.array() - gts.array()).abs() * mask.array()).sum();
}

inline double classify_accuracy(const Eigen::MatrixXd& classify_result, const Eigen::MatrixXd& gt)
{
    return (classify_result.array() == gt.array()).select(Eigen::MatrixXd::Ones(gt.rows(), gt.cols()), 
                                                        Eigen::MatrixXd::Zero(gt.rows(),gt.cols())).sum() / gt.count();
}

#endif