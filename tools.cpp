#include "tools.h"

int loadMatrix_2(const std::string& path, Eigen::MatrixXd &M, const std::string& pattern)
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
        size_t col = 14;
        if (row > 0 && col > 0)
        {
            M = Eigen::MatrixXd::Constant(row, col, 0);
			for(size_t i = 0; i < str_matrix.size(); ++i)
			{
				auto vec_line = str_matrix[i];
				M(i, col-1) = std::stod(vec_line[0]);
				for(size_t j = 1; j < vec_line.size(); ++j)
				{
					auto split_ind = vec_line[j].find(":");
					int col_ind = std::stoi(vec_line[j].substr(0, split_ind)) - 1;
					M(i, col_ind) = std::stod(vec_line[j].substr(split_ind+1));
				}
			}
            return 1;
        }
    }
    return 0;
}

template<typename T>
void Scalar::standardize(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& feature, const std::string& flag)
{
    if(flag == "col")
    {
        auto mean(feature.colwise().mean());
        feature.rowwise() -= mean;
        auto std_dev = feature.colwise().norm() / feature.rows();
        feature.array().rowwise() /= std_dev.array();
    }
    else if (flag == "row")
    {
        feature.transposeInPlace();
        standardize(feature, "col");
        feature.transposeInPlace();
    }
}
template void Scalar::standardize<float>(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& feature, const std::string& flag);
template void Scalar::standardize<double>(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& feature, const std::string& flag);

template<typename T>
void Scalar::minmax(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& feature, const std::string& flag)
{
    if (flag=="col")
    {
        auto arr = feature.array();
        auto min = arr.colwise().minCoeff();
        auto range = arr.colwise().maxCoeff() - min;
        arr.rowwise() -= min;
        arr.rowwise() /= range;
    }
    else
    {
        feature.transposeInPlace();
        minmax(feature, "col");
        feature.transposeInPlace();
    }
}
template void Scalar::minmax<float>(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& feature, const std::string& flag);
template void Scalar::minmax<double>(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& feature, const std::string& flag);

template<typename T>
void Scalar::normalize(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& feature, const std::string& flag, const std::string& norm)
{
    if (flag=="col")
    {
        auto arr = feature.array();
        if (norm=="l2")
            arr.rowwise() /= arr.colwise().norm();
        else if (norm=="l1")
            arr.rowwise() /= arr.abs().colwise().sum();
        else if (norm=="max")
            arr.rowwise() /= arr.abs().colwise().maxCoeff();
    }
    else
    {
        feature.transposeInPlace();
        normalize(feature, "col", norm);
        feature.transposeInPlace();
    }
}
template void Scalar::normalize<float>(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& feature, const std::string& flag, const std::string& norm);
template void Scalar::normalize<double>(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& feature, const std::string& flag, const std::string& norm);
