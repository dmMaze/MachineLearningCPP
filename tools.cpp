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