#ifndef SVM_H
#define SVM_H

#include "utils.h"
#include "data.h"
#include "model.h"
#include "kernelfunctions.h"
#define LINEAR 0;
#define POLY 2;

struct svm_parameter: model_parameter
{
    double tol = 0.001;
};

class SVM: public Model<struct svm_parameter>, public Kernel
{
public:
    static const int Linear = 0;
    static const int Poly = 1;
    static const int Gaussian = 2;
    SVM(shared_ptr<DataLoader> _data_loader): Model(_data_loader) {}
    ~SVM(){}

    void loadData(const std::string& data_path);

    template<typename T>
    void setParam(const std::string param_flag, T value)
    {
        if(param_flag == "kernel_type")
            setKernel(value);
        else if (param_flag == "tol")
            param.tol = value;
        else if(param_flag == "poly_n")
            poly_n = value;
        else
            Model::setParam(param_flag, value);
    }
    
    void updateSupportVectors(int i)
    {
        double _tol_ = 0; 
        if (std::abs(alpha(i)) < _tol_)
        {
            alpha(i) = 0;
            vector_mask(i) = 0;
        }
        else
            vector_mask(i) = 1;
    }
    

protected:
    std::vector<int> support_vector_indices;
    Eigen::MatrixXd support_vectors;
    Eigen::MatrixXd alpha;
    Eigen::VectorXi vector_mask;
};

class SVC: public SVM {
public:
    void train();
    void SMO();
    Eigen::MatrixXd predict(const Eigen::MatrixXd &input, bool post_process = true);
    void postProcess(Eigen::MatrixXd &predict);
    SVC(shared_ptr<DataLoader> _dataloader):SVM(_dataloader){};
    bool examineExamples(int, bool);
    int takeStep(int&, int&);
    ~SVC(){}

private:
    Eigen::MatrixXd g;
    Eigen::MatrixXd E;
    double b;
    double C = 0.1;
};

void testSVC(shared_ptr<DataLoader> dataloader);
#endif