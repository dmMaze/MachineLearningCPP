#ifndef SVM_H
#define SVM_H

#include "tools.h"
#define LINEAR 0;
#define POLY 2;

struct svm_parameter
{
    int kernel_type;
    double tol;
    double min_delta_loss;
    int loss_eval_circle;
    int max_iterations;
    double loss = -1;
    double delta_loss = 1e10;
    int verbose;
    int current_iteration;
    double training_acc;
    double max_acc = -1;
    int poly_n = 1;
};

class SVM
{
friend double calculate_error(const Eigen::MatrixXd& predict, const Eigen::MatrixXd& gts);
friend double classify_accuracy(const Eigen::MatrixXd& classify_result, const Eigen::MatrixXd& gt);
public:
    static const int Linear = 0;
    static const int Poly = 1;
    static const int Gaussian = 2;
    SVM(DataLoader* _data_loader, int kernel_type = 0, double tol = 0.001, int loss_eval_circle = 10, double min_delta_loss = 0.0001, int max_iterations = -1, bool verbose = false)
    {
        dataloader = _data_loader;
        std::string kernel_flag = "kernel_type";
        setParam(kernel_flag, kernel_type);
        param.kernel_type = kernel_type;
        param.tol = tol;
        param.loss_eval_circle = loss_eval_circle;
        param.min_delta_loss = min_delta_loss;
        param.max_iterations = max_iterations;
        param.verbose = verbose;
    }
    ~SVM()
    {
        delete dataloader;
    }

    virtual void train() = 0;
    virtual Eigen::MatrixXd predict(const Eigen::MatrixXd &input, bool post_process) = 0;
    void loadData(const std::string& data_path);
    void setKernel();

    template<typename T>
    void setParam(const std::string param_flag, T value)
    {
        if(param_flag == "kernel_type")
        {
            param.kernel_type = value;
            setKernel();
        }
        else if (param_flag == "tol")
            param.tol = value;
        else if(param_flag == "min_delta_loss")
            param.min_delta_loss = value;
        else if(param_flag == "loss_eval_circle")
            param.loss_eval_circle = value;
        else if(param_flag == "max_iterations")
            param.max_iterations = value;
        else if(param_flag == "verbose")
            param.verbose = value;
        else if(param_flag == "poly_n")
            param.poly_n = value;
    }

    Eigen::MatrixXd (SVM::*kernel_function) (Eigen::MatrixXd &m1, Eigen::MatrixXd &m2);

    Eigen::MatrixXd linear(Eigen::MatrixXd &m1, Eigen::MatrixXd &m2)
    {
        return m1 * m2.transpose();
    }

    Eigen::MatrixXd poly(Eigen::MatrixXd &m1, Eigen::MatrixXd &m2)
    {
        return ((m1 * m2.transpose()).array() - 1).pow(param.poly_n);
    }

    Eigen::MatrixXd gaussian(Eigen::MatrixXd &m1, Eigen::MatrixXd &m2)
    {
        double gamma = 1. / 13;
        Eigen::MatrixXd result(m1.rows(), m2.rows());
        for (int i = 0; i < m1.rows(); ++i)
            result.row(i) = ((m2.rowwise() - m1.row(i)).rowwise().lpNorm<1>().array().pow(2) * (-gamma)).exp();
        return result;
    }

    double K(double x1, double x2)
    {
        Eigen::MatrixXd X1(1, 1), X2(1, 1);
        X1(0, 0) = x1;
        X2(0, 0) = x2;
        return (this->*kernel_function)(X1, X2)(0, 0);
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

    void printVerbose();

protected:
    struct svm_parameter param;
    std::vector<int> support_vector_indices;
    Eigen::MatrixXd support_vectors;
    Eigen::MatrixXd alpha;
    Eigen::VectorXi vector_mask;
    DataLoader *dataloader;
};

class SVC: public SVM {
public:
    void train();
    void SMO();
    Eigen::MatrixXd predict(const Eigen::MatrixXd &input, bool post_process = true);
    void postProcess(Eigen::MatrixXd &predict);
    SVC(DataLoader *_dataloader, int kernel_type = 0, double tol = 0.001, int loss_eval_circle = 10, double min_delta_loss = 0.0001, int max_iterations = -1)
        :SVM(_dataloader, kernel_type, tol, loss_eval_circle, min_delta_loss, max_iterations){};
    bool examineExamples(int, bool);
    int takeStep(int&, int&);

private:
    
    Eigen::MatrixXd g;
    Eigen::MatrixXd E;
    double b;
    double C = 0.1;
};
#endif