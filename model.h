#ifndef MODEL_H
#define MODEL_H
#include "utils.h"

struct model_parameter
{
    double min_delta_loss = -1;
    int loss_eval_circle = 5;
    int max_iterations = 200;
    double loss = -1;
    double delta_loss = 1e10;
    int verbose = 1;
    int current_iteration = -1;
    double training_acc = -1;
    double max_acc = -1;
    double learning_rate = 0.01;
    int batch_size = 1;
};

template <typename ParamStruct>
class Model
{
private:

public:
    Model(){}
    Model(shared_ptr<DataLoader> _data_loader): dataloader(_data_loader){}
    ~Model(){}
    virtual void train() = 0;
    // virtual Eigen::MatrixXd predict(const Eigen::MatrixXd &input, bool post_process) = 0;
    void printVerbose()
    {
        std::cout << param.current_iteration << "/" << param.max_iterations << "\t";
        std::cout << "loss: " << param.loss << "\t" << "acc: " << int(param.training_acc*10000)/100. << "%";
        std::cout <<"\t" << "max: " << int(param.max_acc*10000)/100. << "%" << std::endl;
    }

    template<typename T>
    void setParam(const std::string param_flag, T value)
    {
        if(param_flag == "min_delta_loss")
            param.min_delta_loss = value;
        else if(param_flag == "loss_eval_circle")
            param.loss_eval_circle = value;
        else if(param_flag == "max_iterations")
            param.max_iterations = value;
        else if(param_flag == "verbose")
            param.verbose = value;
    }

protected:
    shared_ptr<DataLoader> dataloader = nullptr;
    ParamStruct param;   
};
#endif