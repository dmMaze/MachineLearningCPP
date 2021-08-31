#include "SVM.h"

void SVM::setKernel()
{
    switch (param.kernel_type)
    {
    case 0:
        kernel_function = &SVM::linear;
        break;
    case 1:
        kernel_function = &SVM::poly;
        break;
    case 2:
        kernel_function = &SVM::gaussian;
        break;
    default:
        kernel_function = &SVM::linear;
        break;
    }
}

void SVM::printVerbose()
{
    std::cout << param.current_iteration << "/" << param.max_iterations << "\t";
    std::cout << "loss: " << param.loss << "\t" << "acc: " << int(param.training_acc*10000)/100. << "%";
    std::cout <<"\t" << "max: " << int(param.max_acc*10000)/100. << "%" << std::endl;
}

int SVC::takeStep(int& i1, int& i2)
{
    
    double L, H;
    if (dataloader->Y(i1) == dataloader->Y(i2))
    {
        L = std::max(0., alpha(i2) + alpha(i1) - C);
        H = std::min(C,  alpha(i2) + alpha(i1));
    }
    else{
        L = std::max(0., alpha(i2) - alpha(i1));
        H = std::min(C,  C + alpha(i2) - alpha(i1));
    }
    if (L == H)
        return 0;
    Eigen::MatrixXd x1 = dataloader->X(i1, Eigen::all);
    Eigen::MatrixXd x2 = dataloader->X(i2, Eigen::all);
    double k12 = (this->*kernel_function)(x1, x2)(0);
    double k11 = (this->*kernel_function)(x1, x1)(0);
    double k22 = (this->*kernel_function)(x2, x2)(0);
    double der = k11 + k22 - 2 * k12;
    if (der <= 0)
        return 0;

    double lr = 1;
    double alpha_unclipped = alpha(i2) + lr * dataloader->Y(i2) * (E(i1) - E(i2)) / der;
    double alpha_new = std::min(std::max(alpha_unclipped, L), H);
    double alpha2_delta = alpha_new - alpha(i2);
    double alpha1_delta = -alpha2_delta * dataloader->Y(i1) * dataloader->Y(i2);
    
    alpha(i1) += alpha1_delta;
    alpha(i2) = alpha_new;
    updateSupportVectors(i1);
    updateSupportVectors(i2);
    // double b1 = E(i1) + dataloader->Y(i1) * alpha1_delta * k11 + dataloader->Y(i2) * alpha2_delta * k12 + b;
    // double b2 = E(i2) + dataloader->Y(i1) * alpha1_delta * k12 + dataloader->Y(i2) * alpha2_delta * k22 + b;
    // b = (b1 + b2) / 2;
    return 1;
}

bool SVC::examineExamples(int examineAll, bool eval_delta_loss)
{
    // examineAll = 1;
    bool numChanged = 0;
    g = predict(dataloader->X, false);
    E.array() = g.array() - dataloader->Y.array();
    if(eval_delta_loss)
    {
        double current_loss = E.array().abs().sum();
        param.delta_loss = std::abs(current_loss - param.loss);
        param.loss = current_loss;
        if (param.min_delta_loss > 0 && param.delta_loss < param.min_delta_loss)
            return 0;
    }

    for (int i = 0; i < dataloader->Y.rows(); ++i)
    {
        double alpha_2 = alpha(i);
        if (!examineAll && alpha_2 == 0)
            continue;
            
        double r = E(i) * dataloader->Y(i);
        if((r < -param.tol && alpha_2 < C) || (r > param.tol && alpha_2 > 0))
        {
            double max_step_size = 0;
            int alpha1_index = -1;
            for (int j = 0; j < dataloader->Y.rows(); ++j)
            {
                if(j == i || E(j) == C || E(j) == 0)
                    continue;
                double step_size = std::abs(E(i) - E(j));
                if (step_size > max_step_size)
                {
                    max_step_size = step_size;
                    alpha1_index = j;
                }   
            }
            if(alpha1_index != -1 && takeStep(alpha1_index, i))
                return 1;

            for (int j = 0; j < dataloader->Y.rows(); ++j)
            {
                if(j == i || E(j) == C || E(j) == 0)
                    continue;
                if(takeStep(j, i))
                    return 1;
            }

            for (int j = 0; j < dataloader->Y.rows(); ++j)
            {
                if(j == i)
                    continue;
                if(takeStep(j, i))
                    return 1;
            }
        }
    }
    return 0;
}

void SVC::SMO()
{
    bool examineAll = true;
    bool numChanged = false;
    bool eval_delta_loss = false;
    param.current_iteration = 0;
    while( true)
    {
        ++param.current_iteration;
        if(param.loss_eval_circle > 0 && param.current_iteration % param.loss_eval_circle == 0)
            eval_delta_loss = true;
        numChanged = examineExamples(examineAll, eval_delta_loss);
        if (examineAll)
            examineAll = 0;
        else if(numChanged == 0)
            examineAll = 1;
        if(eval_delta_loss)
        {
            eval_delta_loss = false;
            postProcess(g);
            param.training_acc = classify_accuracy(g, dataloader->Y);
            param.max_acc = param.training_acc > param.max_acc ? param.training_acc: param.max_acc;
            printVerbose();
        }
        if(param.max_iterations > 0 && param.current_iteration >= param.max_iterations)
            break;
        if(param.min_delta_loss > 0 && param.min_delta_loss > param.delta_loss)
            break;
        dataloader->loadData();
        vector_mask = dataloader->shuffer * vector_mask;
        alpha = dataloader->shuffer * alpha;
    }
}

void SVC::train()
{
    std::cout << "start training..." << std::endl;
    alpha = Eigen::MatrixXd(Eigen::MatrixXd::Constant(dataloader->X.rows(), 1, 0));
    vector_mask = Eigen::VectorXi(Eigen::VectorXi::Zero(dataloader->X.rows()));
    // support_vector_indices = std::set<int>();
    SMO();
}

Eigen::MatrixXd SVC::predict(const Eigen::MatrixXd &input, bool post_process)
{
    Eigen::MatrixXd result;
    support_vector_indices.clear();
    for(int i = 0; i < vector_mask.size(); ++i)
    {
        if (vector_mask(i) != 0)
            support_vector_indices.push_back(i);
    }

    if (support_vector_indices.empty())
        result = Eigen::MatrixXd::Constant(dataloader->X.rows(), 1, K(0, 0));
    else
    {
        Eigen::MatrixXd sv = (dataloader->X)(support_vector_indices, Eigen::all).matrix();
        Eigen::MatrixXd a = (alpha)(support_vector_indices, Eigen::all);
        Eigen::MatrixXd y = (dataloader->Y)(support_vector_indices, Eigen::all);
        result = ((this->*kernel_function)(dataloader->X, sv)) * (a.array() * y.array()).matrix();
    }

    // result = ((this->*kernel_function)(dataloader->X, dataloader->X)) * (alpha.array() * dataloader->Y.array()).matrix();
    // result.array() -= b;
    if (post_process)
        postProcess(result);
    return result;
}

void SVC::postProcess(Eigen::MatrixXd &predict)
{
    predict = (predict.array() > 0).select(Eigen::MatrixXd::Ones(predict.rows(), predict.cols()), Eigen::MatrixXd::Constant(predict.rows(), predict.cols(), -1)); 
}