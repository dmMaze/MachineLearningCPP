#ifndef DATA_H
#define DATA_H
#include "utils.h"
#include "autograd.h"
class DataLoader
{
public:
    DataLoader(Eigen::MatrixXd &_X, Eigen::MatrixXd &_Y, bool shuffle=true)
        :X(_X), Y(_Y)
    {
        setShuffle(shuffle);
    }

    DataLoader(Eigen::MatrixXd& data, bool shuffle=true, const std::string& scale_method="std")
    {
        Y = data.col(data.cols()-1);
        X = data(Eigen::all, Eigen::seq(0, data.cols() - 2));
        preprocess(shuffle, scale_method);
    }

    DataLoader(const std::string& path, bool shuffle=true, const std::string& scale_method="std")
    {
        Eigen::MatrixXd data;
        loadMatrix(path, data);
        DataLoader(data, shuffle, scale_method);
        Y = data.col(data.cols()-1);
        X = data(Eigen::all, Eigen::seq(0, data.cols() - 2));
        preprocess(shuffle, scale_method);
    }

    template<typename ... Args>
    DataLoader(const std::string& path, 
        int (*dataParaser)(const std::string& path, Eigen::MatrixXd &M, const std::string& pattern), 
        bool shuffle=true, const std::string& scale_method="std", Args ... args)
    {
        Eigen::MatrixXd data;
        (*dataParaser)(path, data, " ");
        DataLoader(data, shuffle, scale_method);
        Y = data.col(data.cols()-1);
        X = data(Eigen::all, Eigen::seq(0, data.cols() - 2));
        preprocess(shuffle, scale_method, args...);
    }

    void preprocess(bool shuffle=true, const std::string& scale_method="std", bool append=false)
    {
        if(scale_method == "std")
            Scaler::standardize(X);
        else if(scale_method == "norm" || scale_method == "norml2")
            Scaler::normalize(X);
        else if(scale_method == "minmax")
            Scaler::minmax(X);
        if (append)
        {
            Eigen::MatrixXd X_append = Eigen::MatrixXd::Ones(X.rows(), X.cols()+1);
            X_append(Eigen::all, Eigen::seq(0, X.cols()-1)) = X;
            X = X_append;
        }
        
        setShuffle(shuffle);
    }

    void setShuffle(bool shuffle)
    {
        shuffle_flag = shuffle;
        if (shuffle_flag)
        {
            shuffer = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>(X.rows());
            shuffer.setIdentity();
        }
    }

    void shuffle()
    {
        std::random_shuffle(shuffer.indices().data(), shuffer.indices().data()+shuffer.indices().size());
        X = shuffer * X;
        Y = shuffer * Y;
    }

    void loadData()
    {
        if (shuffle_flag)
            shuffle();
    }

    void loadData(Eigen::MatrixXd& alpha)
    {
        std::random_shuffle(shuffer.indices().data(), shuffer.indices().data()+shuffer.indices().size());
        X = shuffer * X;
        Y = shuffer * Y;
        alpha = shuffer * alpha;
    }

    void loadData(EinsorD &x, EinsorD &y, int batch_size=1, bool append=true)
    {
        if(load_iter >= getDataSize())
        {
            load_iter = 0;
            loadData();
        }
        auto selected_rows =  Eigen::seq(load_iter, std::min(getDataSize()-1, load_iter+batch_size-1));

        x = EinsorD(X(selected_rows, Eigen::all).matrix());
        y = EinsorD(Y(selected_rows, Eigen::all).matrix());
        load_iter += batch_size;
    }

    int getDataSize()
    {
        return X.rows();
    }
    int load_iter = 0;
    Eigen::MatrixXd X;
    Eigen::MatrixXd Y;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> shuffer;
private:
    
    bool shuffle_flag;
};

#endif