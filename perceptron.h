#ifndef PERCEPTRON_H
#define PERCEPTRON_H
#include "data.h"
#include "model.h"
#include "autograd.h"

struct perception_parameter: public model_parameter
{
    /* data */
};



class Perceptron: public Model<perception_parameter>
{
public:
    Perceptron(shared_ptr<DataLoader> dataloader): Model(dataloader) {}
    void train()
    {
        weights = EinsorD(dataloader->X.cols(), 1, true, 0);   
        param.current_iteration = 0;
        while(true)
        {
            EinsorD x, y;
            dataloader->loadData(x, y, param.batch_size);
            EinsorD result = hadamardprod(y, x * weights);
            EinsorD mask((result.data->array() <= 0).matrix().cast<double>());
            result = hadamardprod(result, mask);
            result.sum().backward();
            *weights.data += param.learning_rate * *weights.grad;
            weights.grad_zero();
            if (param.max_iterations > 0 && param.current_iteration > param.max_iterations)
                break;
            ++param.current_iteration;
        }
        cout << weights;
    }
    EinsorD predict(EinsorD &input, bool post_process = true)
    {
        EinsorD result = input * weights;
        activate(result);
        return result;
    }
    void activate(EinsorD& result)
    {

    }
private:
    EinsorD weights;
    EinsorD bias;
};

void testPerceptron(shared_ptr<DataLoader> dataloader);


#endif
