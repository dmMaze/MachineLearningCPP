#include "perceptron.h"

void testPerceptron(shared_ptr<DataLoader> dataloader)
{
	unique_ptr<Perceptron> perceptron(new Perceptron(dataloader));
	perceptron->train();
}