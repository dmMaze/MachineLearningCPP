
#include "SVM.h"
#include <typeinfo>
#include <iostream>
#include <string>
#include <cmath>
#include <unordered_set>

int main()
{
	std::string data_path = "data/testdata";
	std::string heartdata_path = "data/heart_scale";
	
	DataLoader dataloader(heartdata_path, &loadMatrix_2);
	SVM *svc = new SVC(&dataloader);
	svc->setParam("loss_eval_circle", 5);
	svc->setParam("max_iterations", 200);
	svc->setParam("min_delta_loss", -1);
	svc->setParam("kernel_type", SVM::Gaussian);
	svc->setParam("tol", 0.01);
	svc->setParam("poly_n", 5);
	svc->train();
	return 1;
}