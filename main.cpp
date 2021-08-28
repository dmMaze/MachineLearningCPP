#include "SVM.h"
#include <typeinfo>
#include <iostream>
#include <string>
#include <cmath>
// #include <cblas.h>

int main()
{
	std::string data_path = "data/testdata";
	std::string heartdata_path = "data/heart_scale";

	DataLoader dataloader(heartdata_path, &loadMatrix_2);
	SVM *svc = new SVC(&dataloader);
	svc->setParam("loss_eval_circle", 10);
	svc->setParam("max_iterations", 300);
	svc->setParam("min_delta_loss", -1);
	svc->setParam("kernel_type", 2);
	svc->setParam("poly_n", 25);
	svc->train();
	return 1;
}