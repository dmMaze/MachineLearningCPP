
#include "SVM.h"
#include"perceptron.h"

int main()
{
	std::string data_path = "data/testdata";
	std::string heartdata_path = "data/heart_scale";
	shared_ptr<DataLoader> dl1(new DataLoader(data_path, &loadMatrix, true, "none", true));
	shared_ptr<DataLoader> dl2(new DataLoader(heartdata_path, &loadMatrix_2));
	testSVC(dl2);
	testPerceptron(dl1);
	return 1;
}