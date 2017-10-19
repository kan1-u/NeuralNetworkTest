#pragma once

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

namespace MnistDataset {

	class Mnist {
	public:
		vector<vector<double> > readTrainingFile(string filename);
		vector<double> readLabelFile(string filename);
	};

}
