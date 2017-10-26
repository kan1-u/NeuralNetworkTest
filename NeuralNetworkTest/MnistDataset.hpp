#pragma once

#include "FastContainerLibrary.hpp"

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

namespace MnistDataset {

	class Mnist {
	public:
		FastContainer::FastMatrix<double> read_training_file(string filename);
		FastContainer::FastVector<double> read_label_file(string filename);
		FastContainer::FastMatrix<double> read_label_file_onehot(string filename);
	};

}
