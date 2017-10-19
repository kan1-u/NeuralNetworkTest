// NeuralNetworkTest.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"

#define FAST_CONTAONER_OPERATOR_OVERLOAD_AMP_MODE
//#define FAST_CONTAINER_NO_EXCEPTION

#include "NeuralNetworkLibrary.hpp"
#include "MnistDataset.hpp"

#include <chrono>
#include <iostream>
#include <time.h>

using namespace std;

template<typename T>
void cout_calc_span(T func, int cycle_num, std::string str) {
	clock_t start = clock();
	for (int i = 0; i < cycle_num; i++) {
		func();
	}
	clock_t end = clock();
	const auto timeSpan = (double)(end - start) / CLOCKS_PER_SEC / cycle_num;
	cout << str.c_str() << ": " << timeSpan << "s" << endl;
}

int main()
{
	using namespace NeuralNetwork;
	using namespace FastContainer;
	using namespace MnistDataset;

	Mnist mnist;
	auto train_img = FastMatrix<double>(mnist.readTrainingFile("mnist\\train-images.idx3-ubyte"));
	auto train_lbl = FastVector<double>(mnist.readLabelFile("mnist\\train-labels.idx1-ubyte"));

	int train_num = 1000;
	int batch_size = 1000;
	int input_size = train_img.get_columns();
	int hidden_size = 50;
	int output_size = 1;
	double weight_init = 0.01;

	Network<double> net;

	net.layers.push_back(new AffineLayer<double>(weight_init * FastMatrix<double>::random(input_size, hidden_size), FastVector<double>::random(hidden_size)));
	net.layers.push_back(new ReluLayer<double>());
	net.layers.push_back(new AffineLayer<double>(weight_init * FastMatrix<double>::random(hidden_size, output_size), FastVector<double>::random(output_size)));
	net.lastLayer = new SoftmaxWithLossLayer<double>();

	for (int i = 0; i < train_num; i++) {
		auto mask = FastVector<int>::int_hash_random(batch_size, 0, train_img.get_rows() - 1);
		auto x_batch = train_img.random_batch(mask);
		auto t_batch = FastMatrix<double>(train_lbl.random_batch(mask), 1);
		net.training(x_batch, t_batch, weight_init);
	}

	getchar();

	return 0;
}
