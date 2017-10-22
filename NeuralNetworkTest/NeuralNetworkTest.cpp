// NeuralNetworkTest.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"

//#define FAST_CONTAONER_FUNCTIONS_COM_MODE
//#define FAST_CONTAONER_FUNCTIONS_AMP_MODE
#define FAST_CONTAONER_FUNCTIONS_PPL_MODE

//#define FAST_CONTAONER_OPERATOR_OVERLOAD_COM_MODE
//#define FAST_CONTAONER_OPERATOR_OVERLOAD_AMP_MODE
#define FAST_CONTAONER_OPERATOR_OVERLOAD_PPL_MODE

//#define FAST_CONTAINER_NO_EXCEPTION

#include "NeuralNetworkLibrary.hpp"
#include "MnistDataset.hpp"

#include <chrono>
#include <iostream>
#include <time.h>

using namespace std;

using namespace NeuralNetwork;
using namespace FastContainer;
using namespace MnistDataset;

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

void neuralnetwork_test() {
	using fvd = FastVector<double>;
	using fmd = FastMatrix<double>;

	Mnist mnist;
	auto train_img = fmd(mnist.readTrainingFile("mnist\\train-images.idx3-ubyte"));
	auto train_lbl = fvd(mnist.readLabelFile("mnist\\train-labels.idx1-ubyte"));

	int train_num = 1000;
	int batch_size = 300;
	int input_size = train_img.get_column_size();
	int hidden_size = 50;
	int output_size = 1;
	double weight_init = 0.01;

	Network<double> net;

	net.layers.push_back(new AffineLayer<double>(weight_init * fmd::real_random_ppl(input_size, hidden_size), fvd::real_random_ppl(hidden_size)));
	net.layers.push_back(new ReluLayer<double>());
	net.layers.push_back(new AffineLayer<double>(weight_init * fmd::real_random_ppl(hidden_size, output_size), fvd::real_random_ppl(output_size)));
	net.lastLayer = new SoftmaxWithLossLayer<double>();

	for (int i = 0; i < train_num; i++) {
		auto x_batch = train_img.random_batch_ppl(batch_size);
		auto t_batch = fmd(train_lbl.random_batch_ppl(batch_size), 1);
		net.training(x_batch, t_batch, weight_init);
	}
}

void fast_container_test() {
	auto x1 = FastMatrix<double>::real_random_ppl(5, 3);
	auto x2 = FastMatrix<double>::real_random_ppl(5, 3);
	auto y1 = x1.random_batch_com(3);
	auto y2 = x1.random_batch_amp(3);
	auto y3 = x1.random_batch_ppl(3);
	auto y4 = x1.random_batch(3);
	auto y5 = (2.0 + (2.0 - (2.0 * (2.0 / ((((((((x1 + x2) - x2) * x2) / x2) + 2.0) - 2.0) * 2.0) / 2.0)))));

	cout << x1.to_string().c_str() << endl;
	cout << x2.to_string().c_str() << endl;
	cout << y1.to_string().c_str() << endl;
	cout << y2.to_string().c_str() << endl;
	cout << y3.to_string().c_str() << endl;
	cout << y4.to_string().c_str() << endl;
	cout << y5.to_string().c_str() << endl;
}

int main()
{
	fast_container_test();
	//neuralnetwork_test();

	getchar();

	return 0;
}
