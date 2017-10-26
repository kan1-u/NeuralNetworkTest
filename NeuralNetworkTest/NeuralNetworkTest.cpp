// NeuralNetworkTest.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"

//#define FAST_CONTAONER_FUNCTIONS_COM_MODE
#define FAST_CONTAONER_FUNCTIONS_AMP_MODE
//#define FAST_CONTAONER_FUNCTIONS_PPL_MODE

//#define FAST_CONTAONER_OPERATOR_OVERLOAD_COM_MODE
#define FAST_CONTAONER_OPERATOR_OVERLOAD_AMP_MODE
//#define FAST_CONTAONER_OPERATOR_OVERLOAD_PPL_MODE

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
void cout_calc_span(T func, int cycle_num = 1, std::string str = "calc") {
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
	auto train_img = fmd(mnist.read_training_file("mnist\\train-images.idx3-ubyte")).normalization();
	auto train_lbl = fmd(mnist.read_label_file_onehot("mnist\\train-labels.idx1-ubyte"));
	auto test_img = fmd(mnist.read_training_file("mnist\\t10k-images.idx3-ubyte")).normalization();
	auto test_lbl = fmd(mnist.read_label_file_onehot("mnist\\t10k-labels.idx1-ubyte"));

	int train_num = 100;
	int batch_size = 1000;
	int tbatch_size = 100;
	int input_size = train_img.get_column_size();
	int hidden_size = 100;
	int output_size = train_lbl.get_column_size();
	double weight_init = 0.05;

	Network<double> net;

	net.layers.push_back(new AffineLayer<double>(weight_init * fmd::normal_random_ppl(input_size, hidden_size), weight_init * fvd::real_random_ppl(hidden_size)));
	net.layers.push_back(new ReluLayer<double>());
	net.layers.push_back(new AffineLayer<double>(weight_init * fmd::normal_random_ppl(hidden_size, output_size), weight_init * fvd::real_random_ppl(output_size)));
	net.lastLayer = new SoftmaxWithLossLayer<double>();

	for (int i = 0; i < train_num; i++) {
		auto mask = fvd::int_hash_random(batch_size, 0, train_img.get_row_size() - 1);
		auto x_batch = train_img.batch(mask);
		auto t_batch = train_lbl.batch(mask);
		auto tmask = fvd::int_hash_random(tbatch_size, 0, test_img.get_row_size() - 1);
		auto tx_batch = test_img.batch(tmask);
		auto tt_batch = test_lbl.batch(tmask);
		net.training(x_batch, t_batch, weight_init);
		//cout << "loss:" << net.loss(x_batch, t_batch) << endl;
		cout << to_string(i).c_str() << ".train acc: " << net.accuracy(x_batch, t_batch) << endl;
		cout << to_string(i).c_str() << ".test  acc: " << net.accuracy(tx_batch, tt_batch) << endl;
	}
}

int main()
{
	neuralnetwork_test();

	getchar();

	return 0;
}
