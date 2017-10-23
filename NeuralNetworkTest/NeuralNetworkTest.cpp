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
	auto train_img = fmd(mnist.readTrainingFile("mnist\\train-images.idx3-ubyte")).normalization();
	auto train_lbl = fmd(mnist.readLabelFileBinaries("mnist\\train-labels.idx1-ubyte"));
	auto test_img = fmd(mnist.readTrainingFile("mnist\\t10k-images.idx3-ubyte")).normalization();
	auto test_lbl = fmd(mnist.readLabelFileBinaries("mnist\\t10k-labels.idx1-ubyte"));

	int train_num = 100;
	int batch_size = 1000;
	int tbatch_size = 100;
	int input_size = train_img.get_column_size();
	int hidden_size = 50;
	int output_size = train_lbl.get_column_size();
	double weight_init = 0.05;

	Network<double> net;

	net.layers.push_back(new AffineLayer<double>(weight_init * fmd::normal_random_ppl(input_size, hidden_size), fvd(hidden_size)));
	net.layers.push_back(new ReluLayer<double>());
	net.layers.push_back(new AffineLayer<double>(weight_init * fmd::normal_random_ppl(hidden_size, output_size), fvd(output_size)));
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

void fast_container_test() {
	auto x1 = FastMatrix<double>::real_random_ppl(1000, 500);
	auto x2 = FastMatrix<double>::real_random_ppl(500, 50);

	//cout << x1.to_string().c_str() << endl;
	//cout << x2.to_string().c_str() << endl;
	//auto func1 = [&]() {
	//	auto y = x1.dot_com(x2);
	//	cout << y.to_string().c_str() << endl;
	//};
	//auto func2 = [&]() {
	//	auto y = x1.dot_amp(x2);
	//	cout << y.to_string().c_str() << endl;
	//};
	//auto func3 = [&]() {
	//	auto y = x1.dot_ppl(x2);
	//	cout << y.to_string().c_str() << endl;
	//};

	auto y1 = x1.dot_com(x2);
	auto y2 = x1.dot_amp(x2);
	auto y3 = x1.dot_ppl(x2);

	//cout << "com: " << y1.to_string().c_str() << endl << endl;
	//cout << "amp: " << y2.to_string().c_str() << endl << endl;
	//cout << "ppl: " << y3.to_string().c_str() << endl << endl;
	cout << "com != amp: " << (y1 - y2).abs_com().sum() << endl;
	cout << "amp != ppl: " << (y2 - y3).abs_com().sum() << endl;
	cout << "com != ppl: " << (y1 - y3).abs_com().sum() << endl;

	//cout_calc_span(func1, 1, "com");
	//cout_calc_span(func2, 1, "amp");
	//cout_calc_span(func3, 1, "ppl");
}

int main()
{
	//fast_container_test();
	neuralnetwork_test();

	getchar();

	return 0;
}
