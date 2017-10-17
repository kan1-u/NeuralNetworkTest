// NeuralNetworkTest.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"

#define FAST_CONTAONER_OPERATOR_OVERLOAD_AMP_MODE
//#define FAST_CONTAINER_NO_EXCEPTION

#include "NeuralNetworkLibrary.hpp"

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

	int input_size = 200;
	int hidden_size = 100;
	int output_size = 50;

	int row = 50;
	int col = 20;
	int cycle_num = 10;

	auto a = FastMatrix<double>::random(row, col, -10, 10);
	auto b = FastMatrix<double>::random(row, col, -10, 10);

	auto func1 = [&]() {
		//cout << a.to_string().c_str() << endl;
		auto result = a.dot(b.reverse());
		//cout << result.to_string().c_str() << endl;
	};
	cout_calc_span(func1, cycle_num, "norm");

	auto func2 = [&]() {
		auto result = a.amp_dot(b.amp_reverse());
		//cout << result.to_string().c_str() << endl;
	};
	cout_calc_span(func2, cycle_num, "amp");

	auto func3 = [&]() {
		auto result = a.ppl_dot(b.ppl_reverse());
		//cout << result.to_string().c_str() << endl;
	};
	cout_calc_span(func3, cycle_num, "ppl");

	getchar();

	return 0;
}
