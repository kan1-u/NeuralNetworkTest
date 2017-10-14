// NeuralNetworkTest.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"

#define FAST_CONTAONER_OPERATOR_OVERLOAD_AMP_MODE
//#define FAST_CONTAINER_NO_EXCEPTION

#include "FastContainerLibrary.hpp"

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
	using namespace FastContainer;

	int row = 5;
	int col = 4;
	int cycle_num = 1;

	auto a = FastMatrix<int>::random(row, col, 1, 10);

	auto func2 = [&]() {
		cout << a.to_string().c_str() << endl;
		auto result = a.argmax_by_columns();
		cout << result.to_string().c_str() << endl;
	};
	cout_calc_span(func2, cycle_num, "norm");

	auto func3 = [&]() {
		auto result = a.amp_argmax_by_columns();
		cout << result.to_string().c_str() << endl;
	};
	cout_calc_span(func3, cycle_num, "amp");

	auto func4 = [&]() {
		auto result = a.ppl_argmax_by_columns();
		cout << result.to_string().c_str() << endl;
	};
	cout_calc_span(func4, cycle_num, "ppl");

	getchar();

	return 0;
}
