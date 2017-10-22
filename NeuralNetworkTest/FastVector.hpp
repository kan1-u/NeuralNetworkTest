#pragma once

#include "FastContainerLibrary.hpp"

namespace FastContainer {

	/*
	ベクトルクラス
	*/
	template<typename T>
	class FastVector {
	public:
		FastVector() { }
		FastVector(int size) { resize(size); }
		FastVector(const std::vector<T> vec) { this->size = vec.size(); entity = vec; }
		~FastVector() { }

		void resize(int size) { this->size = size; entity.resize(size); }

		std::vector<T> get_entity() { return entity; }
		int get_size() { return size; }

		T& operator[](int index) { return entity[index]; }

		/*そのまま返す*/
		FastVector<T> identity() { return this; }

		/*絶対値 実装モード切替*/
		FastVector<T> abs() { return SWITCH_FAST_CONTAONER_FUNCTION(abs)(); }
		/*絶対値*/
		FastVector<T> abs_com() { return apply_com_func([](T x) { return std::abs(x); }); }
		/*絶対値 AMP実装*/
		FastVector<T> abs_amp() { return apply_amp_func([](T x) restrict(amp) { return concurrency::fast_math::fabs(x); }); }
		/*絶対値 PPL実装*/
		FastVector<T> abs_ppl() { return apply_ppl_func([](T x) { return std::abs(x); }); }

		/*Log e 実装モード切替*/
		FastVector<T> log() { return SWITCH_FAST_CONTAONER_FUNCTION(log)(); }
		/*Log e*/
		FastVector<T> log_com() { return apply_com_func([](T x) { return std::log(x); }); }
		/*Log e AMP実装*/
		FastVector<T> log_amp() { return apply_amp_func([](T x) restrict(amp) { return concurrency::fast_math::log(x); }); }
		/*Log e PPL実装*/
		FastVector<T> log_ppl() { return apply_ppl_func([](T x) { return std::log(x); }); }

		/*Log 10 実装モード切替*/
		FastVector<T> log10() { return SWITCH_FAST_CONTAONER_FUNCTION(log10)(); }
		/*Log 10*/
		FastVector<T> log10_com() { return apply_com_func([](T x) { return std::log10(x); }); }
		/*Log 10*/
		FastVector<T> log10_amp() { return apply_amp_func([](T x) restrict(amp) { return concurrency::fast_math::log10(x); }); }
		/*Log 10*/
		FastVector<T> log10_ppl() { return apply_ppl_func([](T x) { return std::log10(x); }); }

		/*2乗根 実装モード切替*/
		FastVector<T> sqrt() { return SWITCH_FAST_CONTAONER_FUNCTION(sqrt)(); }
		/*2乗根*/
		FastVector<T> sqrt_com() { return apply_com_func([](T x) { return std::sqrt(x); }); }
		/*2乗根 AMP実装*/
		FastVector<T> sqrt_amp() { return apply_amp_func([](T x) restrict(amp) { return concurrency::fast_math::sqrt(x); }); }
		/*2乗根 PPL実装*/
		FastVector<T> sqrt_ppl() { return apply_ppl_func([](T x) { return std::sqrt(x); }); }

		/*階乗 実装モード切替*/
		FastVector<T> pow(T exp) { return SWITCH_FAST_CONTAONER_FUNCTION(pow)(exp); }
		/*階乗*/
		FastVector<T> pow_com(T exp) { return apply_com_func([=](T x) { return std::pow(x, exp); }); }
		/*階乗 AMP実装*/
		FastVector<T> pow_amp(T exp) { return apply_amp_func([=](T x) restrict(amp) { return concurrency::fast_math::pow(x, exp); }); }
		/*階乗 PPL実装*/
		FastVector<T> pow_ppl(T exp) { return apply_ppl_func([=](T x) { return std::pow(x, exp); }); }

		/*e^x 実装モード切替*/
		FastVector<T> exp() { return SWITCH_FAST_CONTAONER_FUNCTION(exp)(); }
		/*e^x*/
		FastVector<T> exp_com() { return apply_com_func([](T x) { return std::exp(x); }); }
		/*e^x AMP実装*/
		FastVector<T> exp_amp() { return apply_amp_func([](T x) restrict(amp) { return concurrency::fast_math::exp(x); }); }
		/*e^x PPL実装*/
		FastVector<T> exp_ppl() { return apply_ppl_func([](T x) { return std::exp(x); }); }

		/*シグモイド関数 実装モード切替*/
		FastVector<T> sigmoid() { return SWITCH_FAST_CONTAONER_FUNCTION(sigmoid)(); }
		/*シグモイド関数*/
		FastVector<T> sigmoid_com() { return apply_com_func([](T x) { return (T)1 / (1 + std::exp(-x)); }); }
		/*シグモイド関数 AMP実装*/
		FastVector<T> sigmoid_amp() { return apply_amp_func([](T x) restrict(amp) { return (T)1 / (1 + concurrency::fast_math::exp(-x)); }); }
		/*シグモイド関数 PPL実装*/
		FastVector<T> sigmoid_ppl() { return apply_ppl_func([](T x) { return (T)1 / (1 + std::exp(-x)); }); }

		/*ReLU関数 実装モード切替*/
		FastVector<T> relu() { return SWITCH_FAST_CONTAONER_FUNCTION(relu)(); }
		/*ReLU関数*/
		FastVector<T> relu_com() { return apply_com_func([](T x) { return x > 0 ? x : 0; }); }
		/*ReLU関数 AMP実装*/
		FastVector<T> relu_amp() { return apply_amp_func([](T x) restrict(amp) { return x > 0 ? x : 0; }); }
		/*ReLU関数 PPL実装*/
		FastVector<T> relu_ppl() { return apply_ppl_func([](T x) { return x > 0 ? x : 0; }); }

		/*正規化 実装モード切替*/
		FastVector<T> normalization() { return SWITCH_FAST_CONTAONER_FUNCTION(normalization)(); }
		/*正規化*/
		FastVector<T> normalization_com() {
			T max = get_max();
			return apply_com_func([=](T x) { return x / max; });
		}
		/*正規化 AMP実装*/
		FastVector<T> normalization_amp() {
			T max = get_max();
			return apply_amp_func([=](T x) restrict(amp) { return x / max; });
		}
		/*正規化 PPL実装*/
		FastVector<T> normalization_ppl() {
			T max = get_max();
			return apply_ppl_func([=](T x) { return x / max; });
		}

		/*ソフトマックス関数 実装モード切替*/
		FastVector<T> softmax() { return SWITCH_FAST_CONTAONER_FUNCTION(softmax)(); }
		/*ソフトマックス関数*/
		FastVector<T> softmax_com() {
			T max = get_max();
			T total = apply_com_func([=](T x) { return std::exp(x - max); }).sum();
			return apply_com_func([=](T x) { return std::exp(x - max) / total; });
		}
		/*ソフトマックス関数 AMP実装*/
		FastVector<T> softmax_amp() {
			T max = get_max();
			T total = apply_amp_func([=](T x) restrict(amp) { return concurrency::fast_math::exp(x - max); }).sum();
			return apply_amp_func([=](T x) restrict(amp) { return concurrency::fast_math::exp(x - max) / total; });
		}
		/*ソフトマックス関数 PPL実装*/
		FastVector<T> softmax_ppl() {
			T max = get_max();
			T total = apply_ppl_func([=](T x) { return std::exp(x - max); }).sum();
			return apply_ppl_func([=](T x) { return std::exp(x - max) / total; });
		}

		/*数値微分 実装モード切替
		func: (T)(*func)(T x)*/
		template<class F>
		FastVector<T> num_diff(F func, T delta = 0.0001) { return SWITCH_FAST_CONTAONER_FUNCTION(num_diff)(func, delta); }
		/*数値微分
		func: (T)(*func)(T x)*/
		template<class F>
		FastVector<T> num_diff_com(F func, T delta = 0.0001) { return apply_com_func([=](T x) { return (func(x + delta) - func(x - delta)) / (2 * delta); }); }
		/*数値微分 AMP実装
		func: (T)(*func)(T x) restrict(amp)*/
		template<class F>
		FastVector<T> num_diff_amp(F func, T delta = 0.0001) { return apply_amp_func([=](T x) restrict(amp) { return (func(x + delta) - func(x - delta)) / (2 * delta); }); }
		/*数値微分 PPL実装
		func: (T)(*func)(T x)*/
		template<class F>
		FastVector<T> num_diff_ppl(F func, T delta = 0.0001) { return apply_ppl_func([=](T x) { return (func(x + delta) - func(x - delta)) / (2 * delta); }); }

		/*交差エントロピー誤差 実装モード切替*/
		T cross_entropy_error(FastVector<T>& teacher, T delta = 0.0000001) { return SWITCH_FAST_CONTAONER_FUNCTION(cross_entropy_error)(teacher, delta); }
		/*交差エントロピー誤差*/
		T cross_entropy_error_com(FastVector<T>& teacher, T delta = 0.0000001) {
			auto buf = apply_com_combo_func([=](T x1, T x2) { return -1 * x2 * std::log(x1 + delta) + (1 - x2) * std::log(1 - x1 + delta); }, teacher);
			return buf.sum();
		}
		/*交差エントロピー誤差 AMP実装*/
		T cross_entropy_error_amp(FastVector<T>& teacher, T delta = 0.0000001) {
			auto buf = apply_amp_combo_func([=](T x1, T x2) restrict(amp) { return -1 * x2 * concurrency::fast_math::log(x1 + delta) + (1 - x2) * concurrency::fast_math::log(1 - x1 + delta); }, teacher);
			return buf.sum();
		}
		/*交差エントロピー誤差 PPL実装*/
		T cross_entropy_error_ppl(FastVector<T>& teacher, T delta = 0.0000001) {
			auto buf = apply_ppl_combo_func([=](T x1, T x2) { return -1 * x2 * std::log(x1 + delta) + (1 - x2) * std::log(1 - x1 + delta); }, teacher);
			return buf.sum();
		}

		/*交差エントロピー誤差 分類問題 実装モード切替*/
		T cross_entropy_error_class(FastVector<T>& teacher, T delta = 0.0000001) { return SWITCH_FAST_CONTAONER_FUNCTION(cross_entropy_error_class)(teacher, delta); }
		/*交差エントロピー誤差 分類問題*/
		T cross_entropy_error_class_com(FastVector<T>& teacher, T delta = 0.0000001) {
			auto buf = apply_com_combo_func([=](T x1, T x2) { return -1 * x2 * std::log(x1 + delta) }, teacher);
			return buf.sum();
		}
		/*交差エントロピー誤差 分類問題 AMP実装*/
		T cross_entropy_error_class_amp(FastVector<T>& teacher, T delta = 0.0000001) {
			auto buf = apply_amp_combo_func([=](T x1, T x2) restrict(amp) { return -1 * x2 * concurrency::fast_math::log(x1 + delta) }, teacher);
			return buf.sum();
		}
		/*交差エントロピー誤差 分類問題 PPL実装*/
		T cross_entropy_error_class_ppl(FastVector<T>& teacher, T delta = 0.0000001) {
			auto buf = apply_ppl_combo_func([=](T x1, T x2) { return -1 * x2 * std::log(x1 + delta) }, teacher);
			return buf.sum();
		}

		/*最小値*/
		T get_min() {
			T result = entity[0];
			for (auto x : entity) result = min(result, x);
			return result;
		}
		/*最大値*/
		T get_max() {
			T result = entity[0];
			for (auto x : entity) result = max(result, x);
			return result;
		}
		/*最大値のインデックス*/
		T get_argmax() {
			int result = 0;
			T max = entity[0];
			for (int i = 0; i < size; i++) {
				if (max < entity[i]) {
					max = entity[i];
					result = i;
				}
			}
			return result;
		}

		/*合計*/
		T sum() {
			T result = 0;
			for (auto x : entity) result += x;
			return result;
		}

		/*平均*/
		T mean() {
			return sum() / size;
		}

		/*内積*/
		T dot(FastVector<T>& vec) {
			FAST_CONTAINER_EXCEPTION_CHECK(size == vec.get_size(), fast_container_exception());
			T result = 0;
			for (int i = 0; i < size; i++) {
				result += entity[i] * vec[i];
			}
			return result;
		}

		/*バッチの取得 実装モード切替*/
		FastVector<T> batch(FastVector<int>& mask) { return SWITCH_FAST_CONTAONER_FUNCTION(batch)(mask); }
		/*バッチの取得*/
		FastVector<T> batch_com(FastVector<int>& mask) {
			FAST_CONTAINER_EXCEPTION_CHECK(size >= mask.get_size(), fast_container_exception());
			int row = mask.get_size();
			FastVector<T> result(row);
			for (int i = 0; i < row; i++) {
				result[i] = entity[mask[i]];
			}
			return result;
		}
		/*バッチの取得 AMP実装*/
		FastVector<T> batch_amp(FastVector<int>& mask) {
			FAST_CONTAINER_EXCEPTION_CHECK(size >= mask.get_size(), fast_container_exception());
			int m_size = mask.get_size();
			FastVector<T> result(m_size);
			concurrency::array_view<const T, 1> av_entity(size, &entity[0]);
			concurrency::array_view<const int, 1> av_mask(m_size, &mask[0]);
			concurrency::array_view<T, 1> av_result(m_size, &result[0]);
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<1> idx) restrict(amp) {
				av_result[idx] = av_entity[av_mask[idx]];
			});
			av_result.synchronize();
			return result;
		}
		/*バッチの取得 PPL実装*/
		FastVector<T> batch_ppl(FastVector<int>& mask) {
			FAST_CONTAINER_EXCEPTION_CHECK(size >= mask.get_size(), fast_container_exception());
			int row = mask.get_size();
			FastVector<T> result(row);
			concurrency::parallel_for<int>(0, row, [&](int i) {
				result[i] = entity[mask[i]];
			});
			return result;
		}

		/*ランダムなバッチの取得 実装モード切替*/
		FastVector<T> random_batch(int size) { return SWITCH_FAST_CONTAONER_FUNCTION(random_batch)(size); }
		/*ランダムなバッチの取得*/
		FastVector<T> random_batch_com(int size) { return batch_com(FastVector<int>::int_hash_random(size, 0, this->size - 1)); }
		/*ランダムなバッチの取得 AMP実装*/
		FastVector<T> random_batch_amp(int size) { return batch_amp(FastVector<int>::int_hash_random(size, 0, this->size - 1)); }
		/*ランダムなバッチの取得 PPL実装*/
		FastVector<T> random_batch_ppl(int size) { return batch_ppl(FastVector<int>::int_hash_random(size, 0, this->size - 1)); }

		/*関数を適用
		func: T(*func)(T x)*/
		template<class F>
		FastVector<T> apply_com_func(F func) {
			FastVector<T> result(size);
			for (int i = 0; i < size; i++) {
				result[i] = func(entity[i]);
			}
			return result;
		}
		/*関数を適用 AMP実装
		func: T(*func)(T x) restrict(amp)*/
		template<class F>
		FastVector<T> apply_amp_func(F func) {
			FastVector<T> result(size);
			concurrency::array_view<const T, 1> av_entity(size, &entity[0]);
			concurrency::array_view<T, 1> av_result(size, &result[0]);
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<1> idx) restrict(amp) {
				av_result[idx] = func(av_entity[idx]);
			});
			av_result.synchronize();
			return result;
		}
		/*関数を適用 PPL実装
		func: T(*func)(T x)*/
		template<class F>
		FastVector<T> apply_ppl_func(F func) {
			FastVector<T> result(size);
			concurrency::parallel_for<int>(0, size, [&](int i) {
				result[i] = func(entity[i]);
			});
			return result;
		}

		/*関数を適用
		func: T(*func)(T x1, T x2)*/
		template<class F>
		FastVector<T> apply_com_combo_func(F func, FastVector<T>& vec) {
			FAST_CONTAINER_EXCEPTION_CHECK(size == vec.get_size(), fast_container_exception());
			FastVector<T> result(size);
			for (int i = 0; i < size; i++) {
				result[i] = func(entity[i], vec[i]);
			}
			return result;
		}
		/*関数を適用 AMP実装
		func: T(*func)(T x1, T x2) restrict(amp)*/
		template<class F>
		FastVector<T> apply_amp_combo_func(F func, FastVector<T>& vec) {
			FAST_CONTAINER_EXCEPTION_CHECK(size == vec.get_size(), fast_container_exception());
			FastVector<T> result(size);
			concurrency::array_view<const T, 1> av_entity(size, &entity[0]);
			concurrency::array_view<const T, 1> av_vec(size, &vec[0]);
			concurrency::array_view<T, 1> av_result(size, &result[0]);
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<1> idx) restrict(amp) {
				av_result[idx] = func(av_entity[idx], av_vec[idx]);
			});
			av_result.synchronize();
			return result;
		}
		/*関数を適用 PPL実装
		func: T(*func)(T x1, T x2)*/
		template<class F>
		FastVector<T> apply_ppl_combo_func(F func, FastVector<T>& vec) {
			FAST_CONTAINER_EXCEPTION_CHECK(size == vec.get_size(), fast_container_exception());
			FastVector<T> result(size);
			concurrency::parallel_for<int>(0, size, [&](int i) {
				result[i] = func(entity[i], vec[i]);
			});
			return result;
		}

		/*[size] (value1,value2,...)*/
		std::string to_string() {
			std::string result = "[" + std::to_string(size) + "](";
			for (int i = 0; i < size; i++) {
				if (i) result += "," + std::to_string(entity[i]);
				else result += std::to_string(entity[i]);
			}
			result += ")";
			return result;
		}

		/*ランダムなFastVectorを生成*/
		static FastVector<T> real_random_com(int size, T min = -1, T max = 1) {
			FastVector<T> result(size);
			RealRandom<T> rnd(min, max);
			return result.apply_com_func([&](T x) { return rnd.generate(); });
		}
		/*ランダムなFastVectorを生成 PPL実装*/
		static FastVector<T> real_random_ppl(int size, T min = -1, T max = 1) {
			FastVector<T> result(size);
			RealRandom<T> rnd(min, max);
			return result.apply_ppl_func([&](T x) { return rnd.generate(); });
		}
		/*ランダムなFastVector<int>を生成*/
		static FastVector<int> int_random_com(int size, int min = -1, int max = 1) {
			FastVector<int> result(size);
			IntRandom rnd(min, max);
			return result.apply_com_func([&](int x) { return rnd.generate(); });
		}
		/*ランダムなFastVector<int>を生成 PPL実装*/
		static FastVector<int> int_random_ppl(int size, int min = -1, int max = 1) {
			FastVector<int> result(size);
			IntRandom rnd(min, max);
			return result.apply_ppl_func([&](int x) { return rnd.generate(); });
		}
		/*重複のないランダムなFastVector<int>を生成*/
		static FastVector<int> int_hash_random(int size, int min, int max) {
			FAST_CONTAINER_EXCEPTION_CHECK(min <= max, fast_container_exception());
			FAST_CONTAINER_EXCEPTION_CHECK(max - min + 1 >= size, fast_container_exception());
			FastVector<int> result(size);
			std::random_device rnd;
			auto mt = std::mt19937(rnd());
			std::unordered_map<int, int> map;
			int index = 0;
			for (int i = 0; i < size; i++) {
				int val = std::uniform_int_distribution<>(min, max)(mt);
				std::unordered_map<int, int>::iterator itr = map.find(val);
				int replaced_val;
				std::unordered_map<int, int>::iterator replaced_itr = map.find(max);
				if (replaced_itr != map.end()) {
					replaced_val = replaced_itr->second;
				}
				else replaced_val = max;
				if (itr == map.end()) {
					result[index] = val;
					if (val != max) map.insert(std::make_pair(val, replaced_val));
				}
				else {
					result[index] = itr->second;
					itr->second = replaced_val;
				}
				index++;
				max--;
			}
			return result;
		}

	private:
		std::vector<T> entity;
		int size;
	};


#ifdef FAST_CONTAONER_OPERATOR_OVERLOAD_COM_MODE

	template<typename T>
	FastVector<T> operator+(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_com_combo_func([](T x1, T x2) {return x1 + x2; }, vec2); }
	template<typename T>
	FastVector<T> operator-(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_com_combo_func([](T x1, T x2) {return x1 - x2; }, vec2); }
	template<typename T>
	FastVector<T> operator*(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_com_combo_func([](T x1, T x2) {return x1 * x2; }, vec2); }
	template<typename T>
	FastVector<T> operator/(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_com_combo_func([](T x1, T x2) {return x1 / x2; }, vec2); }
	template<typename T>
	FastVector<T> operator==(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_com_combo_func([](T x1, T x2) {return x1 == x2; }, vec2); }
	template<typename T>
	FastVector<T> operator!=(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_com_combo_func([](T x1, T x2) {return x1 != x2; }, vec2); }
	template<typename T>
	FastVector<T> operator>(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_com_combo_func([](T x1, T x2) {return x1 > x2; }, vec2); }
	template<typename T>
	FastVector<T> operator<(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_com_combo_func([](T x1, T x2) {return x1 < x2; }, vec2); }
	template<typename T>
	FastVector<T> operator>=(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_com_combo_func([](T x1, T x2) {return x1 >= x2; }, vec2); }
	template<typename T>
	FastVector<T> operator<=(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_com_combo_func([](T x1, T x2) {return x1 <= x2; }, vec2); }

	template<typename T>
	FastVector<T> operator+(T val, FastVector<T>& vec) { return vec.apply_com_func([=](T x) {return val + x; }); }
	template<typename T>
	FastVector<T> operator-(T val, FastVector<T>& vec) { return vec.apply_com_func([=](T x) {return val - x; }); }
	template<typename T>
	FastVector<T> operator*(T val, FastVector<T>& vec) { return vec.apply_com_func([=](T x) {return val * x; }); }
	template<typename T>
	FastVector<T> operator/(T val, FastVector<T>& vec) { return vec.apply_com_func([=](T x) {return val / x; }); }
	template<typename T>
	FastVector<T> operator==(T val, FastVector<T>& vec) { return vec.apply_com_func([=](T x) {return val == x; }); }
	template<typename T>
	FastVector<T> operator!=(T val, FastVector<T>& vec) { return vec.apply_com_func([=](T x) {return val != x; }); }
	template<typename T>
	FastVector<T> operator>(T val, FastVector<T>& vec) { return vec.apply_com_func([=](T x) {return val > x; }); }
	template<typename T>
	FastVector<T> operator<(T val, FastVector<T>& vec) { return vec.apply_com_func([=](T x) {return val < x; }); }
	template<typename T>
	FastVector<T> operator>=(T val, FastVector<T>& vec) { return vec.apply_com_func([=](T x) {return val >= x; }); }
	template<typename T>
	FastVector<T> operator<=(T val, FastVector<T>& vec) { return vec.apply_com_func([=](T x) {return val <= x; }); }

	template<typename T>
	FastVector<T> operator+(FastVector<T>& vec, T val) { return vec.apply_com_func([=](T x) {return x + val; }); }
	template<typename T>
	FastVector<T> operator-(FastVector<T>& vec, T val) { return vec.apply_com_func([=](T x) {return x - val; }); }
	template<typename T>
	FastVector<T> operator*(FastVector<T>& vec, T val) { return vec.apply_com_func([=](T x) {return x * val; }); }
	template<typename T>
	FastVector<T> operator/(FastVector<T>& vec, T val) { return vec.apply_com_func([=](T x) {return x / val; }); }
	template<typename T>
	FastVector<T> operator==(FastVector<T>& vec, T val) { return vec.apply_com_func([=](T x) {return x == val; }); }
	template<typename T>
	FastVector<T> operator!=(FastVector<T>& vec, T val) { return vec.apply_com_func([=](T x) {return x != val; }); }
	template<typename T>
	FastVector<T> operator>(FastVector<T>& vec, T val) { return vec.apply_com_func([=](T x) {return x > val; }); }
	template<typename T>
	FastVector<T> operator<(FastVector<T>& vec, T val) { return vec.apply_com_func([=](T x) {return x < val; }); }
	template<typename T>
	FastVector<T> operator>=(FastVector<T>& vec, T val) { return vec.apply_com_func([=](T x) {return x >= val; }); }
	template<typename T>
	FastVector<T> operator<=(FastVector<T>& vec, T val) { return vec.apply_com_func([=](T x) {return x <= val; }); }

#elif defined FAST_CONTAONER_OPERATOR_OVERLOAD_AMP_MODE

	template<typename T>
	FastVector<T> operator+(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 + x2; }, vec2); }
	template<typename T>
	FastVector<T> operator-(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 - x2; }, vec2); }
	template<typename T>
	FastVector<T> operator*(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 * x2; }, vec2); }
	template<typename T>
	FastVector<T> operator/(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 / x2; }, vec2); }
	template<typename T>
	FastVector<T> operator==(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 == x2; }, vec2); }
	template<typename T>
	FastVector<T> operator!=(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 != x2; }, vec2); }
	template<typename T>
	FastVector<T> operator>(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 > x2; }, vec2); }
	template<typename T>
	FastVector<T> operator<(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 < x2; }, vec2); }
	template<typename T>
	FastVector<T> operator>=(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 >= x2; }, vec2); }
	template<typename T>
	FastVector<T> operator<=(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 <= x2; }, vec2); }

	template<typename T>
	FastVector<T> operator+(T val, FastVector<T>& vec) { return vec.apply_amp_func([=](T x) restrict(amp) {return val + x; }); }
	template<typename T>
	FastVector<T> operator-(T val, FastVector<T>& vec) { return vec.apply_amp_func([=](T x) restrict(amp) {return val - x; }); }
	template<typename T>
	FastVector<T> operator*(T val, FastVector<T>& vec) { return vec.apply_amp_func([=](T x) restrict(amp) {return val * x; }); }
	template<typename T>
	FastVector<T> operator/(T val, FastVector<T>& vec) { return vec.apply_amp_func([=](T x) restrict(amp) {return val / x; }); }
	template<typename T>
	FastVector<T> operator==(T val, FastVector<T>& vec) { return vec.apply_amp_func([=](T x) restrict(amp) {return val == x; }); }
	template<typename T>
	FastVector<T> operator!=(T val, FastVector<T>& vec) { return vec.apply_amp_func([=](T x) restrict(amp) {return val != x; }); }
	template<typename T>
	FastVector<T> operator>(T val, FastVector<T>& vec) { return vec.apply_amp_func([=](T x) restrict(amp) {return val > x; }); }
	template<typename T>
	FastVector<T> operator<(T val, FastVector<T>& vec) { return vec.apply_amp_func([=](T x) restrict(amp) {return val < x; }); }
	template<typename T>
	FastVector<T> operator>=(T val, FastVector<T>& vec) { return vec.apply_amp_func([=](T x) restrict(amp) {return val >= x; }); }
	template<typename T>
	FastVector<T> operator<=(T val, FastVector<T>& vec) { return vec.apply_amp_func([=](T x) restrict(amp) {return val <= x; }); }

	template<typename T>
	FastVector<T> operator+(FastVector<T>& vec, T val) { return vec.apply_amp_func([=](T x) restrict(amp) {return x + val; }); }
	template<typename T>
	FastVector<T> operator-(FastVector<T>& vec, T val) { return vec.apply_amp_func([=](T x) restrict(amp) {return x - val; }); }
	template<typename T>
	FastVector<T> operator*(FastVector<T>& vec, T val) { return vec.apply_amp_func([=](T x) restrict(amp) {return x * val; }); }
	template<typename T>
	FastVector<T> operator/(FastVector<T>& vec, T val) { return vec.apply_amp_func([=](T x) restrict(amp) {return x / val; }); }
	template<typename T>
	FastVector<T> operator==(FastVector<T>& vec, T val) { return vec.apply_amp_func([=](T x) restrict(amp) {return x == val; }); }
	template<typename T>
	FastVector<T> operator!=(FastVector<T>& vec, T val) { return vec.apply_amp_func([=](T x) restrict(amp) {return x != val; }); }
	template<typename T>
	FastVector<T> operator>(FastVector<T>& vec, T val) { return vec.apply_amp_func([=](T x) restrict(amp) {return x > val; }); }
	template<typename T>
	FastVector<T> operator<(FastVector<T>& vec, T val) { return vec.apply_amp_func([=](T x) restrict(amp) {return x < val; }); }
	template<typename T>
	FastVector<T> operator>=(FastVector<T>& vec, T val) { return vec.apply_amp_func([=](T x) restrict(amp) {return x >= val; }); }
	template<typename T>
	FastVector<T> operator<=(FastVector<T>& vec, T val) { return vec.apply_amp_func([=](T x) restrict(amp) {return x <= val; }); }

#elif defined FAST_CONTAONER_OPERATOR_OVERLOAD_PPL_MODE

	template<typename T>
	FastVector<T> operator+(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_ppl_combo_func([](T x1, T x2) {return x1 + x2; }, vec2); }
	template<typename T>
	FastVector<T> operator-(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_ppl_combo_func([](T x1, T x2) {return x1 - x2; }, vec2); }
	template<typename T>
	FastVector<T> operator*(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_ppl_combo_func([](T x1, T x2) {return x1 * x2; }, vec2); }
	template<typename T>
	FastVector<T> operator/(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_ppl_combo_func([](T x1, T x2) {return x1 / x2; }, vec2); }
	template<typename T>
	FastVector<T> operator==(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_ppl_combo_func([](T x1, T x2) {return x1 == x2; }, vec2); }
	template<typename T>
	FastVector<T> operator!=(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_ppl_combo_func([](T x1, T x2) {return x1 != x2; }, vec2); }
	template<typename T>
	FastVector<T> operator>(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_ppl_combo_func([](T x1, T x2) {return x1 > x2; }, vec2); }
	template<typename T>
	FastVector<T> operator<(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_ppl_combo_func([](T x1, T x2) {return x1 < x2; }, vec2); }
	template<typename T>
	FastVector<T> operator>=(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_ppl_combo_func([](T x1, T x2) {return x1 >= x2; }, vec2); }
	template<typename T>
	FastVector<T> operator<=(FastVector<T>& vec1, FastVector<T>& vec2) { return vec1.apply_ppl_combo_func([](T x1, T x2) {return x1 <= x2; }, vec2); }

	template<typename T>
	FastVector<T> operator+(T val, FastVector<T>& vec) { return vec.apply_ppl_func([=](T x) {return val + x; }); }
	template<typename T>
	FastVector<T> operator-(T val, FastVector<T>& vec) { return vec.apply_ppl_func([=](T x) {return val - x; }); }
	template<typename T>
	FastVector<T> operator*(T val, FastVector<T>& vec) { return vec.apply_ppl_func([=](T x) {return val * x; }); }
	template<typename T>
	FastVector<T> operator/(T val, FastVector<T>& vec) { return vec.apply_ppl_func([=](T x) {return val / x; }); }
	template<typename T>
	FastVector<T> operator==(T val, FastVector<T>& vec) { return vec.apply_ppl_func([=](T x) {return val == x; }); }
	template<typename T>
	FastVector<T> operator!=(T val, FastVector<T>& vec) { return vec.apply_ppl_func([=](T x) {return val != x; }); }
	template<typename T>
	FastVector<T> operator>(T val, FastVector<T>& vec) { return vec.apply_ppl_func([=](T x) {return val > x; }); }
	template<typename T>
	FastVector<T> operator<(T val, FastVector<T>& vec) { return vec.apply_ppl_func([=](T x) {return val < x; }); }
	template<typename T>
	FastVector<T> operator>=(T val, FastVector<T>& vec) { return vec.apply_ppl_func([=](T x) {return val >= x; }); }
	template<typename T>
	FastVector<T> operator<=(T val, FastVector<T>& vec) { return vec.apply_ppl_func([=](T x) {return val <= x; }); }

	template<typename T>
	FastVector<T> operator+(FastVector<T>& vec, T val) { return vec.apply_ppl_func([=](T x) {return x + val; }); }
	template<typename T>
	FastVector<T> operator-(FastVector<T>& vec, T val) { return vec.apply_ppl_func([=](T x) {return x - val; }); }
	template<typename T>
	FastVector<T> operator*(FastVector<T>& vec, T val) { return vec.apply_ppl_func([=](T x) {return x * val; }); }
	template<typename T>
	FastVector<T> operator/(FastVector<T>& vec, T val) { return vec.apply_ppl_func([=](T x) {return x / val; }); }
	template<typename T>
	FastVector<T> operator==(FastVector<T>& vec, T val) { return vec.apply_ppl_func([=](T x) {return x == val; }); }
	template<typename T>
	FastVector<T> operator!=(FastVector<T>& vec, T val) { return vec.apply_ppl_func([=](T x) {return x != val; }); }
	template<typename T>
	FastVector<T> operator>(FastVector<T>& vec, T val) { return vec.apply_ppl_func([=](T x) {return x > val; }); }
	template<typename T>
	FastVector<T> operator<(FastVector<T>& vec, T val) { return vec.apply_ppl_func([=](T x) {return x < val; }); }
	template<typename T>
	FastVector<T> operator>=(FastVector<T>& vec, T val) { return vec.apply_ppl_func([=](T x) {return x >= val; }); }
	template<typename T>
	FastVector<T> operator<=(FastVector<T>& vec, T val) { return vec.apply_ppl_func([=](T x) {return x <= val; }); }

#endif

}
