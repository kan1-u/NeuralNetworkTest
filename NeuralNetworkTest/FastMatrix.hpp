#pragma once

#include "FastContainerLibrary.hpp"

namespace FastContainer {

	/*
	行列クラス
	*/
	template<typename T>
	class FastMatrix {
	public:
		FastMatrix() { }
		FastMatrix(int row, int col) { resize(row, col); }
		FastMatrix(std::vector<T> vec, int row) {
			row_size = row;
			column_size = vec.size() / row_size;
			size = row_size * column_size;
			FAST_CONTAINER_EXCEPTION_CHECK(size == vec.size(), fast_container_exception());
			entity = vec;
		}
		FastMatrix(FastVector<T>& vec, int row) {
			row_size = row;
			column_size = vec.get_size() / row_size;
			size = row_size * column_size;
			FAST_CONTAINER_EXCEPTION_CHECK(size == vec.get_size(), fast_container_exception());
			entity = vec.get_entity();
		}
		FastMatrix(std::vector<std::vector<T>> mat) {
			resize(mat.size(), mat[0].size());
			for (int i = 0; i < row_size; i++) {
				FAST_CONTAINER_EXCEPTION_CHECK(column_size == mat[i].size(), fast_container_exception());
				T offset = i * column_size;
				for (int j = 0; j < column_size; j++) {
					entity[offset + j] = mat[i][j];
				}
			}
		}
		~FastMatrix() { }

		void resize(int row, int col) {
			row_size = row;
			column_size = col;
			size = row * col;
			entity.resize(size);
		}

		std::vector<T> get_entity() { return entity; }
		int get_row_size() { return row_size; }
		int get_column_size() { return column_size; }
		int get_size() { return size; }

		T& operator[](int idx) { return entity[idx]; }
		T& operator()(int row, int col) { return entity[row * column_size + col]; }

		/*FastVectorへ変換*/
		FastVector<T> to_FastVector() {
			FastVector<T> result(entity);
			return result;
		}

		/*そのまま返す*/
		FastMatrix<T> identity() { return this; }

		/*絶対値 実装モード切替*/
		FastMatrix<T> abs() { return SWITCH_FAST_CONTAONER_FUNCTION(abs)(); }
		/*絶対値*/
		FastMatrix<T> abs_com() { return apply_com_func([](T x) { return std::abs(x); }); }
		/*絶対値 AMP実装*/
		FastMatrix<T> abs_amp() { return apply_amp_func([](T x) restrict(amp) { return concurrency::fast_math::fabs(x); }); }
		/*絶対値 PPL実装*/
		FastMatrix<T> abs_ppl() { return apply_ppl_func([](T x) { return std::abs(x); }); }

		/*Log e 実装モード切替*/
		FastMatrix<T> log() { return SWITCH_FAST_CONTAONER_FUNCTION(log)(); }
		/*Log e*/
		FastMatrix<T> log_com() { return apply_com_func([](T x) { return std::log(x); }); }
		/*Log e AMP実装*/
		FastMatrix<T> log_amp() { return apply_amp_func([](T x) restrict(amp) { return concurrency::fast_math::log(x); }); }
		/*Log e PPL実装*/
		FastMatrix<T> log_ppl() { return apply_ppl_func([](T x) { return std::log(x); }); }

		/*Log 10 実装モード切替*/
		FastMatrix<T> log10() { return SWITCH_FAST_CONTAONER_FUNCTION(log10)(); }
		/*Log 10*/
		FastMatrix<T> log10_com() { return apply_com_func([](T x) { return std::log10(x); }); }
		/*Log 10*/
		FastMatrix<T> log10_amp() { return apply_amp_func([](T x) restrict(amp) { return concurrency::fast_math::log10(x); }); }
		/*Log 10*/
		FastMatrix<T> log10_ppl() { return apply_ppl_func([](T x) { return std::log10(x); }); }

		/*2乗根 実装モード切替*/
		FastMatrix<T> sqrt() { return SWITCH_FAST_CONTAONER_FUNCTION(sqrt)(); }
		/*2乗根*/
		FastMatrix<T> sqrt_com() { return apply_com_func([](T x) { return std::sqrt(x); }); }
		/*2乗根 AMP実装*/
		FastMatrix<T> sqrt_amp() { return apply_amp_func([](T x) restrict(amp) { return concurrency::fast_math::sqrt(x); }); }
		/*2乗根 PPL実装*/
		FastMatrix<T> sqrt_ppl() { return apply_ppl_func([](T x) { return std::sqrt(x); }); }

		/*階乗 実装モード切替*/
		FastMatrix<T> pow(T exp) { return SWITCH_FAST_CONTAONER_FUNCTION(pow)(exp); }
		/*階乗*/
		FastMatrix<T> pow_com(T exp) { return apply_com_func([=](T x) { return std::pow(x, exp); }); }
		/*階乗 AMP実装*/
		FastMatrix<T> pow_amp(T exp) { return apply_amp_func([=](T x) restrict(amp) { return concurrency::fast_math::pow(x, exp); }); }
		/*階乗 PPL実装*/
		FastMatrix<T> pow_ppl(T exp) { return apply_ppl_func([=](T x) { return std::pow(x, exp); }); }

		/*e^x 実装モード切替*/
		FastMatrix<T> exp() { return SWITCH_FAST_CONTAONER_FUNCTION(exp)(); }
		/*e^x*/
		FastMatrix<T> exp_com() { return apply_com_func([](T x) { return std::exp(x); }); }
		/*e^x AMP実装*/
		FastMatrix<T> exp_amp() { return apply_amp_func([](T x) restrict(amp) { return concurrency::fast_math::exp(x); }); }
		/*e^x PPL実装*/
		FastMatrix<T> exp_ppl() { return apply_ppl_func([](T x) { return std::exp(x); }); }

		/*シグモイド関数 実装モード切替*/
		FastMatrix<T> sigmoid() { return SWITCH_FAST_CONTAONER_FUNCTION(sigmoid)(); }
		/*シグモイド関数*/
		FastMatrix<T> sigmoid_com() { return apply_com_func([](T x) { return (T)1 / (1 + std::exp(-x)); }); }
		/*シグモイド関数 AMP実装*/
		FastMatrix<T> sigmoid_amp() { return apply_amp_func([](T x) restrict(amp) { return (T)1 / (1 + concurrency::fast_math::exp(-x)); }); }
		/*シグモイド関数 PPL実装*/
		FastMatrix<T> sigmoid_ppl() { return apply_ppl_func([](T x) { return (T)1 / (1 + std::exp(-x)); }); }

		/*ReLU関数 実装モード切替*/
		FastMatrix<T> relu() { return SWITCH_FAST_CONTAONER_FUNCTION(relu)(); }
		/*ReLU関数*/
		FastMatrix<T> relu_com() { return apply_com_func([](T x) { return x > 0 ? x : 0; }); }
		/*ReLU関数 AMP実装*/
		FastMatrix<T> relu_amp() { return apply_amp_func([](T x) restrict(amp) { return x > 0 ? x : 0; }); }
		/*ReLU関数 PPL実装*/
		FastMatrix<T> relu_ppl() { return apply_ppl_func([](T x) { return x > 0 ? x : 0; }); }

		/*正規化 実装モード切替*/
		FastMatrix<T> normalization() { return SWITCH_FAST_CONTAONER_FUNCTION(normalization)(); }
		/*正規化*/
		FastMatrix<T> normalization_com() {
			T max = get_max();
			return apply_com_func([=](T x) { return x / max; });
		}
		/*正規化 AMP実装*/
		FastMatrix<T> normalization_amp() {
			T max = get_max();
			return apply_amp_func([=](T x) restrict(amp) { return x / max; });
		}
		/*正規化 PPL実装*/
		FastMatrix<T> normalization_ppl() {
			T max = get_max();
			return apply_ppl_func([=](T x) { return x / max; });
		}

		/*ソフトマックス関数 実装モード切替*/
		FastMatrix<T> softmax() { return SWITCH_FAST_CONTAONER_FUNCTION(softmax)(); }
		/*ソフトマックス関数*/
		FastMatrix<T> softmax_com() {
			T max = get_max();
			auto buf = apply_com_func([=](T x) { return std::exp(x - max); });
			return buf.div_by_columns_com(buf.sum_by_rows_com());
		}
		/*ソフトマックス関数 AMP実装*/
		FastMatrix<T> softmax_amp() {
			T max = get_max();
			auto buf = apply_amp_func([=](T x) restrict(amp) { return concurrency::fast_math::exp(x - max); });
			return buf.div_by_columns_amp(buf.sum_by_rows_amp());
		}
		/*ソフトマックス関数 PPL実装*/
		FastMatrix<T> softmax_ppl() {
			T max = get_max();
			auto buf = apply_ppl_func([=](T x) { return std::exp(x - max); });
			return buf.div_by_columns_ppl(buf.sum_by_rows_ppl());
		}

		/*数値微分 実装モード切替
		func: (T)(*func)(T x)*/
		template<class F>
		FastMatrix<T> num_diff(F func, T delta = 0.0001) { return SWITCH_FAST_CONTAONER_FUNCTION(num_diff)(func, delta); }
		/*数値微分
		func: (T)(*func)(T x)*/
		template<class F>
		FastMatrix<T> num_diff_com(F func, T delta = 0.0001) { return apply_com_func([=](T x) { return (func(x + delta) - func(x - delta)) / (2 * delta); }); }
		/*数値微分 AMP実装
		func: (T)(*func)(T x) restrict(amp)*/
		template<class F>
		FastMatrix<T> num_diff_amp(F func, T delta = 0.0001) { return apply_amp_func([=](T x) restrict(amp) { return (func(x + delta) - func(x - delta)) / (2 * delta); }); }
		/*数値微分 PPL実装
		func: (T)(*func)(T x)*/
		template<class F>
		FastMatrix<T> num_diff_ppl(F func, T delta = 0.0001) { return apply_ppl_func([=](T x) { return (func(x + delta) - func(x - delta)) / (2 * delta); }); }

		/*各行へ対象を加算 実装モード切替*/
		FastMatrix<T> add_by_rows(FastVector<T>& vec) { return SWITCH_FAST_CONTAONER_FUNCTION(add_by_rows)(vec); }
		/*各行へ対象を加算*/
		FastMatrix<T> add_by_rows_com(FastVector<T>& vec) { return apply_com_combo_func_by_rows([](T x1, T x2) {return x1 + x2; }, vec); }
		/*各行へ対象を加算 AMP実装*/
		FastMatrix<T> add_by_rows_amp(FastVector<T>& vec) { return apply_amp_combo_func_by_rows([](T x1, T x2) restrict(amp) {return x1 + x2; }, vec); }
		/*各行へ対象を加算 PPL実装*/
		FastMatrix<T> add_by_rows_ppl(FastVector<T>& vec) { return apply_ppl_combo_func_by_rows([](T x1, T x2) {return x1 + x2; }, vec); }

		/*各列へ対象を加算 実装モード切替*/
		FastMatrix<T> add_by_columns(FastVector<T>& vec) { return SWITCH_FAST_CONTAONER_FUNCTION(add_by_columns)(vec); }
		/*各列へ対象を加算*/
		FastMatrix<T> add_by_columns_com(FastVector<T>& vec) { return apply_com_combo_func_by_columns([](T x1, T x2) {return x1 + x2; }, vec); }
		/*各列へ対象を加算 AMP実装*/
		FastMatrix<T> add_by_columns_amp(FastVector<T>& vec) { return apply_amp_combo_func_by_columns([](T x1, T x2) restrict(amp) {return x1 + x2; }, vec); }
		/*各列へ対象を加算 PPL実装*/
		FastMatrix<T> add_by_columns_ppl(FastVector<T>& vec) { return apply_ppl_combo_func_by_columns([](T x1, T x2) {return x1 + x2; }, vec); }

		/*各行から対象を減算 実装モード切替*/
		FastMatrix<T> sub_by_rows(FastVector<T>& vec) { return SWITCH_FAST_CONTAONER_FUNCTION(sub_by_rows)(vec); }
		/*各行から対象を減算*/
		FastMatrix<T> sub_by_rows_com(FastVector<T>& vec) { return apply_com_combo_func_by_rows([](T x1, T x2) {return x1 - x2; }, vec); }
		/*各行から対象を減算 AMP実装*/
		FastMatrix<T> sub_by_rows_amp(FastVector<T>& vec) { return apply_amp_combo_func_by_rows([](T x1, T x2) restrict(amp) {return x1 - x2; }, vec); }
		/*各行から対象を減算 PPL実装*/
		FastMatrix<T> sub_by_rows_ppl(FastVector<T>& vec) { return apply_ppl_combo_func_by_rows([](T x1, T x2) {return x1 - x2; }, vec); }

		/*各列から対象を減算 実装モード切替*/
		FastMatrix<T> sub_by_columns(FastVector<T>& vec) { return SWITCH_FAST_CONTAONER_FUNCTION(sub_by_columns)(vec); }
		/*各列から対象を減算*/
		FastMatrix<T> sub_by_columns_com(FastVector<T>& vec) { return apply_com_combo_func_by_columns([](T x1, T x2) {return x1 - x2; }, vec); }
		/*各列から対象を減算 AMP実装*/
		FastMatrix<T> sub_by_columns_amp(FastVector<T>& vec) { return apply_amp_combo_func_by_columns([](T x1, T x2) restrict(amp) {return x1 - x2; }, vec); }
		/*各列から対象を減算 PPL実装*/
		FastMatrix<T> sub_by_columns_ppl(FastVector<T>& vec) { return apply_ppl_combo_func_by_columns([](T x1, T x2) {return x1 - x2; }, vec); }

		/*各行へ対象を乗算 実装モード切替*/
		FastMatrix<T> mul_by_rows(FastVector<T>& vec) { return SWITCH_FAST_CONTAONER_FUNCTION(mul_by_rows)(vec); }
		/*各行へ対象を乗算*/
		FastMatrix<T> mul_by_rows_com(FastVector<T>& vec) { return apply_com_combo_func_by_rows([](T x1, T x2) {return x1 * x2; }, vec); }
		/*各行へ対象を乗算 AMP実装*/
		FastMatrix<T> mul_by_rows_amp(FastVector<T>& vec) { return apply_amp_combo_func_by_rows([](T x1, T x2) restrict(amp) {return x1 * x2; }, vec); }
		/*各行へ対象を乗算 PPL実装*/
		FastMatrix<T> mul_by_rows_ppl(FastVector<T>& vec) { return apply_ppl_combo_func_by_rows([](T x1, T x2) {return x1 * x2; }, vec); }

		/*各列へ対象を乗算 実装モード切替*/
		FastMatrix<T> mul_by_columns(FastVector<T>& vec) { return SWITCH_FAST_CONTAONER_FUNCTION(mul_by_columns)(vec); }
		/*各列へ対象を乗算*/
		FastMatrix<T> mul_by_columns_com(FastVector<T>& vec) { return apply_com_combo_func_by_columns([](T x1, T x2) {return x1 * x2; }, vec); }
		/*各列へ対象を乗算 AMP実装*/
		FastMatrix<T> mul_by_columns_amp(FastVector<T>& vec) { return apply_amp_combo_func_by_columns([](T x1, T x2) restrict(amp) {return x1 * x2; }, vec); }
		/*各列へ対象を乗算 PPL実装*/
		FastMatrix<T> mul_by_columns_ppl(FastVector<T>& vec) { return apply_ppl_combo_func_by_columns([](T x1, T x2) {return x1 * x2; }, vec); }

		/*各行から対象を除算 実装モード切替*/
		FastMatrix<T> div_by_rows(FastVector<T>& vec) { return SWITCH_FAST_CONTAONER_FUNCTION(div_by_rows)(vec); }
		/*各行から対象を除算*/
		FastMatrix<T> div_by_rows_com(FastVector<T>& vec) { return apply_com_combo_func_by_rows([](T x1, T x2) {return x1 / x2; }, vec); }
		/*各行から対象を除算 AMP実装*/
		FastMatrix<T> div_by_rows_amp(FastVector<T>& vec) { return apply_amp_combo_func_by_rows([](T x1, T x2) restrict(amp) {return x1 / x2; }, vec); }
		/*各行から対象を除算 PPL実装*/
		FastMatrix<T> div_by_rows_ppl(FastVector<T>& vec) { return apply_ppl_combo_func_by_rows([](T x1, T x2) {return x1 / x2; }, vec); }

		/*各列から対象を除算 実装モード切替*/
		FastMatrix<T> div_by_columns(FastVector<T>& vec) { return SWITCH_FAST_CONTAONER_FUNCTION(div_by_columns)(vec); }
		/*各列から対象を除算*/
		FastMatrix<T> div_by_columns_com(FastVector<T>& vec) { return apply_com_combo_func_by_columns([](T x1, T x2) {return x1 / x2; }, vec); }
		/*各列から対象を除算 AMP実装*/
		FastMatrix<T> div_by_columns_amp(FastVector<T>& vec) { return apply_amp_combo_func_by_columns([](T x1, T x2) restrict(amp) {return x1 / x2; }, vec); }
		/*各列から対象を除算 PPL実装*/
		FastMatrix<T> div_by_columns_ppl(FastVector<T>& vec) { return apply_ppl_combo_func_by_columns([](T x1, T x2) {return x1 / x2; }, vec); }

		/*交差エントロピー誤差 実装モード切替*/
		T cross_entropy_error(FastMatrix<T>& teacher, T delta = 0.0000001) { return SWITCH_FAST_CONTAONER_FUNCTION(cross_entropy_error)(teacher, delta); }
		/*交差エントロピー誤差*/
		T cross_entropy_error_com(FastMatrix<T>& teacher, T delta = 0.0000001) {
			auto buf = apply_com_combo_func([=](T x1, T x2) { return (T)-1 * x2 * std::log(x1 + delta) + (1 - x2) * std::log(1 - x1 + delta); }, teacher);
			return buf.sum();
		}
		/*交差エントロピー誤差 AMP実装*/
		T cross_entropy_error_amp(FastMatrix<T>& teacher, T delta = 0.0000001) {
			auto buf = apply_amp_combo_func([=](T x1, T x2) restrict(amp) { return (T)-1 * x2 * concurrency::fast_math::log(x1 + delta) + (1 - x2) * concurrency::fast_math::log(1 - x1 + delta); }, teacher);
			return buf.sum();
		}
		/*交差エントロピー誤差 PPL実装*/
		T cross_entropy_error_ppl(FastMatrix<T>& teacher, T delta = 0.0000001) {
			auto buf = apply_ppl_combo_func([=](T x1, T x2) { return (T)-1 * x2 * std::log(x1 + delta) + (1 - x2) * std::log(1 - x1 + delta); }, teacher);
			return buf.sum();
		}

		/*交差エントロピー誤差 分類問題 実装モード切替*/
		T cross_entropy_error_class(FastMatrix<T>& teacher, T delta = 0.0000001) { return SWITCH_FAST_CONTAONER_FUNCTION(cross_entropy_error_class)(teacher, delta); }
		/*交差エントロピー誤差 分類問題*/
		T cross_entropy_error_class_com(FastMatrix<T>& teacher, T delta = 0.0000001) {
			auto buf = apply_com_combo_func([=](T x1, T x2) { return (T)-1 * x2 * std::log(x1 + delta); }, teacher);
			return buf.sum() / row_size;
		}
		/*交差エントロピー誤差 分類問題 AMP実装*/
		T cross_entropy_error_class_amp(FastMatrix<T>& teacher, T delta = 0.0000001) {
			auto buf = apply_amp_combo_func([=](T x1, T x2) restrict(amp) { return (T)-1 * x2 * concurrency::fast_math::log(x1 + delta); }, teacher);
			return buf.sum() / row_size;
		}
		/*交差エントロピー誤差 分類問題 PPL実装*/
		T cross_entropy_error_class_ppl(FastMatrix<T>& teacher, T delta = 0.0000001) {
			auto buf = apply_ppl_combo_func([=](T x1, T x2) { return (T)-1 * x2 * std::log(x1 + delta); }, teacher);
			return buf.sum() / row_size;
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

		/*行毎の最小値 実装モード切替*/
		FastVector<T> min_by_rows() { return SWITCH_FAST_CONTAONER_FUNCTION(min_by_rows)(); }
		/*行毎の最小値*/
		FastVector<T> min_by_rows_com() {
			FastVector<T> result(row_size);
			for (int i = 0; i < row_size; i++) {
				int offset = i * column_size;
				result[i] = entity[offset];
				for (int j = 0; j < column_size; j++) {
					T b_result = result[i];
					T b_entity = entity[offset + j];
					result[i] = min(b_result, b_entity);
				}
			}
			return result;
		}
		/*行毎の最小値 AMP実装*/
		FastVector<T> min_by_rows_amp() {
			FastVector<T> result(row_size);
			concurrency::array_view<const T, 2> av_entity(row_size, column_size, &entity[0]);
			concurrency::array_view<T, 1> av_result(row_size, &result[0]);
			int b_col = column_size;
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<1> idx) restrict(amp) {
				av_result[idx] = av_entity[idx[0]][0];
				for (int i = 0; i < b_col; i++) {
					T b_result = av_result[idx];
					T b_entity = av_entity[idx[0]][i];
					av_result[idx] = min(b_result, b_entity);
				}
			});
			av_result.synchronize();
			return result;
		}
		/*行毎の最小値 PPL実装*/
		FastVector<T> min_by_rows_ppl() {
			FastVector<T> result(row_size);
			concurrency::parallel_for<int>(0, row_size, [&](int i) {
				int offset = i * column_size;
				result[i] = entity[offset];
				for (int j = 0; j < column_size; j++) {
					T b_result = result[i];
					T b_entity = entity[offset + j];
					result[i] = min(b_result, b_entity);
				}
			});
			return result;
		}

		/*列毎の最小値 実装モード切替*/
		FastVector<T> min_by_columns() { return SWITCH_FAST_CONTAONER_FUNCTION(min_by_columns)(); }
		/*列毎の最小値*/
		FastVector<T> min_by_columns_com() {
			FastVector<T> result(column_size);
			for (int i = 0; i < column_size; i++) {
				result[i] = entity[i];
				for (int j = 0; j < row_size; j++) {
					T b_result = result[i];
					T b_entity = entity[j * column_size + i];
					result[i] = min(b_result, b_entity);
				}
			}
			return result;
		}
		/*列毎の最小値 AMP実装*/
		FastVector<T> min_by_columns_amp() {
			FastVector<T> result(column_size);
			concurrency::array_view<const T, 2> av_entity(row_size, column_size, &entity[0]);
			concurrency::array_view<T, 1> av_result(column_size, &result[0]);
			int b_row = row_size;
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<1> idx) restrict(amp) {
				av_result[idx] = av_entity[0][idx[0]];
				for (int i = 0; i < b_row; i++) {
					T b_result = av_result[idx];
					T b_entity = av_entity[i][idx[0]];
					av_result[idx] = min(b_result, b_entity);
				}
			});
			av_result.synchronize();
			return result;
		}
		/*列毎の最小値 PPL実装*/
		FastVector<T> min_by_columns_ppl() {
			FastVector<T> result(column_size);
			concurrency::parallel_for<int>(0, column_size, [&](int i) {
				result[i] = entity[i];
				for (int j = 0; j < row_size; j++) {
					T b_result = result[i];
					T b_entity = entity[j * column_size + i];
					result[i] = min(b_result, b_entity);
				}
			});
			return result;
		}

		/*行毎の最大値 実装モード切替*/
		FastVector<T> max_by_rows() { return SWITCH_FAST_CONTAONER_FUNCTION(max_by_rows)(); }
		/*行毎の最大値*/
		FastVector<T> max_by_rows_com() {
			FastVector<T> result(row_size);
			for (int i = 0; i < row_size; i++) {
				int offset = i * column_size;
				result[i] = entity[offset];
				for (int j = 0; j < column_size; j++) {
					T b_result = result[i];
					T b_entity = entity[offset + j];
					result[i] = max(b_result, b_entity);
				}
			}
			return result;
		}
		/*行毎の最大値 AMP実装*/
		FastVector<T> max_by_rows_amp() {
			FastVector<T> result(row_size);
			concurrency::array_view<const T, 2> av_entity(row_size, column_size, &entity[0]);
			concurrency::array_view<T, 1> av_result(row_size, &result[0]);
			int b_col = column_size;
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<1> idx) restrict(amp) {
				av_result[idx] = av_entity[idx[0]][0];
				for (int i = 0; i < b_col; i++) {
					T b_result = av_result[idx];
					T b_entity = av_entity[idx[0]][i];
					av_result[idx] = max(b_result, b_entity);
				}
			});
			av_result.synchronize();
			return result;
		}
		/*行毎の最大値 PPL実装*/
		FastVector<T> max_by_rows_ppl() {
			FastVector<T> result(row_size);
			concurrency::parallel_for<int>(0, row_size, [&](int i) {
				int offset = i * column_size;
				result[i] = entity[offset];
				for (int j = 0; j < column_size; j++) {
					T b_result = result[i];
					T b_entity = entity[offset + j];
					result[i] = max(b_result, b_entity);
				}
			});
			return result;
		}

		/*列毎の最大値 実装モード切替*/
		FastVector<T> max_by_columns() { return SWITCH_FAST_CONTAONER_FUNCTION(max_by_columns)(); }
		/*列毎の最大値*/
		FastVector<T> max_by_columns_com() {
			FastVector<T> result(column_size);
			for (int i = 0; i < column_size; i++) {
				result[i] = entity[i];
				for (int j = 0; j < row_size; j++) {
					T b_result = result[i];
					T b_entity = entity[j * column_size + i];
					result[i] = max(b_result, b_entity);
				}
			}
			return result;
		}
		/*列毎の最大値 AMP実装*/
		FastVector<T> max_by_columns_amp() {
			FastVector<T> result(column_size);
			concurrency::array_view<const T, 2> av_entity(row_size, column_size, &entity[0]);
			concurrency::array_view<T, 1> av_result(column_size, &result[0]);
			int b_row = row_size;
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<1> idx) restrict(amp) {
				av_result[idx] = av_entity[0][idx[0]];
				for (int i = 0; i < b_row; i++) {
					T b_result = av_result[idx];
					T b_entity = av_entity[i][idx[0]];
					av_result[idx] = max(b_result, b_entity);
				}
			});
			av_result.synchronize();
			return result;
		}
		/*列毎の最大値 PPL実装*/
		FastVector<T> max_by_columns_ppl() {
			FastVector<T> result(column_size);
			concurrency::parallel_for<int>(0, column_size, [&](int i) {
				result[i] = entity[i];
				for (int j = 0; j < row_size; j++) {
					T b_result = result[i];
					T b_entity = entity[j * column_size + i];
					result[i] = max(b_result, b_entity);
				}
			});
			return result;
		}

		/*行毎の最大値のインデックス 実装モード切替*/
		FastVector<T> argmax_by_rows() { return SWITCH_FAST_CONTAONER_FUNCTION(argmax_by_rows)(); }
		/*行毎の最大値のインデックス*/
		FastVector<T> argmax_by_rows_com() {
			FastVector<T> result(row_size);
			for (int i = 0; i < row_size; i++) {
				int offset = i * column_size;
				T max = entity[offset];
				for (int j = 0; j < column_size; j++) {
					T b_entity = entity[offset + j];
					if (max < b_entity) {
						max = b_entity;
						result[i] = j;
					}
				}
			}
			return result;
		}
		/*行毎の最大値のインデックス AMP実装*/
		FastVector<T> argmax_by_rows_amp() {
			FastVector<T> result(row_size);
			concurrency::array_view<const T, 2> av_entity(row_size, column_size, &entity[0]);
			concurrency::array_view<T, 1> av_result(row_size, &result[0]);
			int b_col = column_size;
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<1> idx) restrict(amp) {
				T max = av_entity[idx[0]][0];
				for (int i = 0; i < b_col; i++) {
					T b_entity = av_entity[idx[0]][i];
					if (max < b_entity) {
						max = b_entity;
						av_result[idx] = i;
					}
				}
			});
			av_result.synchronize();
			return result;
		}
		/*行毎の最大値のインデックス PPL実装*/
		FastVector<T> argmax_by_rows_ppl() {
			FastVector<T> result(row_size);
			concurrency::parallel_for<int>(0, row_size, [&](int i) {
				int offset = i * column_size;
				T max = entity[offset];
				for (int j = 0; j < column_size; j++) {
					T b_entity = entity[offset + j];
					if (max < b_entity) {
						max = b_entity;
						result[i] = j;
					}
				}
			});
			return result;
		}

		/*列毎の最大値のインデックス 実装モード切替*/
		FastVector<T> argmax_by_columns() { return SWITCH_FAST_CONTAONER_FUNCTION(argmax_by_columns)(); }
		/*列毎の最大値のインデックス*/
		FastVector<T> argmax_by_columns_com() {
			FastVector<T> result(column_size);
			for (int i = 0; i < column_size; i++) {
				T max = entity[i];
				for (int j = 0; j < row_size; j++) {
					T b_entity = entity[j * column_size + i];
					if (max < b_entity) {
						max = b_entity;
						result[i] = j;
					}
				}
			}
			return result;
		}
		/*列毎の最大値のインデックス AMP実装*/
		FastVector<T> argmax_by_columns_amp() {
			FastVector<T> result(column_size);
			concurrency::array_view<const T, 2> av_entity(row_size, column_size, &entity[0]);
			concurrency::array_view<T, 1> av_result(column_size, &result[0]);
			int b_row = row_size;
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<1> idx) restrict(amp) {
				T max = av_entity[0][idx[0]];
				for (int i = 0; i < b_row; i++) {
					T b_entity = av_entity[i][idx[0]];
					if (max < b_entity) {
						max = b_entity;
						av_result[idx] = i;
					}
				}
			});
			av_result.synchronize();
			return result;
		}
		/*列毎の最大値のインデックス PPL実装*/
		FastVector<T> argmax_by_columns_ppl() {
			FastVector<T> result(column_size);
			concurrency::parallel_for<int>(0, column_size, [&](int i) {
				T max = entity[i];
				for (int j = 0; j < row_size; j++) {
					T b_entity = entity[j * column_size + i];
					if (max < b_entity) {
						max = b_entity;
						result[i] = j;
					}
				}
			});
			return result;
		}

		/*行毎の合計 実装モード切替*/
		FastVector<T> sum_by_rows() { return SWITCH_FAST_CONTAONER_FUNCTION(sum_by_rows)(); }
		/*行毎の合計*/
		FastVector<T> sum_by_rows_com() {
			FastVector<T> result(row_size);
			for (int i = 0; i < row_size; i++) {
				int offset = i * column_size;
				for (int j = 0; j < column_size; j++) {
					result[i] += entity[offset + j];
				}
			}
			return result;
		}
		/*行毎の合計 AMP実装*/
		FastVector<T> sum_by_rows_amp() {
			FastVector<T> result(row_size);
			concurrency::array_view<const T, 2> av_entity(row_size, column_size, &entity[0]);
			concurrency::array_view<T, 1> av_result(row_size, &result[0]);
			int b_col = column_size;
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<1> idx) restrict(amp) {
				for (int i = 0; i < b_col; i++) {
					av_result[idx] += av_entity[idx[0]][i];
				}
			});
			av_result.synchronize();
			return result;
		}
		/*行毎の合計 PPL実装*/
		FastVector<T> sum_by_rows_ppl() {
			FastVector<T> result(row_size);
			concurrency::parallel_for<int>(0, row_size, [&](int i) {
				int offset = i * column_size;
				for (int j = 0; j < column_size; j++) {
					result[i] += entity[offset + j];
				}
			});
			return result;
		}

		/*列毎の合計 実装モード切替*/
		FastVector<T> sum_by_columns() { return SWITCH_FAST_CONTAONER_FUNCTION(sum_by_columns)(); }
		/*列毎の合計*/
		FastVector<T> sum_by_columns_com() {
			FastVector<T> result(column_size);
			for (int i = 0; i < column_size; i++) {
				for (int j = 0; j < row_size; j++) {
					result[i] += entity[j * column_size + i];
				}
			}
			return result;
		}
		/*列毎の合計 AMP実装*/
		FastVector<T> sum_by_columns_amp() {
			FastVector<T> result(column_size);
			concurrency::array_view<const T, 2> av_entity(row_size, column_size, &entity[0]);
			concurrency::array_view<T, 1> av_result(column_size, &result[0]);
			int b_row = row_size;
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<1> idx) restrict(amp) {
				for (int i = 0; i < b_row; i++) {
					av_result[idx] += av_entity[i][idx[0]];
				}
			});
			av_result.synchronize();
			return result;
		}
		/*列毎の合計 PPL実装*/
		FastVector<T> sum_by_columns_ppl() {
			FastVector<T> result(column_size);
			concurrency::parallel_for<int>(0, column_size, [&](int i) {
				for (int j = 0; j < row_size; j++) {
					result[i] += entity[j * column_size + i];
				}
			});
			return result;
		}

		/*行毎の平均 実装モード切替*/
		FastVector<T> mean_by_rows() { return SWITCH_FAST_CONTAONER_FUNCTION(mean_by_rows)(); }
		/*行毎の平均*/
		FastVector<T> mean_by_rows_com() {
			FastVector<T> result(row_size);
			for (int i = 0; i < row_size; i++) {
				int offset = i * column_size;
				for (int j = 0; j < column_size; j++) {
					result[i] += entity[offset + j];
				}
				result[i] /= column_size;
			}
			return result;
		}
		/*行毎の平均 AMP実装*/
		FastVector<T> mean_by_rows_amp() {
			FastVector<T> result(row_size);
			concurrency::array_view<const T, 2> av_entity(row_size, column_size, &entity[0]);
			concurrency::array_view<T, 1> av_result(row_size, &result[0]);
			int b_col = column_size;
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<1> idx) restrict(amp) {
				for (int i = 0; i < b_col; i++) {
					av_result[idx] += av_entity[idx[0]][i];
				}
				av_result[idx] /= b_col;
			});
			av_result.synchronize();
			return result;
		}
		/*行毎の平均 PPL実装*/
		FastVector<T> mean_by_rows_ppl() {
			FastVector<T> result(row_size);
			concurrency::parallel_for<int>(0, row_size, [&](int i) {
				int offset = i * column_size;
				for (int j = 0; j < column_size; j++) {
					result[i] += entity[offset + j];
				}
				result[i] /= column_size;
			});
			return result;
		}

		/*列毎の平均 実装モード切替*/
		FastVector<T> mean_by_columns() { return SWITCH_FAST_CONTAONER_FUNCTION(mean_by_columns)(); }
		/*列毎の平均*/
		FastVector<T> mean_by_columns_com() {
			FastVector<T> result(column_size);
			for (int i = 0; i < column_size; i++) {
				for (int j = 0; j < row_size; j++) {
					result[i] += entity[j * column_size + i];
				}
				result[i] /= row_size;
			}
			return result;
		}
		/*列毎の平均 AMP実装*/
		FastVector<T> mean_by_columns_amp() {
			FastVector<T> result(column_size);
			concurrency::array_view<const T, 2> av_entity(row_size, column_size, &entity[0]);
			concurrency::array_view<T, 1> av_result(column_size, &result[0]);
			int b_row = row_size;
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<1> idx) restrict(amp) {
				for (int i = 0; i < b_row; i++) {
					av_result[idx] += av_entity[i][idx[0]];
				}
				av_result[idx] /= b_row;
			});
			av_result.synchronize();
			return result;
		}
		/*列毎の平均 PPL実装*/
		FastVector<T> mean_by_columns_ppl() {
			FastVector<T> result(column_size);
			concurrency::parallel_for<int>(0, column_size, [&](int i) {
				for (int j = 0; j < row_size; j++) {
					result[i] += entity[j * column_size + i];
				}
				result[i] /= row_size;
			});
			return result;
		}

		/*内積 実装モード切替*/
		FastMatrix<T> dot(FastMatrix<T>& mat) { return SWITCH_FAST_CONTAONER_FUNCTION(dot)(mat); }
		/*内積*/
		FastMatrix<T> dot_com(FastMatrix<T>& mat) {
			FAST_CONTAINER_EXCEPTION_CHECK(column_size == mat.get_row_size(), fast_container_exception());
			int mat_col = mat.get_column_size();
			FastMatrix<T> result(row_size, mat_col);
			for (int i = 0; i < row_size; i++) {
				int res_offset = i * mat_col;
				int ent_pos = i * column_size;
				int mat_offset = 0;
				for (int k = 0; k < column_size; k++, ent_pos++, mat_offset += mat_col) {
					for (int j = 0; j < mat_col; j++) {
						result[res_offset + j] += entity[ent_pos] * mat[mat_offset + j];
					}
				}
			}
			return result;
		}
		/*内積 AMP実装*/
		FastMatrix<T> dot_amp(FastMatrix<T>& mat) {
			FAST_CONTAINER_EXCEPTION_CHECK(column_size == mat.get_row_size(), fast_container_exception());
			int col = mat.get_column_size();
			int mid = column_size;
			FastMatrix<T> result(row_size, col);
			concurrency::array_view<const T, 2> av_entity(row_size, mid, &entity[0]);
			concurrency::array_view<const T, 2> av_mat(mid, col, &mat[0]);
			concurrency::array_view<T, 2> av_result(row_size, col, &result[0]);
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<2> idx) restrict(amp) {
				for (int i = 0; i < mid; i++) {
					av_result[idx] += av_entity[idx[0]][i] * av_mat[i][idx[1]];
				}
			});
			av_result.synchronize();
			return result;
		}
		/*内積 PPL実装*/
		FastMatrix<T> dot_ppl(FastMatrix<T>& mat) {
			FAST_CONTAINER_EXCEPTION_CHECK(column_size == mat.get_row_size(), fast_container_exception());
			int mat_col = mat.get_column_size();
			FastMatrix<T> result(row_size, mat_col);
			concurrency::parallel_for<int>(0, row_size, [&](int i) {
				int res_offset = i * mat_col;
				int ent_pos = i * column_size;
				int mat_offset = 0;
				for (int k = 0; k < column_size; k++, ent_pos++, mat_offset += mat_col) {
					for (int j = 0; j < mat_col; j++) {
						result[res_offset + j] += entity[ent_pos] * mat[mat_offset + j];
					}
				}
			});
			return result;
		}

		/*転置行列 実装モード切替*/
		FastMatrix<T> reverse() { return SWITCH_FAST_CONTAONER_FUNCTION(reverse)(); }
		/*転置行列*/
		FastMatrix<T> reverse_com() {
			FastMatrix<T> result(column_size, row_size);
			for (int i = 0; i < row_size; i++) {
				int offset = i * column_size;
				for (int j = 0; j < column_size; j++) {
					result(j, i) = entity[offset + j];
				}
			}
			return result;
		}
		/*転置行列 AMP実装*/
		FastMatrix<T> reverse_amp() {
			FastMatrix<T> result(column_size, row_size);
			concurrency::array_view<const T, 2> av_entity(row_size, column_size, &entity[0]);
			concurrency::array_view<T, 2> av_result(column_size, row_size, &result[0]);
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<2> idx) restrict(amp) {
				av_result[idx] = av_entity[idx[1]][idx[0]];
			});
			av_result.synchronize();
			return result;
		}
		/*転置行列 PPL実装*/
		FastMatrix<T> reverse_ppl() {
			FastMatrix<T> result(column_size, row_size);
			concurrency::parallel_for<int>(0, size, [&](int i) {
				result[i] = entity[(i % row_size) * column_size + i / row_size];
			});
			return result;
		}

		/*指定行を取得 実装モード切替*/
		FastVector<T> row(int row) { return SWITCH_FAST_CONTAONER_FUNCTION(row)(row); }
		/*指定行を取得*/
		FastVector<T> row_com(int row) {
			FAST_CONTAINER_EXCEPTION_CHECK(row >= row_size, fast_container_exception());
			FastVector<T> result(column_size);
			for (int i = 0, j = row * column_size; i < column_size; i++, j++) {
				result[i] = entity[j];
			}
			return result;
		}
		/*指定行を取得 AMP実装*/
		FastVector<T> row_amp(int row) {
			FAST_CONTAINER_EXCEPTION_CHECK(row >= row_size, fast_container_exception());
			FastVector<T> result(column_size);
			concurrency::array_view<const T, 2> av_entity(row_size, column_size, &entity[0]);
			concurrency::array_view<T, 1> av_result(column_size, &result[0]);
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<1> idx) restrict(amp) {
				av_result[idx] = av_entity[row][idx[0]];
			});
			av_result.synchronize();
			return result;
		}
		/*指定行を取得 PPL実装*/
		FastVector<T> row_ppl(int row) {
			FAST_CONTAINER_EXCEPTION_CHECK(row >= row_size, fast_container_exception());
			FastVector<T> result(column_size);
			concurrency::parallel_for<int>(0, column_size, [&](int i) {
				result[i] = entity[row * column_size + i];
			});
			return result;
		}

		/*指定列を取得 実装モード切替*/
		FastVector<T> column(int col) { return SWITCH_FAST_CONTAONER_FUNCTION(column)(col); }
		/*指定列を取得*/
		FastVector<T> column_com(int col) {
			FAST_CONTAINER_EXCEPTION_CHECK(col >= column_size, fast_container_exception());
			FastVector<T> result(row_size);
			for (int i = 0, j = col; i < row_size; i++, j += column_size) {
				result[i] = entity[j];
			}
			return result;
		}
		/*指定列を取得 AMP実装*/
		FastVector<T> column_amp(int col) {
			FAST_CONTAINER_EXCEPTION_CHECK(col >= column_size, fast_container_exception());
			FastVector<T> result(row_size);
			concurrency::array_view<const T, 2> av_entity(row_size, column_size, &entity[0]);
			concurrency::array_view<T, 1> av_result(row_size, &result[0]);
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<1> idx) restrict(amp) {
				av_result[idx] = av_entity[idx[0]][col];
			});
			av_result.synchronize();
			return result;
		}
		/*指定列を取得 PPL実装*/
		FastVector<T> column_ppl(int col) {
			FAST_CONTAINER_EXCEPTION_CHECK(col >= column_size, fast_container_exception());
			FastVector<T> result(row_size);
			concurrency::parallel_for<int>(0, row_size, [&](int i) {
				result[i] = entity[i * column_size + col];
			});
			return result;
		}

		/*バッチの取得 実装モード切替*/
		FastMatrix<T> batch(FastVector<int>& mask) { return SWITCH_FAST_CONTAONER_FUNCTION(batch)(mask); }
		/*バッチの取得*/
		FastMatrix<T> batch_com(FastVector<int>& mask) {
			FAST_CONTAINER_EXCEPTION_CHECK(row_size >= mask.get_size(), fast_container_exception());
			int row = mask.get_size();
			FastMatrix<T> result(row, column_size);
			for (int i = 0; i < row; i++) {
				int r_offset = i * column_size;
				int e_offset = mask[i] * column_size;
				for (int j = 0; j < column_size; j++) {
					result[r_offset + j] = entity[e_offset + j];
				}
			}
			return result;
		}
		/*バッチの取得 AMP実装*/
		FastMatrix<T> batch_amp(FastVector<int>& mask) {
			FAST_CONTAINER_EXCEPTION_CHECK(row_size >= mask.get_size(), fast_container_exception());
			int row = mask.get_size();
			FastMatrix<T> result(row, column_size);
			concurrency::array_view<const T, 2> av_entity(row_size, column_size, &entity[0]);
			concurrency::array_view<const int, 1> av_mask(row, &mask[0]);
			concurrency::array_view<T, 2> av_result(row, column_size, &result[0]);
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<2> idx) restrict(amp) {
				av_result[idx] = av_entity[av_mask[idx[0]]][idx[1]];
			});
			av_result.synchronize();
			return result;
		}
		/*バッチの取得 PPL実装*/
		FastMatrix<T> batch_ppl(FastVector<int>& mask) {
			FAST_CONTAINER_EXCEPTION_CHECK(row_size >= mask.get_size(), fast_container_exception());
			int row = mask.get_size();
			int res_size = row * column_size;
			FastMatrix<T> result(row, column_size);
			concurrency::parallel_for<int>(0, res_size, [&](int i) {
				result[i] = entity[mask[i / column_size] * column_size + (i % column_size)];
			});
			return result;
		}

		/*ランダムなバッチの取得 実装モード切替*/
		FastMatrix<T> random_batch(int size) { return SWITCH_FAST_CONTAONER_FUNCTION(random_batch)(size); }
		/*ランダムなバッチの取得*/
		FastMatrix<T> random_batch_com(int size) { return batch_com(FastVector<int>::int_hash_random(size, 0, row_size - 1)); }
		/*ランダムなバッチの取得 AMP実装*/
		FastMatrix<T> random_batch_amp(int size) { return batch_amp(FastVector<int>::int_hash_random(size, 0, row_size - 1)); }
		/*ランダムなバッチの取得 PPL実装*/
		FastMatrix<T> random_batch_ppl(int size) { return batch_ppl(FastVector<int>::int_hash_random(size, 0, row_size - 1)); }

		/*関数を適用
		func: T(*func)(T x)*/
		template<class F>
		FastMatrix<T> apply_com_func(F func) {
			FastMatrix<T> result(row_size, column_size);
			for (int i = 0; i < size; i++) {
				result[i] = func(entity[i]);
			}
			return result;
		}
		/*関数を適用 AMP実装
		func: T(*func)(T x) restrict(amp)*/
		template<class F>
		FastMatrix<T> apply_amp_func(F func) {
			FastMatrix<T> result(row_size, column_size);
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
		FastMatrix<T> apply_ppl_func(F func) {
			FastMatrix<T> result(row_size, column_size);
			concurrency::parallel_for<int>(0, size, [&](int i) {
				result[i] = func(entity[i]);
			});
			return result;
		}

		/*関数を適用
		func: T(*func)(T x1, T x2)*/
		template<class F>
		FastMatrix<T> apply_com_combo_func(F func, FastMatrix<T>& mat) {
			FAST_CONTAINER_EXCEPTION_CHECK(row_size == mat.get_row_size(), fast_container_exception());
			FAST_CONTAINER_EXCEPTION_CHECK(column_size == mat.get_column_size(), fast_container_exception());
			FastMatrix<T> result(row_size, column_size);
			for (int i = 0; i < size; i++) {
				result[i] = func(entity[i], mat[i]);
			}
			return result;
		}
		/*関数を適用 AMP実装
		func: T(*func)(T x1, T x2) restrict(amp)*/
		template<class F>
		FastMatrix<T> apply_amp_combo_func(F func, FastMatrix<T>& mat) {
			FAST_CONTAINER_EXCEPTION_CHECK(row_size == mat.get_row_size(), fast_container_exception());
			FAST_CONTAINER_EXCEPTION_CHECK(column_size == mat.get_column_size(), fast_container_exception());
			FastMatrix<T> result(row_size, column_size);
			concurrency::array_view<const T, 1> av_entity(size, &entity[0]);
			concurrency::array_view<const T, 1> av_mat(size, &mat[0]);
			concurrency::array_view<T, 1> av_result(size, &result[0]);
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<1> idx) restrict(amp) {
				av_result[idx] = func(av_entity[idx], av_mat[idx]);
			});
			av_result.synchronize();
			return result;
		}
		/*関数を適用 PPL実装
		func: T(*func)(T x1, T x2)*/
		template<class F>
		FastMatrix<T> apply_ppl_combo_func(F func, FastMatrix<T>& mat) {
			FAST_CONTAINER_EXCEPTION_CHECK(row_size == mat.get_row_size(), fast_container_exception());
			FAST_CONTAINER_EXCEPTION_CHECK(column_size == mat.get_column_size(), fast_container_exception());
			FastMatrix<T> result(row_size, column_size);
			concurrency::parallel_for<int>(0, size, [&](int i) {
				result[i] = func(entity[i], mat[i]);
			});
			return result;
		}

		/*行毎に関数を適用
		func: T(*func)(T x1, T x2)*/
		template<class F>
		FastMatrix<T> apply_com_combo_func_by_rows(F func, FastVector<T>& vec) {
			FAST_CONTAINER_EXCEPTION_CHECK(column_size == vec.get_size(), fast_container_exception());
			FastMatrix<T> result(row_size, column_size);
			for (int i = 0; i < size; i++) {
				result[i] = func(entity[i], vec[i % column_size]);
			}
			return result;
		}
		/*行毎に関数を適用 AMP実装
		func: T(*func)(T x1, T x2) restrict(amp)*/
		template<class F>
		FastMatrix<T> apply_amp_combo_func_by_rows(F func, FastVector<T>& vec) {
			FAST_CONTAINER_EXCEPTION_CHECK(column_size == vec.get_size(), fast_container_exception());
			FastMatrix<T> result(row_size, column_size);
			concurrency::array_view<const T, 2> av_entity(row_size, column_size, &entity[0]);
			concurrency::array_view<const T, 1> av_vec(column_size, &vec[0]);
			concurrency::array_view<T, 2> av_result(row_size, column_size, &result[0]);
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<2> idx) restrict(amp) {
				av_result[idx] = func(av_entity[idx], av_vec[idx[1]]);
			});
			av_result.synchronize();
			return result;
		}
		/*行毎に関数を適用 PPL実装
		func: T(*func)(T x1, T x2)*/
		template<class F>
		FastMatrix<T> apply_ppl_combo_func_by_rows(F func, FastVector<T>& vec) {
			FAST_CONTAINER_EXCEPTION_CHECK(column_size == vec.get_size(), fast_container_exception());
			FastMatrix<T> result(row_size, column_size);
			concurrency::parallel_for<int>(0, size, [&](int i) {
				result[i] = func(entity[i], vec[i % column_size]);
			});
			return result;
		}

		/*列毎に関数を適用
		func: T(*func)(T x1, T x2)*/
		template<class F>
		FastMatrix<T> apply_com_combo_func_by_columns(F func, FastVector<T>& vec) {
			FAST_CONTAINER_EXCEPTION_CHECK(row_size == vec.get_size(), fast_container_exception());
			FastMatrix<T> result(row_size, column_size);
			for (int i = 0; i < size; i++) {
				result[i] = func(entity[i], vec[i / column_size]);
			}
			return result;
		}
		/*列毎に関数を適用 AMP実装
		func: T(*func)(T x1, T x2) restrict(amp)*/
		template<class F>
		FastMatrix<T> apply_amp_combo_func_by_columns(F func, FastVector<T>& vec) {
			FAST_CONTAINER_EXCEPTION_CHECK(row_size == vec.get_size(), fast_container_exception());
			FastMatrix<T> result(row_size, column_size);
			concurrency::array_view<const T, 2> av_entity(row_size, column_size, &entity[0]);
			concurrency::array_view<const T, 1> av_vec(row_size, &vec[0]);
			concurrency::array_view<T, 2> av_result(row_size, column_size, &result[0]);
			av_result.discard_data();
			concurrency::parallel_for_each(av_result.extent, [=](concurrency::index<2> idx) restrict(amp) {
				av_result[idx] = func(av_entity[idx], av_vec[idx[0]]);
			});
			av_result.synchronize();
			return result;
		}
		/*列毎に関数を適用 PPL実装
		func: T(*func)(T x1, T x2)*/
		template<class F>
		FastMatrix<T> apply_ppl_combo_func_by_columns(F func, FastVector<T>& vec) {
			FAST_CONTAINER_EXCEPTION_CHECK(row_size == vec.get_size(), fast_container_exception());
			FastMatrix<T> result(row_size, column_size);
			concurrency::parallel_for<int>(0, size, [&](int i) {
				result[i] = func(entity[i], vec[i / column_size]);
			});
			return result;
		}

		/*[row,column] ((values1),(values2),...)*/
		std::string to_string() {
			std::string result = "[" + std::to_string(row_size) + "," + std::to_string(column_size) + "](";
			for (int i = 0; i < row_size; i++) {
				if (i) result += ",(";
				else result += "(";
				for (int j = 0; j < column_size; j++) {
					if (j) result += "," + std::to_string(entity[i * column_size + j]);
					else result += std::to_string(entity[i * column_size + j]);
				}
				result += ")";
			}
			result += ")";
			return result;
		}

		/*ランダムなFastMatrixを生成*/
		static FastMatrix<T> real_random_com(int row, int col, T min = -1, T max = 1) {
			FastMatrix<T> result(row, col);
			RealRandom<T> rnd(min, max);
			return result.apply_com_func([&](T x) { return rnd.generate(); });
		}
		/*ランダムなFastMatrixを生成 PPL実装*/
		static FastMatrix<T> real_random_ppl(int row, int col, T min = -1, T max = 1) {
			FastMatrix<T> result(row, col);
			RealRandom<T> rnd(min, max);
			return result.apply_ppl_func([&](T x) { return rnd.generate(); });
		}
		/*ランダムなFastMatrix<int>を生成*/
		static FastMatrix<int> int_random_com(int row, int col, int min = -1, int max = 1) {
			FastMatrix<int> result(row, col);
			IntRandom rnd(min, max);
			return result.apply_com_func([&](int x) { return rnd.generate(); });
		}
		/*ランダムなFastMatrix<int>を生成 PPL実装*/
		static FastMatrix<int> int_random_ppl(int row, int col, int min = -1, int max = 1) {
			FastMatrix<int> result(row, col);
			IntRandom rnd(min, max);
			return result.apply_ppl_func([&](int x) { return rnd.generate(); });
		}
		/*平均:mean, 標準偏差:sd のランダムなFastMatrixを生成*/
		static FastMatrix<T> normal_random_com(int row, int col, T mean = 0, T sd = 1) {
			FastMatrix<T> result(row, col);
			NormalRandom<T> rnd(mean, sd);
			return result.apply_com_func([&](T x) { return rnd.generate(); });
		}
		/*平均:mean, 標準偏差:sd ランダムなFastMatrixを生成 PPL実装*/
		static FastMatrix<T> normal_random_ppl(int row, int col, T mean = 0, T sd = 1) {
			FastMatrix<T> result(row, col);
			NormalRandom<T> rnd(mean, sd);
			return result.apply_ppl_func([&](T x) { return rnd.generate(); });
		}

	private:
		std::vector<T> entity;
		int row_size;
		int column_size;
		int size;
	};


#ifdef FAST_CONTAONER_OPERATOR_OVERLOAD_COM_MODE

	template<typename T>
	FastMatrix<T> operator+(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_com_combo_func([](T x1, T x2) {return x1 + x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator-(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_com_combo_func([](T x1, T x2) {return x1 - x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator*(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_com_combo_func([](T x1, T x2) {return x1 * x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator/(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_com_combo_func([](T x1, T x2) {return x1 / x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator==(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_com_combo_func([](T x1, T x2) {return x1 == x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator!=(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_com_combo_func([](T x1, T x2) {return x1 != x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator>(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_com_combo_func([](T x1, T x2) {return x1 > x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator<(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_com_combo_func([](T x1, T x2) {return x1 < x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator>=(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_com_combo_func([](T x1, T x2) {return x1 >= x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator<=(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_com_combo_func([](T x1, T x2) {return x1 <= x2; }, mat2); }

	template<typename T>
	FastMatrix<T> operator+(T val, FastMatrix<T>& mat) { return mat.apply_com_func([=](T x) {return val + x; }); }
	template<typename T>
	FastMatrix<T> operator-(T val, FastMatrix<T>& mat) { return mat.apply_com_func([=](T x) {return val - x; }); }
	template<typename T>
	FastMatrix<T> operator*(T val, FastMatrix<T>& mat) { return mat.apply_com_func([=](T x) {return val * x; }); }
	template<typename T>
	FastMatrix<T> operator/(T val, FastMatrix<T>& mat) { return mat.apply_com_func([=](T x) {return val / x; }); }
	template<typename T>
	FastMatrix<T> operator==(T val, FastMatrix<T>& mat) { return mat.apply_com_func([=](T x) {return val == x; }); }
	template<typename T>
	FastMatrix<T> operator!=(T val, FastMatrix<T>& mat) { return mat.apply_com_func([=](T x) {return val != x; }); }
	template<typename T>
	FastMatrix<T> operator>(T val, FastMatrix<T>& mat) { return mat.apply_com_func([=](T x) {return val > x; }); }
	template<typename T>
	FastMatrix<T> operator<(T val, FastMatrix<T>& mat) { return mat.apply_com_func([=](T x) {return val < x; }); }
	template<typename T>
	FastMatrix<T> operator>=(T val, FastMatrix<T>& mat) { return mat.apply_com_func([=](T x) {return val >= x; }); }
	template<typename T>
	FastMatrix<T> operator<=(T val, FastMatrix<T>& mat) { return mat.apply_com_func([=](T x) {return val <= x; }); }

	template<typename T>
	FastMatrix<T> operator+(FastMatrix<T>& mat, T val) { return mat.apply_com_func([=](T x) {return x + val; }); }
	template<typename T>
	FastMatrix<T> operator-(FastMatrix<T>& mat, T val) { return mat.apply_com_func([=](T x) {return x - val; }); }
	template<typename T>
	FastMatrix<T> operator*(FastMatrix<T>& mat, T val) { return mat.apply_com_func([=](T x) {return x * val; }); }
	template<typename T>
	FastMatrix<T> operator/(FastMatrix<T>& mat, T val) { return mat.apply_com_func([=](T x) {return x / val; }); }
	template<typename T>
	FastMatrix<T> operator==(FastMatrix<T>& mat, T val) { return mat.apply_com_func([=](T x) {return x == val; }); }
	template<typename T>
	FastMatrix<T> operator!=(FastMatrix<T>& mat, T val) { return mat.apply_com_func([=](T x) {return x != val; }); }
	template<typename T>
	FastMatrix<T> operator>(FastMatrix<T>& mat, T val) { return mat.apply_com_func([=](T x) {return x > val; }); }
	template<typename T>
	FastMatrix<T> operator<(FastMatrix<T>& mat, T val) { return mat.apply_com_func([=](T x) {return x < val; }); }
	template<typename T>
	FastMatrix<T> operator>=(FastMatrix<T>& mat, T val) { return mat.apply_com_func([=](T x) {return x >= val; }); }
	template<typename T>
	FastMatrix<T> operator<=(FastMatrix<T>& mat, T val) { return mat.apply_com_func([=](T x) {return x <= val; }); }

#elif defined FAST_CONTAONER_OPERATOR_OVERLOAD_AMP_MODE

	template<typename T>
	FastMatrix<T> operator+(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 + x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator-(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 - x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator*(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 * x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator/(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 / x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator==(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 == x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator!=(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 != x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator>(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 > x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator<(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 < x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator>=(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 >= x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator<=(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_amp_combo_func([](T x1, T x2) restrict(amp) {return x1 <= x2; }, mat2); }

	template<typename T>
	FastMatrix<T> operator+(T val, FastMatrix<T>& mat) { return mat.apply_amp_func([=](T x) restrict(amp) {return val + x; }); }
	template<typename T>
	FastMatrix<T> operator-(T val, FastMatrix<T>& mat) { return mat.apply_amp_func([=](T x) restrict(amp) {return val - x; }); }
	template<typename T>
	FastMatrix<T> operator*(T val, FastMatrix<T>& mat) { return mat.apply_amp_func([=](T x) restrict(amp) {return val * x; }); }
	template<typename T>
	FastMatrix<T> operator/(T val, FastMatrix<T>& mat) { return mat.apply_amp_func([=](T x) restrict(amp) {return val / x; }); }
	template<typename T>
	FastMatrix<T> operator==(T val, FastMatrix<T>& mat) { return mat.apply_amp_func([=](T x) restrict(amp) {return val == x; }); }
	template<typename T>
	FastMatrix<T> operator!=(T val, FastMatrix<T>& mat) { return mat.apply_amp_func([=](T x) restrict(amp) {return val != x; }); }
	template<typename T>
	FastMatrix<T> operator>(T val, FastMatrix<T>& mat) { return mat.apply_amp_func([=](T x) restrict(amp) {return val > x; }); }
	template<typename T>
	FastMatrix<T> operator<(T val, FastMatrix<T>& mat) { return mat.apply_amp_func([=](T x) restrict(amp) {return val < x; }); }
	template<typename T>
	FastMatrix<T> operator>=(T val, FastMatrix<T>& mat) { return mat.apply_amp_func([=](T x) restrict(amp) {return val >= x; }); }
	template<typename T>
	FastMatrix<T> operator<=(T val, FastMatrix<T>& mat) { return mat.apply_amp_func([=](T x) restrict(amp) {return val <= x; }); }

	template<typename T>
	FastMatrix<T> operator+(FastMatrix<T>& mat, T val) { return mat.apply_amp_func([=](T x) restrict(amp) {return x + val; }); }
	template<typename T>
	FastMatrix<T> operator-(FastMatrix<T>& mat, T val) { return mat.apply_amp_func([=](T x) restrict(amp) {return x - val; }); }
	template<typename T>
	FastMatrix<T> operator*(FastMatrix<T>& mat, T val) { return mat.apply_amp_func([=](T x) restrict(amp) {return x * val; }); }
	template<typename T>
	FastMatrix<T> operator/(FastMatrix<T>& mat, T val) { return mat.apply_amp_func([=](T x) restrict(amp) {return x / val; }); }
	template<typename T>
	FastMatrix<T> operator==(FastMatrix<T>& mat, T val) { return mat.apply_amp_func([=](T x) restrict(amp) {return x == val; }); }
	template<typename T>
	FastMatrix<T> operator!=(FastMatrix<T>& mat, T val) { return mat.apply_amp_func([=](T x) restrict(amp) {return x != val; }); }
	template<typename T>
	FastMatrix<T> operator>(FastMatrix<T>& mat, T val) { return mat.apply_amp_func([=](T x) restrict(amp) {return x > val; }); }
	template<typename T>
	FastMatrix<T> operator<(FastMatrix<T>& mat, T val) { return mat.apply_amp_func([=](T x) restrict(amp) {return x < val; }); }
	template<typename T>
	FastMatrix<T> operator>=(FastMatrix<T>& mat, T val) { return mat.apply_amp_func([=](T x) restrict(amp) {return x >= val; }); }
	template<typename T>
	FastMatrix<T> operator<=(FastMatrix<T>& mat, T val) { return mat.apply_amp_func([=](T x) restrict(amp) {return x <= val; }); }

#elif defined FAST_CONTAONER_OPERATOR_OVERLOAD_PPL_MODE

	template<typename T>
	FastMatrix<T> operator+(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_ppl_combo_func([](T x1, T x2) {return x1 + x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator-(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_ppl_combo_func([](T x1, T x2) {return x1 - x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator*(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_ppl_combo_func([](T x1, T x2) {return x1 * x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator/(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_ppl_combo_func([](T x1, T x2) {return x1 / x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator==(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_ppl_combo_func([](T x1, T x2) {return x1 == x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator!=(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_ppl_combo_func([](T x1, T x2) {return x1 != x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator>(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_ppl_combo_func([](T x1, T x2) {return x1 > x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator<(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_ppl_combo_func([](T x1, T x2) {return x1 < x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator>=(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_ppl_combo_func([](T x1, T x2) {return x1 >= x2; }, mat2); }
	template<typename T>
	FastMatrix<T> operator<=(FastMatrix<T>& mat1, FastMatrix<T>& mat2) { return mat1.apply_ppl_combo_func([](T x1, T x2) {return x1 <= x2; }, mat2); }

	template<typename T>
	FastMatrix<T> operator+(T val, FastMatrix<T>& mat) { return mat.apply_ppl_func([=](T x) {return val + x; }); }
	template<typename T>
	FastMatrix<T> operator-(T val, FastMatrix<T>& mat) { return mat.apply_ppl_func([=](T x) {return val - x; }); }
	template<typename T>
	FastMatrix<T> operator*(T val, FastMatrix<T>& mat) { return mat.apply_ppl_func([=](T x) {return val * x; }); }
	template<typename T>
	FastMatrix<T> operator/(T val, FastMatrix<T>& mat) { return mat.apply_ppl_func([=](T x) {return val / x; }); }
	template<typename T>
	FastMatrix<T> operator==(T val, FastMatrix<T>& mat) { return mat.apply_ppl_func([=](T x) {return val == x; }); }
	template<typename T>
	FastMatrix<T> operator!=(T val, FastMatrix<T>& mat) { return mat.apply_ppl_func([=](T x) {return val != x; }); }
	template<typename T>
	FastMatrix<T> operator>(T val, FastMatrix<T>& mat) { return mat.apply_ppl_func([=](T x) {return val > x; }); }
	template<typename T>
	FastMatrix<T> operator<(T val, FastMatrix<T>& mat) { return mat.apply_ppl_func([=](T x) {return val < x; }); }
	template<typename T>
	FastMatrix<T> operator>=(T val, FastMatrix<T>& mat) { return mat.apply_ppl_func([=](T x) {return val >= x; }); }
	template<typename T>
	FastMatrix<T> operator<=(T val, FastMatrix<T>& mat) { return mat.apply_ppl_func([=](T x) {return val <= x; }); }

	template<typename T>
	FastMatrix<T> operator+(FastMatrix<T>& mat, T val) { return mat.apply_ppl_func([=](T x) {return x + val; }); }
	template<typename T>
	FastMatrix<T> operator-(FastMatrix<T>& mat, T val) { return mat.apply_ppl_func([=](T x) {return x - val; }); }
	template<typename T>
	FastMatrix<T> operator*(FastMatrix<T>& mat, T val) { return mat.apply_ppl_func([=](T x) {return x * val; }); }
	template<typename T>
	FastMatrix<T> operator/(FastMatrix<T>& mat, T val) { return mat.apply_ppl_func([=](T x) {return x / val; }); }
	template<typename T>
	FastMatrix<T> operator==(FastMatrix<T>& mat, T val) { return mat.apply_ppl_func([=](T x) {return x == val; }); }
	template<typename T>
	FastMatrix<T> operator!=(FastMatrix<T>& mat, T val) { return mat.apply_ppl_func([=](T x) {return x != val; }); }
	template<typename T>
	FastMatrix<T> operator>(FastMatrix<T>& mat, T val) { return mat.apply_ppl_func([=](T x) {return x > val; }); }
	template<typename T>
	FastMatrix<T> operator<(FastMatrix<T>& mat, T val) { return mat.apply_ppl_func([=](T x) {return x < val; }); }
	template<typename T>
	FastMatrix<T> operator>=(FastMatrix<T>& mat, T val) { return mat.apply_ppl_func([=](T x) {return x >= val; }); }
	template<typename T>
	FastMatrix<T> operator<=(FastMatrix<T>& mat, T val) { return mat.apply_ppl_func([=](T x) {return x <= val; }); }

#endif

}
