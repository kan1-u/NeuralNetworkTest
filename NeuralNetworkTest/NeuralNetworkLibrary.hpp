#pragma once

#include "FastContainerLibrary.hpp"

#define FAST_CONTAONER_OPERATOR_OVERLOAD_AMP_MODE
//#define FAST_CONTAONER_OPERATOR_OVERLOAD_PPL_MODE
//#define FAST_CONTAONER_OPERATOR_OVERLOAD_NORMAL_MODE

namespace NeuralNetwork {

	using namespace std;
	using namespace FastContainer;

#pragma region Layer

	template<typename T>
	class Layer {
	public:
		virtual FastMatrix<T> forward(FastMatrix<T>& target) = 0;
		virtual FastMatrix<T> backward(FastMatrix<T>& target) = 0;
		virtual void update(T learningRate) = 0;
	};

	template<typename T>
	class SigmoidLayer :public Layer<T> {
	public:
		FastMatrix<T> forward(FastMatrix<T>& target) {
			return target.amp_sigmoid();
		}
		FastMatrix<T> backward(const FastMatrix<T>& target) {
			return out * (1 - out) * target;
		}
		void update(T learningRate) {
		}
	private:
		FastMatrix<T> out;
	};

	template<typename T>
	class ReluLayer :public Layer<T> {
	public:
		FastMatrix<T> forward(FastMatrix<T>& target) {
			mask = target > 0;
			return target * mask;
		}
		FastMatrix<T> backward(FastMatrix<T>& target) {
			return target * mask;
		}
		void update(T learningRate) {
		}
	private:
		FastMatrix<T> mask;
	};

	template<typename T>
	class PReluLayer :public Layer<T> {
	public:
		PReluLayer(T slope) {
			this->slope = slope;
		}
		FastMatrix<T> forward(FastMatrix<T>& target) {
			mask = target > 0;
			mask = mask + (slope * (mask == 0));
			return target * mask;
		}
		FastMatrix<T> backward(FastMatrix<T>& target) {
			return target * mask;
		}
		void update(T learningRate) {
		}
	private:
		FastMatrix<T> mask;
		T slope;
	};

	template<typename T>
	class RReluLayer :public Layer<T> {
	public:
		RReluLayer(T slope_min, T slope_max) {
			this->slope_min = slope_min;
			this->slope_max = slope_max;
		}
		FastMatrix<T> forward(FastMatrix<T>& target) {
			mask = target > 0;
			auto rnd = FastMatrix<T>::ppl_random(target.get_rows(), target.get_columns(), slope_min, slope_max);
			mask = mask + (rnd * (mask == 0));
			return target * mask;
		}
		FastMatrix<T> backward(FastMatrix<T>& target) {
			return target * mask;
		}
		void update(T learningRate) {
		}
	private:
		FastMatrix<T> mask;
		T slope_min;
		T slope_max;
	};

	template<typename T>
	class AffineLayer :public Layer<T> {
	public:
		AffineLayer(const FastMatrix<T>& w, const FastVector<T>& b) {
			this->w = w;
			this->b = b;
		}
		FastMatrix<T> forward(FastMatrix<T>& target) {
			x = target;
			return target.amp_dot(w).amp_add_by_rows(b);
		}
		FastMatrix<T> backward(FastMatrix<T>& target) {
			auto dx = target.amp_dot(w.amp_reverse());
			dw = x.amp_reverse().amp_dot(target);
			db = target.amp_sum_by_columns();
			return dx;
		}
		void update(T learningRate) {
			w = w - (learningRate * dw);
			b = b - (learningRate * db);
		}
		FastMatrix<T> getdw() {
			return dw;
		}
		FastVector<T> getdb() {
			return db;
		}
	private:
		FastMatrix<T> w;
		FastVector<T> b;
		FastMatrix<T> x;
		FastMatrix<T> dw;
		FastVector<T> db;
	};

#pragma endregion

#pragma region LastLayer

	template<typename T>
	class LastLayer {
	public:
		virtual T forward(FastMatrix<T>& target, FastMatrix<T>& teacher) = 0;
		virtual FastMatrix<T> backward() = 0;
	};

	template<typename T>
	class SoftmaxWithLossLayer :public LastLayer<T> {
	public:
		T forward(FastMatrix<T>& target, FastMatrix<T>& teacher) {
			this->teacher = teacher;
			out = target.amp_softmax();
			return out.amp_cross_entropy_error_class(teacher);
		}
		FastMatrix<T> backward() {
			if (teacher.get_rows() == 1) {
				out.sub_by_rows(teacher.to_FastVector()) / (-1 * teacher.get_columns());
			}
			else {
				return (out - teacher) / teacher.get_rows();
			}
		}
	private:
		T loss;
		FastMatrix<T> out;
		FastMatrix<T> teacher;
	};

#pragma endregion

#pragma region NeuralNetwork

	template<typename T>
	class NeuralNetwork {
	public:
		std::vector<Layer<T> *> layers;
		LastLayer<T> *lastLayer;
		FastMatrix<T> predict(FastMatrix<T>& input) {
			auto result = input;
			for each (auto layer in layers)
			{
				result = layer->forward(result);
			}
			return result;
		}
		template<typename CT>
		T loss(FastMatrix<T>& input, const CT& teacher) {
			auto y = predict(input);
			return lastLayer->forward(y, teacher);
		}
		T accuracy(FastMatrix<T>& input, FastMatrix<T>& teacher) {
			auto y = predict(input).amp_argmax_by_rows();
			auto t = teacher.amp_argmax_by_rows();
			return (y == t).sum() / input.get_rows();
		}
		T accuracy(FastMatrix<T>& input, const FastVector<T>& teacher) {
			auto y = predict(input).amp_argmax_by_rows();
			return (y == teacher).sum() / input.get_rows();
		}
		void set_gradient() {
			auto out = lastLayer->backward();
			for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
				out = (*it)->backward(out);
			}
		}
		void update(T learningRate) {
			for each (auto layer in layers)
			{
				layer->update(learningRate);
			}
		}
		template<typename CT>
		void training(FastMatrix<T>& input, const CT& teacher, T learningRate) {
			loss(input, teacher);
			set_gradient();
			update(learningRate);
		}
	private:
	};

#pragma endregion

}
