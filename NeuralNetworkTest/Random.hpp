#pragma once

#include "FastContainerLibrary.hpp"

namespace FastContainer {

	/*min`max‚Ì—”*/
	template<typename T>
	class RealRandom {
	public:
		RealRandom(T min = -1, T max = 1) { set_param(min, max); }
		T generate() {
			return random(mt);
		}
		void set_param(T min, T max) {
			if (min > max) throw fast_container_exception();
			mt = std::mt19937(rnd());
			this->random = std::uniform_real_distribution<>(min, max);
		}
	private:
		std::random_device rnd;
		std::mt19937 mt;
		std::uniform_real_distribution<> random;
	};

	/*min`max‚Ì—” int*/
	class IntRandom {
	public:
		IntRandom(int min = -1, int max = 1) { set_param(min, max); }
		int generate() {
			return random(mt);
		}
		void set_param(int min, int max) {
			if (min > max) throw fast_container_exception();
			mt = std::mt19937(rnd());
			this->random = std::uniform_int_distribution<>(min, max);
		}
	private:
		std::random_device rnd;
		std::mt19937 mt;
		std::uniform_int_distribution<> random;
	};

	/*•½‹Ï:mean, •W€•Î·:sd ‚Ì—”*/
	template<typename T>
	class NormalRandom {
	public:
		NormalRandom(T mean = 0, T sd = 1) { set_param(mean, sd); }
		T generate() {
			return random(mt);
		}
		void set_param(T mean, T sd) {
			mt = std::mt19937(rnd());
			this->random = std::normal_distribution<>(mean, sd);
		}
	private:
		std::random_device rnd;
		std::mt19937 mt;
		std::normal_distribution<> random;
	};

	/*min`max‚Ìd•¡‚Ì‚È‚¢®”—”
	(ˆê“x¶¬‚µ‚½’l‚Í¶¬‚µ‚È‚¢)*/
	class IntHashRandom {
	public:
		IntHashRandom(int min, int max) { set_param(min, max); }
		int generate() {
			if (min > now_max) throw fast_container_exception();
			int result;
			int val = std::uniform_int_distribution<>(min, now_max)(mt);
			auto itr = map.find(val);
			int replaced_val;
			auto replaced_itr = map.find(now_max);
			if (replaced_itr != map.end()) {
				replaced_val = replaced_itr->second;
			}
			else replaced_val = now_max;
			if (itr == map.end()) {
				result = val;
				if (val != now_max) map.insert(std::make_pair(val, replaced_val));
			}
			else {
				result = itr->second;
				itr->second = replaced_val;
			}
			now_max--;
			return result;
		}
		void set_param(int min, int max) {
			if (min > max) throw fast_container_exception();
			this->min = min;
			this->max = max;
			now_max = max;
			mt = std::mt19937(rnd());
		}
		void reset_param() {
			map.clear();
			now_max = max;
		}
	private:
		std::random_device rnd;
		std::mt19937 mt;
		std::unordered_map<int, int> map;
		int min;
		int max;
		int now_max;
	};

}
