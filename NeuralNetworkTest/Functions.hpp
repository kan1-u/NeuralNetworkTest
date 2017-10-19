#pragma once

#include "FastContainerLibrary.hpp"

#ifdef FAST_CONTAINER_NO_EXCEPTION

#define	EXCEPTION_CHECK(expression, e)

#else

#define	EXCEPTION_CHECK(expression, e) \
	if (!(expression)) { \
		std::cerr << "Check failed in file " << __FILE__ << " at line " << __LINE__ << ":" << std::endl; \
		std::cerr << #expression << std::endl; \
		e.raise(); \
	}

#endif

namespace FastContainer {

	/*ó·äOèàóù*/
	struct fast_container_exception
#if ! defined FAST_CONTAINER_NO_EXCEPTION
		: public std::exception {
		explicit fast_container_exception(const char *s = "fast_container_exception") : std::exception(s) {}
		void raise() { throw *this; }
	};
#else
	{
		fast_container_exception() {}
		explicit fast_container_exception(const char *) {}
		void raise() { std::abort(); }
	};
#endif

	/*minÅ`maxÇÃóêêî*/
	template<typename T>
	class Random {
	public:
		Random(T min = -1, T max = 1) { set_param(min, max); }
		T generate() {
			return random(mt);
		}
		void set_param(T min, T max) {
			EXCEPTION_CHECK(min <= max, fast_container_exception());
			mt = std::mt19937(rnd());
			this->random = std::uniform_real_distribution<>(min, max);
		}
	private:
		std::random_device rnd;
		std::mt19937 mt;
		std::uniform_real_distribution<> random;
	};

	/*minÅ`maxÇÃóêêî int*/
	class IntRandom {
	public:
		IntRandom(int min = -1, int max = 1) { set_param(min, max); }
		int generate() {
			return random(mt);
		}
		void set_param(int min, int max) {
			EXCEPTION_CHECK(min <= max, fast_container_exception());
			mt = std::mt19937(rnd());
			this->random = std::uniform_int_distribution<>(min, max);
		}
	private:
		std::random_device rnd;
		std::mt19937 mt;
		std::uniform_int_distribution<> random;
	};

}
