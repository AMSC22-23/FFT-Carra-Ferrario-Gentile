#include <iostream>
#include <memory>
#include <type_traits>

template<typename T, int Rank>
class DataStruct{
	public:
		void print(T t){
			std::cout<<Rank<<std::endl;
		}
		virtual ~DataStruct() = default;

};

template<typename T, int Rank>
class A{
	public:
		virtual void f(DataStruct<T, Rank> d) const = 0;
		virtual ~A() = default;
};


template<typename T>
class A_1D:
public A<T, 1>{
	public:
		virtual void f(DataStruct<T, 1> d) const = 0;
		virtual ~A_1D() = default;
};

template<typename T>
class A_2D:
public A<T, 2>{
	public:
		virtual void f(DataStruct<T, 2> d) const = 0;
		virtual ~A_2D() = default;
};

template<typename T, int Rank>
//class A_ND:
//public A<T, Rank>{
class A_ND: public A<T, Rank>{
	public:
		virtual void f(DataStruct<T, Rank> d) const = 0;
		virtual ~A_ND() = default;
};

template<typename T, int Rank>
class A_CUDA:
public A_1D<T>,
public A_2D<T>,
public A_ND<T, Rank>{
//public A<T,Rank>{
	public:
		virtual void f(DataStruct<T, 1> d) const {
			T x = 5;
			std::cout<<"1-D"<<std::endl;
			d.print(x);	
		};
		virtual void f(DataStruct<T, 2> d) const {
			std::cout<<"2-D"<<std::endl;
			T x = 5;
			d.print(x);	
		};
		virtual void f(DataStruct<T, Rank> d) const {
			T x = 5;
			std::cout<<"N-D"<<std::endl;
			d.print(x);	
		};
		virtual ~A_CUDA() = default;
};


template<typename T, int Rank>
class Solver{
	public:
		Solver(std::unique_ptr<A<T,Rank>>&& strategy) : _strategy(std::move(strategy)) {};

		virtual void compute_f(DataStruct<T, Rank> d){
			_strategy->f(d);
		};
	private:
		std::unique_ptr<A<T, Rank>> _strategy;
};

int main(){
	DataStruct<int, 2> d;
	Solver<int, 2> s(std::make_unique<A_CUDA<int, 2>>());
	s.compute_f(d);
}
