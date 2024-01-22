#ifndef FFTSTRATEGY_HPP
#define FFTSTRATEGY_HPP

#include "../utils/FFTDataTypes.hpp"

namespace fftcore{

	/**
	 * @brief This class describes a general strategy for computing the FFT, templated on the dimension and on the floating type. 
	 *  
	 * This class is abstract and represents, with FFTSolver.hpp, the basic structure of the Strategy Pattern. It exposes three methods
	 * for computing the FFT :
	 * - An out-of-place fft from complex data to complex data.
	 * - An out-of-place fft from real data to complex data.
	 * - An in-place fft from complex data to complex data.
	 * 
	 * Each method allows to specify in which direction compute the fft with #FFTDirection. Pay attention, not all the methods are supported.
	 * Even though, following the *interface segregation principle*, this should not happen, this design allows more flexibility in order to obtain
	 * better performance.
	 * 
	 * When a method is not supported, NotSupportedException is thrown.
     */
	template<int Rank, typename FloatingType>
	class FFTStrategy{


		public:
			/**
			 * Describes a general complex tensor templated on the dimension and on the floating type. 	
			 */
			using CTensor = Eigen::Tensor<std::complex<FloatingType>, Rank>;
			/**
			 * Describes a general real tensor templated on the dimension and on the floating type. 	
			 */
			using RTensor = Eigen::Tensor<FloatingType, Rank>;        

            //needed in test_template to access the floating type from outside
            using FloatTypeAlias = FloatingType;

			/**
			* Complex to Complex out-of-place FFT. This is the most general case, in which no symmetry of the Fourier transform
			* can be used effectively.
			* 
			* @param [in] data_in a constant refeference to the input complex tensor that won't be modified by the method.
			* @param [out] data_out a reference to the output complex tensor that contains the transformed data.
			* @param [in] dir the fft direction.
			*/ 
			virtual void fft(const CTensor& data_in, CTensor& data_out, FFTDirection dir) const = 0;
			
			/**
			* Real to Complex out-of-place FFT. This method can save memory, since real numbers requires less memory.
			* This can be used also to exploit Fourier's transform symmetries to gain performance. 
			*
			* @param [in] data_in a constant refeference to the input real tensor that won't be modified by the method.
			* @param [out] data_out a reference to the output complex tensor that contains the transformed data.
			* @param [in] dir the fft direction.
			*/ 
			virtual void fft(const RTensor& data_in, CTensor& data_out, FFTDirection dir) const = 0;

			/**
			* In-place FFT. No conservation of the original data.
			*
			* @param [in,out] data_in_out a refeference to the input real tensor that will contain the transformed data,
			*	loosing the original tensor.
			* @param [in] dir the fft direction.
			*/ 
			virtual void fft(CTensor& data_in_out, FFTDirection dir) const = 0;
			
			virtual ~FFTStrategy() = default;
	};


	/**
	 * @brief This class describes a general strategy for computing the FFT on 1 dimensional data.
	 */
	template<typename FloatingType>
	class FFT_1D :
	public FFTStrategy<1, FloatingType>{
		public:
			/**
			 * Describes a general complex 1 dimensional tensor templated on the floating type. 	
			 */
			using RTensor_1D = Eigen::Tensor<FloatingType, 1>;
			/**
			 * Describes a general real 1 dimensional tensor templated on the floating type. 	
			 */
			using CTensor_1D = Eigen::Tensor<std::complex<FloatingType>, 1>;

			virtual ~FFT_1D() = default;
	};

	/**
	 * @brief This class describes a general strategy for computing the FFT on 2 dimensional data.
	*/
	template<typename FloatingType>
	class FFT_2D :
	public FFTStrategy<2, FloatingType>{

		public:
			/**
			 * Describes a general complex 2 dimensional tensor templated on the floating type. 	
			 */
			using RTensor_2D = Eigen::Tensor<FloatingType, 2>;
			/**
			 * Describes a general real 2 dimensional tensor templated on the floating type. 	
			 */
			using CTensor_2D = Eigen::Tensor<std::complex<FloatingType>, 2>;
			//@TODO CHANGE HERE
			using RTensor_1D = Eigen::Tensor<FloatingType, 1>;
			using CTensor_1D = Eigen::Tensor<std::complex<FloatingType>, 1>;

			virtual ~FFT_2D() = default;
	};

	/**
	 * @brief This class describes a general strategy for computing the FFT on 3 dimensional data.
	*/
	template<typename FloatingType>
	class FFT_3D :
	public FFTStrategy<3, FloatingType>{

		public:
			
			/**
			 * Describes a general complex 3 dimensional tensor templated on the floating type. 	
			 */
			using CTensor_3D = Eigen::Tensor<std::complex<FloatingType>, 3>;
			/**
			 * Describes a general real 3 dimensional tensor templated on the floating type. 	
			 */
			using RTensor_3D = Eigen::Tensor<FloatingType, 3>;
			
			//@TODO CHANGE HERE
			using RTensor_2D = Eigen::Tensor<FloatingType, 2>;
			using CTensor_2D = Eigen::Tensor<std::complex<FloatingType>, 2>;

			virtual ~FFT_3D() = default;
	};

}

#endif // FFTSTRATEGY_HPP