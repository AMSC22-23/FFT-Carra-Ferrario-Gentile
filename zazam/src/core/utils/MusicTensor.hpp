#ifndef MUSICTENSOR_HPP
#define MUSICTENSOR_HPP

#include "ffft/fftcore.hpp"
#include "Utils.hpp"
#include "ZazamDataTypes.hpp"
#include <string>
#include <complex>

namespace zazamcore{

   /**
    * @brief An implementation of the FFFT Tensor wrapper which takes care of the 
    * tensor loading from different sources.
   */
   template<typename Scalar>
   class MusicTensor: public fftcore::TensorFFTBase<std::complex<Scalar>, 1>{
      public:

         /**
            * @brief Constructor for MusicTensor.
            * @param path Path of the WAV or AIFF file 
         */
         MusicTensor(const std::string &path);

         /**
            * @brief Constructor for MusicTensor
            * @param raw Raw input vector to load 
         */
         MusicTensor(const std::vector<Scalar> &raw);


         void slice_tensor(e_index &, e_index &);        

         AudioFile<Scalar> get_audio_file(){
               return audio_file;
         };

         void normalize();

      private:

         /**
          * Internal AudioFile object to manage file loads.
         */
         AudioFile<Scalar> audio_file; 

         /**
          * Converts audio file samples into a mono signal on cahnnel 0.
          * @param audio_file AudioFile object 
         */
         void convert_to_mono(AudioFile<Scalar> &audio_file);

         /**
          * Converts a raw stereo audio signal to a raw mono audio signal.
          * @param stereo Stereo raw signal 
          * @param mono Mono raw signal
         */
         void convert_to_mono(const std::vector<Scalar> &stereo, std::vector<Scalar> &mono);
   }; 

   template<typename Scalar>
   MusicTensor<Scalar>::MusicTensor(const std::string &path){
       // Load file form path
      bool loadedOK = audio_file.load(path);
      assert(loadedOK);

      // Cut to mono
      convert_to_mono(audio_file);

      this->get_tensor() = Vector<std::complex<Scalar>>(audio_file.getNumSamplesPerChannel());
      // Copy samples into the tensor 
      zazamcore::utils::std_to_complex_eigen_vector(audio_file.samples[0], this->get_tensor(), 0, audio_file.getNumSamplesPerChannel());
   }

   template<typename Scalar>
   MusicTensor<Scalar>::MusicTensor(const std::vector<Scalar> &vector){

      std::vector<Scalar> mono;
      // Cut to mono
      convert_to_mono(vector, mono);

      assert(mono.size()*2 == vector.size());
      this->get_tensor() = Vector<std::complex<Scalar>>(mono.size());
      utils::std_to_complex_eigen_vector(mono, this->get_tensor(), 0, mono.size());
   }


   template<typename Scalar>
   void MusicTensor<Scalar>::slice_tensor(e_index &begin, e_index &dimension){
      this->get_tensor() = Vector<std::complex<Scalar>>(dimension); 
      zazamcore::utils::std_to_complex_eigen_vector(audio_file.samples[0], this->get_tensor(), begin, dimension);
   }

   template<typename Scalar>
   void MusicTensor<Scalar>::convert_to_mono(const std::vector<Scalar> &raw, std::vector<Scalar> &mono){
      
      for(int i=0; i<raw.size()/2; i++){
         mono.push_back((raw[2*i]+raw[2*i+1])/2.0);
      }
   }

   template<typename Scalar>
   void MusicTensor<Scalar>::convert_to_mono(AudioFile<Scalar> &a){
      // Only if is stereo
      if(a.getNumChannels()==2){
         for(auto &s : a.samples){
            s[0] = (s[0] + s[1])/2.0;
         }
         a.setNumChannels(1);
      }
   }

   template<typename Scalar>
   void MusicTensor<Scalar>::normalize(){
      this->get_tensor() /= this->get_tensor().constant(this->get_tensor().size()); 
   }
}

#endif