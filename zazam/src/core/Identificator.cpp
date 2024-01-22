#include "Identificator.hpp"
#include <iostream>
#include <filesystem>

using namespace zazamcore;
/**
 * Identify a sample from a dataset of songs hases
 * @param sample_hash The hash of the sample
 * @return The Song object that has been identified by the algorithm.
 * 
*/
void Identificator::identify(const Vector_ui &sample_hash, Song &result) const{


   // Dataset hashes and titles
   std::vector<Vector_ui> music_hashes;
   std::vector<std::string> file_names;

   // --- Load dataset --- 
   Vector_ui tmp;
   for (const auto & entry : std::filesystem::directory_iterator(hashes_dataset_path)){
      // Load hash from file
      fftcore::utils::load_tensor_mtx(tmp, entry.path().string());
      music_hashes.push_back(tmp);
      // Load also the name of the file 
      file_names.push_back(entry.path().filename());
   }

   // --- Identification algorithm ---
   std::vector<double> all_ratios; 
   std::vector<int> matches; 
   int mode, n_occurrences;
   double ratio;

   Vector_ui normalized_sample_hash(sample_hash);
   // "Normalize" the sample hash 
   normalize_and_round(normalized_sample_hash);

   // For each sampled song in the dataset
   for(auto &music_hash: music_hashes){
      matches.clear();

      // "Normalize" the current song hash 
      normalize_and_round(music_hash);
       
      // Fill the matches vector according to the algorithm specifications 
      calculate_matches_scores(music_hash, normalized_sample_hash, matches);
      
      if(matches.size() == 0){
         ratio = 0; 
      }else{
         // If there is at least one match, calculate the ratio between matches
         // and total tries 

         // Find the element which appear the most in the matches vector
         zazamcore::utils::mode_of_vector(matches, mode, n_occurrences);
         ratio = n_occurrences/(double)sample_hash.size();
      }
      
      // Save the song matching ratio
      all_ratios.push_back(ratio);
   }


   std::cout << "============================================================";
   for(int i=0; i<all_ratios.size(); i++){
      std::cout << "i: " << i << " | ratio: " << all_ratios[i] << std::endl;
   }

   for(int i=0; i<all_ratios.size(); i++){
      std::cout << i << ": " << file_names[i] << std::endl; 
   }

   const int res_i = utils::find_max_element_index(all_ratios);

   double match_ratio = all_ratios[res_i]; 
   all_ratios.erase(all_ratios.begin() + res_i);
   double no_match_average_ratio = accumulate(all_ratios.begin(), all_ratios.end(), 0.0)/all_ratios.size();              

   std::cout << "============================================================" << std::endl;
   std::cout << "ID: " << res_i << std::endl; 
   std::cout << "Ratio: " << match_ratio << std::endl; 
   std::cout << "Average no matches ratio: " << no_match_average_ratio << std::endl; 


   result.hash = music_hashes[res_i];   
   result.title = file_names[res_i];
}


void Identificator::normalize_and_round(Vector_ui &hash) const{

   double normalized_value;
   for(int i=0; i<hash.size(); i++){
      normalized_value = hash(i) / 1000;
      hash(i) = std::round(normalized_value);
   }
}

void Identificator::calculate_matches_scores(const Vector_ui &song_hash, const Vector_ui &sample_hash, std::vector<int> &matches_vector) const{
   assert(song_hash.size()>0 && sample_hash.size() > 0);
   for(int i=0; i<sample_hash.size(); i++){
      for(int j=0; j<song_hash.size(); j++){
         if(song_hash(j) == sample_hash(i)){
            matches_vector.push_back( j - i );
         }
      }
   }

}
