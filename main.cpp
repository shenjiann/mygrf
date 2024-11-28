#include <iostream>
#include "utility.h"
#include "ForestTrainers.h"
#include "ForestTrainer.h"
#include "RandomSampler.h"

using namespace grf;

int main() {
    int outcome_index = 10;
    bool check = equal_doubles(1.0, 2.0, 3.0);
    std::cout << check << std::endl;
    
    auto data_vec = load_data("/Users/js/Desktop/mygrf/test/forest/resources/gaussian_data.csv");
    Data data(data_vec);
    data.set_outcome_index(outcome_index);
    
    ForestTrainer trainer = regression_trainer();

    uint num_trees = 50;
    size_t ci_group_size = 1;
    double sample_fraction = ci_group_size > 1 ? 0.35 : 0.7;
    uint mtry = 3;
    uint min_node_size = 1;
    bool honesty = true;
    double honesty_fraction = 0.5;
    bool prune = true;
    double alpha = 0.0;
    double imbalance_penalty = 0.0;
    uint num_threads = 4;
    uint seed = 42;
    bool legacy_seed = true;
    std::vector<size_t> empty_clusters;
    uint samples_per_cluster = 0;
    ForestOptions options = ForestOptions(num_trees, ci_group_size, sample_fraction, 
        mtry, min_node_size, honesty, honesty_fraction, prune, alpha, imbalance_penalty, 
        num_threads, seed, legacy_seed, empty_clusters, samples_per_cluster);

    size_t start = 13;
    std::mt19937_64 random_number_generator(options.get_random_seed() + start);
    nonstd::uniform_int_distribution<uint> udist;
    uint tree_seed = udist(random_number_generator);;
    RandomSampler sampler(tree_seed, options.get_sampling_options());


    std::vector<size_t> clusters;
    sampler.sample(data.get_num_rows(), options.get_sample_fraction(), clusters);
    if (sampler.options.get_clusters().empty()) {
        std::cout << "is empty !" << std::endl;
    }

    std::unique_ptr<Tree> tree = trainer.tree_trainer.train(data, sampler, clusters, options.get_tree_options());

    std::cout << "asdf" << std::endl;
}
