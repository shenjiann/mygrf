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

    // 创建ForestOptions
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
    // std::vector<size_t> empty_clusters;
    std::vector<size_t> province_id(data.get_num_rows());
    size_t n = 0;
    std::generate(province_id.begin(), province_id.end(), [&n]() { return 10 * (n++ / 5 + 1); });
    uint samples_per_cluster = 4;
    ForestOptions options = ForestOptions(num_trees, ci_group_size, sample_fraction, 
        mtry, min_node_size, honesty, honesty_fraction, prune, alpha, imbalance_penalty, 
        num_threads, seed, legacy_seed, province_id, samples_per_cluster);

    // 创建Sampler
    size_t start = 13;
    std::mt19937_64 random_number_generator(options.get_random_seed() + start);
    nonstd::uniform_int_distribution<uint> udist;
    uint tree_seed = udist(random_number_generator);;
    RandomSampler sampler(tree_seed, options.get_sampling_options());

    // 创建clusters
    std::vector<size_t> clusters;
    sampler.sample_clusters(data.get_num_rows(), options.get_sample_fraction(), clusters);
    
    // 训练一棵树
    // std::unique_ptr<Tree> tree = trainer.tree_trainer.train(data, sampler, clusters, options.get_tree_options());
    std::vector<std::vector<size_t>> child_nodes;
    std::vector<std::vector<size_t>> nodes;
    std::vector<size_t> split_vars;
    std::vector<double> split_values;
    std::vector<bool> send_missing_left;

    child_nodes.emplace_back();
    child_nodes.emplace_back();
    trainer.tree_trainer.create_empty_node(child_nodes, nodes, split_vars, split_values, send_missing_left);
    
    std::vector<size_t> new_leaf_samples;
    std::vector<size_t> tree_growing_clusters;
    std::vector<size_t> new_leaf_clusters;
    sampler.subsample(clusters, options.get_tree_options().get_honesty_fraction(), tree_growing_clusters, new_leaf_clusters);
    sampler.sample_from_clusters(tree_growing_clusters, nodes[0]);
    sampler.sample_from_clusters(new_leaf_clusters, new_leaf_samples);

    std::unique_ptr<SplittingRule> splitting_rule = trainer.tree_trainer.splitting_rule_factory->create(
      nodes[0].size(), options.get_tree_options());

    size_t num_open_nodes = 1;
    size_t i = 0;
    Eigen::ArrayXXd responses_by_sample(data.get_num_rows(), trainer.tree_trainer.relabeling_strategy->get_response_length());

    // 结束运行
    std::cout << "Down !" << std::endl;
    std::cout << std::cin.get() << std::endl;
}
