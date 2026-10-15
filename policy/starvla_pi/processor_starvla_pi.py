"""Use the shared StarVLA pipeline with StarvlaPi quantile normalization."""
from policy.starvla_groot.processor_starvla_groot import make_starvla_groot_pre_post_processors

def make_starvla_pi_pre_post_processors(config, dataset_stats=None):
    return make_starvla_groot_pre_post_processors(config, dataset_stats)
