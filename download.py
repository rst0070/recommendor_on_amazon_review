#!/usr/bin/env python
import os

# Set cache directory under the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(current_dir, "cache")
os.environ["HF_HOME"] = cache_dir

# Load the dataset
from datasets import load_dataset

dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Home_and_Kitchen",
    trust_remote_code=True,
)
