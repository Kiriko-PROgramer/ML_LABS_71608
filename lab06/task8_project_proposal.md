# Individual Project Proposal

Student name: Kyrylo

Student ID: 71608

## Problem

Fashion-MNIST image classification using the best CNN architecture from Lab 3 for compact deployment.

## Dataset

Fashion-MNIST: 60,000 training and 10,000 test grayscale images, 28x28 pixels, 10 clothing classes.

## Success metric

Test accuracy, using the Lab 3 CNN result of 89.11% after 5 epochs as the reference baseline.

## Deployment constraints

Latency must be at most 50 ms per sample and model size must be at most 5 MB.

## Planned optimization

Post-training INT8 quantization, with structured pruning as a fallback if further compression is needed.

## Reproducibility and evaluation plan

Training uses a YAML-driven configuration, a fixed random seed, checkpointing, MLflow experiment tracking, throughput and memory measurements, and controlled comparisons of batch size, worker count, optimizer, and learning-rate scheduling. Fast mode uses one deterministic subset for every controlled comparison so the experiments remain directly comparable while finishing faster on a CPU. The final project version can disable fast mode to repeat the same pipeline on the complete Fashion-MNIST training and test sets.
