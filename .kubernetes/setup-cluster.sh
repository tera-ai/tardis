#!/bin/bash
set -eux

# Taken from https://cloud.google.com/blog/products/containers-kubernetes/machine-learning-with-jax-on-kubernetes-with-nvidia-gpus
GKE_NETWORK_NAME=default
GKE_CLUSTER_NAME=world-model-cluster
GKE_CLUSTER_ZONE=us-central1-c
GKE_NODE_POOL_NAME=gpus-node-pool
GPU_NODE_POOL_MACHINE_TYPE=a2-ultragpu-8g
GPU_NODE_POOL_ACCELERATOR_TYPE=nvidia-a100-80gb
# GPU_NODE_POOL_DRIVER_VERSION=default|latest|disabled
GPU_NODE_POOL_ACCELERATOR_COUNT=8
GPU_NODE_POOL_NODE_COUNT=2

### Run all of these commands separately ###


# Create VPC network (if not done already)
# NOTE: if VPC network name changes, we would have to change
gcloud compute networks create $GKE_NETWORK_NAME \
    --subnet-mode auto \
    --bgp-routing-mode regional \
    --mtu 1460

# Create Kubernetes Cluster
gcloud container clusters create $GKE_CLUSTER_NAME --zone=$GKE_CLUSTER_ZONE

# Add kubectl credentials for the Kubernetes Cluster
gcloud container clusters get-credentials $GKE_CLUSTER_NAME --region $GKE_CLUSTER_ZONE

# Create GPU NodePool
# NOTE: Need enable-fast-socket and enable-gvnic for multi-node performance
# NOTE: The preemptible flag removes need for quotas
gcloud container node-pools create $GKE_NODE_POOL_NAME \
    --accelerator type=$GPU_NODE_POOL_ACCELERATOR_TYPE,count=$GPU_NODE_POOL_ACCELERATOR_COUNT \
    --machine-type=$GPU_NODE_POOL_MACHINE_TYPE \
    --cluster=$GKE_CLUSTER_NAME \
    --zone=$GKE_CLUSTER_ZONE \
    --num-nodes=$GPU_NODE_POOL_NODE_COUNT \
    --enable-fast-socket \
    --enable-gvnic

# Install NVIDIA CUDA Drivers on the compute nodes (if not done already)
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml

# Submit the job
kubectl apply -k .
