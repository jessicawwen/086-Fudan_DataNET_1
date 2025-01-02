# 086-Fudan_DataNET_1

### Performance Optimization for Graph Analysis on Open Data

**EasyGraph** is a high-performance graph analytics library optimized for large-scale graph processing. Designed to efficiently handle open data, this library leverages GPU acceleration to provide swift implementations of various graph algorithms and supplements EasyGraph with GPU versions of several critical functions, including PageRank, Minimum Spanning Tree (MST), clustering coefficients, and structural hole-related metrics.

#### Prerequisites

Before compiling and using the GPU extension module, ensure that your system meets the following requirements:

- **CUDA Toolkit**: Install the NVIDIA CUDA Toolkit appropriate for your GPU. [Download CUDA](https://developer.nvidia.com/cuda-downloads)

- **C++ Compiler**: Compatible with C++11 or later.

- **NVIDIA GPU**: A CUDA-capable NVIDIA GPU is required for GPU acceleration.

- **Build Tools**: Ensure you have nvcc (CUDA compiler) installed and properly configured in your system’s PATH.

#### Usage

The GPU-accelerated modules are written in CUDA and need to be compiled using nvcc. Below are the compilation and execution commands for each file.

- structural_hole/effective_size.cu

```shell
nvcc -std=c++11 effective_size.cu -o effective_size
./effective_size <path_to_dataset>
```

- structural_hole/constraint.cu

```shell
nvcc -std=c++11 constraint.cu -o constraint
./constraint <path_to_dataset>
```

- structural_hole/hierarchy.cu

```shell
nvcc -std=c++11 hierarchy.cu -o hierarchy
./hierarchy <path_to_dataset>
```

**Note**: Replace <path_to_dataset> with the actual path to your graph dataset. The dataset should be formatted with each line representing an edge in the format source_node destination_node [weight]. If the weight is not specified, it defaults to 1.0.

#### Integration into EasyGraph

To integrate the GPU-accelerated structural hole-related functions into EasyGraph, follow the steps below:

**1.Clone the EasyGraph Repository**

```bash
git clone https://github.com/easy-graph/Easy-Graph.git
cd Easy-Graph
```

**2.Add GPU and CPU Modules**

- **GPU Modules**: Add the gpu_structural_hole folder under the path Easy-Graph/gpu_easygraph.

- **CPU Modules**: Add the cpu_structural_hole folder under the path Easy-Graph/cpu_easygraph.

**3.Expose API Interfaces**

To declare the GPU-accelerated functions, add declaration to Easy-Graph/gpu_easygraph/gpu_easygraph.h.

```cpp
int constraint(
    int num_nodes,
    const std::vector<int>& rowPtrOut,
    const std::vector<int>& colIdxOut,
    const std::vector<double>& valOut,
    const std::vector<int>& rowPtrIn,
    const std::vector<int>& colIdxIn,
    const std::vector<double>& valIn,
    bool is_directed,
    std::vector<int>& node_mask,
    std::vector<double>& constraint
);

int hierarchy(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& row,
    const std::vector<int>& col,
    int num_nodes,
    const std::vector<double>& W,
    bool is_directed,
    std::vector<int>& node_mask, 
    std::vector<double>& hierarchy
);

int effective_size(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& row,
    const std::vector<int>& col,
    int num_nodes,
    const std::vector<double>& W,
    bool is_directed,
    std::vector<int>& node_mask, 
    std::vector<double>& effective_size
);
```

 **4.Testing the Integration**

After integrating, compile EasyGraph and run existing or new datasets.

```bash
export EASYGRAPH_ENABLE_GPU="True"
pip install ./Easy-Graph/
```
