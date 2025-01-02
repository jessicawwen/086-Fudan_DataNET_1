#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <chrono>

enum norm_t { SUM = 0, MAX = 1 };

bool loadGraph(
    const std::string& filename,
    std::vector<int>& srcVec,
    std::vector<int>& dstVec,
    std::vector<double>& wVec
)
{
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        int s, d;
        double w;
        if (!(ss >> s >> d)) {
            std::cerr << "Error: parse failed (need at least two columns), line=" << line << std::endl;
            return false;
        }
        if (!(ss >> w)) {
            w = 1.0;
        }
        srcVec.push_back(s);
        dstVec.push_back(d);
        wVec.push_back(w);
    }
    fin.close();
    return true;
}

void compressNodes(
    const std::vector<int>& srcVec,
    const std::vector<int>& dstVec,
    std::unordered_map<int,int>& nodeMap,
    std::vector<int>& invNodeMap
) {
    std::vector<int> allNodes;
    allNodes.reserve(srcVec.size() + dstVec.size());
    for (size_t i = 0; i < srcVec.size(); i++) {
        allNodes.push_back(srcVec[i]);
        allNodes.push_back(dstVec[i]);
    }
    std::sort(allNodes.begin(), allNodes.end());
    allNodes.erase(std::unique(allNodes.begin(), allNodes.end()), allNodes.end());

    nodeMap.clear();
    invNodeMap.clear();
    invNodeMap.reserve(allNodes.size());

    for (size_t i = 0; i < allNodes.size(); i++) {
        int originalID = allNodes[i];
        nodeMap[originalID] = (int)i;
        invNodeMap.push_back(originalID);
    }
}

void buildCSR(
    const std::vector<int>& srcVec,
    const std::vector<int>& dstVec,
    const std::vector<double>& wVec,
    const std::unordered_map<int,int>& nodeMap,
    int& num_nodes,
    int& num_edges,
    std::vector<int>& rowPtrOut,
    std::vector<int>& colIdxOut,
    std::vector<double>& valOut,
    std::vector<int>& rowPtrIn,
    std::vector<int>& colIdxIn,
    std::vector<double>& valIn
)
{
    num_nodes = (int)nodeMap.size();
    num_edges = (int)srcVec.size();

    rowPtrOut.assign(num_nodes + 1, 0);
    rowPtrIn.assign(num_nodes + 1, 0);

    for (size_t i = 0; i < srcVec.size(); i++) {
        int u = nodeMap.at(srcVec[i]);
        int v = nodeMap.at(dstVec[i]);
        rowPtrOut[u + 1]++;
        rowPtrIn[v + 1]++;
    }

    for (int i = 1; i <= num_nodes; i++) {
        rowPtrOut[i] += rowPtrOut[i - 1];
        rowPtrIn[i]  += rowPtrIn[i - 1];
    }

    colIdxOut.resize(num_edges);
    valOut.resize(num_edges);
    colIdxIn.resize(num_edges);
    valIn.resize(num_edges);

    std::vector<int> offsetOut(num_nodes, 0);
    std::vector<int> offsetIn(num_nodes, 0);

    for (size_t i = 0; i < srcVec.size(); i++){
        int u = nodeMap.at(srcVec[i]);
        int v = nodeMap.at(dstVec[i]);
        double w = wVec[i];

        int posOut = rowPtrOut[u] + offsetOut[u];
        colIdxOut[posOut] = v;
        valOut[posOut]    = w;
        offsetOut[u]++;

        int posIn = rowPtrIn[v] + offsetIn[v];
        colIdxIn[posIn] = u;
        valIn[posIn]    = w;
        offsetIn[v]++;
    }
}

//-------------------GPU kernel--------------------
__device__ double mutual_weight(
    const int* rowPtrOut,
    const int* colIdxOut,
    const double* valOut,
    const int* rowPtrIn,
    const int* colIdxIn,
    const double* valIn,
    int u,
    int v
)
{
    double w_uv = 0.0, w_vu = 0.0;
    for (int i = rowPtrOut[u]; i < rowPtrOut[u+1]; i++) {
        if (colIdxOut[i] == v) {
            w_uv = valOut[i];
            break;
        }
    }
    for (int i = rowPtrIn[u]; i < rowPtrIn[u+1]; i++) {
        if (colIdxIn[i] == v) {
            w_vu = valIn[i];
            break;
        }
    }
    // printf("u=%d, v=%d, w_uv=%f, w_vu=%f\n", u, v, w_uv, w_vu);
    return w_uv + w_vu;
}



__global__ void compute_out_in_sum(
    const int* rowPtrOut,
    const double* valOut,
    const int* rowPtrIn,
    const double* valIn,
    int num_nodes,
    double* d_sum
)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if(u >= num_nodes) return;

    double sum_val = 0.0;

    for(int i = rowPtrOut[u]; i < rowPtrOut[u+1]; i++){
        sum_val += valOut[i];
    }

    for(int i = rowPtrIn[u]; i < rowPtrIn[u+1]; i++){
        sum_val += valIn[i];
    }
    d_sum[u] = sum_val;
    // printf("Node %d: sum_val = %f\n", u, sum_val);
}

static __device__ double normalized_mutual_weight(
    const int* rowPtrOut,
    const int* colIdxOut,
    const double* valOut,
    const int* rowPtrIn,
    const int* colIdxIn,
    const double* valIn,
    int u,
    int v,
    norm_t norm
) {
    double weight_uv = mutual_weight(rowPtrOut, colIdxOut, valOut, rowPtrIn, colIdxIn, valIn, u, v);

    double scale = 0.0;
    if(norm == SUM){
        for (int i = rowPtrOut[u]; i < rowPtrOut[u+1]; i++) {
            scale += valOut[i];
        }
        for (int i = rowPtrIn[u]; i < rowPtrIn[u+1]; i++) {
            scale += valIn[i];
        }
    }else if(norm == MAX){
        for (int i = rowPtrOut[u]; i < rowPtrOut[u+1]; i++) {
            scale = fmax(scale, valOut[i]);
        }
        for (int i = rowPtrIn[u]; i < rowPtrIn[u+1]; i++) {
            scale = fmax(scale, valIn[i]);
        }
    }
    return (scale==0.0) ? 0.0 : (weight_uv / scale);
}

__device__ double redundancy(
    const int* rowPtrOut,
    const int* colIdxOut,
    const double* valOut,
    const int* rowPtrIn,
    const int* colIdxIn,
    const double* valIn,
    const double* d_sum,
    int u,
    int v
) {
    double r = 0.0;
    for (int i = rowPtrOut[u]; i < rowPtrOut[u + 1]; i++) {
        int w = colIdxOut[i];
        double norm_u_w = normalized_mutual_weight(rowPtrOut, colIdxOut, valOut, rowPtrIn, colIdxIn, valIn, u, w, SUM);
        double norm_v_w = normalized_mutual_weight(rowPtrOut, colIdxOut, valOut, rowPtrIn, colIdxIn, valIn, v, w, MAX);
        r += norm_u_w * norm_v_w;
        // printf("Node %d -> Node %d via Node %d: norm_u_w = %f, norm_v_w = %f, r = %f\n", u, v, w, norm_u_w, norm_v_w, r);
    }
    for (int i = rowPtrIn[u]; i < rowPtrIn[u + 1]; i++) {
        int w = colIdxIn[i];
        double norm_u_w = normalized_mutual_weight(rowPtrOut, colIdxOut, valOut, rowPtrIn, colIdxIn, valIn, u, w, SUM);
        double norm_v_w = normalized_mutual_weight(rowPtrOut, colIdxOut, valOut, rowPtrIn, colIdxIn, valIn, v, w, MAX);
        r += norm_u_w * norm_v_w;
        // printf("Node %d <- Node %d via Node %d: norm_u_w = %f, norm_v_w = %f, r = %f\n", u, v, w, norm_u_w, norm_v_w, r);
    }
    return 1.0 - r;
}

__global__ void calculate_effective_sizes(
    int num_nodes,
    int NODES_PER_BLOCK,
    const int* rowPtrOut,
    const int* colIdxOut,
    const double* valOut,
    const int* rowPtrIn,
    const int* colIdxIn,
    const double* valIn,
    const double* d_sum,
    // const int* node_mask,
    double* effective_sizes
){
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;
    double redundancy_sum = 0.0;
    bool is_nan = true;
    for (int i = rowPtrOut[u]; i < rowPtrOut[u + 1]; i++) {
        int v = colIdxOut[i];
        if (v == u) continue;
        is_nan = false;
        // printf("Node %d -> Node %d: Weight = %f\n", u, v, valOut[i]);
        redundancy_sum += redundancy(rowPtrOut, colIdxOut, valOut, rowPtrIn, colIdxIn, valIn, d_sum, u, v);
        // printf("Node %d -> Node %d: redundancy_sum = %f\n", u, v, redundancy_sum);
    }

    for (int i = rowPtrIn[u]; i < rowPtrIn[u + 1]; i++) {
        int v = colIdxIn[i];
        // if (v == u) continue;
        // is_nan = false;
        redundancy_sum += redundancy(rowPtrOut, colIdxOut, valOut, rowPtrIn, colIdxIn, valIn, d_sum, u, v);
        // printf("Node %d <- Node %d: redundancy_sum = %f\n", u, v, redundancy_sum);
    }
    effective_sizes[u] = is_nan ? NAN : redundancy_sum;
}

void cuda_effective_size(
    int num_nodes,
    const std::vector<int>& rowPtrOut,
    const std::vector<int>& colIdxOut,
    const std::vector<double>& valOut,
    const std::vector<int>& rowPtrIn,
    const std::vector<int>& colIdxIn,
    const std::vector<double>& valIn,
    std::vector<double>& effective_sizes
) {
    // Allocate device memory
    int* d_rowPtrOut = nullptr;
    int* d_colIdxOut = nullptr;
    double* d_valOut = nullptr;
    int* d_rowPtrIn = nullptr;
    int* d_colIdxIn = nullptr;
    double* d_valIn = nullptr;
    double* d_effective_sizes = nullptr;
    double* d_sum = nullptr;

    cudaMalloc((void**)&d_rowPtrOut, rowPtrOut.size() * sizeof(int));
    cudaMalloc((void**)&d_colIdxOut, colIdxOut.size() * sizeof(int));
    cudaMalloc((void**)&d_valOut, valOut.size() * sizeof(double));
    cudaMalloc((void**)&d_rowPtrIn, rowPtrIn.size() * sizeof(int));
    cudaMalloc((void**)&d_colIdxIn, colIdxIn.size() * sizeof(int));
    cudaMalloc((void**)&d_valIn, valIn.size() * sizeof(double));
    cudaMalloc((void**)&d_effective_sizes, num_nodes * sizeof(double));
    cudaMalloc((void**)&d_sum, num_nodes * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_rowPtrOut, rowPtrOut.data(), rowPtrOut.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdxOut, colIdxOut.data(), colIdxOut.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valOut, valOut.data(), valOut.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtrIn, rowPtrIn.data(), rowPtrIn.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdxIn, colIdxIn.data(), colIdxIn.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valIn, valIn.data(), valIn.size() * sizeof(double), cudaMemcpyHostToDevice);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int maxBlocks = prop.maxGridSize[0];
    int NODES_PER_BLOCK =16;
    int blockSize = (num_nodes < maxThreadsPerBlock) ? num_nodes : 256;
    int numBlocks = (num_nodes + blockSize - 1) / blockSize;

    if (numBlocks > maxBlocks) {
        NODES_PER_BLOCK = numBlocks / maxBlocks + 1;
        numBlocks = (num_nodes + NODES_PER_BLOCK * blockSize - 1) / (NODES_PER_BLOCK * blockSize);
    }
    // int blockSize = 256; 
    // int numBlocks = (num_nodes + NODES_PER_BLOCK - 1) / NODES_PER_BLOCK;
    
    compute_out_in_sum<<<numBlocks, blockSize>>>(d_rowPtrOut, d_valOut, d_rowPtrIn, d_valIn, num_nodes, d_sum);
    cudaDeviceSynchronize();

    calculate_effective_sizes<<<numBlocks, blockSize>>>(
        num_nodes,
        NODES_PER_BLOCK,
        d_rowPtrOut,
        d_colIdxOut,
        d_valOut,
        d_rowPtrIn,
        d_colIdxIn,
        d_valIn,
        d_sum, 
        d_effective_sizes
    );

    cudaDeviceSynchronize();

    effective_sizes.resize(num_nodes);
    cudaMemcpy(effective_sizes.data(), d_effective_sizes, num_nodes * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_rowPtrOut);
    cudaFree(d_colIdxOut);
    cudaFree(d_valOut);
    cudaFree(d_rowPtrIn);
    cudaFree(d_colIdxIn);
    cudaFree(d_valIn);
    cudaFree(d_sum);
    cudaFree(d_effective_sizes);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file_path>" << std::endl;
        return -1;
    }

    const std::string filename = argv[1];
    
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<int> srcVec, dstVec;
    std::vector<double> wVec;

    if (!loadGraph(filename, srcVec, dstVec, wVec)) {
        return -1;
    }

    std::unordered_map<int, int> nodeMap;
    std::vector<int> invNodeMap;
    compressNodes(srcVec, dstVec, nodeMap, invNodeMap);

    int num_nodes, num_edges;
    std::vector<int> rowPtrOut, colIdxOut, rowPtrIn, colIdxIn;
    std::vector<double> valOut, valIn;

    buildCSR(
        srcVec, dstVec, wVec, nodeMap,
        num_nodes, num_edges,
        rowPtrOut, colIdxOut, valOut,
        rowPtrIn, colIdxIn, valIn
    );

    std::vector<double> effective_sizes;

    cuda_effective_size(
        num_nodes,
        rowPtrOut, colIdxOut, valOut,
        rowPtrIn, colIdxIn, valIn,
        effective_sizes
    );

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    // for (int i = 0; i < effective_sizes.size(); i++) {
    //     std::cout << "Node " << invNodeMap[i] << ": Effective Size = " << effective_sizes[i] << std::endl;
    // }
        // Write results to a file in the current directory
    std::ofstream outFile("effective_sizes.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error: Cannot open file for writing results. File will be created." << std::endl;
        std::ofstream createFile("effective_sizes.txt");
        if (!createFile.is_open()) {
            std::cerr << "Error: Unable to create the file." << std::endl;
            return -1;
        }
        createFile.close();
    }

    for (int i = 0; i < effective_sizes.size(); i++) {
        outFile << "Node " << invNodeMap[i] << ": Effective Size = " << effective_sizes[i] << std::endl;
    }

    outFile.close();

    return 0;
}