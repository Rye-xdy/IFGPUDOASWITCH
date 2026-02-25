#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

// --- 常量定义 ---
#define PI 3.14159265358979323846f
#define C_LIGHT 3e8f
#define FREQ 2.4e9f
#define N_ANTENNAS 8
#define RADIUS 0.075f // 0.6 * lambda (approx for 2.4G)
#define N_CHANNELS (N_ANTENNAS - 1) // 轮采产生的相位差数量

// 网格分辨率
#define AZ_START 0.0f
#define AZ_END 360.0f
#define AZ_STEP 0.5f  // 0.5度步进
#define EL_START 0.0f
#define EL_END 90.0f
#define EL_STEP 0.5f

// --- 辅助函数：计算网格大小 ---
int get_grid_size() {
    int az_points = (int)((AZ_END - AZ_START) / AZ_STEP);
    int el_points = (int)((EL_END - EL_START) / EL_STEP);
    return az_points * el_points;
}

// --- CUDA Kernel: 相关性计算 ---
// 每个线程计算一个网格点的相关系数
__global__ void correlation_kernel(
    const cuFloatComplex* __restrict__ d_manifold, // 样本库 [Grid_Size * 7]
    const cuFloatComplex* __restrict__ d_measured, // 实测数据 [7]
    float* d_scores,                               // 输出分数 [Grid_Size]
    int n_scenarios
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_scenarios) {
        cuFloatComplex correlation_sum = make_cuFloatComplex(0.0f, 0.0f);
        
        // 循环计算 7 个通道的内积: sum += measured * conj(theory)
        for (int i = 0; i < N_CHANNELS; i++) {
            // 获取当前网格点对应通道 i 的理论值
            // d_manifold 是扁平化的一维数组，布局为 [Scenario 0 (7 chans), Scenario 1 (7 chans)...]
            cuFloatComplex theory = d_manifold[tid * N_CHANNELS + i];
            cuFloatComplex meas = d_measured[i];
            
            // Conjugate of theory
            cuFloatComplex theory_conj = make_cuFloatComplex(cuCrealf(theory), -cuCimagf(theory));
            
            // Multiply
            cuFloatComplex prod = cuCmulf(meas, theory_conj);
            
            // Accumulate
            correlation_sum = cuCaddf(correlation_sum, prod);
        }

        // 计算模值作为分数 (也可以除以 N_CHANNELS 进行归一化，但这不影响最大值位置)
        d_scores[tid] = cuCabsf(correlation_sum);
    }
}

// --- Host端辅助: 理论导向矢量计算 ---
// 简单的C++实现，用于建库
void calculate_manifold_point(float az_deg, float el_deg, std::vector<std::complex<float>>& vec) {
    float az_rad = az_deg * PI / 180.0f;
    float el_rad = el_deg * PI / 180.0f;
    float lambda = C_LIGHT / FREQ;
    float k = 2.0f * PI / lambda;

    // 天线坐标 (假设都在 z=0 平面)
    // 信号方向矢量 u (指向信号源)
    // 注意：波矢量 k_vec 通常指向传播方向，即 -u
    // Phase = k_vec dot r_vec = -k * (u dot r)
    // u = [sin(el)cos(az), sin(el)sin(az), cos(el)]
    
    // 为了简化，直接计算相位 phi_n
    std::vector<float> phases(N_ANTENNAS);
    for(int n=0; n<N_ANTENNAS; ++n) {
        float angle_ant = 2.0f * PI * n / N_ANTENNAS;
        float x = RADIUS * cos(angle_ant);
        float y = RADIUS * sin(angle_ant);
        
        // 空间相位延迟 (假设入射波矢量方向)
        // dot product
        float u_x = sin(el_rad) * cos(az_rad);
        float u_y = sin(el_rad) * sin(az_rad);
        // float u_z = cos(el_rad); // 平面阵 z=0, 这项为0
        
        // 这里相位定义为 -k*(r*u)
        float phase = -k * (x * u_x + y * u_y);
        phases[n] = phase;
    }

    // 计算相对于天线0的相位差，并转为复数
    for(int i=1; i<N_ANTENNAS; ++i) {
        float diff = phases[i] - phases[0];
        vec.push_back(std::polar(1.0f, diff));
    }
}

int main() {
    // 1. 初始化参数
    int az_steps = (int)((AZ_END - AZ_START) / AZ_STEP);
    int el_steps = (int)((EL_END - EL_START) / EL_STEP);
    int total_scenarios = az_steps * el_steps;
    
    size_t manifold_size = total_scenarios * N_CHANNELS * sizeof(cuFloatComplex);
    size_t meas_size = N_CHANNELS * sizeof(cuFloatComplex);
    size_t score_size = total_scenarios * sizeof(float);

    std::cout << "=== CUDA 2D Interferometer Simulation ===" << std::endl;
    std::cout << "Grid Size: " << az_steps << "x" << el_steps << " = " << total_scenarios << " points." << std::endl;
    std::cout << "Building Manifold Library on Host (CPU)..." << std::endl;

    // 2. Host端建库 (模拟“已知样本库”)
    // 使用 std::vector 方便管理，最后 memcpy
    std::vector<cuFloatComplex> h_manifold(total_scenarios * N_CHANNELS);
    
    int idx = 0;
    for(int i_el = 0; i_el < el_steps; ++i_el) {
        for(int i_az = 0; i_az < az_steps; ++i_az) {
            float az = AZ_START + i_az * AZ_STEP;
            float el = EL_START + i_el * EL_STEP;
            
            std::vector<std::complex<float>> theory_vec;
            calculate_manifold_point(az, el, theory_vec);
            
            for(int k=0; k<N_CHANNELS; ++k) {
                h_manifold[idx * N_CHANNELS + k] = make_cuFloatComplex(theory_vec[k].real(), theory_vec[k].imag());
            }
            idx++;
        }
    }
    std::cout << "Manifold Built." << std::endl;

    // 3. 模拟一个实测信号 (用于测试)
    // 假设真实角度 Az=135, El=45
    std::vector<cuFloatComplex> h_measured(N_CHANNELS);
    std::vector<std::complex<float>> true_vec;
    float true_az = 135.0f;
    float true_el = 45.0f;
    calculate_manifold_point(true_az, true_el, true_vec);
    
    // 添加一点噪声并赋值给 h_measured
    // 这里简单直接用真值，主要测算法流程
    for(int k=0; k<N_CHANNELS; ++k) {
        h_measured[k] = make_cuFloatComplex(true_vec[k].real(), true_vec[k].imag());
    }

    // 4. GPU 资源分配
    cuFloatComplex *d_manifold, *d_measured;
    float *d_scores;
    
    cudaMalloc(&d_manifold, manifold_size);
    cudaMalloc(&d_measured, meas_size);
    cudaMalloc(&d_scores, score_size);

    // 5. 预加载样本库到 GPU (模拟系统启动时的加载)
    cudaMemcpy(d_manifold, h_manifold.data(), manifold_size, cudaMemcpyHostToDevice);

    std::cout << "Starting GPU Processing..." << std::endl;

    // A. 拷贝实测数据 (Host -> Device)
    cudaMemcpy(d_measured, h_measured.data(), meas_size, cudaMemcpyHostToDevice);

    // B. 执行相关计算 Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_scenarios + threadsPerBlock - 1) / threadsPerBlock;
    correlation_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_manifold, d_measured, d_scores, total_scenarios);

    // C. 在 GPU 上搜索最大值 (使用 Thrust)
    // thrust::max_element 返回指向最大值的迭代器(指针)，这是个同步操作，自带阻塞等待 Kernel 完成的效果
    thrust::device_ptr<float> score_ptr(d_scores);
    thrust::device_ptr<float> max_ptr = thrust::max_element(score_ptr, score_ptr + total_scenarios);
    
    // 计算索引
    int max_idx = max_ptr - score_ptr;
    
    // 获取最大值 (如果需要，可选)
    // float max_val = *max_ptr; 

    // 6. 结果解析 (从索引反推角度)
    // idx = i_el * az_steps + i_az
    int res_i_el = max_idx / az_steps;
    int res_i_az = max_idx % az_steps;
    
    float res_az = AZ_START + res_i_az * AZ_STEP;
    float res_el = EL_START + res_i_el * EL_STEP;

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "True Angle:      Az=" << true_az << ", El=" << true_el << std::endl;
    std::cout << "Estimated Angle: Az=" << res_az << ", El=" << res_el << std::endl;
    
    // 7. 清理
    cudaFree(d_manifold);
    cudaFree(d_measured);
    cudaFree(d_scores);

    return 0;
}