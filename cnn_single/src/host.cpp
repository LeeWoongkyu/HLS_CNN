/**
* Copyright (C) 2020 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#include "xcl2.hpp"
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <omp.h>

#include "cnn.h"

void LoadData(
	int _R, int _C, int _M, int _N, int _K,
	std::vector<short, aligned_allocator<short> > & inp,
	std::vector<short, aligned_allocator<short> > & ker,
	std::vector<short, aligned_allocator<short> > & out_sw,
	std::vector<short, aligned_allocator<short> > & out_hw
	){

	// initialize input
	for(int row = 0; row < _R+_K-1; row++) {
		for(int col = 0; col < _C+_K-1; col++) {
			for(int chi = 0; chi < _N; chi++) {
				if (row >= (_K-1)/2 && row < _R+(_K-1)/2 && col >= (_K-1)/2 && col < _C+(_K-1)/2) {
					inp[chi*(_R+_K-1)*(_C+_K-1) + row*(_C+_K-1) + col] = rand() % 3 - 1;
				}
				else {
					inp[chi*(_R+_K-1)*(_C+_K-1) + row*(_C+_K-1) + col] = 0;
	}}}}
	// initialize kernel
	for(int cho = 0; cho < _M; cho++) {
		for(int chi = 0; chi < _N; chi++) {
          		 	for (int ki = 0; ki < _K; ki++) {
           				for (int kj = 0; kj < _K; kj++) {    
					ker[cho*(_N*_K*_K) + chi*(_K*_K) + ki*_K + kj] = rand() % 3 - 1;
	}}}}
	// initialize output
	for(int row = 0; row < _R; row++) {
		for(int col = 0; col < _C; col++) {
			for(int cho = 0; cho < _M; cho++) {
				out_sw[cho*(_R*_C) + row*_C + col] = 0;
				out_hw[cho*(_R*_C) + row*_C + col] = 0;
	}}}

}
//--------------------------------------------------------------------------------------------------------------------

short IsError(short a, short b) {
	return fabs((a - b) / (a + b)) > 1e-3f && fabs(a - b) > 0.05f;
}

//--------------------------------------------------------------------------------------------------------------------

void cnn_sw(
	int _R, int _C, int _M, int _N,
	std::vector<short, aligned_allocator<short> > inp,
	std::vector<short, aligned_allocator<short> > ker,
	std::vector<short, aligned_allocator<short> > & out
	){


	// Convolution
#pragma omp parallel for
	for(int row = 0; row < _R; row++) {
		for(int col = 0; col < _C; col++) {
			for(int cho = 0; cho < _M; cho++) {
				for(int chi = 0; chi < _N; chi++) {
                				for (int ki = 0; ki < K; ki++) {
                					for (int kj = 0; kj < K; kj++) {
							out[row*_M*_C + col*_M + cho] += ker[ki*_M*_N*K + kj*_M*_N + cho*_N + chi] * inp[(row+ki)*_N*(_C+K-1) + (col+kj)*_N + chi];
						}
					}
				}
			}
		}
	}
}

//--------------------------------------------------------------------------------------------------------------------

int main(int argc, char** argv) {
	if (argc != 2) {
		std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
		return EXIT_FAILURE;
	}
	
	std::vector<short, aligned_allocator<short> > inp1(N[0]*(R[0]+K-1)*(C[0]+K-1));
	std::vector<short, aligned_allocator<short> > ker1(M[0]*N[0]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw1(M[0]*R[0]*C[0]);
	std::vector<short, aligned_allocator<short> > out_hw1(M[0]*R[0]*C[0]);

	std::vector<short, aligned_allocator<short> > inp2(N[1]*(R[1]+K-1)*(C[1]+K-1));
	std::vector<short, aligned_allocator<short> > ker2(M[1]*N[1]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw2(M[1]*R[1]*C[1]);
	std::vector<short, aligned_allocator<short> > out_hw2(M[1]*R[1]*C[1]);

	std::vector<short, aligned_allocator<short> > inp3(N[2]*(R[2]+K-1)*(C[2]+K-1));
	std::vector<short, aligned_allocator<short> > ker3(M[2]*N[2]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw3(M[2]*R[2]*C[2]);
	std::vector<short, aligned_allocator<short> > out_hw3(M[2]*R[2]*C[2]);

	std::vector<short, aligned_allocator<short> > inp4(N[3]*(R[3]+K-1)*(C[3]+K-1));
	std::vector<short, aligned_allocator<short> > ker4(M[3]*N[3]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw4(M[3]*R[3]*C[3]);
	std::vector<short, aligned_allocator<short> > out_hw4(M[3]*R[3]*C[3]);

	std::vector<short, aligned_allocator<short> > inp5(N[4]*(R[4]+K-1)*(C[4]+K-1));
	std::vector<short, aligned_allocator<short> > ker5(M[4]*N[4]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw5(M[4]*R[4]*C[4]);
	std::vector<short, aligned_allocator<short> > out_hw5(M[4]*R[4]*C[4]);


	std::cout << "Loading input data...\n";
	LoadData(R[0], C[0], M[0], N[0], K, inp1, ker1, out_sw1, out_hw1);
	LoadData(R[1], C[1], M[1], N[1], K, inp2, ker2, out_sw2, out_hw2);
	LoadData(R[2], C[2], M[2], N[2], K, inp3, ker3, out_sw3, out_hw3);
	LoadData(R[3], C[3], M[3], N[3], K, inp4, ker4, out_sw4, out_hw4);
	LoadData(R[4], C[4], M[4], N[4], K, inp5, ker5, out_sw5, out_hw5);

	std::cout << "Done.\n";

#pragma omp parallel
{
	int tid = omp_get_thread_num();
	if( tid == 0 ){
		int nthreads = omp_get_num_threads();
		std::cout << "Running CPU CNN with " << nthreads << " threads...\n";
	}
}
/*
#pragma omp parallel for
	// initialize output
	for(int row = 0; row < R; row++) {
		for(int col = 0; col < C; col++) {
			for(int cho = 0; cho < M; cho++) {
				out_sw[cho*(R*C) + row*C + col] = 0;
				out_hw[cho*(R*C) + row*C + col] = 0;
	}}}
*/
	auto start = std::chrono::steady_clock::now();

	cnn_sw(R[0], C[0], M[0], N[0], inp1, ker1, out_sw1);
	cnn_sw(R[1], C[1], M[1], N[1], inp2, ker2, out_sw2);
	cnn_sw(R[2], C[2], M[2], N[2], inp3, ker3, out_sw3);
	cnn_sw(R[3], C[3], M[3], N[3], inp4, ker4, out_sw4);
	cnn_sw(R[4], C[4], M[4], N[4], inp5, ker5, out_sw5);

	auto end = std::chrono::steady_clock::now();
	std::cout << "Done.\n";

	double exec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	
	double gflops1 = double(N[0]) * M[0] * R[0] * C[0] * K * K * 2 / (exec_time);
	double gflops2 = double(N[1]) * M[1] * R[1] * C[1] * K * K * 2 / (exec_time);
	double gflops3 = double(N[2]) * M[2] * R[2] * C[2] * K * K * 2 / (exec_time);
	double gflops4 = double(N[3]) * M[3] * R[3] * C[3] * K * K * 2 / (exec_time);
	double gflops5 = double(N[4]) * M[4] * R[4] * C[4] * K * K * 2 / (exec_time);
	std::cout << "Time: " << exec_time*1e-9 << ", GFLOPS: " << (gflops1 + gflops2 + gflops3 + gflops4 + gflops5) << std::endl;

	// double gflops = double(N) * M * R * C * K * K * 2 / (exec_time);
	// std::cout << "Time: " << exec_time*1e-9 << ", GFLOPS: " << gflops << std::endl;

	std::string binaryFile = argv[1];
	cl_int err;
	cl::Context context;
	cl::Kernel kernel;
	cl::CommandQueue q;


	// OPENCL HOST CODE AREA START
	// get_xil_devices() is a utility API which will find the xilinx
	// platforms and will return list of devices connected to Xilinx platform
	auto devices = xcl::get_xil_devices();
	// read_binary_file() is a utility API which will load the binaryFile
	// and will return the pointer to file buffer.
	auto fileBuf = xcl::read_binary_file(binaryFile);
	cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
	bool valid_device = false;
	for (unsigned int i = 0; i < devices.size(); i++) {
		auto device = devices[i];
		// Creating Context and Command Queue for selected Device
		OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
		OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
		std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		cl::Program program(context, {device}, bins, nullptr, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
		} else {
			std::cout << "Device[" << i << "]: program successful!\n";
			OCL_CHECK(err, kernel = cl::Kernel(program, "cnn", &err));
			valid_device = true;
			break; // we break because we found a valid device
		}
	}
	if (!valid_device) {
		std::cout << "Failed to program any device found, exit!\n";
		exit(EXIT_FAILURE);
	}

	// Allocate Buffer in Global Memory
	OCL_CHECK(err, cl::Buffer buffer_input1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[0]*(R[0]+K-1)*(C[0]+K-1), inp1.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[0]*M[0]*K*K, ker1.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output1(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[0]*R[0]*C[0], out_hw1.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[1]*(R[1]+K-1)*(C[1]+K-1), inp2.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[1]*M[1]*K*K, ker2.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output2(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[1]*R[1]*C[1], out_hw2.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[2]*(R[2]+K-1)*(C[2]+K-1), inp3.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[2]*M[2]*K*K, ker3.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output3(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[2]*R[2]*C[2], out_hw3.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input4(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[3]*(R[3]+K-1)*(C[3]+K-1), inp4.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight4(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[3]*M[3]*K*K, ker4.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output4(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[3]*R[3]*C[3], out_hw4.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input5(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[4]*(R[4]+K-1)*(C[4]+K-1), inp5.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight5(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[4]*M[4]*K*K, ker5.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output5(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[4]*R[4]*C[4], out_hw5.data(), &err));

	OCL_CHECK(err, err = kernel.setArg(0, buffer_input1));
	OCL_CHECK(err, err = kernel.setArg(1, buffer_input2));
	OCL_CHECK(err, err = kernel.setArg(2, buffer_input3));
	OCL_CHECK(err, err = kernel.setArg(3, buffer_input4));
	OCL_CHECK(err, err = kernel.setArg(4, buffer_input5));

	OCL_CHECK(err, err = kernel.setArg(5, buffer_weight1));
	OCL_CHECK(err, err = kernel.setArg(6, buffer_weight2));
	OCL_CHECK(err, err = kernel.setArg(7, buffer_weight3));
	OCL_CHECK(err, err = kernel.setArg(8, buffer_weight4));
	OCL_CHECK(err, err = kernel.setArg(9, buffer_weight5));

	OCL_CHECK(err, err = kernel.setArg(10, buffer_output1));
	OCL_CHECK(err, err = kernel.setArg(11, buffer_output2));
	OCL_CHECK(err, err = kernel.setArg(12, buffer_output3));
	OCL_CHECK(err, err = kernel.setArg(13, buffer_output4));
	OCL_CHECK(err, err = kernel.setArg(14, buffer_output5));


	// Copy input data to device global memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
		{buffer_input1, buffer_input2, buffer_input3, buffer_input4, buffer_input5, 
		buffer_weight1, buffer_weight2, buffer_weight3, buffer_weight4, buffer_weight5, 
		buffer_output1, buffer_output2, buffer_output3, buffer_output4, buffer_output5}, 0 /* 0 means from host*/));

	q.finish();
	
	std::cout << "Running FPGA CNN...\n";
	start = std::chrono::steady_clock::now();

	// Launch the Kernel
	OCL_CHECK(err, err = q.enqueueTask(kernel));
	q.finish();

	end = std::chrono::steady_clock::now();
	std::cout << "Done.\n";

	exec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	gflops1 = double(N[0]) * M[0] * R[0] * C[0] * K * K * 2 / (exec_time);
	gflops2 = double(N[1]) * M[1] * R[1] * C[1] * K * K * 2 / (exec_time);
	gflops3 = double(N[2]) * M[2] * R[2] * C[2] * K * K * 2 / (exec_time);
	gflops4 = double(N[3]) * M[3] * R[3] * C[3] * K * K * 2 / (exec_time);
	gflops5 = double(N[4]) * M[4] * R[4] * C[4] * K * K * 2 / (exec_time);
	std::cout << "Time: " << exec_time*1e-9 << ", GFLOPS: " << (gflops1 + gflops2 + gflops3 + gflops4 + gflops5) << std::endl;

	// Copy Result from Device Global Memory to Host Local Memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output1, buffer_output2, buffer_output3, buffer_output4, buffer_output5}, CL_MIGRATE_MEM_OBJECT_HOST));
	q.finish();
	// OPENCL HOST CODE AREA END
	// Verification
	
	std::cout << "Conv1 verification\n";

	int err_cnt0 = 0;
	int err_cnt1 = 0;
	int err_cnt2 = 0;
	int err_cnt3 = 0;
	int err_cnt4 = 0;

	for(int row = 0; row < R[0]; row++) {
		for(int col = 0; col < C[0]; col++) {
			for(int cho = 0; cho < M[0]; cho++) {
				if(out_sw1[cho*(R[0]*C[0]) + row*C[0] + col] != out_hw1[cho*(R[0]*C[0]) + row*C[0] + col]) {
					err_cnt0++;
					if( err_cnt0 == 1 ){
						printf("cho:%d row:%d col:%d sw:%d hw:%d\n", cho, row, col, out_sw1[cho*(R[0]*C[0]) + row*C[0] + col], out_hw1[cho*(R[0]*C[0]) + row*C[0] + col]);
					}
				}
			}
		}
	}
	printf("Conv1 Error count : %d\n", err_cnt0);

	std::cout << "Conv2 verification\n";
	for(int row = 0; row < R[1]; row++) {
		for(int col = 0; col < C[1]; col++) {
			for(int cho = 0; cho < M[1]; cho++) {
				if(out_sw2[cho*(R[1]*C[1]) + row*C[1] + col] != out_hw2[cho*(R[1]*C[1]) + row*C[1] + col]) {
					err_cnt1++;
					if( err_cnt1 == 1 ){
						printf("cho:%d row:%d col:%d sw:%d hw:%d\n", cho, row, col, out_sw2[cho*(R[1]*C[1]) + row*C[1] + col], out_hw2[cho*(R[1]*C[1]) + row*C[1] + col]);
					}
				}
			}
		}
	}
	printf("Conv2 Error count : %d\n", err_cnt2);
	
	std::cout << "Conv3 verification\n";
	for(int row = 0; row < R[2]; row++) {
		for(int col = 0; col < C[2]; col++) {
			for(int cho = 0; cho < M[2]; cho++) {
				if(out_sw3[cho*(R[2]*C[2]) + row*C[2] + col] != out_hw3[cho*(R[2]*C[2]) + row*C[2] + col]) {
					err_cnt2++;
					if( err_cnt2 == 1 ){
						printf("cho:%d row:%d col:%d sw:%d hw:%d\n", cho, row, col, out_sw3[cho*(R[2]*C[2]) + row*C[2] + col], out_hw3[cho*(R[2]*C[2]) + row*C[2] + col]);
					}
				}
			}
		}
	}
	printf("Conv3 Error count : %d\n", err_cnt2);

	std::cout << "Conv4 verification\n";
	for(int row = 0; row < R[3]; row++) {
		for(int col = 0; col < C[3]; col++) {
			for(int cho = 0; cho < M[3]; cho++) {
				if(out_sw4[cho*(R[3]*C[3]) + row*C[3] + col] != out_hw4[cho*(R[3]*C[3]) + row*C[3] + col]) {
					err_cnt3++;
					if( err_cnt3 == 1 ){
						printf("cho:%d row:%d col:%d sw:%d hw:%d\n", cho, row, col, out_sw4[cho*(R[3]*C[3]) + row*C[3] + col], out_hw4[cho*(R[3]*C[3]) + row*C[3] + col]);
					}
				}
			}
		}
	}
	printf("Conv4 Error count : %d\n", err_cnt3);

	std::cout << "Conv5 verification\n";
	for(int row = 0; row < R[4]; row++) {
		for(int col = 0; col < C[4]; col++) {
			for(int cho = 0; cho < M[4]; cho++) {
				if(out_sw5[cho*(R[4]*C[4]) + row*C[4] + col] != out_hw5[cho*(R[4]*C[4]) + row*C[4] + col]) {
					err_cnt4++;
					if( err_cnt4 == 1 ){
						printf("cho:%d row:%d col:%d sw:%d hw:%d\n", cho, row, col, out_sw5[cho*(R[4]*C[4]) + row*C[4] + col], out_hw5[cho*(R[4]*C[4]) + row*C[4] + col]);
					}
				}
			}
		}
	}
	printf("Conv5 Error count : %d\n", err_cnt4);
	
	int err_cnt = 0;
	err_cnt = err_cnt0 + err_cnt1 + err_cnt2 + err_cnt3 + err_cnt4;

	if (err_cnt != 0){
		printf("FAILED! Error count : %d\n", err_cnt);
	}
	else {
		printf("PASSED!\n");
	}
	
	return EXIT_SUCCESS;
}
