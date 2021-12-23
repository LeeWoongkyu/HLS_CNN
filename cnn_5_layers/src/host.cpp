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
	std::vector<short, aligned_allocator<short> > & inp_hw,
	std::vector<short, aligned_allocator<short> > & ker,
	std::vector<short, aligned_allocator<short> > & out_sw,
	std::vector<short, aligned_allocator<short> > & out_hw,
	std::vector<short, aligned_allocator<short> > & mp,
	std::vector<short, aligned_allocator<short> > & mp_hw
	){

	// initialize input
	for(int row = 0; row < _R+_K-1; row++) {
		for(int col = 0; col < _C+_K-1; col++) {
			for(int chi = 0; chi < _N; chi++) {
				inp[chi*(_R+_K-1)*(_C+_K-1) + row*(_C+_K-1) + col] = 0;
				inp_hw[chi*(_R+_K-1)*(_C+_K-1) + row*(_C+_K-1) + col] = 0;
				/*
				if (row >= (_K-1)/2 && row < _R+(_K-1)/2 && col >= (_K-1)/2 && col < _C+(_K-1)/2) {
					inp[chi*(_R+_K-1)*(_C+_K-1) + row*(_C+_K-1) + col] = 0;
					inp_hw[chi*(_R+_K-1)*(_C+_K-1) + row*(_C+_K-1) + col] = 0;
				}
				else {
					inp[chi*(_R+_K-1)*(_C+_K-1) + row*(_C+_K-1) + col] = 0;
				}
				*/
	}}}
	// initialize kernel
	for(int cho = 0; cho < _M; cho++) {
		for(int chi = 0; chi < _N; chi++) {
          		 	for (int ki = 0; ki < _K; ki++) {
           				for (int kj = 0; kj < _K; kj++) {    
					ker[cho*(_N*_K*_K) + chi*(_K*_K) + ki*_K + kj] = rand() % 8;
	}}}}
	// initialize conv output
	for(int row = 0; row < _R; row++) {
		for(int col = 0; col < _C; col++) {
			for(int cho = 0; cho < _M; cho++) {
				out_sw[cho*(_R*_C) + row*_C + col] = 0;
				out_hw[cho*(_R*_C) + row*_C + col] = 0;
	}}}
	// initialize pooling output
	for(int row = 0; row < _R/2; row++) {
		for(int col = 0; col < _C/2; col++) {
			for(int cho = 0; cho < _M; cho++) {
				mp[cho*(_R/2*_C/2) + row*_C/2 + col] = 0;
				mp_hw[cho*(_R/2*_C/2) + row*_C/2 + col] = 0;
	}}}
}

// copy output to input with padding
void ZeroPad2d(
        int _R, int _C, int _M, int _K,
        std::vector<short, aligned_allocator<short> > & max_pool_out,
        std::vector<short, aligned_allocator<short> > & inp
        ){
#pragma omp parallel for
        for(int row = 0; row < _R; row++) {
                for(int col = 0; col < _C; col++) {
                        for(int cho = 0; cho < _M; cho++) {
							inp[cho*(_R+1)*(_C+1) + (row+1)*(_C+1) + (col+1)] = max_pool_out[cho*_R*_C + row*_C + col];
        }}}
}

// Max Pooling
void MaxPool2d(
	int _R, int _C, int _M, int _S,
	std::vector<short, aligned_allocator<short> > & out,
	std::vector<short, aligned_allocator<short> > & maxpool_out
	){
#pragma omp parallel for
	for(int row = 0; row < _R; row+=_S) {
		for(int col = 0; col < _C; col+=_S) {
			for(int cho = 0; cho < _M; cho++) {
				// initialize maxpool_out value
				short maxpool_out_val = out[cho*_R*_C + row*_C + col];
				for (int si = 0; si < _S; si++) {
					for (int sj = 0; sj < _S; sj++) {
						/*
						if(si == 0 && sj == 0){
							maxpool_out_val = out[cho*_R*_C + (row+si)*_C + (col+sj)];
						}
						*/
						maxpool_out_val = std::max(maxpool_out_val, out[cho*_R*_C + (row+si)*_C + (col+sj)]);
					}
				}
				maxpool_out[cho*(_R/2)*(_C/2) + (row/2)*(_C/2) + (col/2)] = maxpool_out_val;
			}
		}
	}
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
	
	/////////////////////////////////////////////////////////////////////////////////////

	std::vector<short, aligned_allocator<short> > inp11(N[0]*(R[0]+K-1)*(C[0]+K-1));
	std::vector<short, aligned_allocator<short> > inp11_hw(N[0]*(R[0]+K-1)*(C[0]+K-1));
	std::vector<short, aligned_allocator<short> > ker11(M[0]*N[0]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw11(M[0]*R[0]*C[0]);
	std::vector<short, aligned_allocator<short> > out_hw11(M[0]*R[0]*C[0]);
	std::vector<short, aligned_allocator<short> > mp11((M[0])*(R[1])*(C[1]));
	std::vector<short, aligned_allocator<short> > mp11_hw((M[0])*(R[1])*(C[1]));

	std::vector<short, aligned_allocator<short> > inp12(N[1]*(R[1]+K-1)*(C[1]+K-1));
	std::vector<short, aligned_allocator<short> > inp12_hw(N[1]*(R[1]+K-1)*(C[1]+K-1));
	std::vector<short, aligned_allocator<short> > ker12(M[1]*N[1]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw12(M[1]*R[1]*C[1]);
	std::vector<short, aligned_allocator<short> > out_hw12(M[1]*R[1]*C[1]);
	std::vector<short, aligned_allocator<short> > mp12((M[1])*(R[2])*(C[2]));
	std::vector<short, aligned_allocator<short> > mp12_hw((M[1])*(R[2])*(C[2]));

	std::vector<short, aligned_allocator<short> > inp13(N[2]*(R[2]+K-1)*(C[2]+K-1));
	std::vector<short, aligned_allocator<short> > inp13_hw(N[2]*(R[2]+K-1)*(C[2]+K-1));
	std::vector<short, aligned_allocator<short> > ker13(M[2]*N[2]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw13(M[2]*R[2]*C[2]);
	std::vector<short, aligned_allocator<short> > out_hw13(M[2]*R[2]*C[2]);
	std::vector<short, aligned_allocator<short> > mp13((M[2])*(R[3])*(C[3]));
	std::vector<short, aligned_allocator<short> > mp13_hw((M[2])*(R[3])*(C[3]));

	std::vector<short, aligned_allocator<short> > inp14(N[3]*(R[3]+K-1)*(C[3]+K-1));
	std::vector<short, aligned_allocator<short> > inp14_hw(N[3]*(R[3]+K-1)*(C[3]+K-1));
	std::vector<short, aligned_allocator<short> > ker14(M[3]*N[3]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw14(M[3]*R[3]*C[3]);
	std::vector<short, aligned_allocator<short> > out_hw14(M[3]*R[3]*C[3]);
	std::vector<short, aligned_allocator<short> > mp14((M[3])*(R[4])*(C[4]));
	std::vector<short, aligned_allocator<short> > mp14_hw((M[3])*(R[4])*(C[4]));

	std::vector<short, aligned_allocator<short> > inp15(N[4]*(R[4]+K-1)*(C[4]+K-1));
	std::vector<short, aligned_allocator<short> > inp15_hw(N[4]*(R[4]+K-1)*(C[4]+K-1));
	std::vector<short, aligned_allocator<short> > ker15(M[4]*N[4]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw15(M[4]*R[4]*C[4]);
	std::vector<short, aligned_allocator<short> > out_hw15(M[4]*R[4]*C[4]);
	std::vector<short, aligned_allocator<short> > mp15((M[4])*(R[5])*(C[5]));
	std::vector<short, aligned_allocator<short> > mp15_hw((M[4])*(R[5])*(C[5]));

	/////////////////////////////////////////////////////////////////////////////////////

	std::vector<short, aligned_allocator<short> > inp21(N[0]*(R[0]+K-1)*(C[0]+K-1));
	std::vector<short, aligned_allocator<short> > inp21_hw(N[0]*(R[0]+K-1)*(C[0]+K-1));	
	std::vector<short, aligned_allocator<short> > ker21(M[0]*N[0]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw21(M[0]*R[0]*C[0]);
	std::vector<short, aligned_allocator<short> > out_hw21(M[0]*R[0]*C[0]);
	std::vector<short, aligned_allocator<short> > mp21((M[0])*(R[1])*(C[1]));
	std::vector<short, aligned_allocator<short> > mp21_hw((M[0])*(R[1])*(C[1]));

	std::vector<short, aligned_allocator<short> > inp22(N[1]*(R[1]+K-1)*(C[1]+K-1));
	std::vector<short, aligned_allocator<short> > inp22_hw(N[1]*(R[1]+K-1)*(C[1]+K-1));
	std::vector<short, aligned_allocator<short> > ker22(M[1]*N[1]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw22(M[1]*R[1]*C[1]);
	std::vector<short, aligned_allocator<short> > out_hw22(M[1]*R[1]*C[1]);
	std::vector<short, aligned_allocator<short> > mp22((M[1])*(R[2])*(C[2]));
	std::vector<short, aligned_allocator<short> > mp22_hw((M[1])*(R[2])*(C[2]));

	std::vector<short, aligned_allocator<short> > inp23(N[2]*(R[2]+K-1)*(C[2]+K-1));
	std::vector<short, aligned_allocator<short> > inp23_hw(N[2]*(R[2]+K-1)*(C[2]+K-1));
	std::vector<short, aligned_allocator<short> > ker23(M[2]*N[2]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw23(M[2]*R[2]*C[2]);
	std::vector<short, aligned_allocator<short> > out_hw23(M[2]*R[2]*C[2]);
	std::vector<short, aligned_allocator<short> > mp23((M[2])*(R[3])*(C[3]));
	std::vector<short, aligned_allocator<short> > mp23_hw((M[2])*(R[3])*(C[3]));

	std::vector<short, aligned_allocator<short> > inp24(N[3]*(R[3]+K-1)*(C[3]+K-1));
	std::vector<short, aligned_allocator<short> > inp24_hw(N[3]*(R[3]+K-1)*(C[3]+K-1));
	std::vector<short, aligned_allocator<short> > ker24(M[3]*N[3]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw24(M[3]*R[3]*C[3]);
	std::vector<short, aligned_allocator<short> > out_hw24(M[3]*R[3]*C[3]);
	std::vector<short, aligned_allocator<short> > mp24((M[3])*(R[4])*(C[4]));
	std::vector<short, aligned_allocator<short> > mp24_hw((M[3])*(R[4])*(C[4]));

	std::vector<short, aligned_allocator<short> > inp25(N[4]*(R[4]+K-1)*(C[4]+K-1));
	std::vector<short, aligned_allocator<short> > inp25_hw(N[4]*(R[4]+K-1)*(C[4]+K-1));
	std::vector<short, aligned_allocator<short> > ker25(M[4]*N[4]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw25(M[4]*R[4]*C[4]);
	std::vector<short, aligned_allocator<short> > out_hw25(M[4]*R[4]*C[4]);
	std::vector<short, aligned_allocator<short> > mp25((M[4])*(R[5])*(C[5]));
	std::vector<short, aligned_allocator<short> > mp25_hw((M[4])*(R[5])*(C[5]));

	/////////////////////////////////////////////////////////////////////////////////////

	std::vector<short, aligned_allocator<short> > inp31(N[0]*(R[0]+K-1)*(C[0]+K-1));
	std::vector<short, aligned_allocator<short> > inp31_hw(N[0]*(R[0]+K-1)*(C[0]+K-1));
	std::vector<short, aligned_allocator<short> > ker31(M[0]*N[0]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw31(M[0]*R[0]*C[0]);
	std::vector<short, aligned_allocator<short> > out_hw31(M[0]*R[0]*C[0]);
	std::vector<short, aligned_allocator<short> > mp31((M[0])*(R[1])*(C[1]));
	std::vector<short, aligned_allocator<short> > mp31_hw((M[0])*(R[1])*(C[1]));

	std::vector<short, aligned_allocator<short> > inp32(N[1]*(R[1]+K-1)*(C[1]+K-1));
	std::vector<short, aligned_allocator<short> > inp32_hw(N[1]*(R[1]+K-1)*(C[1]+K-1));
	std::vector<short, aligned_allocator<short> > ker32(M[1]*N[1]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw32(M[1]*R[1]*C[1]);
	std::vector<short, aligned_allocator<short> > out_hw32(M[1]*R[1]*C[1]);
	std::vector<short, aligned_allocator<short> > mp32((M[1])*(R[2])*(C[2]));
	std::vector<short, aligned_allocator<short> > mp32_hw((M[1])*(R[2])*(C[2]));

	std::vector<short, aligned_allocator<short> > inp33(N[2]*(R[2]+K-1)*(C[2]+K-1));
	std::vector<short, aligned_allocator<short> > inp33_hw(N[2]*(R[2]+K-1)*(C[2]+K-1));
	std::vector<short, aligned_allocator<short> > ker33(M[2]*N[2]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw33(M[2]*R[2]*C[2]);
	std::vector<short, aligned_allocator<short> > out_hw33(M[2]*R[2]*C[2]);
	std::vector<short, aligned_allocator<short> > mp33((M[2])*(R[3])*(C[3]));
	std::vector<short, aligned_allocator<short> > mp33_hw((M[2])*(R[3])*(C[3]));

	std::vector<short, aligned_allocator<short> > inp34(N[3]*(R[3]+K-1)*(C[3]+K-1));
	std::vector<short, aligned_allocator<short> > inp34_hw(N[3]*(R[3]+K-1)*(C[3]+K-1));
	std::vector<short, aligned_allocator<short> > ker34(M[3]*N[3]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw34(M[3]*R[3]*C[3]);
	std::vector<short, aligned_allocator<short> > out_hw34(M[3]*R[3]*C[3]);
	std::vector<short, aligned_allocator<short> > mp34((M[3])*(R[4])*(C[4]));
	std::vector<short, aligned_allocator<short> > mp34_hw((M[3])*(R[4])*(C[4]));

	std::vector<short, aligned_allocator<short> > inp35(N[4]*(R[4]+K-1)*(C[4]+K-1));
	std::vector<short, aligned_allocator<short> > inp35_hw(N[4]*(R[4]+K-1)*(C[4]+K-1));	
	std::vector<short, aligned_allocator<short> > ker35(M[4]*N[4]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw35(M[4]*R[4]*C[4]);
	std::vector<short, aligned_allocator<short> > out_hw35(M[4]*R[4]*C[4]);
	std::vector<short, aligned_allocator<short> > mp35((M[4])*(R[5])*(C[5]));
	std::vector<short, aligned_allocator<short> > mp35_hw((M[4])*(R[5])*(C[5]));

	/////////////////////////////////////////////////////////////////////////////////////

	std::vector<short, aligned_allocator<short> > inp41(N[0]*(R[0]+K-1)*(C[0]+K-1));
	std::vector<short, aligned_allocator<short> > inp41_hw(N[0]*(R[0]+K-1)*(C[0]+K-1));
	std::vector<short, aligned_allocator<short> > ker41(M[0]*N[0]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw41(M[0]*R[0]*C[0]);
	std::vector<short, aligned_allocator<short> > out_hw41(M[0]*R[0]*C[0]);
	std::vector<short, aligned_allocator<short> > mp41((M[0])*(R[1])*(C[1]));
	std::vector<short, aligned_allocator<short> > mp41_hw((M[0])*(R[1])*(C[1]));

	std::vector<short, aligned_allocator<short> > inp42(N[1]*(R[1]+K-1)*(C[1]+K-1));
	std::vector<short, aligned_allocator<short> > inp42_hw(N[1]*(R[1]+K-1)*(C[1]+K-1));
	std::vector<short, aligned_allocator<short> > ker42(M[1]*N[1]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw42(M[1]*R[1]*C[1]);
	std::vector<short, aligned_allocator<short> > out_hw42(M[1]*R[1]*C[1]);
	std::vector<short, aligned_allocator<short> > mp42((M[1])*(R[2])*(C[2]));
	std::vector<short, aligned_allocator<short> > mp42_hw((M[1])*(R[2])*(C[2]));

	std::vector<short, aligned_allocator<short> > inp43(N[2]*(R[2]+K-1)*(C[2]+K-1));
	std::vector<short, aligned_allocator<short> > inp43_hw(N[2]*(R[2]+K-1)*(C[2]+K-1));
	std::vector<short, aligned_allocator<short> > ker43(M[2]*N[2]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw43(M[2]*R[2]*C[2]);
	std::vector<short, aligned_allocator<short> > out_hw43(M[2]*R[2]*C[2]);
	std::vector<short, aligned_allocator<short> > mp43((M[2])*(R[3])*(C[3]));
	std::vector<short, aligned_allocator<short> > mp43_hw((M[2])*(R[3])*(C[3]));

	std::vector<short, aligned_allocator<short> > inp44(N[3]*(R[3]+K-1)*(C[3]+K-1));
	std::vector<short, aligned_allocator<short> > inp44_hw(N[3]*(R[3]+K-1)*(C[3]+K-1));
	std::vector<short, aligned_allocator<short> > ker44(M[3]*N[3]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw44(M[3]*R[3]*C[3]);
	std::vector<short, aligned_allocator<short> > out_hw44(M[3]*R[3]*C[3]);
	std::vector<short, aligned_allocator<short> > mp44((M[3])*(R[4])*(C[4]));
	std::vector<short, aligned_allocator<short> > mp44_hw((M[3])*(R[4])*(C[4]));

	std::vector<short, aligned_allocator<short> > inp45(N[4]*(R[4]+K-1)*(C[4]+K-1));
	std::vector<short, aligned_allocator<short> > inp45_hw(N[4]*(R[4]+K-1)*(C[4]+K-1));
	std::vector<short, aligned_allocator<short> > ker45(M[4]*N[4]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw45(M[4]*R[4]*C[4]);
	std::vector<short, aligned_allocator<short> > out_hw45(M[4]*R[4]*C[4]);
	std::vector<short, aligned_allocator<short> > mp45((M[4])*(R[5])*(C[5]));
	std::vector<short, aligned_allocator<short> > mp45_hw((M[4])*(R[5])*(C[5]));

	/////////////////////////////////////////////////////////////////////////////////////

	std::vector<short, aligned_allocator<short> > inp51(N[0]*(R[0]+K-1)*(C[0]+K-1));
	std::vector<short, aligned_allocator<short> > inp51_hw(N[0]*(R[0]+K-1)*(C[0]+K-1));
	std::vector<short, aligned_allocator<short> > ker51(M[0]*N[0]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw51(M[0]*R[0]*C[0]);
	std::vector<short, aligned_allocator<short> > out_hw51(M[0]*R[0]*C[0]);
	std::vector<short, aligned_allocator<short> > mp51((M[0])*(R[1])*(C[1]));
	std::vector<short, aligned_allocator<short> > mp51_hw((M[0])*(R[1])*(C[1]));

	std::vector<short, aligned_allocator<short> > inp52(N[1]*(R[1]+K-1)*(C[1]+K-1));
	std::vector<short, aligned_allocator<short> > inp52_hw(N[1]*(R[1]+K-1)*(C[1]+K-1));
	std::vector<short, aligned_allocator<short> > ker52(M[1]*N[1]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw52(M[1]*R[1]*C[1]);
	std::vector<short, aligned_allocator<short> > out_hw52(M[1]*R[1]*C[1]);
	std::vector<short, aligned_allocator<short> > mp52((M[1])*(R[2])*(C[2]));
	std::vector<short, aligned_allocator<short> > mp52_hw((M[1])*(R[2])*(C[2]));

	std::vector<short, aligned_allocator<short> > inp53(N[2]*(R[2]+K-1)*(C[2]+K-1));
	std::vector<short, aligned_allocator<short> > inp53_hw(N[2]*(R[2]+K-1)*(C[2]+K-1));
	std::vector<short, aligned_allocator<short> > ker53(M[2]*N[2]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw53(M[2]*R[2]*C[2]);
	std::vector<short, aligned_allocator<short> > out_hw53(M[2]*R[2]*C[2]);
	std::vector<short, aligned_allocator<short> > mp53((M[2])*(R[3])*(C[3]));
	std::vector<short, aligned_allocator<short> > mp53_hw((M[2])*(R[3])*(C[3]));

	std::vector<short, aligned_allocator<short> > inp54(N[3]*(R[3]+K-1)*(C[3]+K-1));
	std::vector<short, aligned_allocator<short> > inp54_hw(N[3]*(R[3]+K-1)*(C[3]+K-1));
	std::vector<short, aligned_allocator<short> > ker54(M[3]*N[3]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw54(M[3]*R[3]*C[3]);
	std::vector<short, aligned_allocator<short> > out_hw54(M[3]*R[3]*C[3]);
	std::vector<short, aligned_allocator<short> > mp54((M[3])*(R[4])*(C[4]));
	std::vector<short, aligned_allocator<short> > mp54_hw((M[3])*(R[4])*(C[4]));

	std::vector<short, aligned_allocator<short> > inp55(N[4]*(R[4]+K-1)*(C[4]+K-1));
	std::vector<short, aligned_allocator<short> > inp55_hw(N[4]*(R[4]+K-1)*(C[4]+K-1));
	std::vector<short, aligned_allocator<short> > ker55(M[4]*N[4]*K*K);
	std::vector<short, aligned_allocator<short> > out_sw55(M[4]*R[4]*C[4]);
	std::vector<short, aligned_allocator<short> > out_hw55(M[4]*R[4]*C[4]);
	std::vector<short, aligned_allocator<short> > mp55((M[4])*(R[5])*(C[5]));
	std::vector<short, aligned_allocator<short> > mp55_hw((M[4])*(R[5])*(C[5]));

	/////////////////////////////////////////////////////////////////////////////////////

	std::cout << "Loading input data...\n";
	LoadData(R[0], C[0], M[0], N[0], K, inp11, inp11_hw, ker11, out_sw11, out_hw11, mp11, mp11_hw);
	LoadData(R[1], C[1], M[1], N[1], K, inp12, inp12_hw,  ker12, out_sw12, out_hw12, mp12, mp12_hw);
	LoadData(R[2], C[2], M[2], N[2], K, inp13, inp13_hw, ker13, out_sw13, out_hw13, mp13, mp13_hw);
	LoadData(R[3], C[3], M[3], N[3], K, inp14, inp14_hw, ker14, out_sw14, out_hw14, mp14, mp14_hw);
	LoadData(R[4], C[4], M[4], N[4], K, inp15, inp15_hw, ker15, out_sw15, out_hw15, mp15, mp15_hw);
	
	LoadData(R[0], C[0], M[0], N[0], K, inp21, inp21_hw, ker21, out_sw21, out_hw21, mp21, mp21_hw);
	LoadData(R[1], C[1], M[1], N[1], K, inp22, inp22_hw, ker22, out_sw22, out_hw22, mp22, mp22_hw);
	LoadData(R[2], C[2], M[2], N[2], K, inp23, inp23_hw, ker23, out_sw23, out_hw23, mp23, mp23_hw);
	LoadData(R[3], C[3], M[3], N[3], K, inp24, inp24_hw, ker24, out_sw24, out_hw24, mp24, mp24_hw);
	LoadData(R[4], C[4], M[4], N[4], K, inp25, inp25_hw, ker25, out_sw25, out_hw25, mp25, mp25_hw);

	LoadData(R[0], C[0], M[0], N[0], K, inp31, inp31_hw, ker31, out_sw31, out_hw31, mp31, mp31_hw);
	LoadData(R[1], C[1], M[1], N[1], K, inp32, inp32_hw, ker32, out_sw32, out_hw32, mp32, mp32_hw);
	LoadData(R[2], C[2], M[2], N[2], K, inp33, inp33_hw, ker33, out_sw33, out_hw33, mp33, mp33_hw);
	LoadData(R[3], C[3], M[3], N[3], K, inp34, inp34_hw, ker34, out_sw34, out_hw34, mp34, mp34_hw);
	LoadData(R[4], C[4], M[4], N[4], K, inp35, inp35_hw, ker35, out_sw35, out_hw35, mp35, mp35_hw);

	LoadData(R[0], C[0], M[0], N[0], K, inp41, inp41_hw, ker41, out_sw41, out_hw41, mp41, mp41_hw);
	LoadData(R[1], C[1], M[1], N[1], K, inp42, inp42_hw, ker42, out_sw42, out_hw42, mp42, mp42_hw);
	LoadData(R[2], C[2], M[2], N[2], K, inp43, inp43_hw, ker43, out_sw43, out_hw43, mp43, mp43_hw);
	LoadData(R[3], C[3], M[3], N[3], K, inp44, inp44_hw, ker44, out_sw44, out_hw44, mp44, mp44_hw);
	LoadData(R[4], C[4], M[4], N[4], K, inp45, inp45_hw, ker45, out_sw45, out_hw45, mp45, mp45_hw);

	LoadData(R[0], C[0], M[0], N[0], K, inp51, inp51_hw, ker51, out_sw51, out_hw51, mp51, mp51_hw);
	LoadData(R[1], C[1], M[1], N[1], K, inp52, inp52_hw, ker52, out_sw52, out_hw52, mp52, mp52_hw);
	LoadData(R[2], C[2], M[2], N[2], K, inp53, inp53_hw, ker53, out_sw53, out_hw53, mp53, mp53_hw);
	LoadData(R[3], C[3], M[3], N[3], K, inp54, inp54_hw, ker54, out_sw54, out_hw54, mp54, mp54_hw);
	LoadData(R[4], C[4], M[4], N[4], K, inp55, inp55_hw, ker55, out_sw55, out_hw55, mp55, mp55_hw);

	std::cout << "Done.\n";

#pragma omp parallel
{
	int tid = omp_get_thread_num();
	if( tid == 0 ){
		int nthreads = omp_get_num_threads();
		std::cout << "Running CPU CNN with " << nthreads << " threads...\n";
	}
}

#pragma omp parallel for
	// initialize input
	for(int row = 0; row < R[0]+K-1; row++) {
		for(int col = 0; col < C[0]+K-1; col++) {
			for(int chi = 0; chi < N[0]; chi++) {
				if (row >= (K-1)/2 && row < R[0]+(K-1)/2 && col >= (K-1)/2 && col < C[0]+(K-1)/2) {
					inp11[chi*(R[0]+K-1)*(C[0]+K-1) + row*(C[0]+K-1) + col] = rand() % 8;
					inp11_hw[chi*(R[0]+K-1)*(C[0]+K-1) + row*(C[0]+K-1) + col] = inp11[chi*(R[0]+K-1)*(C[0]+K-1) + row*(C[0]+K-1) + col];
				}
				else {
					inp11[chi*(R[0]+K-1)*(C[0]+K-1) + row*(C[0]+K-1) + col] = 0;
					inp11_hw[chi*(R[0]+K-1)*(C[0]+K-1) + row*(C[0]+K-1) + col] = inp11[chi*(R[0]+K-1)*(C[0]+K-1) + row*(C[0]+K-1) + col];
				}
			}
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

	// Step 1

	cnn_sw(R[0], C[0], M[0], N[0], inp11, ker11, out_sw11);
	cnn_sw(R[1], C[1], M[1], N[1], inp12, ker12, out_sw12);
	cnn_sw(R[2], C[2], M[2], N[2], inp13, ker13, out_sw13);
	cnn_sw(R[3], C[3], M[3], N[3], inp14, ker14, out_sw14);
	cnn_sw(R[4], C[4], M[4], N[4], inp15, ker15, out_sw15);

	MaxPool2d(R[0], C[0], M[0], 2, out_sw11, mp11);
	MaxPool2d(R[1], C[1], M[1], 2, out_sw12, mp12);
	MaxPool2d(R[2], C[2], M[2], 2, out_sw13, mp13);
	MaxPool2d(R[3], C[3], M[3], 2, out_sw14, mp14);
	MaxPool2d(R[4], C[4], M[4], 2, out_sw15, mp15);

	// Step 2

	ZeroPad2d(R[1], C[1], M[1], K, mp11, inp22);
	ZeroPad2d(R[2], C[2], M[2], K, mp12, inp23);
	ZeroPad2d(R[3], C[3], M[3], K, mp13, inp24);
	ZeroPad2d(R[4], C[4], M[4], K, mp14, inp25);

	cnn_sw(R[0], C[0], M[0], N[0], inp21, ker21, out_sw21);
	cnn_sw(R[1], C[1], M[1], N[1], inp22, ker22, out_sw22);
	cnn_sw(R[2], C[2], M[2], N[2], inp23, ker23, out_sw23);
	cnn_sw(R[3], C[3], M[3], N[3], inp24, ker24, out_sw24);
	cnn_sw(R[4], C[4], M[4], N[4], inp25, ker25, out_sw25);

	MaxPool2d(R[0], C[0], M[0], 2, out_sw21, mp21);
	MaxPool2d(R[1], C[1], M[1], 2, out_sw22, mp22);
	MaxPool2d(R[2], C[2], M[2], 2, out_sw23, mp23);
	MaxPool2d(R[3], C[3], M[3], 2, out_sw24, mp24);
	MaxPool2d(R[4], C[4], M[4], 2, out_sw25, mp25);

	// Step 3

	ZeroPad2d(R[1], C[1], M[0], K, mp21, inp32);
	ZeroPad2d(R[2], C[2], M[1], K, mp22, inp33);
	ZeroPad2d(R[3], C[3], M[2], K, mp23, inp34);
	ZeroPad2d(R[4], C[4], M[3], K, mp24, inp35);

	cnn_sw(R[0], C[0], M[0], N[0], inp31, ker31, out_sw31);
	cnn_sw(R[1], C[1], M[1], N[1], inp32, ker32, out_sw32);
	cnn_sw(R[2], C[2], M[2], N[2], inp33, ker33, out_sw33);
	cnn_sw(R[3], C[3], M[3], N[3], inp34, ker34, out_sw34);
	cnn_sw(R[4], C[4], M[4], N[4], inp35, ker35, out_sw35);

	MaxPool2d(R[0], C[0], M[0], 2, out_sw31, mp31);
	MaxPool2d(R[1], C[1], M[1], 2, out_sw32, mp32);
	MaxPool2d(R[2], C[2], M[2], 2, out_sw33, mp33);
	MaxPool2d(R[3], C[3], M[3], 2, out_sw34, mp34);
	MaxPool2d(R[4], C[4], M[4], 2, out_sw35, mp35);

	// Step 4

	ZeroPad2d(R[1], C[1], M[0], K, mp31, inp42);
	ZeroPad2d(R[2], C[2], M[1], K, mp32, inp43);
	ZeroPad2d(R[3], C[3], M[2], K, mp33, inp44);
	ZeroPad2d(R[4], C[4], M[3], K, mp34, inp45);

	cnn_sw(R[0], C[0], M[0], N[0], inp41, ker41, out_sw41);
	cnn_sw(R[1], C[1], M[1], N[1], inp42, ker42, out_sw42);
	cnn_sw(R[2], C[2], M[2], N[2], inp43, ker43, out_sw43);
	cnn_sw(R[3], C[3], M[3], N[3], inp44, ker44, out_sw44);
	cnn_sw(R[4], C[4], M[4], N[4], inp45, ker45, out_sw45);

	MaxPool2d(R[0], C[0], M[0], 2, out_sw41, mp41);
	MaxPool2d(R[1], C[1], M[1], 2, out_sw42, mp42);
	MaxPool2d(R[2], C[2], M[2], 2, out_sw43, mp43);
	MaxPool2d(R[3], C[3], M[3], 2, out_sw44, mp44);
	MaxPool2d(R[4], C[4], M[4], 2, out_sw45, mp45);

	// Step 5

	ZeroPad2d(R[1], C[1], M[0], K, mp41, inp52);
	ZeroPad2d(R[2], C[2], M[1], K, mp42, inp53);
	ZeroPad2d(R[3], C[3], M[2], K, mp43, inp54);
	ZeroPad2d(R[4], C[4], M[3], K, mp44, inp55);

	cnn_sw(R[0], C[0], M[0], N[0], inp51, ker51, out_sw51);
	cnn_sw(R[1], C[1], M[1], N[1], inp52, ker52, out_sw52);
	cnn_sw(R[2], C[2], M[2], N[2], inp53, ker53, out_sw53);
	cnn_sw(R[3], C[3], M[3], N[3], inp54, ker54, out_sw54);
	cnn_sw(R[4], C[4], M[4], N[4], inp55, ker55, out_sw55);

	MaxPool2d(R[0], C[0], M[0], 2, out_sw51, mp51);
	MaxPool2d(R[1], C[1], M[1], 2, out_sw52, mp52);
	MaxPool2d(R[2], C[2], M[2], 2, out_sw53, mp53);
	MaxPool2d(R[3], C[3], M[3], 2, out_sw54, mp54);
	MaxPool2d(R[4], C[4], M[4], 2, out_sw55, mp55);
	
	auto end = std::chrono::steady_clock::now();
	std::cout << "Done.\n";

	double exec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	/*
	double gflops1 = double(N[0]) * M[0] * R[0] * C[0] * K * K * 2 / (exec_time);
	double gflops2 = double(N[1]) * M[1] * R[1] * C[1] * K * K * 2 / (exec_time);
	double gflops3 = double(N[2]) * M[2] * R[2] * C[2] * K * K * 2 / (exec_time);
	double gflops4 = double(N[3]) * M[3] * R[3] * C[3] * K * K * 2 / (exec_time);
	double gflops5 = double(N[4]) * M[4] * R[4] * C[4] * K * K * 2 / (exec_time);
	std::cout << "Time: " << exec_time*1e-9 << ", GFLOPS: " << (gflops1 + gflops2 + gflops3 + gflops4 + gflops5) << std::endl;
	*/
	std::cout << "Time: " << exec_time*1e-9 << std::endl;

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


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	std::cout << "Running FPGA CNN...\n";
	auto start0 = std::chrono::steady_clock::now();

	// Step 1 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	std::cout << "******************* Step 1 *******************\n";

	// Allocate Buffer in Global Memory
	OCL_CHECK(err, cl::Buffer buffer_input11(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[0]*(R[0]+K-1)*(C[0]+K-1), inp11_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight11(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[0]*M[0]*K*K, ker11.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output11(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[0]*R[0]*C[0], out_hw11.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input12(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[1]*(R[1]+K-1)*(C[1]+K-1), inp12_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight12(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[1]*M[1]*K*K, ker12.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output12(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[1]*R[1]*C[1], out_hw12.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input13(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[2]*(R[2]+K-1)*(C[2]+K-1), inp13_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight13(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[2]*M[2]*K*K, ker13.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output13(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[2]*R[2]*C[2], out_hw13.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input14(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[3]*(R[3]+K-1)*(C[3]+K-1), inp14_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight14(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[3]*M[3]*K*K, ker14.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output14(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[3]*R[3]*C[3], out_hw14.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input15(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[4]*(R[4]+K-1)*(C[4]+K-1), inp15_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight15(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[4]*M[4]*K*K, ker15.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output15(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[4]*R[4]*C[4], out_hw15.data(), &err));

	OCL_CHECK(err, err = kernel.setArg(0, buffer_input11));
	OCL_CHECK(err, err = kernel.setArg(1, buffer_input12));
	OCL_CHECK(err, err = kernel.setArg(2, buffer_input13));
	OCL_CHECK(err, err = kernel.setArg(3, buffer_input14));
	OCL_CHECK(err, err = kernel.setArg(4, buffer_input15));

	OCL_CHECK(err, err = kernel.setArg(5, buffer_weight11));
	OCL_CHECK(err, err = kernel.setArg(6, buffer_weight12));
	OCL_CHECK(err, err = kernel.setArg(7, buffer_weight13));
	OCL_CHECK(err, err = kernel.setArg(8, buffer_weight14));
	OCL_CHECK(err, err = kernel.setArg(9, buffer_weight15));

	OCL_CHECK(err, err = kernel.setArg(10, buffer_output11));
	OCL_CHECK(err, err = kernel.setArg(11, buffer_output12));
	OCL_CHECK(err, err = kernel.setArg(12, buffer_output13));
	OCL_CHECK(err, err = kernel.setArg(13, buffer_output14));
	OCL_CHECK(err, err = kernel.setArg(14, buffer_output15));

	// Copy input data to device global memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
		{buffer_input11, buffer_input12, buffer_input13, buffer_input14, buffer_input15, 
		buffer_weight11, buffer_weight12, buffer_weight13, buffer_weight14, buffer_weight15, 
		buffer_output11, buffer_output12, buffer_output13, buffer_output14, buffer_output15}, 0 /* 0 means from host*/));

	q.finish();
	
	std::cout << "Convolution...\n";
	// Launch the Kernel
	OCL_CHECK(err, err = q.enqueueTask(kernel));
	q.finish();

	// Copy Result from Device Global Memory to Host Local Memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output11, buffer_output12, buffer_output13, buffer_output14, buffer_output15}, CL_MIGRATE_MEM_OBJECT_HOST));
	q.finish();

	std::cout << "Max pooling...\n";
	MaxPool2d(R[0], C[0], M[0], 2, out_hw11, mp11_hw);
	MaxPool2d(R[1], C[1], M[1], 2, out_hw12, mp12_hw);
	MaxPool2d(R[2], C[2], M[2], 2, out_hw13, mp13_hw);
	MaxPool2d(R[3], C[3], M[3], 2, out_hw14, mp14_hw);
	MaxPool2d(R[4], C[4], M[4], 2, out_hw15, mp15_hw);

	// Step 2 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	std::cout << "******************* Step 2 *******************\n";
	std::cout << "Zero padding...\n";
	ZeroPad2d(R[1], C[1], M[0], K, mp11_hw, inp22_hw);
	ZeroPad2d(R[2], C[2], M[1], K, mp12_hw, inp23_hw);
	ZeroPad2d(R[3], C[3], M[2], K, mp13_hw, inp24_hw);
	ZeroPad2d(R[4], C[4], M[3], K, mp14_hw, inp25_hw);

	// Allocate Buffer in Global Memory
	OCL_CHECK(err, cl::Buffer buffer_input21(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[0]*(R[0]+K-1)*(C[0]+K-1), inp21_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight21(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[0]*M[0]*K*K, ker21.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output21(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[0]*R[0]*C[0], out_hw21.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input22(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[1]*(R[1]+K-1)*(C[1]+K-1), inp22.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight22(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[1]*M[1]*K*K, ker22.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output22(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[1]*R[1]*C[1], out_hw22.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input23(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[2]*(R[2]+K-1)*(C[2]+K-1), inp23_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight23(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[2]*M[2]*K*K, ker23.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output23(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[2]*R[2]*C[2], out_hw23.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input24(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[3]*(R[3]+K-1)*(C[3]+K-1), inp24_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight24(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[3]*M[3]*K*K, ker24.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output24(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[3]*R[3]*C[3], out_hw24.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input25(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[4]*(R[4]+K-1)*(C[4]+K-1), inp25_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight25(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[4]*M[4]*K*K, ker25.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output25(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[4]*R[4]*C[4], out_hw25.data(), &err));

	OCL_CHECK(err, err = kernel.setArg(0, buffer_input21));
	OCL_CHECK(err, err = kernel.setArg(1, buffer_input22));
	OCL_CHECK(err, err = kernel.setArg(2, buffer_input23));
	OCL_CHECK(err, err = kernel.setArg(3, buffer_input24));
	OCL_CHECK(err, err = kernel.setArg(4, buffer_input25));

	OCL_CHECK(err, err = kernel.setArg(5, buffer_weight21));
	OCL_CHECK(err, err = kernel.setArg(6, buffer_weight22));
	OCL_CHECK(err, err = kernel.setArg(7, buffer_weight23));
	OCL_CHECK(err, err = kernel.setArg(8, buffer_weight24));
	OCL_CHECK(err, err = kernel.setArg(9, buffer_weight25));

	OCL_CHECK(err, err = kernel.setArg(10, buffer_output21));
	OCL_CHECK(err, err = kernel.setArg(11, buffer_output22));
	OCL_CHECK(err, err = kernel.setArg(12, buffer_output23));
	OCL_CHECK(err, err = kernel.setArg(13, buffer_output24));
	OCL_CHECK(err, err = kernel.setArg(14, buffer_output25));

	// Copy input data to device global memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
		{buffer_input21, buffer_input22, buffer_input23, buffer_input24, buffer_input25, 
		buffer_weight21, buffer_weight22, buffer_weight23, buffer_weight24, buffer_weight25, 
		buffer_output21, buffer_output22, buffer_output23, buffer_output24, buffer_output25}, 0 /* 0 means from host*/));

	q.finish();
	
	start = std::chrono::steady_clock::now();
	std::cout << "Convolution...\n";
	// Launch the Kernel
	OCL_CHECK(err, err = q.enqueueTask(kernel));
	q.finish();

	// Copy Result from Device Global Memory to Host Local Memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output21, buffer_output22, buffer_output23, buffer_output24, buffer_output25}, CL_MIGRATE_MEM_OBJECT_HOST));
	q.finish();

	std::cout << "Max pooling...\n";
	MaxPool2d(R[0], C[0], M[0], 2, out_hw21, mp21_hw);
	MaxPool2d(R[1], C[1], M[1], 2, out_hw22, mp22_hw);
	MaxPool2d(R[2], C[2], M[2], 2, out_hw23, mp23_hw);
	MaxPool2d(R[3], C[3], M[3], 2, out_hw24, mp24_hw);
	MaxPool2d(R[4], C[4], M[4], 2, out_hw25, mp25_hw);

	// Step 3 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	std::cout << "******************* Step 3 *******************\n";
	std::cout << "Zero padding...\n";
	ZeroPad2d(R[1], C[1], M[0], K, mp21_hw, inp32_hw);
	ZeroPad2d(R[2], C[2], M[1], K, mp22_hw, inp33_hw);
	ZeroPad2d(R[3], C[3], M[2], K, mp23_hw, inp34_hw);
	ZeroPad2d(R[4], C[4], M[3], K, mp24_hw, inp35_hw);

	// Allocate Buffer in Global Memory
	OCL_CHECK(err, cl::Buffer buffer_input31(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[0]*(R[0]+K-1)*(C[0]+K-1), inp31_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight31(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[0]*M[0]*K*K, ker31.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output31(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[0]*R[0]*C[0], out_hw31.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input32(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[1]*(R[1]+K-1)*(C[1]+K-1), inp32_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight32(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[1]*M[1]*K*K, ker32.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output32(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[1]*R[1]*C[1], out_hw32.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input33(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[2]*(R[2]+K-1)*(C[2]+K-1), inp33_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight33(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[2]*M[2]*K*K, ker33.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output33(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[2]*R[2]*C[2], out_hw33.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input34(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[3]*(R[3]+K-1)*(C[3]+K-1), inp34_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight34(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[3]*M[3]*K*K, ker34.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output34(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[3]*R[3]*C[3], out_hw34.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input35(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[4]*(R[4]+K-1)*(C[4]+K-1), inp35_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight35(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[4]*M[4]*K*K, ker35.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output35(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[4]*R[4]*C[4], out_hw35.data(), &err));

	OCL_CHECK(err, err = kernel.setArg(0, buffer_input31));
	OCL_CHECK(err, err = kernel.setArg(1, buffer_input32));
	OCL_CHECK(err, err = kernel.setArg(2, buffer_input33));
	OCL_CHECK(err, err = kernel.setArg(3, buffer_input34));
	OCL_CHECK(err, err = kernel.setArg(4, buffer_input35));

	OCL_CHECK(err, err = kernel.setArg(5, buffer_weight31));
	OCL_CHECK(err, err = kernel.setArg(6, buffer_weight32));
	OCL_CHECK(err, err = kernel.setArg(7, buffer_weight33));
	OCL_CHECK(err, err = kernel.setArg(8, buffer_weight34));
	OCL_CHECK(err, err = kernel.setArg(9, buffer_weight35));

	OCL_CHECK(err, err = kernel.setArg(10, buffer_output31));
	OCL_CHECK(err, err = kernel.setArg(11, buffer_output32));
	OCL_CHECK(err, err = kernel.setArg(12, buffer_output33));
	OCL_CHECK(err, err = kernel.setArg(13, buffer_output34));
	OCL_CHECK(err, err = kernel.setArg(14, buffer_output35));

	// Copy input data to device global memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
		{buffer_input31, buffer_input32, buffer_input33, buffer_input34, buffer_input35, 
		buffer_weight31, buffer_weight32, buffer_weight33, buffer_weight34, buffer_weight35, 
		buffer_output31, buffer_output32, buffer_output33, buffer_output34, buffer_output35}, 0 /* 0 means from host*/));

	q.finish();
	
	start = std::chrono::steady_clock::now();
	std::cout << "Convolution...\n";
	// Launch the Kernel
	OCL_CHECK(err, err = q.enqueueTask(kernel));
	q.finish();

	// Copy Result from Device Global Memory to Host Local Memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output31, buffer_output32, buffer_output33, buffer_output34, buffer_output35}, CL_MIGRATE_MEM_OBJECT_HOST));
	q.finish();

	std::cout << "Max pooling...\n";
	MaxPool2d(R[0], C[0], M[0], 2, out_hw31, mp31_hw);
	MaxPool2d(R[1], C[1], M[1], 2, out_hw32, mp32_hw);
	MaxPool2d(R[2], C[2], M[2], 2, out_hw33, mp33_hw);
	MaxPool2d(R[3], C[3], M[3], 2, out_hw34, mp34_hw);
	MaxPool2d(R[4], C[4], M[4], 2, out_hw35, mp35_hw);


	// Step 4 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	std::cout << "******************* Step 4 *******************\n";
	std::cout << "Zero padding...\n";
	ZeroPad2d(R[1], C[1], M[0], K, mp31_hw, inp42_hw);
	ZeroPad2d(R[2], C[2], M[1], K, mp32_hw, inp43_hw);
	ZeroPad2d(R[3], C[3], M[2], K, mp33_hw, inp44_hw);
	ZeroPad2d(R[4], C[4], M[3], K, mp34_hw, inp45_hw);

	// Allocate Buffer in Global Memory
	OCL_CHECK(err, cl::Buffer buffer_input41(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[0]*(R[0]+K-1)*(C[0]+K-1), inp41_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight41(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[0]*M[0]*K*K, ker41.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output41(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[0]*R[0]*C[0], out_hw41.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input42(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[1]*(R[1]+K-1)*(C[1]+K-1), inp42_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight42(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[1]*M[1]*K*K, ker42.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output42(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[1]*R[1]*C[1], out_hw42.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input43(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[2]*(R[2]+K-1)*(C[2]+K-1), inp43_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight43(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[2]*M[2]*K*K, ker43.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output43(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[2]*R[2]*C[2], out_hw43.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input44(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[3]*(R[3]+K-1)*(C[3]+K-1), inp44_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight44(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[3]*M[3]*K*K, ker44.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output44(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[3]*R[3]*C[3], out_hw44.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input45(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[4]*(R[4]+K-1)*(C[4]+K-1), inp45_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight45(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[4]*M[4]*K*K, ker45.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output45(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[4]*R[4]*C[4], out_hw45.data(), &err));

	OCL_CHECK(err, err = kernel.setArg(0, buffer_input41));
	OCL_CHECK(err, err = kernel.setArg(1, buffer_input42));
	OCL_CHECK(err, err = kernel.setArg(2, buffer_input43));
	OCL_CHECK(err, err = kernel.setArg(3, buffer_input44));
	OCL_CHECK(err, err = kernel.setArg(4, buffer_input45));

	OCL_CHECK(err, err = kernel.setArg(5, buffer_weight41));
	OCL_CHECK(err, err = kernel.setArg(6, buffer_weight42));
	OCL_CHECK(err, err = kernel.setArg(7, buffer_weight43));
	OCL_CHECK(err, err = kernel.setArg(8, buffer_weight44));
	OCL_CHECK(err, err = kernel.setArg(9, buffer_weight45));

	OCL_CHECK(err, err = kernel.setArg(10, buffer_output41));
	OCL_CHECK(err, err = kernel.setArg(11, buffer_output42));
	OCL_CHECK(err, err = kernel.setArg(12, buffer_output43));
	OCL_CHECK(err, err = kernel.setArg(13, buffer_output44));
	OCL_CHECK(err, err = kernel.setArg(14, buffer_output45));

	// Copy input data to device global memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
		{buffer_input41, buffer_input42, buffer_input43, buffer_input44, buffer_input45, 
		buffer_weight41, buffer_weight42, buffer_weight43, buffer_weight44, buffer_weight45, 
		buffer_output41, buffer_output42, buffer_output43, buffer_output44, buffer_output45}, 0 /* 0 means from host*/));

	q.finish();
	
	start = std::chrono::steady_clock::now();
	std::cout << "Convolution...\n";
	// Launch the Kernel
	OCL_CHECK(err, err = q.enqueueTask(kernel));
	q.finish();

	// Copy Result from Device Global Memory to Host Local Memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output41, buffer_output42, buffer_output43, buffer_output44, buffer_output45}, CL_MIGRATE_MEM_OBJECT_HOST));
	q.finish();

	std::cout << "Max pooling...\n";
	MaxPool2d(R[0], C[0], M[0], 2, out_hw41, mp41_hw);
	MaxPool2d(R[1], C[1], M[1], 2, out_hw42, mp42_hw);
	MaxPool2d(R[2], C[2], M[2], 2, out_hw43, mp43_hw);
	MaxPool2d(R[3], C[3], M[3], 2, out_hw44, mp44_hw);
	MaxPool2d(R[4], C[4], M[4], 2, out_hw45, mp45_hw);

	// Step 5 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	std::cout << "******************* Step 5 *******************\n";
	std::cout << "Zero padding...\n";
	ZeroPad2d(R[1], C[1], M[0], K, mp41_hw, inp52_hw);
	ZeroPad2d(R[2], C[2], M[1], K, mp42_hw, inp53_hw);
	ZeroPad2d(R[3], C[3], M[2], K, mp43_hw, inp54_hw);
	ZeroPad2d(R[4], C[4], M[3], K, mp44_hw, inp55_hw);

	// Allocate Buffer in Global Memory
	OCL_CHECK(err, cl::Buffer buffer_input51(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[0]*(R[0]+K-1)*(C[0]+K-1), inp51_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight51(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[0]*M[0]*K*K, ker51.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output51(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[0]*R[0]*C[0], out_hw51.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input52(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[1]*(R[1]+K-1)*(C[1]+K-1), inp52_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight52(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[1]*M[1]*K*K, ker52.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output52(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[1]*R[1]*C[1], out_hw52.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input53(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[2]*(R[2]+K-1)*(C[2]+K-1), inp53_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight53(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[2]*M[2]*K*K, ker53.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output53(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[2]*R[2]*C[2], out_hw53.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input54(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[3]*(R[3]+K-1)*(C[3]+K-1), inp54_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight54(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[3]*M[3]*K*K, ker54.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output54(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[3]*R[3]*C[3], out_hw54.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input55(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[4]*(R[4]+K-1)*(C[4]+K-1), inp55_hw.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight55(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N[4]*M[4]*K*K, ker55.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output55(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * M[4]*R[4]*C[4], out_hw55.data(), &err));

	OCL_CHECK(err, err = kernel.setArg(0, buffer_input51));
	OCL_CHECK(err, err = kernel.setArg(1, buffer_input52));
	OCL_CHECK(err, err = kernel.setArg(2, buffer_input53));
	OCL_CHECK(err, err = kernel.setArg(3, buffer_input54));
	OCL_CHECK(err, err = kernel.setArg(4, buffer_input55));

	OCL_CHECK(err, err = kernel.setArg(5, buffer_weight51));
	OCL_CHECK(err, err = kernel.setArg(6, buffer_weight52));
	OCL_CHECK(err, err = kernel.setArg(7, buffer_weight53));
	OCL_CHECK(err, err = kernel.setArg(8, buffer_weight54));
	OCL_CHECK(err, err = kernel.setArg(9, buffer_weight55));

	OCL_CHECK(err, err = kernel.setArg(10, buffer_output51));
	OCL_CHECK(err, err = kernel.setArg(11, buffer_output52));
	OCL_CHECK(err, err = kernel.setArg(12, buffer_output53));
	OCL_CHECK(err, err = kernel.setArg(13, buffer_output54));
	OCL_CHECK(err, err = kernel.setArg(14, buffer_output55));

	// Copy input data to device global memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
		{buffer_input51, buffer_input52, buffer_input53, buffer_input54, buffer_input55, 
		buffer_weight51, buffer_weight52, buffer_weight53, buffer_weight54, buffer_weight55, 
		buffer_output51, buffer_output52, buffer_output53, buffer_output54, buffer_output55}, 0 /* 0 means from host*/));

	q.finish();
	
	start = std::chrono::steady_clock::now();
	std::cout << "Convolution...\n";
	// Launch the Kernel
	OCL_CHECK(err, err = q.enqueueTask(kernel));
	q.finish();

	// Copy Result from Device Global Memory to Host Local Memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output51, buffer_output52, buffer_output53, buffer_output54, buffer_output55}, CL_MIGRATE_MEM_OBJECT_HOST));
	q.finish();

	std::cout << "Max pooling...\n";
	MaxPool2d(R[0], C[0], M[0], 2, out_hw51, mp51_hw);
	MaxPool2d(R[1], C[1], M[1], 2, out_hw52, mp52_hw);
	MaxPool2d(R[2], C[2], M[2], 2, out_hw53, mp53_hw);
	MaxPool2d(R[3], C[3], M[3], 2, out_hw54, mp54_hw);
	MaxPool2d(R[4], C[4], M[4], 2, out_hw55, mp55_hw);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	auto end0 = std::chrono::steady_clock::now();
	std::cout << "Done.\n";

	exec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end0 - start0).count();
	/*
	gflops1 = double(N[0]) * M[0] * R[0] * C[0] * K * K * 2 / (exec_time);
	gflops2 = double(N[1]) * M[1] * R[1] * C[1] * K * K * 2 / (exec_time);
	gflops3 = double(N[2]) * M[2] * R[2] * C[2] * K * K * 2 / (exec_time);
	gflops4 = double(N[3]) * M[3] * R[3] * C[3] * K * K * 2 / (exec_time);
	gflops5 = double(N[4]) * M[4] * R[4] * C[4] * K * K * 2 / (exec_time);
	std::cout << "Time: " << exec_time*1e-9 << ", GFLOPS: " << (gflops1 + gflops2 + gflops3 + gflops4 + gflops5) << std::endl;
	*/
	
	std::cout << "Time: " << exec_time*1e-9 << std::endl;

	// OPENCL HOST CODE AREA END
	// Verification
	int err_cnt = 0;
	for(int cho = 0; cho < M[4]; cho++) {
		for(int row = 0; row < R[5]; row++) {
			for(int col = 0; col < C[5]; col++) {
				if(mp55[cho*(R[5]*C[5]) + row*C[5] + col] != mp55_hw[cho*(R[5]*C[5]) + row*C[5] + col]) {
					err_cnt++;
					if( err_cnt == 1 ){
						printf("cho:%d row:%d col:%d sw:%d hw:%d\n", cho, row, col, mp55[cho*(R[5]*C[5]) + row*C[5] + col], mp55_hw[cho*(R[5]*C[5]) + row*C[5] + col]);
					}
				}
			}
		}
	}

	if(err_cnt != 0){
		printf("FAILED! Error count : %d\n", err_cnt);
	}
	else{
		printf("PASSED!\n");
	}

	return EXIT_SUCCESS;
}
