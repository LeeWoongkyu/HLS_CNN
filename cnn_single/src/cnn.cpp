#include "hls_vector.h"
#include "hls_stream.h"
#include "ap_int.h"
#include "assert.h"

#include "cnn.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// CONV 1 
///
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void load_input1(hls::vector<short, BUSWIDTH> *inp, hls::stream<hls::vector<short, Tn1>> & inp_stream) {
	
	hls::vector<short, Tn1> tinp;
	hls::vector<short, BUSWIDTH> temp_inp;
	
	r_loop: for(int row = 0; row < R[0]; row+=Tr1) {
		c_loop: for(int col = 0; col < C[0]; col+=Tc1) {
			m_loop: for(int cho = 0; cho < M[0]; cho+=Tm1) {

				n_loop: for(int chi = 0; chi < N[0]; chi+=Tn1) {

					init_tinp_r: for (int tr = 0; tr < Tr1+K-1; tr++) {
						int r = row + tr;
						init_tinp_c: for (int tc = 0; tc < Tc1+K-1; tc++) {
#pragma HLS pipeline II = 1
							int c = col + tc;
							init_tinp_n: for (int tn = 0; tn < Tn1; tn+=BUSWIDTH) {
#pragma HLS unroll
								int n = chi + tn;
								temp_inp = inp[( r*N[0]*(C[0]+K-1) + c*N[0] + n)/BUSWIDTH];
								for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
									tinp[tn+b] = temp_inp[b];
								}
							}
							inp_stream.write(tinp);
						}
					}		
				}
			}
		}
	}
}
static void load_weight1(hls::vector<short, BUSWIDTH> *ker, hls::stream<hls::vector<short, Tn1>> & ker_stream) {

	hls::vector<short, Tn1> tker;
	hls::vector<short, BUSWIDTH> temp_ker;
	
	r_loop: for(int row = 0; row < R[0]; row+=Tr1) {
		c_loop: for(int col = 0; col < C[0]; col+=Tc1) {
			m_loop: for(int cho = 0; cho < M[0]; cho+=Tm1) {

				n_loop: for(int chi = 0; chi < N[0]; chi+=Tn1) {
					init_tker_ki: for (int ki = 0; ki < K; ki++) {
						init_tker_kj: for (int kj = 0; kj < K; kj++) {
							init_tker_m: for (int tm = 0; tm < Tm1; tm++) {
#pragma HLS pipeline II = 1
								int m = cho + tm;
								init_tker_n: for (int tn = 0; tn < Tn1; tn+=BUSWIDTH) {
#pragma HLS unroll
									int n = chi + tn;
									temp_ker = ker[( ki*N[0]*M[0]*K + kj*N[0]*M[0] + m*N[0] + n )/BUSWIDTH];
									for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
										tker[tn+b] = temp_ker[b];
									}
								}
								ker_stream.write(tker);
							}
						}
					}
				}
			}
		}
	}
}
static void Tiled_cnn1(
		hls::stream<hls::vector<short, Tn1>> & ker_stream,
		hls::stream<hls::vector<short, Tn1>> & inp_stream,
		hls::stream<hls::vector<short, Tm1>> & out_stream) {

	static short tinp[Tr1+K-1][Tc1+K-1][Tn1];
	static short tker[K][K][Tm1][Tn1];
	static short tout[Tr1][Tc1][Tm1];

	hls::vector<short, Tn1> temp_inp;
	hls::vector<short, Tn1> temp_ker;
	hls::vector<short, Tm1> temp_out;

#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=tinp
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=tker
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=tout


	r_loop: for(int row = 0; row < R[0]; row+=Tr1) {
		c_loop: for(int col = 0; col < C[0]; col+=Tc1) {
			m_loop: for(int cho = 0; cho < M[0]; cho+=Tm1) {
				
				init_tout_r: for (int tr = 0; tr < Tr1; tr++) {
					init_tout_c: for (int tc = 0; tc < Tc1; tc++) {
#pragma HLS pipeline II = 1
						init_tout_m: for (int tm = 0; tm < Tm1; tm++) {
#pragma HLS unroll
							tout[tr][tc][tm] = 0;
						}
					}
				}

				n_loop: for(int chi = 0; chi < N[0]; chi+=Tn1) {
					// Initialize tile of input
					init_tinp_r: for (int tr = 0; tr < Tr1+K-1; tr++) {
						init_tinp_c: for (int tc = 0; tc < Tc1+K-1; tc++) {
#pragma HLS pipeline II = 1
							temp_inp = inp_stream.read();
							init_tinp_n: for (int tn = 0; tn < Tn1; tn++) {
#pragma HLS unroll
								tinp[tr][tc][tn] = temp_inp[tn];
					}}}
					// Initialize tile of kernel
					init_tker_ki: for (int ki = 0; ki < K; ki++) {
						init_tker_kj: for (int kj = 0; kj < K; kj++) {
							init_tker_m: for (int tm = 0; tm < Tm1; tm++) {
#pragma HLS pipeline II = 1
								temp_ker = ker_stream.read();
								init_tker_n: for (int tn = 0; tn < Tn1; tn++) {
#pragma HLS unroll
									tker[ki][kj][tm][tn]= temp_ker[tn];
					}}}}
					// Main computation
					ki: for (int ki = 0; ki < K; ki++) {
						kj: for (int kj = 0; kj < K; kj++) {
							tr: for (int tr = 0; tr < Tr1; tr++) {
								tc: for (int tc = 0; tc < Tc1; tc++) {
#pragma HLS pipeline II = 1
									tm: for (int tm = 0; tm < Tm1; tm++) {
#pragma HLS unroll
										tn: for (int tn = 0; tn < Tn1; tn++) {
#pragma HLS unroll
											L: tout[tr][tc][tm] += tker[ki][kj][tm][tn] * tinp[tr+ki][tc+kj][tn];

					}}}}}}
				}
				// Writeback tile of output (Loop promotion)
				wb_tout_r: for (int tr = 0; tr < Tr1; tr++) {
					wb_tout_c: for (int tc = 0; tc < Tc1; tc++) {
#pragma HLS pipeline II = 1
						wb_tout_m: for (int tm = 0; tm < Tm1; tm++) {
#pragma HLS unroll
							temp_out[tm] = tout[tr][tc][tm];
						}
						out_stream.write(temp_out);
				}}
			
	}}}

}
static void store_result1(hls::vector<short, BUSWIDTH> *out, hls::stream<hls::vector<short, Tm1>> & out_stream) {
	
	hls::vector<short, Tm1> tout;
	hls::vector<short, BUSWIDTH> temp_out;
	
	r_loop: for(int row = 0; row < R[0]; row+=Tr1) {
		c_loop: for(int col = 0; col < C[0]; col+=Tc1) {
			m_loop: for(int cho = 0; cho < M[0]; cho+=Tm1) {
				
				wb_tout_r: for (int tr = 0; tr < Tr1; tr++) {
					int r = row + tr;
					wb_tout_c: for (int tc = 0; tc < Tc1; tc++) {
#pragma HLS pipeline II = 1
						int c = col + tc;
						tout = out_stream.read();
						wb_tout_m: for (int tm = 0; tm < Tm1; tm+=BUSWIDTH) {
#pragma HLS unroll
							for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
								temp_out[b] = tout[tm+b];
							}
							int m = cho + tm;
							out[( r*M[0]*C[0] + c*M[0] + m )/BUSWIDTH] = temp_out;
						}
					}
				}
			}
		}
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// CONV 2
///
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void load_input2(hls::vector<short, BUSWIDTH> *inp, hls::stream<hls::vector<short, Tn2>> & inp_stream) {
	
	hls::vector<short, Tn2> tinp;
	hls::vector<short, BUSWIDTH> temp_inp;
	
	r_loop: for(int row = 0; row < R[1]; row+=Tr2) {
		c_loop: for(int col = 0; col < C[1]; col+=Tc2) {
			m_loop: for(int cho = 0; cho < M[1]; cho+=Tm2) {

				n_loop: for(int chi = 0; chi < N[1]; chi+=Tn2) {

					init_tinp_r: for (int tr = 0; tr < Tr2+K-1; tr++) {
						int r = row + tr;
						init_tinp_c: for (int tc = 0; tc < Tc2+K-1; tc++) {
#pragma HLS pipeline II = 1
							int c = col + tc;
							init_tinp_n: for (int tn = 0; tn < Tn2; tn+=BUSWIDTH) {
#pragma HLS unroll
								int n = chi + tn;
								temp_inp = inp[( r*N[1]*(C[1]+K-1) + c*N[1] + n)/BUSWIDTH];
								for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
									tinp[tn+b] = temp_inp[b];
								}
							}
							inp_stream.write(tinp);
						}
					}		
				}
			}
		}
	}
}
static void load_weight2(hls::vector<short, BUSWIDTH> *ker, hls::stream<hls::vector<short, Tn2>> & ker_stream) {

	hls::vector<short, Tn2> tker;
	hls::vector<short, BUSWIDTH> temp_ker;
	
	r_loop: for(int row = 0; row < R[1]; row+=Tr2) {
		c_loop: for(int col = 0; col < C[1]; col+=Tc2) {
			m_loop: for(int cho = 0; cho < M[1]; cho+=Tm2) {

				n_loop: for(int chi = 0; chi < N[1]; chi+=Tn2) {
					init_tker_ki: for (int ki = 0; ki < K; ki++) {
						init_tker_kj: for (int kj = 0; kj < K; kj++) {
							init_tker_m: for (int tm = 0; tm < Tm2; tm++) {
#pragma HLS pipeline II = 1
								int m = cho + tm;
								init_tker_n: for (int tn = 0; tn < Tn2; tn+=BUSWIDTH) {
#pragma HLS unroll
									int n = chi + tn;
									temp_ker = ker[( ki*N[1]*M[1]*K + kj*N[1]*M[1] + m*N[1] + n )/BUSWIDTH];
									for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
										tker[tn+b] = temp_ker[b];
									}
								}
								ker_stream.write(tker);
							}
						}
					}
				}
			}
		}
	}
}
static void Tiled_cnn2(
		hls::stream<hls::vector<short, Tn2>> & ker_stream,
		hls::stream<hls::vector<short, Tn2>> & inp_stream,
		hls::stream<hls::vector<short, Tm2>> & out_stream) {

	static short tinp[Tr2+K-1][Tc2+K-1][Tn2];
	static short tker[K][K][Tm2][Tn2];
	static short tout[Tr2][Tc2][Tm2];

	hls::vector<short, Tn2> temp_inp;
	hls::vector<short, Tn2> temp_ker;
	hls::vector<short, Tm2> temp_out;

#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=tinp
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=tker
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=tout


	r_loop: for(int row = 0; row < R[1]; row+=Tr2) {
		c_loop: for(int col = 0; col < C[1]; col+=Tc2) {
			m_loop: for(int cho = 0; cho < M[1]; cho+=Tm2) {
				
				init_tout_r: for (int tr = 0; tr < Tr2; tr++) {
					init_tout_c: for (int tc = 0; tc < Tc2; tc++) {
#pragma HLS pipeline II = 1
						init_tout_m: for (int tm = 0; tm < Tm2; tm++) {
#pragma HLS unroll
							tout[tr][tc][tm] = 0;
						}
					}
				}

				n_loop: for(int chi = 0; chi < N[1]; chi+=Tn2) {
					// Initialize tile of input
					init_tinp_r: for (int tr = 0; tr < Tr2+K-1; tr++) {
						init_tinp_c: for (int tc = 0; tc < Tc2+K-1; tc++) {
#pragma HLS pipeline II = 1
							temp_inp = inp_stream.read();
							init_tinp_n: for (int tn = 0; tn < Tn2; tn++) {
#pragma HLS unroll
								tinp[tr][tc][tn] = temp_inp[tn];
					}}}
					// Initialize tile of kernel
					init_tker_ki: for (int ki = 0; ki < K; ki++) {
						init_tker_kj: for (int kj = 0; kj < K; kj++) {
							init_tker_m: for (int tm = 0; tm < Tm2; tm++) {
#pragma HLS pipeline II = 1
								temp_ker = ker_stream.read();
								init_tker_n: for (int tn = 0; tn < Tn2; tn++) {
#pragma HLS unroll
									tker[ki][kj][tm][tn]= temp_ker[tn];
					}}}}
					// Main computation
					ki: for (int ki = 0; ki < K; ki++) {
						kj: for (int kj = 0; kj < K; kj++) {
							tr: for (int tr = 0; tr < Tr2; tr++) {
								tc: for (int tc = 0; tc < Tc2; tc++) {
#pragma HLS pipeline II = 1
									tm: for (int tm = 0; tm < Tm2; tm++) {
#pragma HLS unroll
										tn: for (int tn = 0; tn < Tn2; tn++) {
#pragma HLS unroll
											L: tout[tr][tc][tm] += tker[ki][kj][tm][tn] * tinp[tr+ki][tc+kj][tn];

					}}}}}}
				}
				// Writeback tile of output (Loop promotion)
				wb_tout_r: for (int tr = 0; tr < Tr2; tr++) {
					wb_tout_c: for (int tc = 0; tc < Tc2; tc++) {
#pragma HLS pipeline II = 1
						wb_tout_m: for (int tm = 0; tm < Tm2; tm++) {
#pragma HLS unroll
							temp_out[tm] = tout[tr][tc][tm];
						}
						out_stream.write(temp_out);
				}}
			
	}}}

}
static void store_result2(hls::vector<short, BUSWIDTH> *out, hls::stream<hls::vector<short, Tm2>> & out_stream) {
	
	hls::vector<short, Tm2> tout;
	hls::vector<short, BUSWIDTH> temp_out;
	
	r_loop: for(int row = 0; row < R[1]; row+=Tr2) {
		c_loop: for(int col = 0; col < C[1]; col+=Tc2) {
			m_loop: for(int cho = 0; cho < M[1]; cho+=Tm2) {
				
				wb_tout_r: for (int tr = 0; tr < Tr2; tr++) {
					int r = row + tr;
					wb_tout_c: for (int tc = 0; tc < Tc2; tc++) {
#pragma HLS pipeline II = 1
						int c = col + tc;
						tout = out_stream.read();
						wb_tout_m: for (int tm = 0; tm < Tm2; tm+=BUSWIDTH) {
#pragma HLS unroll
							for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
								temp_out[b] = tout[tm+b];
							}
							int m = cho + tm;
							out[( r*M[1]*C[1] + c*M[1] + m )/BUSWIDTH] = temp_out;
						}
					}
				}
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// CONV 3
///
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void load_input3(hls::vector<short, BUSWIDTH> *inp, hls::stream<hls::vector<short, Tn3>> & inp_stream) {
	
	hls::vector<short, Tn3> tinp;
	hls::vector<short, BUSWIDTH> temp_inp;
	
	r_loop: for(int row = 0; row < R[2]; row+=Tr3) {
		c_loop: for(int col = 0; col < C[2]; col+=Tc3) {
			m_loop: for(int cho = 0; cho < M[2]; cho+=Tm3) {

				n_loop: for(int chi = 0; chi < N[2]; chi+=Tn3) {

					init_tinp_r: for (int tr = 0; tr < Tr3+K-1; tr++) {
						int r = row + tr;
						init_tinp_c: for (int tc = 0; tc < Tc3+K-1; tc++) {
#pragma HLS pipeline II = 1
							int c = col + tc;
							init_tinp_n: for (int tn = 0; tn < Tn3; tn+=BUSWIDTH) {
#pragma HLS unroll
								int n = chi + tn;
								temp_inp = inp[( r*N[2]*(C[2]+K-1) + c*N[2] + n)/BUSWIDTH];
								for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
									tinp[tn+b] = temp_inp[b];
								}
							}
							inp_stream.write(tinp);
						}
					}		
				}
			}
		}
	}
}
static void load_weight3(hls::vector<short, BUSWIDTH> *ker, hls::stream<hls::vector<short, Tn3>> & ker_stream) {

	hls::vector<short, Tn3> tker;
	hls::vector<short, BUSWIDTH> temp_ker;
	
	r_loop: for(int row = 0; row < R[2]; row+=Tr3) {
		c_loop: for(int col = 0; col < C[2]; col+=Tc3) {
			m_loop: for(int cho = 0; cho < M[2]; cho+=Tm3) {

				n_loop: for(int chi = 0; chi < N[2]; chi+=Tn3) {
					init_tker_ki: for (int ki = 0; ki < K; ki++) {
						init_tker_kj: for (int kj = 0; kj < K; kj++) {
							init_tker_m: for (int tm = 0; tm < Tm3; tm++) {
#pragma HLS pipeline II = 1
								int m = cho + tm;
								init_tker_n: for (int tn = 0; tn < Tn3; tn+=BUSWIDTH) {
#pragma HLS unroll
									int n = chi + tn;
									temp_ker = ker[( ki*N[2]*M[2]*K + kj*N[2]*M[2] + m*N[2] + n )/BUSWIDTH];
									for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
										tker[tn+b] = temp_ker[b];
									}
								}
								ker_stream.write(tker);
							}
						}
					}
				}
			}
		}
	}
}
static void Tiled_cnn3(
		hls::stream<hls::vector<short, Tn3>> & ker_stream,
		hls::stream<hls::vector<short, Tn3>> & inp_stream,
		hls::stream<hls::vector<short, Tm3>> & out_stream) {

	static short tinp[Tr3+K-1][Tc3+K-1][Tn3];
	static short tker[K][K][Tm3][Tn3];
	static short tout[Tr3][Tc3][Tm3];

	hls::vector<short, Tn3> temp_inp;
	hls::vector<short, Tn3> temp_ker;
	hls::vector<short, Tm3> temp_out;

#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=tinp
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=tker
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=tout


	r_loop: for(int row = 0; row < R[2]; row+=Tr3) {
		c_loop: for(int col = 0; col < C[2]; col+=Tc3) {
			m_loop: for(int cho = 0; cho < M[2]; cho+=Tm3) {
				
				init_tout_r: for (int tr = 0; tr < Tr3; tr++) {
					init_tout_c: for (int tc = 0; tc < Tc3; tc++) {
#pragma HLS pipeline II = 1
						init_tout_m: for (int tm = 0; tm < Tm3; tm++) {
#pragma HLS unroll
							tout[tr][tc][tm] = 0;
						}
					}
				}

				n_loop: for(int chi = 0; chi < N[2]; chi+=Tn3) {
					// Initialize tile of input
					init_tinp_r: for (int tr = 0; tr < Tr3+K-1; tr++) {
						init_tinp_c: for (int tc = 0; tc < Tc3+K-1; tc++) {
#pragma HLS pipeline II = 1
							temp_inp = inp_stream.read();
							init_tinp_n: for (int tn = 0; tn < Tn3; tn++) {
#pragma HLS unroll
								tinp[tr][tc][tn] = temp_inp[tn];
					}}}
					// Initialize tile of kernel
					init_tker_ki: for (int ki = 0; ki < K; ki++) {
						init_tker_kj: for (int kj = 0; kj < K; kj++) {
							init_tker_m: for (int tm = 0; tm < Tm3; tm++) {
#pragma HLS pipeline II = 1
								temp_ker = ker_stream.read();
								init_tker_n: for (int tn = 0; tn < Tn3; tn++) {
#pragma HLS unroll
									tker[ki][kj][tm][tn]= temp_ker[tn];
					}}}}
					// Main computation
					ki: for (int ki = 0; ki < K; ki++) {
						kj: for (int kj = 0; kj < K; kj++) {
							tr: for (int tr = 0; tr < Tr3; tr++) {
								tc: for (int tc = 0; tc < Tc3; tc++) {
#pragma HLS pipeline II = 1
									tm: for (int tm = 0; tm < Tm3; tm++) {
#pragma HLS unroll
										tn: for (int tn = 0; tn < Tn3; tn++) {
#pragma HLS unroll
											L: tout[tr][tc][tm] += tker[ki][kj][tm][tn] * tinp[tr+ki][tc+kj][tn];

					}}}}}}
				}
				// Writeback tile of output (Loop promotion)
				wb_tout_r: for (int tr = 0; tr < Tr3; tr++) {
					wb_tout_c: for (int tc = 0; tc < Tc3; tc++) {
#pragma HLS pipeline II = 1
						wb_tout_m: for (int tm = 0; tm < Tm3; tm++) {
#pragma HLS unroll
							temp_out[tm] = tout[tr][tc][tm];
						}
						out_stream.write(temp_out);
				}}
			
	}}}

}
static void store_result3(hls::vector<short, BUSWIDTH> *out, hls::stream<hls::vector<short, Tm3>> & out_stream) {
	
	hls::vector<short, Tm3> tout;
	hls::vector<short, BUSWIDTH> temp_out;
	
	r_loop: for(int row = 0; row < R[2]; row+=Tr3) {
		c_loop: for(int col = 0; col < C[2]; col+=Tc3) {
			m_loop: for(int cho = 0; cho < M[2]; cho+=Tm3) {
				
				wb_tout_r: for (int tr = 0; tr < Tr3; tr++) {
					int r = row + tr;
					wb_tout_c: for (int tc = 0; tc < Tc3; tc++) {
#pragma HLS pipeline II = 1
						int c = col + tc;
						tout = out_stream.read();
						wb_tout_m: for (int tm = 0; tm < Tm3; tm+=BUSWIDTH) {
#pragma HLS unroll
							for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
								temp_out[b] = tout[tm+b];
							}
							int m = cho + tm;
							out[( r*M[2]*C[2] + c*M[2] + m )/BUSWIDTH] = temp_out;
						}
					}
				}
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// CONV 4
///
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void load_input4(hls::vector<short, BUSWIDTH> *inp, hls::stream<hls::vector<short, Tn4>> & inp_stream) {
	
	hls::vector<short, Tn4> tinp;
	hls::vector<short, BUSWIDTH> temp_inp;
	
	r_loop: for(int row = 0; row < R[3]; row+=Tr4) {
		c_loop: for(int col = 0; col < C[3]; col+=Tc4) {
			m_loop: for(int cho = 0; cho < M[3]; cho+=Tm4) {

				n_loop: for(int chi = 0; chi < N[3]; chi+=Tn4) {

					init_tinp_r: for (int tr = 0; tr < Tr4+K-1; tr++) {
						int r = row + tr;
						init_tinp_c: for (int tc = 0; tc < Tc4+K-1; tc++) {
#pragma HLS pipeline II = 1
							int c = col + tc;
							init_tinp_n: for (int tn = 0; tn < Tn4; tn+=BUSWIDTH) {
#pragma HLS unroll
								int n = chi + tn;
								temp_inp = inp[( r*N[3]*(C[3]+K-1) + c*N[3] + n)/BUSWIDTH];
								for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
									tinp[tn+b] = temp_inp[b];
								}
							}
							inp_stream.write(tinp);
						}
					}		
				}
			}
		}
	}
}
static void load_weight4(hls::vector<short, BUSWIDTH> *ker, hls::stream<hls::vector<short, Tn4>> & ker_stream) {

	hls::vector<short, Tn4> tker;
	hls::vector<short, BUSWIDTH> temp_ker;
	
	r_loop: for(int row = 0; row < R[3]; row+=Tr4) {
		c_loop: for(int col = 0; col < C[3]; col+=Tc4) {
			m_loop: for(int cho = 0; cho < M[3]; cho+=Tm4) {

				n_loop: for(int chi = 0; chi < N[3]; chi+=Tn4) {
					init_tker_ki: for (int ki = 0; ki < K; ki++) {
						init_tker_kj: for (int kj = 0; kj < K; kj++) {
							init_tker_m: for (int tm = 0; tm < Tm4; tm++) {
#pragma HLS pipeline II = 1
								int m = cho + tm;
								init_tker_n: for (int tn = 0; tn < Tn4; tn+=BUSWIDTH) {
#pragma HLS unroll
									int n = chi + tn;
									temp_ker = ker[( ki*N[3]*M[3]*K + kj*N[3]*M[3] + m*N[3] + n )/BUSWIDTH];
									for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
										tker[tn+b] = temp_ker[b];
									}
								}
								ker_stream.write(tker);
							}
						}
					}
				}
			}
		}
	}
}
static void Tiled_cnn4(
		hls::stream<hls::vector<short, Tn4>> & ker_stream,
		hls::stream<hls::vector<short, Tn4>> & inp_stream,
		hls::stream<hls::vector<short, Tm4>> & out_stream) {

	static short tinp[Tr4+K-1][Tc4+K-1][Tn4];
	static short tker[K][K][Tm4][Tn4];
	static short tout[Tr4][Tc4][Tm4];

	hls::vector<short, Tn4> temp_inp;
	hls::vector<short, Tn4> temp_ker;
	hls::vector<short, Tm4> temp_out;

#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=tinp
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=tker
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=tout


	r_loop: for(int row = 0; row < R[3]; row+=Tr4) {
		c_loop: for(int col = 0; col < C[3]; col+=Tc4) {
			m_loop: for(int cho = 0; cho < M[3]; cho+=Tm4) {
				
				init_tout_r: for (int tr = 0; tr < Tr4; tr++) {
					init_tout_c: for (int tc = 0; tc < Tc4; tc++) {
#pragma HLS pipeline II = 1
						init_tout_m: for (int tm = 0; tm < Tm4; tm++) {
#pragma HLS unroll
							tout[tr][tc][tm] = 0;
						}
					}
				}

				n_loop: for(int chi = 0; chi < N[3]; chi+=Tn4) {
					// Initialize tile of input
					init_tinp_r: for (int tr = 0; tr < Tr4+K-1; tr++) {
						init_tinp_c: for (int tc = 0; tc < Tc4+K-1; tc++) {
#pragma HLS pipeline II = 1
							temp_inp = inp_stream.read();
							init_tinp_n: for (int tn = 0; tn < Tn4; tn++) {
#pragma HLS unroll
								tinp[tr][tc][tn] = temp_inp[tn];
					}}}
					// Initialize tile of kernel
					init_tker_ki: for (int ki = 0; ki < K; ki++) {
						init_tker_kj: for (int kj = 0; kj < K; kj++) {
							init_tker_m: for (int tm = 0; tm < Tm4; tm++) {
#pragma HLS pipeline II = 1
								temp_ker = ker_stream.read();
								init_tker_n: for (int tn = 0; tn < Tn4; tn++) {
#pragma HLS unroll
									tker[ki][kj][tm][tn]= temp_ker[tn];
					}}}}
					// Main computation
					ki: for (int ki = 0; ki < K; ki++) {
						kj: for (int kj = 0; kj < K; kj++) {
							tr: for (int tr = 0; tr < Tr4; tr++) {
								tc: for (int tc = 0; tc < Tc4; tc++) {
#pragma HLS pipeline II = 1
									tm: for (int tm = 0; tm < Tm4; tm++) {
#pragma HLS unroll
										tn: for (int tn = 0; tn < Tn4; tn++) {
#pragma HLS unroll
											L: tout[tr][tc][tm] += tker[ki][kj][tm][tn] * tinp[tr+ki][tc+kj][tn];

					}}}}}}
				}
				// Writeback tile of output (Loop promotion)
				wb_tout_r: for (int tr = 0; tr < Tr4; tr++) {
					wb_tout_c: for (int tc = 0; tc < Tc4; tc++) {
#pragma HLS pipeline II = 1
						wb_tout_m: for (int tm = 0; tm < Tm4; tm++) {
#pragma HLS unroll
							temp_out[tm] = tout[tr][tc][tm];
						}
						out_stream.write(temp_out);
				}}
			
	}}}

}
static void store_result4(hls::vector<short, BUSWIDTH> *out, hls::stream<hls::vector<short, Tm4>> & out_stream) {
	
	hls::vector<short, Tm4> tout;
	hls::vector<short, BUSWIDTH> temp_out;
	
	r_loop: for(int row = 0; row < R[3]; row+=Tr4) {
		c_loop: for(int col = 0; col < C[3]; col+=Tc4) {
			m_loop: for(int cho = 0; cho < M[3]; cho+=Tm4) {
				
				wb_tout_r: for (int tr = 0; tr < Tr4; tr++) {
					int r = row + tr;
					wb_tout_c: for (int tc = 0; tc < Tc4; tc++) {
#pragma HLS pipeline II = 1
						int c = col + tc;
						tout = out_stream.read();
						wb_tout_m: for (int tm = 0; tm < Tm4; tm+=BUSWIDTH) {
#pragma HLS unroll
							for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
								temp_out[b] = tout[tm+b];
							}
							int m = cho + tm;
							out[( r*M[3]*C[3] + c*M[3] + m )/BUSWIDTH] = temp_out;
						}
					}
				}
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// CONV 5 
///
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void load_input5(hls::vector<short, BUSWIDTH> *inp, hls::stream<hls::vector<short, Tn5>> & inp_stream) {
	
	hls::vector<short, Tn5> tinp;
	hls::vector<short, BUSWIDTH> temp_inp;
	
	r_loop: for(int row = 0; row < R[4]; row+=Tr5) {
		c_loop: for(int col = 0; col < C[4]; col+=Tc5) {
			m_loop: for(int cho = 0; cho < M[4]; cho+=Tm5) {

				n_loop: for(int chi = 0; chi < N[4]; chi+=Tn5) {

					init_tinp_r: for (int tr = 0; tr < Tr5+K-1; tr++) {
						int r = row + tr;
						init_tinp_c: for (int tc = 0; tc < Tc5+K-1; tc++) {
#pragma HLS pipeline II = 1
							int c = col + tc;
							init_tinp_n: for (int tn = 0; tn < Tn5; tn+=BUSWIDTH) {
#pragma HLS unroll
								int n = chi + tn;
								temp_inp = inp[( r*N[4]*(C[4]+K-1) + c*N[4] + n)/BUSWIDTH];
								for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
									tinp[tn+b] = temp_inp[b];
								}
							}
							inp_stream.write(tinp);
						}
					}		
				}
			}
		}
	}
}
static void load_weight5(hls::vector<short, BUSWIDTH> *ker, hls::stream<hls::vector<short, Tn5>> & ker_stream) {

	hls::vector<short, Tn5> tker;
	hls::vector<short, BUSWIDTH> temp_ker;
	
	r_loop: for(int row = 0; row < R[4]; row+=Tr5) {
		c_loop: for(int col = 0; col < C[4]; col+=Tc5) {
			m_loop: for(int cho = 0; cho < M[4]; cho+=Tm5) {

				n_loop: for(int chi = 0; chi < N[4]; chi+=Tn5) {
					init_tker_ki: for (int ki = 0; ki < K; ki++) {
						init_tker_kj: for (int kj = 0; kj < K; kj++) {
							init_tker_m: for (int tm = 0; tm < Tm5; tm++) {
#pragma HLS pipeline II = 1
								int m = cho + tm;
								init_tker_n: for (int tn = 0; tn < Tn5; tn+=BUSWIDTH) {
#pragma HLS unroll
									int n = chi + tn;
									temp_ker = ker[( ki*N[4]*M[4]*K + kj*N[4]*M[4] + m*N[4] + n )/BUSWIDTH];
									for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
										tker[tn+b] = temp_ker[b];
									}
								}
								ker_stream.write(tker);
							}
						}
					}
				}
			}
		}
	}
}
static void Tiled_cnn5(
		hls::stream<hls::vector<short, Tn5>> & ker_stream,
		hls::stream<hls::vector<short, Tn5>> & inp_stream,
		hls::stream<hls::vector<short, Tm5>> & out_stream) {

	static short tinp[Tr5+K-1][Tc5+K-1][Tn5];
	static short tker[K][K][Tm5][Tn5];
	static short tout[Tr5][Tc5][Tm5];

	hls::vector<short, Tn5> temp_inp;
	hls::vector<short, Tn5> temp_ker;
	hls::vector<short, Tm5> temp_out;

#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=tinp
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=tker
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=tout


	r_loop: for(int row = 0; row < R[4]; row+=Tr5) {
		c_loop: for(int col = 0; col < C[4]; col+=Tc5) {
			m_loop: for(int cho = 0; cho < M[4]; cho+=Tm5) {
				
				init_tout_r: for (int tr = 0; tr < Tr5; tr++) {
					init_tout_c: for (int tc = 0; tc < Tc5; tc++) {
#pragma HLS pipeline II = 1
						init_tout_m: for (int tm = 0; tm < Tm5; tm++) {
#pragma HLS unroll
							tout[tr][tc][tm] = 0;
						}
					}
				}

				n_loop: for(int chi = 0; chi < N[4]; chi+=Tn5) {
					// Initialize tile of input
					init_tinp_r: for (int tr = 0; tr < Tr5+K-1; tr++) {
						init_tinp_c: for (int tc = 0; tc < Tc5+K-1; tc++) {
#pragma HLS pipeline II = 1
							temp_inp = inp_stream.read();
							init_tinp_n: for (int tn = 0; tn < Tn5; tn++) {
#pragma HLS unroll
								tinp[tr][tc][tn] = temp_inp[tn];
					}}}
					// Initialize tile of kernel
					init_tker_ki: for (int ki = 0; ki < K; ki++) {
						init_tker_kj: for (int kj = 0; kj < K; kj++) {
							init_tker_m: for (int tm = 0; tm < Tm5; tm++) {
#pragma HLS pipeline II = 1
								temp_ker = ker_stream.read();
								init_tker_n: for (int tn = 0; tn < Tn5; tn++) {
#pragma HLS unroll
									tker[ki][kj][tm][tn]= temp_ker[tn];
					}}}}
					// Main computation
					ki: for (int ki = 0; ki < K; ki++) {
						kj: for (int kj = 0; kj < K; kj++) {
							tr: for (int tr = 0; tr < Tr5; tr++) {
								tc: for (int tc = 0; tc < Tc5; tc++) {
#pragma HLS pipeline II = 1
									tm: for (int tm = 0; tm < Tm5; tm++) {
#pragma HLS unroll
										tn: for (int tn = 0; tn < Tn5; tn++) {
#pragma HLS unroll
											L: tout[tr][tc][tm] += tker[ki][kj][tm][tn] * tinp[tr+ki][tc+kj][tn];

					}}}}}}
				}
				// Writeback tile of output (Loop promotion)
				wb_tout_r: for (int tr = 0; tr < Tr5; tr++) {
					wb_tout_c: for (int tc = 0; tc < Tc5; tc++) {
#pragma HLS pipeline II = 1
						wb_tout_m: for (int tm = 0; tm < Tm5; tm++) {
#pragma HLS unroll
							temp_out[tm] = tout[tr][tc][tm];
						}
						out_stream.write(temp_out);
				}}
			
	}}}

}
static void store_result5(hls::vector<short, BUSWIDTH> *out, hls::stream<hls::vector<short, Tm5>> & out_stream) {
	
	hls::vector<short, Tm5> tout;
	hls::vector<short, BUSWIDTH> temp_out;
	
	r_loop: for(int row = 0; row < R[4]; row+=Tr5) {
		c_loop: for(int col = 0; col < C[4]; col+=Tc5) {
			m_loop: for(int cho = 0; cho < M[4]; cho+=Tm5) {
				
				wb_tout_r: for (int tr = 0; tr < Tr5; tr++) {
					int r = row + tr;
					wb_tout_c: for (int tc = 0; tc < Tc5; tc++) {
#pragma HLS pipeline II = 1
						int c = col + tc;
						tout = out_stream.read();
						wb_tout_m: for (int tm = 0; tm < Tm5; tm+=BUSWIDTH) {
#pragma HLS unroll
							for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
								temp_out[b] = tout[tm+b];
							}
							int m = cho + tm;
							out[( r*M[4]*C[4] + c*M[4] + m )/BUSWIDTH] = temp_out;
						}
					}
				}
			}
		}
	}
}

extern "C" {

void cnn(
		hls::vector<short, BUSWIDTH>* inp1,
		hls::vector<short, BUSWIDTH>* inp2,
		hls::vector<short, BUSWIDTH>* inp3,
		hls::vector<short, BUSWIDTH>* inp4,
		hls::vector<short, BUSWIDTH>* inp5,

		hls::vector<short, BUSWIDTH>* ker1,
		hls::vector<short, BUSWIDTH>* ker2,
		hls::vector<short, BUSWIDTH>* ker3,
		hls::vector<short, BUSWIDTH>* ker4,
		hls::vector<short, BUSWIDTH>* ker5,

		hls::vector<short, BUSWIDTH>* out1,
		hls::vector<short, BUSWIDTH>* out2,
		hls::vector<short, BUSWIDTH>* out3,
		hls::vector<short, BUSWIDTH>* out4,
		hls::vector<short, BUSWIDTH>* out5
		) 
		{

#pragma HLS INTERFACE m_axi port = inp1 bundle = gmem0
#pragma HLS INTERFACE m_axi port = inp2 bundle = gmem1
#pragma HLS INTERFACE m_axi port = inp3 bundle = gmem2
#pragma HLS INTERFACE m_axi port = inp4 bundle = gmem3
#pragma HLS INTERFACE m_axi port = inp5 bundle = gmem4

#pragma HLS INTERFACE m_axi port = ker1 bundle = gmem5
#pragma HLS INTERFACE m_axi port = ker2 bundle = gmem6
#pragma HLS INTERFACE m_axi port = ker3 bundle = gmem7
#pragma HLS INTERFACE m_axi port = ker4 bundle = gmem8
#pragma HLS INTERFACE m_axi port = ker5 bundle = gmem9

#pragma HLS INTERFACE m_axi port = out1 bundle = gmem10
#pragma HLS INTERFACE m_axi port = out2 bundle = gmem11
#pragma HLS INTERFACE m_axi port = out3 bundle = gmem12
#pragma HLS INTERFACE m_axi port = out4 bundle = gmem13
#pragma HLS INTERFACE m_axi port = out5 bundle = gmem14

	static hls::stream<hls::vector<short, Tn1> > inp1_stream("input_stream");
	static hls::stream<hls::vector<short, Tn1> > ker1_stream("weight_stream");
	static hls::stream<hls::vector<short, Tm1> > out1_stream("output_stream");

	static hls::stream<hls::vector<short, Tn2> > inp2_stream("input_stream");
	static hls::stream<hls::vector<short, Tn2> > ker2_stream("weight_stream");
	static hls::stream<hls::vector<short, Tm2> > out2_stream("output_stream");

	static hls::stream<hls::vector<short, Tn3> > inp3_stream("input_stream");
	static hls::stream<hls::vector<short, Tn3> > ker3_stream("weight_stream");
	static hls::stream<hls::vector<short, Tm3> > out3_stream("output_stream");

	static hls::stream<hls::vector<short, Tn4> > inp4_stream("input_stream");
	static hls::stream<hls::vector<short, Tn4> > ker4_stream("weight_stream");
	static hls::stream<hls::vector<short, Tm4> > out4_stream("output_stream");

	static hls::stream<hls::vector<short, Tn5> > inp5_stream("input_stream");
	static hls::stream<hls::vector<short, Tn5> > ker5_stream("weight_stream");
	static hls::stream<hls::vector<short, Tm5> > out5_stream("output_stream");

#pragma HLS dataflow
	
	load_weight1(ker1, ker1_stream);
	load_input1(inp1, inp1_stream);
	Tiled_cnn1(ker1_stream, inp1_stream, out1_stream);
	store_result1(out1, out1_stream);

	load_weight2(ker2, ker2_stream);
	load_input2(inp2, inp2_stream);
	Tiled_cnn2(ker2_stream, inp2_stream, out2_stream);
	store_result2(out2, out2_stream);

	load_weight3(ker3, ker3_stream);
	load_input3(inp3, inp3_stream);
	Tiled_cnn3(ker3_stream, inp3_stream, out3_stream);
	store_result3(out3, out3_stream);

	load_weight4(ker4, ker4_stream);
	load_input4(inp4, inp4_stream);
	Tiled_cnn4(ker4_stream, inp4_stream, out4_stream);
	store_result4(out4, out4_stream);

	load_weight5(ker5, ker5_stream);
	load_input5(inp5, inp5_stream);
	Tiled_cnn5(ker5_stream, inp5_stream, out5_stream);
	store_result5(out5, out5_stream);

}


}

