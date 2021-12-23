#ifndef CNN_H_
#define CNN_H_

typedef short DTYPE;

int R[6] = {224, 112, 56, 28, 14, 7};
int C[6] = {224, 112, 56, 28, 14, 7};
int M[5] = {128, 256, 512, 512, 512};
int N[5] = {64, 128, 256, 512, 512};
// int K[6] = {3, 3, 3, 3, 3, 3};

/*
const int R = 224;
const int C = 224;
const int M = 64;
const int N = 64;
*/

const int K = 3;

const int Tr1 = 14;
const int Tc1 = 14;
const int Tm1 = 16;
const int Tn1 = 16;

const int Tr2 = 14;
const int Tc2 = 14;
const int Tm2 = 16;
const int Tn2 = 16;

const int Tr3 = 14;
const int Tc3 = 14;
const int Tm3 = 16;
const int Tn3 = 16;

const int Tr4 = 14;
const int Tc4 = 14;
const int Tm4 = 16;
const int Tn4 = 16;

const int Tr5 = 14;
const int Tc5 = 14;
const int Tm5 = 16;
const int Tn5 = 16;

const int BUSWIDTH = 16;

#endif 
