/*

Copyright (c) 2023 Yrrid Software, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define uint128_t unsigned __int128
#define root32k 0x5A1597A0CAull

uint64_t mul(uint64_t a, uint64_t b) {
  uint128_t p=0xFFFFFFFF00000001ull;
  uint128_t la=a, lb=b, lp=la*lb%p;
  uint64_t  res=(uint64_t)lp;

  return res;
}

uint64_t power(uint64_t a, uint64_t k) {
  uint64_t current=1, sqr=a;

  while(k!=0) {
	if((k & 0x01)==1)
	  current=mul(current, sqr);
	sqr=mul(sqr, sqr);
	k=k>>1;
  }
  return current;
}

uint64_t add(uint64_t a, uint64_t b) {
  uint64_t sum=a+b;

  if(sum<a)
    sum=sum + 0xFFFFFFFFull;
  return sum;
}

uint64_t sub(uint64_t a, uint64_t b) {
  if(a>=b)
    return a-b;
  else
    return a-b-0xFFFFFFFFull;
}

void slow(uint64_t* out, uint64_t* in, int size) {
  uint128_t p=0xFFFFFFFF00000001ull;
  uint64_t  root=power(root32k, 0x100000000ull/size);
  uint64_t  table[size];
  uint128_t sum;

  // slow as dirt!  compute ntt using n^2 algorithm

  for(int i=0;i<size;i++)
    table[i]=power(root, i);

  for(int i=0;i<size;i++) {
	sum=0;
    for(int j=0;j<size;j++)
	  sum+=mul(in[j], table[i*j%size]);
	out[i]=sum%p;
  }
}

void copy1024(uint64_t* out, uint64_t* in) {
  for(int i=0;i<1024;i++)
    out[i]=in[i];
}

void copy4096(uint64_t* out, uint64_t* in) {
  for(int i=0;i<4096;i++)
    out[i]=in[i];
}

void fast4(uint64_t* out, uint64_t* in) {
  uint64_t X[4], T;
  uint64_t root18=0x1000000, root28=mul(root18, root18);

  for(int i=0;i<4;i++)
    X[i]=in[i];

  T    = add(X[0], X[2]);
  X[2] = sub(X[0], X[2]);
  X[0] = add(X[1], X[3]);
  X[1] = sub(X[1], X[3]);   // T has X0, X0 has X1, X2 has X2, X1 has X3

  X[1] = mul(X[1], root28);

  X[3] = sub(X[2], X[1]);
  X[1] = add(X[2], X[1]);
  X[2] = sub(T, X[0]);
  X[0] = add(T, X[0]);

  for(int i=0;i<4;i++)
    out[i]=X[i];
}

void fast8(uint64_t* out, uint64_t* in) {
  uint64_t X[8], T;
  uint64_t root18=0x1000000, root28=mul(root18, root18), root38=mul(root18, root28);

  for(int i=0;i<8;i++)
    X[i]=in[i];

  // out of 56,623,104 possible mappings, we have:

  T    = sub(X[3], X[7]);
  X[7] = add(X[3], X[7]);
  X[3] = sub(X[1], X[5]);
  X[5] = add(X[1], X[5]);
  X[1] = add(X[2], X[6]);
  X[2] = sub(X[2], X[6]);
  X[6] = add(X[0], X[4]);
  X[0] = sub(X[0], X[4]);

  X[4] = add(X[6], X[1]);
  X[6] = sub(X[6], X[1]);
  X[1] = add(X[3], mul(T, root28));
  X[3] = sub(X[3], mul(T, root28));
  T    = add(X[5], X[7]);
  X[5] = sub(X[5], X[7]);
  X[7] = add(X[0], mul(X[2], root28));
  X[0] = sub(X[0], mul(X[2], root28));

  X[2] = add(X[6], mul(X[5], root28));
  X[6] = sub(X[6], mul(X[5], root28));
  X[5] = sub(X[7], mul(X[1], root18));
  X[1] = add(X[7], mul(X[1], root18));
  X[7] = sub(X[0], mul(X[3], root38));
  X[3] = add(X[0], mul(X[3], root38));
  X[0] = add(X[4], T);
  X[4] = sub(X[4], T);

  for(int i=0;i<8;i++)
    out[i]=X[i];
}

void gpu4_strided(uint64_t* data, uint32_t offset) {
  uint64_t Y[4], X[4];

  for(int i=0;i<4;i++)
    X[i]=data[i*8+offset];
  fast4(Y, X);
  for(int i=0;i<4;i++)
    data[i*8+offset]=Y[i];
}

void gpu8_strided(uint64_t* data, uint32_t offset) {
  uint64_t Y[8], X[8];

  for(int i=0;i<8;i++)
    X[i]=data[i*8+offset];
  fast8(Y, X);
  for(int i=0;i<8;i++)
    data[i*8+offset]=Y[i];
}

void gpu8_contiguous(uint64_t* data, uint32_t base) {
  uint64_t Y[8], X[8];

  for(int i=0;i<8;i++)
    X[i]=data[i+base*8];
  fast8(Y, X);
  for(int i=0;i<8;i++)
    data[i+base*8]=Y[i];
}

void ntt1024(uint64_t* out, uint64_t* in) {
  uint64_t registers[32][32], shared[1024];
  uint32_t thread;
  uint64_t root32=0x40, root1k=power(root32k, 1<<22);

  // Simulate the GPU implementation with 32 threads, each thread has 32 points in registers.
  // This implementation does two 4-step methods, one for 32x32 across 32 threads and 4x8 within each thread.

  // The following resources may be helpful to understand this algorithm:
  //   https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
  // see section about breaking a 1D FFT into 2D FFts.
  //
  // This is sometimes referred to as Bailey's 4 step method.  See:
  //   https://en.wikipedia.org/wiki/Bailey%27s_FFT_algorithm

  // load strided
  for(thread=0;thread<32;thread++)
	for(int i=0;i<32;i++)
      registers[thread][i]=in[thread+i*32];

  // FFT1024
  for(thread=0;thread<32;thread++) {
    // FFT32
    for(int i=0;i<8;i++)
      gpu4_strided(registers[thread], i);
    for(int i=0;i<8;i++)
      for(int j=0;j<4;j++)
        registers[thread][i+j*8]=mul(registers[thread][i+j*8], power(root32, i*j));
    for(int i=0;i<4;i++) {
      gpu8_contiguous(registers[thread], i);
    }
    // the registers are transposed (4x8) at this point
  }

  // transpose the registers, store strided to shared
  for(thread=0;thread<32;thread++)
    for(int i=0;i<8;i++)
      for(int j=0;j<4;j++)
        shared[thread+(i*4+j)*32]=registers[thread][i+j*8];

  // load it contiguous from shared (thus transposing 32x32)
  for(thread=0;thread<32;thread++)
    for(int i=0;i<32;i++)
      registers[thread][i]=mul(shared[thread*32+i], power(root1k, thread*i));

  for(thread=0;thread<32;thread++) {
    // FFT32
    for(int i=0;i<8;i++)
      gpu4_strided(registers[thread], i);
    for(int i=0;i<8;i++)
      for(int j=0;j<4;j++)
        registers[thread][i+j*8]=mul(registers[thread][i+j*8], power(root32, i*j));
    for(int i=0;i<4;i++) {
      gpu8_contiguous(registers[thread], i);
    }
    // the registers are transposed (4x8) at this point
  }

  // tranpose the registers, store strided to global mem
  for(thread=0;thread<32;thread++) {
	for(int i=0;i<8;i++)
      for(int j=0;j<4;j++)
        out[thread+(i*4+j)*32]=registers[thread][i+j*8];
  }
}

uint64_t random_sample() {
  uint64_t x;

  x=rand() & 0xFFFF;
  x=(x<<16) + (rand() & 0xFFFF);
  x=(x<<16) + (rand() & 0xFFFF);
  x=(x<<16) + (rand() & 0xFFFF);
  if(x>0xFFFFFFFF00000001ull)
    x=x + 0xFFFFFFFFull;
  return x;
}

void random_samples(uint64_t* res, uint32_t count) {
  for(int i=0;i<count;i++)
    res[i]=random_sample();
}

int main(int argc, const char** argv) {
  int       ntts, repeatCount;
  uint64_t* nttData;
  uint64_t  samples[1024];

  if(argc!=3) {
    fprintf(stderr, "Usage:  correct <nttCount> <repeatCount>\n");
    fprintf(stderr, "Where <nttCount> is the number of 1024-point NTTs to run\n");
    fprintf(stderr, "and <repeatCount> is the number of consecutive NTT runs\n");
  }

  ntts=atoi(argv[1]);
  repeatCount=atoi(argv[2]);

  nttData=(uint64_t*)malloc(ntts*1024*sizeof(uint64_t));
  if(nttData==NULL) {
    fprintf(stderr, "Malloc failed\n");
    exit(1);
  }
  random_samples(nttData, 1024*ntts);

  for(int i=0;i<ntts;i++) {
    copy1024(samples, nttData + i*1024);
    for(int j=0;j<repeatCount;j++)
      ntt1024(samples, samples);
    for(int j=0;j<1024;j+=4)
      printf("%016lX %016lX %016lX %016lX\n", samples[j], samples[j+1], samples[j+2], samples[j+3]);
  }
}
