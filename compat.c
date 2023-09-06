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

#define __device__
#define __forceinline__

typedef struct {
  int32_t x, y, z, w;
} int4;

typedef struct {
  uint32_t x, y, z, w;
} uint4;

uint4 make_uint4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  uint4 res;

  res.x=x;
  res.y=y;
  res.z=z;
  res.w=w;
  return res;
}

uint32_t carry;

uint32_t __funnelshift_l(uint32_t a, uint32_t b, uint32_t bits) {
  bits=bits & 0x1F;
  if(bits==0)
    return b;
  return (b<<bits) | (a>>32-bits);
}

uint32_t uhigh(uint64_t x) {
  return (uint32_t)(x>>32);
}

uint32_t ulow(uint64_t x) {
  return (uint32_t)x;
}

uint64_t make_wide(uint32_t lo, uint32_t hi) {
  return (((uint64_t)hi)<<32) + lo;
}

uint64_t mulwide(uint32_t a, uint32_t b) {
  uint64_t la=a, lb=b;

  return la*lb;
}

uint64_t madwidec(uint32_t a, uint32_t b, uint64_t c) {
  uint64_t la=a, lb=b, p=la*lb+c+carry;

  return p;
}

uint64_t madwide_cc(uint32_t a, uint32_t b, uint64_t c) {
  uint64_t la=a, lb=b, p=la*lb+c;

  carry=(p<c) ? 1 : 0;
  return p;
}

uint32_t add_cc(uint32_t a, uint32_t b) {
  uint32_t s=a+b;

  carry=(s<a) ? 1 : 0;
  return s;
}

uint32_t addc_cc(uint32_t a, uint32_t b) {
  uint64_t la=a, lb=b, lc=carry, s=la+lb+lc;

  carry=(s>>32);
  return (uint32_t)s;
}

uint32_t addc(uint32_t a, uint32_t b) {
  return a+b+carry;
}

uint32_t sub_cc(uint32_t a, uint32_t b) {
  uint64_t la=a, lb=~b, s=la+lb+1;

  carry=(s>>32);
  return (uint32_t)s;
}

uint32_t subc_cc(uint32_t a, uint32_t b) {
  uint64_t la=a, lb=~b, lc=carry, s=la+lb+lc;

  carry=(s>>32);
  return (uint32_t)s;
}

uint32_t subc(uint32_t a, uint32_t b) {
  return a+~b+carry;
}

uint32_t prmt(uint32_t low, uint32_t high, uint32_t c) {
  int32_t  bytes[16];
  uint32_t res=0;

  for(int i=0;i<4;i++) {
    bytes[i]=(low>>i*8) & 0xFF;
    bytes[i+4]=(high>>i*8) & 0xFF;
  }
  for(int i=0;i<8;i++)
    bytes[i+8]=((bytes[i]<<24)>>31) & 0xFF;

  for(int i=3;i>=0;i--)
    res=(res<<8) + bytes[(c>>i*4) & 0xF];
  return res;
}
