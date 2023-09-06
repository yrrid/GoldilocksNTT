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
#include "compat.c"

#define uint128_t unsigned __int128
#define int128_t __int128

int128_t reduce(int128_t value) {
  int128_t p=0xFFFFFFFF00000001ull;

  if(value<0) {
    value=-value%p;
    if(value!=0)
      value=p-value;
  }
  value=value%p;
  return value;
}

int128_t correct(int128_t value, int shift) {
  value=reduce(value);
  while(shift>=48) {
   value=reduce(value<<48);
   shift-=48;
  }
  return reduce(value<<shift);
}

uint32_t sign(uint32_t x) {
  return ((int32_t)x)>>31;
}

void dump(int128_t x) {
  printf("%016llX%016llX\n", (uint64_t)(x>>64), (uint64_t)x);
}

class TestSampled {
  public:

  typedef struct {
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t w;
  } Value;

  static void dumpValue(Value x) {
    printf("%08X %08X %08X %08X\n", x.w, x.z, x.y, x.x);
  }

  static Value make_int4(int32_t x, int32_t y, int32_t z, int32_t w) {
    Value r;

    r.x=x;
    r.y=y;
    r.z=z;
    r.w=w;
    return r;
  }

  static Value shiftRoot8(Value a, int amount) {
    if(amount==0)
      return a;
    else if(amount==1)
      return make_int4(-a.w, a.x, a.y, a.z);
    else if(amount==2)
      return make_int4(-a.z, -a.w, a.x, a.y);
    else if(amount==3)
      return make_int4(-a.y, -a.z, -a.w, a.x);
    else if(amount==4)
      return make_int4(-a.x, -a.y, -a.z, -a.w);
    else if(amount==5)
      return make_int4(a.w, -a.x, -a.y, -a.z);
    else if(amount==6)
      return make_int4(a.z, a.w, -a.x, -a.y);
    else if(amount==7)
      return make_int4(a.y, a.z, a.w, -a.x);
    return a;
  }

  static __device__ __forceinline__ Value shiftRoot32(const Value& a, int amount) {
    uint32_t left=(amount%4)*6, right=24-left;
    Value    rotated;
    Value    r;

    rotated=shiftRoot8(a, amount/4);
    if(left==0)
      return rotated;

    r.x=rotated.x<<left;
    r.y=rotated.y<<left;
    r.z=rotated.z<<left;
    r.w=rotated.w<<left;

    r.x=(r.x & 0x00FFFFFF) - (rotated.w>>right);
    r.y=(r.y & 0x00FFFFFF) + (rotated.x>>right);
    r.z=(r.z & 0x00FFFFFF) + (rotated.y>>right);
    r.w=(r.w & 0x00FFFFFF) + (rotated.z>>right);
    return r;
  }

  static Value shiftRoot64(Value a, int amount) {
    Value    rotate;
    Value    resolved, r;
    uint32_t l, m, h, hh, s;

    // output a condensed value

    rotate=shiftRoot8(a, amount/8);

    resolved.x=rotate.x;
    resolved.y=rotate.y + (resolved.x>>24);
    resolved.z=rotate.z + (resolved.y>>24);
    resolved.w=rotate.w + (resolved.z>>24);

    l=prmt(resolved.x, resolved.y, 0x4210);
    m=prmt(resolved.y, resolved.z, 0x5421);
    h=prmt(resolved.z, resolved.w, 0x6542);
    hh=resolved.w>>24;
    s=sign(resolved.w);

    if(amount%8>0) {
      hh=__funnelshift_l(h, hh, amount%8*3);
      h=__funnelshift_l(m, h, amount%8*3);
      m=__funnelshift_l(l, m, amount%8*3);
      l=l<<amount%8*3;
    }

    r.x=sub_cc(l, h);
    r.y=subc(h, 0);

    r.y=add_cc(r.y, m);
    r.z=addc(0, 0);

    r.x=sub_cc(r.x, hh);
    r.y=subc_cc(r.y, s);
    r.z=subc(r.z, s);

    return r;
  }

  static Value   normalize(Value a) {
    Value    resolved;
    Value    r;
    uint32_t l, m, h, hh, s;

    resolved.x=a.x - (a.w>>24);
    resolved.y=a.y + (resolved.x>>24);
    resolved.z=a.z + (resolved.y>>24);
    resolved.w=(a.w & 0x00FFFFFF) + (resolved.z>>24);

    l=prmt(resolved.x, resolved.y, 0x4210);
    m=prmt(resolved.y, resolved.z, 0x5421);
    h=prmt(resolved.z, resolved.w, 0x6542);
    hh=resolved.w>>24;
    s=sign(resolved.w);

    r.x=sub_cc(l, h);
    r.y=subc(h, 0);

    r.y=add_cc(r.y, m);
    r.z=addc(0, 0);

    r.x=sub_cc(r.x, hh);
    r.y=subc_cc(r.y, s);
    r.z=subc(r.z, s);

    s=((r.z | ~r.y)==0 && r.x!=0) ? 1 : r.z;

    r.x=add_cc(r.x, -s);
    r.y=addc(r.y, sign(s));
    r.z=0;
    return r;
  }

  static int32_t hardRandom32() {
    int32_t current;
    int     total=0, count;

    if((rand() & 0xFFFF)%10==0) {
      current=rand() & 0xFFFF;
      current=(current<<11) & (rand() & 0x7FF);
      return (current<<5)>>5;
    }
    while(total<27) {
      count=(rand() & 0xFFFF)%10;
      if(total+count>27)
        count=27-total;
      total+=count;
      if((rand()&1)==0)
        current=current<<count;
      else
        current=(current+1<<count)-1;
    }
    current=(current<<5)>>5;
    return current;
  }

  static Value hardRandom() {
    return make_int4(hardRandom32(), hardRandom32(), hardRandom32(), hardRandom32());
  }

  static int128_t unchop(Value a) {
    int128_t x;

    x=a.w;
    x=(x<<24) + a.z;
    x=(x<<24) + a.y;
    x=(x<<24) + a.x;
    return x;
  }

  static int128_t uncondense(Value a) {
    int128_t x;

    x=a.z;
    x=(x<<32) + (uint32_t)a.y;
    x=(x<<32) + (uint32_t)a.x;
    return x;
  }

  static void test() {
    hardRandom();
    for(int i=0;i<1000000;i++) {
      Value    vIn=hardRandom(), vOut;
      int128_t x=unchop(vIn);

      for(int rotate=0;rotate<32;rotate++) {
        vOut=shiftRoot32(vIn, rotate);
        if(reduce(unchop(vOut))!=correct(x, rotate*6)) {
          printf("Failed on shiftRoot32 %08X %08X %08X %08X, rotate=%d\n", vIn.w, vIn.z, vIn.y, vIn.x, rotate);
          return;
        }
      }
      for(int rotate=0;rotate<64;rotate++) {
        vOut=shiftRoot64(vIn, rotate);
        if(reduce(uncondense(vOut))!=correct(x, rotate*3)) {
          printf("Failed on shiftRoot64 %08X %08X %08X %08X, rotate=%d\n", vIn.w, vIn.z, vIn.y, vIn.x, rotate);
          return;
        }
      }
      vOut=normalize(vIn);
      if(uncondense(vOut)!=reduce(x)) {
        printf("Normalized failed on %08X %08X %08X %08X\n", vIn.w, vIn.z, vIn.y, vIn.x);
        dump(uncondense(vOut));
        dump(reduce(x));
        return;
      }
    }
    printf("All sampled tests successful\n");
  }
};

int main() {
  printf("Running sampled tests\n");
  TestSampled::test();
}