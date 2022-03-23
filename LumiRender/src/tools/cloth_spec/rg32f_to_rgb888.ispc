

typedef float<2> float2;
typedef int<2> int2;
typedef uint8<2> uchar2;

export void rg32f_to_rgb888(
  uniform uint32 count,
  uniform const float src[],
  uniform uint8 dest[]
) {

  foreach(i = 0 ... count) {
    float val = src[i];
    uint8 val2 = clamp((int)(val * 256.0), (int)0, (int)256);

    int k = i >> 1;
    int r = i & 1;
    dest[3*k + r] = val2;
    dest[3*k + 2] = 0;
  }

}