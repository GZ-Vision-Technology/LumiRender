#include "../core/ispc_common.isph"

static inline uint8 cvt_to_8bits(float v) {
  const int u8_lower = 0;
  const int u8_upper = 255;
  return (uint8)clamp((int)(v * 256.0), u8_lower, u8_upper);
}

static inline float cvt_from_8bits(uint8 v) {
  return (float)v / 255.0f;
}

static inline float srgb_to_linear_impl(float v) {
  return select((v < 0.04045f), (v / 12.92f),
    (pow((v + 0.055f) / 1.055f, 2.4f)));
}

static inline float linear_to_srgb_impl(float v) {
  return select((v < 0.0031308f),
          (v * 12.92f),
          (1.055f * pow(v, 1.0f / 2.4f - 0.055f)));
}

export enum TransformType {
  TO_SRGBA,
  TO_LINEAR,
  SCALE_VALUE
};

export void LM_EXTERNAL_API_ENTRY(transform_rgba8888_stream)(
  uniform TransformType type,
  uniform uint32 count,
  uniform const uint8 src[],
  uniform const float scale[4],
  uniform uint8 dest[]
) {

#if (TARGET_WIDTH %4 != 0)
#error "transform_rgba8888_stream" does not support lane with less than 4.
#endif

  float scale_in_lane = scale[programIndex & 3];

  if(type == TO_SRGBA) {

      for(int i = programIndex; i < count; i += programCount) {
        float val = cvt_from_8bits(src[i]) * scale_in_lane;
        val = linear_to_srgb_impl(val);
        dest[i] = cvt_to_8bits(val);
      }

  } else if(type == TO_LINEAR) {

    for(int i = programIndex; i < count; i += programCount) {
      float val = cvt_from_8bits(src[i]) * scale_in_lane;
      val = srgb_to_linear_impl(val);
      dest[i] = cvt_to_8bits(val);
    }

  } else if(type == SCALE_VALUE) {

    for(int i = programIndex; i < count; i += programCount) {
      float val = cvt_from_8bits(src[i]);
      val *= scale_in_lane;
      dest[i] = cvt_to_8bits(val);
    }

  }
}

