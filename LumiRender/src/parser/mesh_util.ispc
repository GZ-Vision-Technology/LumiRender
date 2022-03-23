#include "../core/ispc_common.isph"


static unmasked void transform_vertex_layout_stream_packed_lane4f(
  uniform const uint64 count,
  uniform const float vertices_p3[],
  uniform const float normals_p3[],
  uniform const float *texcoords_p3,
  uniform float out_vertices_p4[],
  uniform float out_normals_p4[],
  uniform float out_texcoords_p2[],
  uniform float aabb_p4[]
) {

  uniform uint64 count_s4 = count * 4;
  uniform uint64 packed_pos = 0;

  float aabb_min = FLT_MAX, aabb_max = -FLT_MAX;

  assume(count_s4 % programCount == 0);
  for(uint64 i = programIndex, j = programIndex; j < count_s4; i += 3, j += 4, packed_pos += 4) {

      float pos = 1.0;
      if(programIndex < 3) pos = vertices_p3[i];
      packed_store_active((uniform int32 * uniform)out_vertices_p4 + packed_pos, intbits(pos));
      // out_vertices_p4[j] = val;

      aabb_min = min(aabb_min, pos);
      aabb_max = max(aabb_max, pos);

      float normal = 0.0;
      if(programIndex < 3) normal = normals_p3[i];
      // out_normals_p4[j] = val;
      packed_store_active((uniform int32 * uniform)out_normals_p4 + packed_pos, intbits(normal));
  }

  aabb_p4[programIndex] = aabb_min;
  // streaming_store(aabb_p4 + programIndex, aabb_min[programIndex]);
  aabb_p4[programIndex + 4] = aabb_max;
  // streaming_store(aabb_p4 + programIndex, aabb_max[programIndex]);

  if(texcoords_p3 != NULL) {
    uniform uint64 count_s2 = count * 2;

    for(uint64 i = programIndex, j = programIndex; j < count_s2; i += 6, j += 4) {

      float val;
      if(programIndex < 2)
        val = texcoords_p3[i];
      else
        val = texcoords_p3[i+1];
      
      out_texcoords_p2[j] = val;
    }
  }
}

static unmasked void transform_vertex_layout_stream_packed_lane8f(
  uniform const uint64 count,
  uniform const float vertices_p3[],
  uniform const float normals_p3[],
  uniform const float texcoords_p3[],
  uniform float out_vertices_p4[],
  uniform float out_normals_p4[],
  uniform float out_texcoords_p2[],
  uniform float aabb_p4[]
) {

  uniform uint64 count_s4 = count * 4;
  uniform int64 packed_pos = 0;

  float aabb_max_min = FLT_MAX;

  for(uint64 i = programIndex, j = programIndex; j < count_s4; i += 6, j += 8, packed_pos += 8) {

      float pos = 1.0;
      if(programIndex < 3) pos = vertices_p3[i];
      else if(programIndex > 3 && programIndex < 7) pos = vertices_p3[i-1];
      packed_store_active((uniform int32 * uniform)out_vertices_p4 + packed_pos, intbits(pos));
      // out_vertices_p4[j] = pos;

      if(programIndex < 4)
        pos *= -1;

      aabb_max_min = min(aabb_max_min, pos);

      float normal = 0.0;
      if(programIndex < 3) normal = normals_p3[i];
      else if(programIndex > 3 && programIndex < 7) normal = normals_p3[i-1];
      packed_store_active((uniform int32 * uniform)out_normals_p4 + packed_pos, intbits(normal));
      // out_normals_p4[j] = normal;
  }

  if(programIndex < 4)
    aabb_p4[programIndex] = -aabb_max_min;
  // streaming_store(aabb_p4 + programIndex, aabb_min[programIndex]);
  else
    aabb_p4[programIndex] = aabb_max_min;
  // streaming_store(aabb_p4 + programIndex, aabb_max[programIndex]);

  if(texcoords_p3 != NULL) {
    uniform uint64 count_s2 = count * 2;

    for(uint64 i = programIndex, j = programIndex; j < count_s2; i += 12, j += 8) {

      float val;
      if(programIndex < 2)
        val = texcoords_p3[i];
      else if(programIndex < 4)
        val = texcoords_p3[i+1];
      else if(programIndex < 6)
        val = texcoords_p3[i+2];
      else
        val = texcoords_p3[i+3];
      
      out_texcoords_p2[j] = val;
    }
  }
}

export void LM_EXTERNAL_API_ENTRY(transform_vertex_layout_stream)(
  uniform const uint64 count,
  uniform const float vertices_p3[],
  uniform const float normals_p3[],
  uniform const float texcoords_p3[],
  uniform float out_vertices_p4[],
  uniform float out_normals_p4[],
  uniform float out_texcoords_p2[],
  uniform float aabb_p4[]
) {

#if TARGET_WIDTH == 4
  transform_vertex_layout_stream_packed_lane4f(
    count,
    vertices_p3,
    normals_p3,
    texcoords_p3,
    out_vertices_p4,
    out_normals_p4,
    out_texcoords_p2,
    aabb_p4
  );
#elif TARGET_WIDTH == 8
  transform_vertex_layout_stream_packed_lane8f(
    count,
    vertices_p3,
    normals_p3,
    texcoords_p3,
    out_vertices_p4,
    out_normals_p4,
    out_texcoords_p2,
    aabb_p4
  );
#else
#error unsupported target lane with
#endif
}