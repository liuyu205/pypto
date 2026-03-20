#include "pto/pto-inst.hpp"
#include "runtime/rt.h"
#include <pto/npu/a2a3/custom/TSyncCVID.hpp>
using namespace pto;
constexpr uint16_t FA_QK_READY = 7;
constexpr uint16_t FA_P_READY = 8;
constexpr uint16_t FA_PV_READY = 9;
__global__ AICORE void flash_attention_kernel(__gm__ uint16_t* ffts_addr, __gm__ half* v1, __gm__ half* v2, __gm__ half* v3, __gm__ float* v4, __gm__ half* v5, __gm__ half* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ half* v10, int32_t v11, int32_t v12) {
  set_ffts_base_addr((uint64_t)ffts_addr);
  unsigned v13 = 8192;
  RoundMode v14 = RoundMode::CAST_ROUND;
  unsigned v15 = 16384;
  unsigned v16 = 64;
  unsigned v17 = 128;
  unsigned v18 = 1;
  unsigned v19 = 0;
  int64_t v20 = 163840;
  int64_t v21 = 131072;
  int64_t v22 = 98304;
  int64_t v23 = 81920;
  int64_t v24 = 65536;
  int64_t v25 = 32768;
  int64_t v26 = 16384;
  int64_t v27 = 0;
  float v28 = 0.0883883535f;
  int32_t v29 = 2;
  int32_t v30 = 0;
  int32_t v31 = 64;
  int32_t v32 = 1;
  int32_t v33 = 128;
  using T = float;
  size_t v34 = (size_t) v32;
  size_t v35 = (size_t) v30;
  Tile<TileType::Mat, half, 128, 64, BLayout::ColMajor, 128, 64, SLayout::RowMajor, 512, PadValue::Null> v36;
  TASSIGN(v36, v27);
  Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null> v37;
  TASSIGN(v37, v26);
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v38;
  TASSIGN(v38, v25);
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v39;
  TASSIGN(v39, v24);
  Tile<TileType::Left, half, 128, 64, BLayout::RowMajor, 128, 64, SLayout::RowMajor, 512, PadValue::Null> v40;
  TASSIGN(v40, v27);
  Tile<TileType::Right, half, 64, 128, BLayout::RowMajor, 64, 128, SLayout::ColMajor, 512, PadValue::Null> v41;
  TASSIGN(v41, v27);
  Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v42;
  TASSIGN(v42, v25);
  Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v43;
  TASSIGN(v43, v25);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v44;
  TASSIGN(v44, v27);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v45;
  TASSIGN(v45, v24);
  Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, 64, 128, SLayout::NoneBox, 512, PadValue::Null> v46;
  TASSIGN(v46, v27);
  Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, 64, 128, SLayout::NoneBox, 512, PadValue::Null> v47;
  TASSIGN(v47, v25);
  Tile<TileType::Vec, half, 64, 128, BLayout::RowMajor, 64, 128, SLayout::NoneBox, 512, PadValue::Null> v48;
  TASSIGN(v48, v24);
  Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, 64, 1, SLayout::NoneBox, 512, PadValue::Null> v49;
  TASSIGN(v49, v23);
  Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, 64, 128, SLayout::NoneBox, 512, PadValue::Null> v50;
  TASSIGN(v50, v22);
  Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, 64, 128, SLayout::NoneBox, 512, PadValue::Null> v51;
  TASSIGN(v51, v21);
  Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, 64, 128, SLayout::NoneBox, 512, PadValue::Null> v52;
  TASSIGN(v52, v20);

  #if defined(__DAV_CUBE__)
  int64_t v53 = get_block_idx();
  int64_t v54 = get_block_num();
  for (size_t v55 = (size_t) ((int32_t) (int64_t) v53); v55 < ((size_t) (v11 / v33)); v55 += (size_t) ((int32_t) (int64_t) v54)) {
    int32_t v56 = (int32_t) v55;
    for (size_t v57 = v35; v57 < ((size_t) (v12 / v33)); v57 += v34) {
      int32_t v58 = (int32_t) v57;
      for (size_t v59 = v35; v59 < ((size_t) v29); v59 += v34) {
        int32_t v60 = (int32_t) v59;
        int32_t v61 = (int32_t) ((uint32_t) v60 * (uint32_t) v31);
        pto::Shape<1, 1, 1, 128, 64> v62 = pto::Shape<1, 1, 1, 128, 64>();
        pto::Stride<16384, 16384, 16384, 128, 1> v63 = pto::Stride<16384, 16384, 16384, 128, 1>();
        GlobalTensor<half, pto::Shape<1, 1, 1, 128, 64>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v64 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 64>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v1 + (v19 + (unsigned) ((int32_t) (uint32_t) v56 * (uint32_t) v33) * (unsigned) v33 + (unsigned) v61 * (unsigned) v32), v62, v63);
        TLOAD(v36, v64);
        unsigned v65 = (unsigned) v12;
        unsigned v66 = v16 * v65;
        pto::Shape<1, 1, 1, 64, 128> v67 = pto::Shape<1, 1, 1, 64, 128>();
        pto::Stride<-1, -1, -1, -1, 1> v68 = pto::Stride<-1, -1, -1, -1, 1>(v66, v66, v66, v65);
        GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v69 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v2 + (v19 + (unsigned) v61 * (unsigned) v12 + (unsigned) ((int32_t) (uint32_t) v58 * (uint32_t) v33) * (unsigned) v32), v67, v68);
        TLOAD(v37, v69);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        TMOV(v40, v36);
        TMOV(v41, v37);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        if (v60 == v30) {
          TMATMUL(v44, v40, v41);
        } else {
          TMATMUL_ACC(v44, v44, v40, v41);
        };
        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID2);
      };
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      unsigned v70 = (unsigned) v12;
      unsigned v71 = v17 * v70;
      pto::Shape<1, 1, 1, 128, 128> v72 = pto::Shape<1, 1, 1, 128, 128>();
      pto::Stride<-1, -1, -1, -1, 1> v73 = pto::Stride<-1, -1, -1, -1, 1>(v71, v71, v71, v70);
      GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v74 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v4 + (v19 + (unsigned) ((int32_t) (uint32_t) v56 * (uint32_t) v33) * (unsigned) v12 + (unsigned) ((int32_t) (uint32_t) v58 * (uint32_t) v33) * (unsigned) v32), v72, v73);
      TSTORE(v74, v44);
      pipe_barrier(PIPE_ALL);
    };
    ffts_cross_core_sync(PIPE_FIX, pto::_getFFTSMsg(pto::CV_CORE_SYNC, FA_QK_READY));
  }
  #endif // __DAV_CUBE__


  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v75 = get_block_idx();
  int64_t v76 = get_block_num();
  int64_t v77 = get_subblockid();
  size_t v78 = (size_t) (v12 / v33);
  for (size_t v79 = (size_t) ((int32_t) (int64_t) v75); v79 < ((size_t) (v11 / v33)); v79 += (size_t) ((int32_t) (int64_t) v76)) {
    wait_flag_dev(FA_QK_READY);
    int32_t v80 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) ((int32_t) v79) * (uint32_t) v29) + (uint32_t) ((int32_t) (int64_t) v77));
    for (size_t v81 = v35; v81 < v78; v81 += v34) {
      int32_t v82 = (int32_t) v81;
      pipe_barrier(PIPE_ALL);
      int32_t v83 = (int32_t) ((uint32_t) v80 * (uint32_t) v31);
      int32_t v84 = (int32_t) ((uint32_t) v82 * (uint32_t) v33);
      unsigned v85 = (unsigned) v12;
      unsigned v86 = v16 * v85;
      pto::Shape<1, 1, 1, 64, 128> v87 = pto::Shape<1, 1, 1, 64, 128>();
      pto::Stride<-1, -1, -1, -1, 1> v88 = pto::Stride<-1, -1, -1, -1, 1>(v86, v86, v86, v85);
      GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v89 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v4 + (v19 + (unsigned) v83 * (unsigned) v12 + (unsigned) v84 * (unsigned) v32), v87, v88);
      TLOAD(v46, v89);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      if (v82 == v30) {
        TROWMAX(v49, v46, v47);
        TROWEXPAND(v50, v49);
        TSUB(v47, v46, v50);
        TMULS(v47, v47, v28);
        TEXP(v46, v47);
        TROWSUM(v49, v46, v47);
        TROWEXPAND(v51, v49);
      } else {
        TROWMAX(v49, v46, v47);
        TROWEXPAND(v47, v49);
        TMAX(v47, v47, v50);
        TSUB(v52, v50, v47);
        TMULS(v52, v52, v28);
        TEXP(v52, v52);
        TMUL(v51, v51, v52);
        TMOV(v50, v47);
        TSUB(v47, v46, v50);
        TMULS(v47, v47, v28);
        TEXP(v46, v47);
        TROWSUM(v49, v46, v47);
        TROWEXPAND(v47, v49);
        TADD(v51, v51, v47);
      };
      TCVT(v48, v46, v14);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      unsigned v90 = (unsigned) v12;
      unsigned v91 = v16 * v90;
      pto::Shape<1, 1, 1, 64, 128> v92 = pto::Shape<1, 1, 1, 64, 128>();
      pto::Stride<-1, -1, -1, -1, 1> v93 = pto::Stride<-1, -1, -1, -1, 1>(v91, v91, v91, v90);
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v94 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v5 + (v19 + (unsigned) v83 * (unsigned) v12 + (unsigned) v84 * (unsigned) v32), v92, v93);
      TSTORE(v94, v48);
    };
    for (size_t v95 = v35; v95 < v78; v95 += v34) {
      pipe_barrier(PIPE_ALL);
      int32_t v96 = (int32_t) ((uint32_t) v80 * (uint32_t) v31);
      int32_t v97 = (int32_t) ((uint32_t) ((int32_t) v95) * (uint32_t) v33);
      unsigned v98 = (unsigned) v12;
      unsigned v99 = v16 * v98;
      pto::Shape<1, 1, 1, 64, 128> v100 = pto::Shape<1, 1, 1, 64, 128>();
      pto::Stride<-1, -1, -1, -1, 1> v101 = pto::Stride<-1, -1, -1, -1, 1>(v99, v99, v99, v98);
      GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v102 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v4 + (v19 + (unsigned) v96 * (unsigned) v12 + (unsigned) v97 * (unsigned) v32), v100, v101);
      TLOAD(v46, v102);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TSUB(v47, v46, v50);
      TMULS(v47, v47, v28);
      TEXP(v46, v47);
      TDIV(v46, v46, v51);
      TCVT(v48, v46, v14);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      unsigned v103 = (unsigned) v12;
      unsigned v104 = v16 * v103;
      pto::Shape<1, 1, 1, 64, 128> v105 = pto::Shape<1, 1, 1, 64, 128>();
      pto::Stride<-1, -1, -1, -1, 1> v106 = pto::Stride<-1, -1, -1, -1, 1>(v104, v104, v104, v103);
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v107 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v6 + (v19 + (unsigned) v96 * (unsigned) v12 + (unsigned) v97 * (unsigned) v32), v105, v106);
      TSTORE(v107, v48);
    };
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TROWMAX(v49, v50, v47);
    int32_t v108 = (int32_t) ((uint32_t) v80 * (uint32_t) v31);
    pto::Shape<1, 1, 1, 64, 1> v109 = pto::Shape<1, 1, 1, 64, 1>();
    pto::Stride<64, 64, 64, 1, 1> v110 = pto::Stride<64, 64, 64, 1, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 64, 1>, pto::Stride<64, 64, 64, 1, 1>, pto::Layout::ND> v111 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 1>, pto::Stride<64, 64, 64, 1, 1>, pto::Layout::ND>(v8 + (v19 + (unsigned) v108 * (unsigned) v32 + v19 * (unsigned) v32), v109, v110);
    TSTORE(v111, v49);
    TROWMAX(v49, v51, v47);
    pto::Shape<1, 1, 1, 64, 1> v112 = pto::Shape<1, 1, 1, 64, 1>();
    pto::Stride<64, 64, 64, 1, 1> v113 = pto::Stride<64, 64, 64, 1, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 64, 1>, pto::Stride<64, 64, 64, 1, 1>, pto::Layout::ND> v114 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 1>, pto::Stride<64, 64, 64, 1, 1>, pto::Layout::ND>(v9 + (v19 + (unsigned) v108 * (unsigned) v32 + v19 * (unsigned) v32), v112, v113);
    TSTORE(v114, v49);
    ffts_cross_core_sync(PIPE_MTE3, pto::_getFFTSMsg(pto::CV_CORE_SYNC, FA_P_READY));
  }
  #endif // __DAV_VEC__


  #if defined(__DAV_CUBE__)
  int64_t v115 = get_block_idx();
  int64_t v116 = get_block_num();
  for (size_t v117 = (size_t) ((int32_t) (int64_t) v115); v117 < ((size_t) (v11 / v33)); v117 += (size_t) ((int32_t) (int64_t) v116)) {
    wait_flag_dev(FA_P_READY);
    int32_t v118 = (int32_t) v117;
    for (size_t v119 = v35; v119 < ((size_t) (v12 / v33)); v119 += v34) {
      int32_t v120 = (int32_t) v119;
      int32_t v121 = (int32_t) ((uint32_t) v120 * (uint32_t) v33);
      unsigned v122 = (unsigned) v12;
      unsigned v123 = v17 * v122;
      pto::Shape<1, 1, 1, 128, 128> v124 = pto::Shape<1, 1, 1, 128, 128>();
      pto::Stride<-1, -1, -1, -1, 1> v125 = pto::Stride<-1, -1, -1, -1, 1>(v123, v123, v123, v122);
      GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v126 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v6 + (v19 + (unsigned) ((int32_t) (uint32_t) v118 * (uint32_t) v33) * (unsigned) v12 + (unsigned) v121 * (unsigned) v32), v124, v125);
      TLOAD(v38, v126);
      pto::Shape<1, 1, 1, 128, 128> v127 = pto::Shape<1, 1, 1, 128, 128>();
      pto::Stride<16384, 16384, 16384, 128, 1> v128 = pto::Stride<16384, 16384, 16384, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v129 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v3 + (v19 + (unsigned) v121 * (unsigned) v33 + v19 * (unsigned) v32), v127, v128);
      TLOAD(v39, v129);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      TMOV(v42, v38);
      TMOV(v43, v39);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      if (v120 == v30) {
        TMATMUL(v45, v42, v43);
      } else {
        TMATMUL_ACC(v45, v45, v42, v43);
      };
      set_flag(PIPE_M, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID2);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 128, 128> v130 = pto::Shape<1, 1, 1, 128, 128>();
    pto::Stride<16384, 16384, 16384, 128, 1> v131 = pto::Stride<16384, 16384, 16384, 128, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v132 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v7 + (v19 + (unsigned) ((int32_t) (uint32_t) v118 * (uint32_t) v33) * (unsigned) v33 + v19 * (unsigned) v32), v130, v131);
    TSTORE(v132, v45);
    pipe_barrier(PIPE_ALL);
    ffts_cross_core_sync(PIPE_FIX, pto::_getFFTSMsg(pto::CV_CORE_SYNC, FA_PV_READY));
  }
  #endif // __DAV_CUBE__


  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v133 = get_block_idx();
  int64_t v134 = get_block_num();
  int64_t v135 = get_subblockid();
  for (size_t v136 = (size_t) ((int32_t) (int64_t) v133); v136 < ((size_t) (v11 / v33)); v136 += (size_t) ((int32_t) (int64_t) v134)) {
    wait_flag_dev(FA_PV_READY);
    pipe_barrier(PIPE_ALL);
    int32_t v137 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) ((int32_t) v136) * (uint32_t) v29) + (uint32_t) ((int32_t) (int64_t) v135)) * (uint32_t) v31);
    pto::Shape<1, 1, 1, 64, 128> v138 = pto::Shape<1, 1, 1, 64, 128>();
    pto::Stride<8192, 8192, 8192, 128, 1> v139 = pto::Stride<8192, 8192, 8192, 128, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v140 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v7 + (v19 + (unsigned) v137 * (unsigned) v33 + v19 * (unsigned) v32), v138, v139);
    TLOAD(v46, v140);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(v48, v46, v14);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    pto::Shape<1, 1, 1, 64, 128> v141 = pto::Shape<1, 1, 1, 64, 128>();
    pto::Stride<8192, 8192, 8192, 128, 1> v142 = pto::Stride<8192, 8192, 8192, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v143 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v10 + (v19 + (unsigned) v137 * (unsigned) v33 + v19 * (unsigned) v32), v141, v142);
    TSTORE(v143, v48);
  }
  #endif // __DAV_VEC__

  return;
}

extern "C" void call_kernel(
    uint32_t blockDim, void* stream,
    uint8_t* v1, uint8_t* v2, uint8_t* v3, uint8_t* v4, uint8_t* v5, uint8_t* v6, uint8_t* v7, uint8_t* v8, uint8_t* v9, uint8_t* v10, int32_t v11, int32_t v12)
{
    uint64_t ffts = 0;
    uint32_t fftsLen = 0;
    rtGetC2cCtrlAddr(&ffts, &fftsLen);
    flash_attention_kernel<<<blockDim, nullptr, stream>>>((uint16_t *)ffts, (half *)v1, (half *)v2, (half *)v3, (float *)v4, (half *)v5, (half *)v6, (float *)v7, (float *)v8, (float *)v9, (half *)v10, v11, v12);
}
