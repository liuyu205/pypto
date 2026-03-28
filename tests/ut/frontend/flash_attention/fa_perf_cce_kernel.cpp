
#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

__global__ AICORE void fa_perf_kernel(__gm__ half* q_0, __gm__ half* k_0, __gm__ half* v_0, __gm__ half* o_0, __gm__ float* qk_buf_0, __gm__ half* p_buf_0, __gm__ float* pv_buf_0, int32_t Sq, int32_t D, int32_t Skv, int32_t SqFifo, __gm__ int64_t* ffts_addr)
{
    int32_t _local_Sq = Sq;
    int32_t _local_D = D;
    int32_t _local_Skv = Skv;
    int32_t _local_SqFifo = SqFifo;
    set_ffts_base_addr((unsigned long)ffts_addr);

    using q_0GlobalShapeDim5 = Shape<1, 1, 1, 64, 128>;
    using q_0GlobalStrideDim5 = Stride<1, 1, 1, 128, 1>;
    using q_0GlobalType = GlobalTensor<half, q_0GlobalShapeDim5, q_0GlobalStrideDim5>;
    q_0GlobalType q_0Global(q_0);

    using k_0GlobalShapeDim5 = Shape<1, 1, 1, 128, 128>;
    using k_0GlobalStrideDim5 = Stride<128, 128, 128, 1, 128>;
    using k_0GlobalType = GlobalTensor<half, k_0GlobalShapeDim5, k_0GlobalStrideDim5, Layout::DN>;
    k_0GlobalType k_0Global(k_0);

    using v_0GlobalShapeDim5 = Shape<1, 1, 1, 128, 128>;
    using v_0GlobalStrideDim5 = Stride<1, 1, 1, 128, 1>;
    using v_0GlobalType = GlobalTensor<half, v_0GlobalShapeDim5, v_0GlobalStrideDim5>;
    v_0GlobalType v_0Global(v_0);

    using o_0GlobalShapeDim5 = Shape<1, 1, 1, 64, 128>;
    using o_0GlobalStrideDim5 = Stride<1, 1, 1, 128, 1>;
    using o_0GlobalType = GlobalTensor<half, o_0GlobalShapeDim5, o_0GlobalStrideDim5>;
    o_0GlobalType o_0Global(o_0);

    using qk_buf_0GlobalShapeDim5 = Shape<1, 1, 1, 64, 128>;
    using qk_buf_0GlobalStrideDim5 = Stride<1, 1, 1, 128, 1>;
    using qk_buf_0GlobalType = GlobalTensor<float, qk_buf_0GlobalShapeDim5, qk_buf_0GlobalStrideDim5>;
    qk_buf_0GlobalType qk_buf_0Global(qk_buf_0);

    using p_buf_0GlobalShapeDim5 = Shape<1, 1, 1, 64, 128>;
    using p_buf_0GlobalStrideDim5 = Stride<1, 1, 1, 128, 1>;
    using p_buf_0GlobalType = GlobalTensor<half, p_buf_0GlobalShapeDim5, p_buf_0GlobalStrideDim5>;
    p_buf_0GlobalType p_buf_0Global(p_buf_0);

    using pv_buf_0GlobalShapeDim5 = Shape<1, 1, 1, 64, 128>;
    using pv_buf_0GlobalStrideDim5 = Stride<1, 1, 1, 128, 1>;
    using pv_buf_0GlobalType = GlobalTensor<float, pv_buf_0GlobalShapeDim5, pv_buf_0GlobalStrideDim5>;
    pv_buf_0GlobalType pv_buf_0Global(pv_buf_0);

    #if defined(__DAV_CUBE__)
    // Tile declarations (Cube)
    using q_mat_buf_0_0Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    q_mat_buf_0_0Type q_mat_buf_0_0(64, 128);
    TASSIGN(q_mat_buf_0_0, 0x0);
    using q_mat_buf_0_1Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    q_mat_buf_0_1Type q_mat_buf_0_1(64, 128);
    TASSIGN(q_mat_buf_0_1, 0x4000);
    using k_mat_buf_0_0Type = Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    k_mat_buf_0_0Type k_mat_buf_0_0(128, 128);
    TASSIGN(k_mat_buf_0_0, 0x8000);
    using k_mat_buf_0_1Type = Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    k_mat_buf_0_1Type k_mat_buf_0_1(128, 128);
    TASSIGN(k_mat_buf_0_1, 0x10000);
    using p_mat_buf_0_0Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    p_mat_buf_0_0Type p_mat_buf_0_0(64, 128);
    TASSIGN(p_mat_buf_0_0, 0x18000);
    using p_mat_buf_0_1Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    p_mat_buf_0_1Type p_mat_buf_0_1(64, 128);
    TASSIGN(p_mat_buf_0_1, 0x1c000);
    using v_mat_buf_0_0Type = Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    v_mat_buf_0_0Type v_mat_buf_0_0(128, 128);
    TASSIGN(v_mat_buf_0_0, 0x20000);
    using v_mat_buf_0_1Type = Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    v_mat_buf_0_1Type v_mat_buf_0_1(128, 128);
    TASSIGN(v_mat_buf_0_1, 0x28000);
    using left_buf_0_0Type = Tile<TileType::Left, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512>;
    left_buf_0_0Type left_buf_0_0(64, 128);
    TASSIGN(left_buf_0_0, 0x0);
    using left_buf_0_1Type = Tile<TileType::Left, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512>;
    left_buf_0_1Type left_buf_0_1(64, 128);
    TASSIGN(left_buf_0_1, 0x4000);
    using right_buf_0_0Type = Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    right_buf_0_0Type right_buf_0_0(128, 128);
    TASSIGN(right_buf_0_0, 0x0);
    using right_buf_0_1Type = Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    right_buf_0_1Type right_buf_0_1(128, 128);
    TASSIGN(right_buf_0_1, 0x8000);
    using acc_buf_0_0Type = Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
    acc_buf_0_0Type acc_buf_0_0(64, 128);
    TASSIGN(acc_buf_0_0, 0x0);
    using acc_buf_0_1Type = Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
    acc_buf_0_1Type acc_buf_0_1(64, 128);
    TASSIGN(acc_buf_0_1, 0x8000);
    #endif  // __DAV_CUBE__

    #if defined(__DAV_VEC__)
    // Tile declarations (Vector)
    using qk_vec_0Type = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    qk_vec_0Type qk_vec_0(64, 128);
    TASSIGN(qk_vec_0, 0x0);
    using tmp_vec_0Type = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    tmp_vec_0Type tmp_vec_0(64, 128);
    TASSIGN(tmp_vec_0, 0x8000);
    using p_f16_0Type = Tile<TileType::Vec, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    p_f16_0Type p_f16_0(64, 128);
    TASSIGN(p_f16_0, 0x10000);
    using reduce_dst_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    reduce_dst_0Type reduce_dst_0(64, 1);
    TASSIGN(reduce_dst_0, 0x14000);
    using reduce_dst_rm_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    reduce_dst_rm_0Type reduce_dst_rm_0(1, 64);
    TASSIGN(reduce_dst_rm_0, 0x14000);
    using global_max_buf_0_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    global_max_buf_0_0Type global_max_buf_0_0(64, 1);
    TASSIGN(global_max_buf_0_0, 0x14100);
    using global_max_buf_0_1Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    global_max_buf_0_1Type global_max_buf_0_1(64, 1);
    TASSIGN(global_max_buf_0_1, 0x14200);
    using global_max_rm_buf_0_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    global_max_rm_buf_0_0Type global_max_rm_buf_0_0(1, 64);
    TASSIGN(global_max_rm_buf_0_0, 0x14100);
    using global_max_rm_buf_0_1Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    global_max_rm_buf_0_1Type global_max_rm_buf_0_1(1, 64);
    TASSIGN(global_max_rm_buf_0_1, 0x14200);
    using global_sum_buf_0_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    global_sum_buf_0_0Type global_sum_buf_0_0(64, 1);
    TASSIGN(global_sum_buf_0_0, 0x14300);
    using global_sum_buf_0_1Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    global_sum_buf_0_1Type global_sum_buf_0_1(64, 1);
    TASSIGN(global_sum_buf_0_1, 0x14400);
    using global_sum_rm_buf_0_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    global_sum_rm_buf_0_0Type global_sum_rm_buf_0_0(1, 64);
    TASSIGN(global_sum_rm_buf_0_0, 0x14300);
    using global_sum_rm_buf_0_1Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    global_sum_rm_buf_0_1Type global_sum_rm_buf_0_1(1, 64);
    TASSIGN(global_sum_rm_buf_0_1, 0x14400);
    using exp_corr_fifo_0_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    exp_corr_fifo_0_0Type exp_corr_fifo_0_0(64, 1);
    TASSIGN(exp_corr_fifo_0_0, 0x14500);
    using exp_corr_fifo_0_1Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    exp_corr_fifo_0_1Type exp_corr_fifo_0_1(64, 1);
    TASSIGN(exp_corr_fifo_0_1, 0x14600);
    using exp_corr_rm_fifo_0_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    exp_corr_rm_fifo_0_0Type exp_corr_rm_fifo_0_0(1, 64);
    TASSIGN(exp_corr_rm_fifo_0_0, 0x14500);
    using exp_corr_rm_fifo_0_1Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    exp_corr_rm_fifo_0_1Type exp_corr_rm_fifo_0_1(1, 64);
    TASSIGN(exp_corr_rm_fifo_0_1, 0x14600);
    using running_o_0Type = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    running_o_0Type running_o_0(64, 128);
    TASSIGN(running_o_0, 0x14700);
    using pv_vec_0Type = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    pv_vec_0Type pv_vec_0(64, 128);
    TASSIGN(pv_vec_0, 0x1c700);
    using o_f16_0Type = Tile<TileType::Vec, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    o_f16_0Type o_f16_0(64, 128);
    TASSIGN(o_f16_0, 0x24700);
    #endif  // __DAV_VEC__


    // Function body
    auto sq_dim_0 = _local_Sq;
    auto skv_dim_0 = _local_Skv;
    auto sq_tiles_0 = ((sq_dim_0 + 127) / 128);
    auto skv_tiles_0 = ((skv_dim_0 + 127) / 128);
    auto num_cores_0 = (int32_t)(get_block_num());
    auto core_id_0 = (int32_t)(get_block_idx());
    #if defined(__DAV_CUBE__)
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    auto _ctx_sq_off_0 = 0;
    auto _ctx_row_off_0 = 0;
    auto _ctx_task_id_0 = 0;
    auto _ctx_q_count_0 = 0;
    auto _ctx_buf_idx_0 = 0;
    auto _ctx_l0ab_idx_0 = 0;
    auto _ctx_l0c_idx_0 = 0;
    auto _ctx_core_id_0 = core_id_0;
    // Loop-carried values initialization
    auto _ctx_buf_idx_iter_1 = _ctx_buf_idx_0;
    auto _ctx_l0ab_idx_iter_1 = _ctx_l0ab_idx_0;
    auto _ctx_l0c_idx_iter_1 = _ctx_l0c_idx_0;
    auto _ctx_q_count_iter_1 = _ctx_q_count_0;
    auto _ctx_row_off_iter_1 = _ctx_row_off_0;
    auto _ctx_sq_off_iter_1 = _ctx_sq_off_0;
    auto _ctx_task_id_iter_1 = _ctx_task_id_0;

    const event_t _eid_0_1[] = {(event_t)0, (event_t)1};
    Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512> q_mat_buf_0[] = {q_mat_buf_0_0, q_mat_buf_0_1};
    Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512> k_mat_buf_0[] = {k_mat_buf_0_0, k_mat_buf_0_1};
    Tile<TileType::Left, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512> left_buf_0[] = {left_buf_0_0, left_buf_0_1};
    Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512> right_buf_0[] = {right_buf_0_0, right_buf_0_1};
    Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024> acc_buf_0[] = {acc_buf_0_0, acc_buf_0_1};
    const event_t _eid_2_3[] = {(event_t)2, (event_t)3};
    Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512> v_mat_buf_0[] = {v_mat_buf_0_0, v_mat_buf_0_1};
    Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512> p_mat_buf_0[] = {p_mat_buf_0_0, p_mat_buf_0_1};
    const event_t _eid_4_5[] = {(event_t)4, (event_t)5};

    for (uint64_t qi_0 = core_id_0; qi_0 < sq_tiles_0; qi_0 += num_cores_0) {
        auto _ctx_sq_off_3 = (qi_0 * 128);
        // Loop-carried values initialization
        auto _ctx_buf_idx_iter_3 = _ctx_buf_idx_iter_1;
        auto _ctx_l0ab_idx_iter_3 = _ctx_l0ab_idx_iter_1;
        auto _ctx_l0c_idx_iter_3 = _ctx_l0c_idx_iter_1;
        auto _ctx_q_count_iter_3 = _ctx_q_count_iter_1;
        auto _ctx_row_off_iter_3 = _ctx_row_off_iter_1;
        auto _ctx_task_id_iter_3 = _ctx_task_id_iter_1;

        for (uint64_t row_idx_0 = 0; row_idx_0 < 2; row_idx_0 += 1) {
            auto _ctx_row_off_5 = (row_idx_0 * 64);
            // Loop-carried values initialization
            auto _ctx_buf_idx_iter_5 = _ctx_buf_idx_iter_3;
            auto _ctx_l0ab_idx_iter_5 = _ctx_l0ab_idx_iter_3;
            auto _ctx_l0c_idx_iter_5 = _ctx_l0c_idx_iter_3;
            auto _ctx_task_id_iter_5 = _ctx_task_id_iter_3;

            for (uint64_t pre_0 = 0; pre_0 < 1; pre_0 += 1) {
                auto _ctx_task_id_7 = pre_0;
                auto _ctx_buf_idx_7 = (((_ctx_q_count_iter_3 * skv_tiles_0) + pre_0) % 2);
                auto q_mat_idx_0 = (_ctx_q_count_iter_3 % 2);
                auto qk_fifo_slot_0 = (_ctx_task_id_7 % 2);
                auto skv_off_0 = (_ctx_task_id_7 * 128);

                wait_flag(PIPE_MTE1, PIPE_MTE2, _eid_0_1[_ctx_buf_idx_7]);
                if ((_ctx_task_id_7 == 0)) {

                    TASSIGN(q_0Global, q_0 + ((_ctx_sq_off_3 + _ctx_row_off_5) * _local_D + 0));
                    TLOAD(q_mat_buf_0[q_mat_idx_0], q_0Global);
                }


                TASSIGN(k_0Global, k_0 + (skv_off_0 * _local_D + 0));
                TLOAD(k_mat_buf_0[_ctx_buf_idx_7], k_0Global);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

                wait_flag(PIPE_M, PIPE_MTE1, _eid_0_1[_ctx_l0ab_idx_iter_5]);


                TMOV(left_buf_0[_ctx_l0ab_idx_iter_5], q_mat_buf_0[q_mat_idx_0]);


                TMOV(right_buf_0[_ctx_l0ab_idx_iter_5], k_mat_buf_0[_ctx_buf_idx_7]);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

                set_flag(PIPE_MTE1, PIPE_MTE2, _eid_0_1[_ctx_buf_idx_7]);

                wait_flag(PIPE_FIX, PIPE_M, _eid_0_1[_ctx_l0c_idx_iter_5]);



                TMATMUL(acc_buf_0[_ctx_l0c_idx_iter_5], left_buf_0[_ctx_l0ab_idx_iter_5], right_buf_0[_ctx_l0ab_idx_iter_5]);
                set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

                set_flag(PIPE_M, PIPE_MTE1, _eid_0_1[_ctx_l0ab_idx_iter_5]);

                TASSIGN(qk_buf_0Global, qk_buf_0 + ((((qk_fifo_slot_0 * sq_dim_0) + _ctx_sq_off_3) + _ctx_row_off_5) * _local_Skv + skv_off_0));
                TSTORE(qk_buf_0Global, acc_buf_0[_ctx_l0c_idx_iter_5]);

                set_flag(PIPE_FIX, PIPE_M, _eid_0_1[_ctx_l0c_idx_iter_5]);
                auto _ctx_l0ab_idx_7 = (1 - _ctx_l0ab_idx_iter_5);
                auto _ctx_l0c_idx_7 = (1 - _ctx_l0c_idx_iter_5);

                ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, _eid_0_1[qk_fifo_slot_0]));
                _ctx_buf_idx_iter_5 = _ctx_buf_idx_7;
                _ctx_l0ab_idx_iter_5 = _ctx_l0ab_idx_7;
                _ctx_l0c_idx_iter_5 = _ctx_l0c_idx_7;
                _ctx_task_id_iter_5 = _ctx_task_id_7;
            }

            // Loop-carried values initialization
            auto _ctx_buf_idx_iter_8 = _ctx_buf_idx_iter_5;
            auto _ctx_l0ab_idx_iter_8 = _ctx_l0ab_idx_iter_5;
            auto _ctx_l0c_idx_iter_8 = _ctx_l0c_idx_iter_5;
            auto _ctx_task_id_iter_8 = _ctx_task_id_iter_5;

            for (uint64_t ki_0 = 0; ki_0 < skv_tiles_0; ki_0 += 1) {
                auto next_ki_0 = (ki_0 + 1);
                int64_t _ctx_buf_idx_11;
                int64_t _ctx_l0ab_idx_11;
                int64_t _ctx_l0c_idx_11;
                int64_t _ctx_task_id_11;

                if ((next_ki_0 < skv_tiles_0)) {
                    auto _ctx_task_id_10 = next_ki_0;
                    auto _ctx_buf_idx_10 = (((_ctx_q_count_iter_3 * skv_tiles_0) + next_ki_0) % 2);
                    auto q_mat_idx_1 = (_ctx_q_count_iter_3 % 2);
                    auto qk_fifo_slot_1 = (_ctx_task_id_10 % 2);
                    auto skv_off_1 = (_ctx_task_id_10 * 128);

                    wait_flag(PIPE_MTE1, PIPE_MTE2, _eid_0_1[_ctx_buf_idx_10]);
                    if ((_ctx_task_id_10 == 0)) {

                        TASSIGN(q_0Global, q_0 + ((_ctx_sq_off_3 + _ctx_row_off_5) * _local_D + 0));
                        TLOAD(q_mat_buf_0[q_mat_idx_1], q_0Global);
                    }


                    TASSIGN(k_0Global, k_0 + (skv_off_1 * _local_D + 0));
                    TLOAD(k_mat_buf_0[_ctx_buf_idx_10], k_0Global);
                    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

                    wait_flag(PIPE_M, PIPE_MTE1, _eid_0_1[_ctx_l0ab_idx_iter_8]);


                    TMOV(left_buf_0[_ctx_l0ab_idx_iter_8], q_mat_buf_0[q_mat_idx_1]);


                    TMOV(right_buf_0[_ctx_l0ab_idx_iter_8], k_mat_buf_0[_ctx_buf_idx_10]);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

                    set_flag(PIPE_MTE1, PIPE_MTE2, _eid_0_1[_ctx_buf_idx_10]);

                    wait_flag(PIPE_FIX, PIPE_M, _eid_0_1[_ctx_l0c_idx_iter_8]);



                    TMATMUL(acc_buf_0[_ctx_l0c_idx_iter_8], left_buf_0[_ctx_l0ab_idx_iter_8], right_buf_0[_ctx_l0ab_idx_iter_8]);
                    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

                    set_flag(PIPE_M, PIPE_MTE1, _eid_0_1[_ctx_l0ab_idx_iter_8]);

                    TASSIGN(qk_buf_0Global, qk_buf_0 + ((((qk_fifo_slot_1 * sq_dim_0) + _ctx_sq_off_3) + _ctx_row_off_5) * _local_Skv + skv_off_1));
                    TSTORE(qk_buf_0Global, acc_buf_0[_ctx_l0c_idx_iter_8]);

                    set_flag(PIPE_FIX, PIPE_M, _eid_0_1[_ctx_l0c_idx_iter_8]);
                    auto _ctx_l0ab_idx_10 = (1 - _ctx_l0ab_idx_iter_8);
                    auto _ctx_l0c_idx_10 = (1 - _ctx_l0c_idx_iter_8);

                    ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, _eid_0_1[qk_fifo_slot_1]));
                    _ctx_buf_idx_11 = _ctx_buf_idx_10;
                    _ctx_l0ab_idx_11 = _ctx_l0ab_idx_10;
                    _ctx_l0c_idx_11 = _ctx_l0c_idx_10;
                    _ctx_task_id_11 = _ctx_task_id_10;
                } else {
                    _ctx_buf_idx_11 = _ctx_buf_idx_iter_8;
                    _ctx_l0ab_idx_11 = _ctx_l0ab_idx_iter_8;
                    _ctx_l0c_idx_11 = _ctx_l0c_idx_iter_8;
                    _ctx_task_id_11 = _ctx_task_id_iter_8;
                }

                auto _ctx_task_id_12 = ki_0;
                auto _ctx_buf_idx_12 = (((_ctx_q_count_iter_3 * skv_tiles_0) + ki_0) % 2);
                auto q_mat_idx_2 = (_ctx_q_count_iter_3 % 2);
                auto pv_task_slot_0 = (_ctx_task_id_12 % 2);
                auto sv_off_0 = (_ctx_task_id_12 * 128);
                auto pv_fifo_slot_0 = (_ctx_task_id_12 % 2);

                wait_flag(PIPE_MTE1, PIPE_MTE2, _eid_2_3[_ctx_buf_idx_12]);

                TASSIGN(v_0Global, v_0 + (sv_off_0 * _local_D + 0));
                TLOAD(v_mat_buf_0[_ctx_buf_idx_12], v_0Global);

                wait_flag_dev(_eid_2_3[pv_fifo_slot_0]);

                TASSIGN(p_buf_0Global, p_buf_0 + ((((pv_fifo_slot_0 * sq_dim_0) + _ctx_sq_off_3) + _ctx_row_off_5) * _local_Skv + sv_off_0));
                TLOAD(p_mat_buf_0[_ctx_buf_idx_12], p_buf_0Global);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

                wait_flag(PIPE_M, PIPE_MTE1, _eid_0_1[_ctx_l0ab_idx_11]);


                TMOV(left_buf_0[_ctx_l0ab_idx_11], p_mat_buf_0[_ctx_buf_idx_12]);


                TMOV(right_buf_0[_ctx_l0ab_idx_11], v_mat_buf_0[_ctx_buf_idx_12]);

                set_flag(PIPE_MTE1, PIPE_MTE2, _eid_2_3[_ctx_buf_idx_12]);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

                wait_flag(PIPE_FIX, PIPE_M, _eid_0_1[_ctx_l0c_idx_11]);



                TMATMUL(acc_buf_0[_ctx_l0c_idx_11], left_buf_0[_ctx_l0ab_idx_11], right_buf_0[_ctx_l0ab_idx_11]);
                set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

                set_flag(PIPE_M, PIPE_MTE1, _eid_0_1[_ctx_l0ab_idx_11]);

                TASSIGN(pv_buf_0Global, pv_buf_0 + (((((_ctx_core_id_0 * 512) + ((q_mat_idx_2 * 2) * 128)) + (pv_task_slot_0 * 128)) + _ctx_row_off_5) * _local_D + 0));
                TSTORE(pv_buf_0Global, acc_buf_0[_ctx_l0c_idx_11]);

                set_flag(PIPE_FIX, PIPE_M, _eid_0_1[_ctx_l0c_idx_11]);
                auto _ctx_l0ab_idx_12 = (1 - _ctx_l0ab_idx_11);
                auto _ctx_l0c_idx_12 = (1 - _ctx_l0c_idx_11);

                ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, _eid_4_5[pv_task_slot_0]));
                _ctx_buf_idx_iter_8 = _ctx_buf_idx_12;
                _ctx_l0ab_idx_iter_8 = _ctx_l0ab_idx_12;
                _ctx_l0c_idx_iter_8 = _ctx_l0c_idx_12;
                _ctx_task_id_iter_8 = _ctx_task_id_12;
            }

            auto _ctx_q_count_5 = (_ctx_q_count_iter_3 + 1);
            _ctx_buf_idx_iter_3 = _ctx_buf_idx_iter_8;
            _ctx_l0ab_idx_iter_3 = _ctx_l0ab_idx_iter_8;
            _ctx_l0c_idx_iter_3 = _ctx_l0c_idx_iter_8;
            _ctx_q_count_iter_3 = _ctx_q_count_5;
            _ctx_row_off_iter_3 = _ctx_row_off_5;
            _ctx_task_id_iter_3 = _ctx_task_id_iter_8;
        }

        _ctx_buf_idx_iter_1 = _ctx_buf_idx_iter_3;
        _ctx_l0ab_idx_iter_1 = _ctx_l0ab_idx_iter_3;
        _ctx_l0c_idx_iter_1 = _ctx_l0c_idx_iter_3;
        _ctx_q_count_iter_1 = _ctx_q_count_iter_3;
        _ctx_row_off_iter_1 = _ctx_row_off_iter_3;
        _ctx_sq_off_iter_1 = _ctx_sq_off_3;
        _ctx_task_id_iter_1 = _ctx_task_id_iter_3;
    }

    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    #endif  // __DAV_CUBE__
    #if defined(__DAV_VEC__)
    auto q_count_0 = 0;
    // Loop-carried values initialization
    auto q_count_iter_1 = q_count_0;

    Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512> global_max_buf_0[] = {global_max_buf_0_0, global_max_buf_0_1};
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512> global_max_rm_buf_0[] = {global_max_rm_buf_0_0, global_max_rm_buf_0_1};
    Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512> global_sum_buf_0[] = {global_sum_buf_0_0, global_sum_buf_0_1};
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512> global_sum_rm_buf_0[] = {global_sum_rm_buf_0_0, global_sum_rm_buf_0_1};
    const event_t _eid_0_1[] = {(event_t)0, (event_t)1};
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512> exp_corr_rm_fifo_0[] = {exp_corr_rm_fifo_0_0, exp_corr_rm_fifo_0_1};
    const event_t _eid_2_3[] = {(event_t)2, (event_t)3};
    const event_t _eid_4_5[] = {(event_t)4, (event_t)5};
    Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512> exp_corr_fifo_0[] = {exp_corr_fifo_0_0, exp_corr_fifo_0_1};

    for (uint64_t qi_1 = core_id_0; qi_1 < sq_tiles_0; qi_1 += num_cores_0) {
        auto sq_off_0 = (qi_1 * 128);
        // Loop-carried values initialization
        auto q_count_iter_3 = q_count_iter_1;

        for (uint64_t row_idx_1 = 0; row_idx_1 < 2; row_idx_1 += 1) {
            auto row_off_0 = (row_idx_1 * 64);
            auto q_idx_0 = (q_count_iter_3 % 2);




            for (uint64_t pre_1 = 0; pre_1 < 1; pre_1 += 1) {
                auto p_fifo_slot_0 = (pre_1 % 2);

                wait_flag_dev(_eid_0_1[p_fifo_slot_0]);
                auto p_fifo_slot_1 = (pre_1 % 2);
                auto skv_off_2 = (pre_1 * 128);
                TASSIGN(qk_buf_0Global, qk_buf_0 + ((((p_fifo_slot_1 * sq_dim_0) + sq_off_0) + row_off_0) * _local_Skv + skv_off_2));
                TLOAD(qk_vec_0, qk_buf_0Global);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                if ((pre_1 == 0)) {
                    TROWMAX(reduce_dst_0, qk_vec_0, tmp_vec_0);
                    pipe_barrier(PIPE_V);
                    TROWEXPANDSUB(tmp_vec_0, qk_vec_0, reduce_dst_0);
                    TMULS(global_max_rm_buf_0[q_idx_0], reduce_dst_rm_0, 1.000000);
                    TMULS(tmp_vec_0, tmp_vec_0, 0.088388);
                    TEXP(qk_vec_0, tmp_vec_0);
                    pipe_barrier(PIPE_V);
                    TROWSUM(reduce_dst_0, qk_vec_0, tmp_vec_0);
                    pipe_barrier(PIPE_V);
                    TMULS(global_sum_rm_buf_0[q_idx_0], reduce_dst_rm_0, 1.000000);
                    TCVT(p_f16_0, qk_vec_0, RoundMode::CAST_ROUND);
                }

                if ((pre_1 > 0)) {
                    TROWMAX(reduce_dst_0, qk_vec_0, tmp_vec_0);
                    pipe_barrier(PIPE_V);
                    TMAX(reduce_dst_rm_0, reduce_dst_rm_0, global_max_rm_buf_0[q_idx_0]);
                    pipe_barrier(PIPE_V);

                    TSUB(exp_corr_rm_fifo_0[p_fifo_slot_1], global_max_rm_buf_0[q_idx_0], reduce_dst_rm_0);
                    pipe_barrier(PIPE_V);
                    TMULS(global_max_rm_buf_0[q_idx_0], reduce_dst_rm_0, 1.000000);
                    pipe_barrier(PIPE_V);
                    TROWEXPANDSUB(tmp_vec_0, qk_vec_0, reduce_dst_0);


                    TMULS(exp_corr_rm_fifo_0[p_fifo_slot_1], exp_corr_rm_fifo_0[p_fifo_slot_1], 0.088388);
                    TMULS(tmp_vec_0, tmp_vec_0, 0.088388);


                    TEXP(exp_corr_rm_fifo_0[p_fifo_slot_1], exp_corr_rm_fifo_0[p_fifo_slot_1]);
                    TEXP(qk_vec_0, tmp_vec_0);
                    TCVT(p_f16_0, qk_vec_0, RoundMode::CAST_ROUND);
                    pipe_barrier(PIPE_V);

                    TMUL(global_sum_rm_buf_0[q_idx_0], global_sum_rm_buf_0[q_idx_0], exp_corr_rm_fifo_0[p_fifo_slot_1]);
                    TROWSUM(reduce_dst_0, qk_vec_0, tmp_vec_0);
                    pipe_barrier(PIPE_V);
                    TADD(global_sum_rm_buf_0[q_idx_0], global_sum_rm_buf_0[q_idx_0], reduce_dst_rm_0);
                }

                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TASSIGN(p_buf_0Global, p_buf_0 + ((((p_fifo_slot_1 * sq_dim_0) + sq_off_0) + row_off_0) * _local_Skv + skv_off_2));
                TSTORE(p_buf_0Global, p_f16_0);

                ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(FFTS_MODE_VAL, _eid_2_3[p_fifo_slot_1]));
            }

            for (uint64_t ki_1 = 0; ki_1 < skv_tiles_0; ki_1 += 1) {
                auto next_ki_1 = (ki_1 + 1);
                if ((next_ki_1 < skv_tiles_0)) {
                    auto p_fifo_slot_2 = (next_ki_1 % 2);

                    wait_flag_dev(_eid_0_1[p_fifo_slot_2]);
                    auto p_fifo_slot_3 = (next_ki_1 % 2);
                    auto skv_off_3 = (next_ki_1 * 128);
                    TASSIGN(qk_buf_0Global, qk_buf_0 + ((((p_fifo_slot_3 * sq_dim_0) + sq_off_0) + row_off_0) * _local_Skv + skv_off_3));
                    TLOAD(qk_vec_0, qk_buf_0Global);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    if ((next_ki_1 == 0)) {
                        TROWMAX(reduce_dst_0, qk_vec_0, tmp_vec_0);
                        pipe_barrier(PIPE_V);
                        TROWEXPANDSUB(tmp_vec_0, qk_vec_0, reduce_dst_0);
                        TMULS(global_max_rm_buf_0[q_idx_0], reduce_dst_rm_0, 1.000000);
                        TMULS(tmp_vec_0, tmp_vec_0, 0.088388);
                        TEXP(qk_vec_0, tmp_vec_0);
                        pipe_barrier(PIPE_V);
                        TROWSUM(reduce_dst_0, qk_vec_0, tmp_vec_0);
                        pipe_barrier(PIPE_V);
                        TMULS(global_sum_rm_buf_0[q_idx_0], reduce_dst_rm_0, 1.000000);
                        TCVT(p_f16_0, qk_vec_0, RoundMode::CAST_ROUND);
                    }

                    if ((next_ki_1 > 0)) {
                        TROWMAX(reduce_dst_0, qk_vec_0, tmp_vec_0);
                        pipe_barrier(PIPE_V);
                        TMAX(reduce_dst_rm_0, reduce_dst_rm_0, global_max_rm_buf_0[q_idx_0]);
                        pipe_barrier(PIPE_V);

                        TSUB(exp_corr_rm_fifo_0[p_fifo_slot_3], global_max_rm_buf_0[q_idx_0], reduce_dst_rm_0);
                        pipe_barrier(PIPE_V);
                        TMULS(global_max_rm_buf_0[q_idx_0], reduce_dst_rm_0, 1.000000);
                        pipe_barrier(PIPE_V);
                        TROWEXPANDSUB(tmp_vec_0, qk_vec_0, reduce_dst_0);


                        TMULS(exp_corr_rm_fifo_0[p_fifo_slot_3], exp_corr_rm_fifo_0[p_fifo_slot_3], 0.088388);
                        TMULS(tmp_vec_0, tmp_vec_0, 0.088388);


                        TEXP(exp_corr_rm_fifo_0[p_fifo_slot_3], exp_corr_rm_fifo_0[p_fifo_slot_3]);
                        TEXP(qk_vec_0, tmp_vec_0);
                        TCVT(p_f16_0, qk_vec_0, RoundMode::CAST_ROUND);
                        pipe_barrier(PIPE_V);

                        TMUL(global_sum_rm_buf_0[q_idx_0], global_sum_rm_buf_0[q_idx_0], exp_corr_rm_fifo_0[p_fifo_slot_3]);
                        TROWSUM(reduce_dst_0, qk_vec_0, tmp_vec_0);
                        pipe_barrier(PIPE_V);
                        TADD(global_sum_rm_buf_0[q_idx_0], global_sum_rm_buf_0[q_idx_0], reduce_dst_rm_0);
                    }

                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    TASSIGN(p_buf_0Global, p_buf_0 + ((((p_fifo_slot_3 * sq_dim_0) + sq_off_0) + row_off_0) * _local_Skv + skv_off_3));
                    TSTORE(p_buf_0Global, p_f16_0);

                    ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(FFTS_MODE_VAL, _eid_2_3[p_fifo_slot_3]));
                }

                auto q_mat_idx_3 = (q_count_iter_3 % 2);
                auto pv_slot_0 = (ki_1 % 2);

                wait_flag_dev(_eid_4_5[pv_slot_0]);
                if ((ki_1 == 0)) {
                    TASSIGN(pv_buf_0Global, pv_buf_0 + (((((core_id_0 * 512) + ((q_mat_idx_3 * 2) * 128)) + (pv_slot_0 * 128)) + row_off_0) * _local_D + 0));
                    TLOAD(running_o_0, pv_buf_0Global);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                }

                if ((ki_1 > 0)) {
                    auto gu_fifo_slot_0 = (ki_1 % 2);
                    TASSIGN(pv_buf_0Global, pv_buf_0 + (((((core_id_0 * 512) + ((q_mat_idx_3 * 2) * 128)) + (pv_slot_0 * 128)) + row_off_0) * _local_D + 0));
                    TLOAD(pv_vec_0, pv_buf_0Global);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

                    TROWEXPANDMUL(running_o_0, running_o_0, exp_corr_fifo_0[gu_fifo_slot_0]);
                    TADD(running_o_0, running_o_0, pv_vec_0);
                }

            }

            auto q_count_5 = (q_count_iter_3 + 1);
            TROWEXPANDDIV(running_o_0, running_o_0, global_sum_buf_0[q_idx_0]);
            TCVT(o_f16_0, running_o_0, RoundMode::CAST_ROUND);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TASSIGN(o_0Global, o_0 + ((sq_off_0 + row_off_0) * _local_D + 0));
            TSTORE(o_0Global, o_f16_0);
            q_count_iter_3 = q_count_5;
        }

        q_count_iter_1 = q_count_iter_3;
    }

    #endif  // __DAV_VEC__
}
