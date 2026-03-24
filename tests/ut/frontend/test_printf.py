# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime validation example for plm.printf."""

import torch
import torch_npu

import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


@fe.kernel
def printf_kernel(
    x: pl.Tensor[[16, 16], pl.INT32],
    y: pl.Tensor[[16, 16], pl.INT32],
) -> pl.Tensor[[16, 16], pl.INT32]:
    with pl.section_vector():
        tile = pl.load(x, offsets=[0, 0], shapes=[16, 16])
        value_i: pl.Scalar[pl.INT32] = 12
        value_x: pl.Scalar[pl.INT32] = 255
        value_f: pl.Scalar[pl.FP32] = 3.5
        plm.printf("i=%d x=%x f=%f\n", value_i, value_x, value_f)
        pl.store(tile, offsets=[0, 0], shapes=[16, 16], output_tensor=y)

    return y


@fe.jit()
def test_printf():
    compiled_lib = fe.compile(printf_kernel, arch="dav-c220-vec", enable_print_debug=True)
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:1"
    torch.npu.set_device(device)

    x = torch.arange(16 * 16, device=device, dtype=torch.int32).reshape(16, 16)
    y = torch.empty_like(x)

    fe.launch(None, 1, compiled_lib, x, y)
    torch.npu.synchronize()

    print("***********npu output***********")
    print(y.shape, y.dtype)
    print(y)

    print("***********golden output***********")
    print(x.shape, x.dtype)
    print(x)

    torch.testing.assert_close(y, x)
    print("result equal!")


if __name__ == "__main__":
    test_printf()
    print("\nAll tests passed!")
