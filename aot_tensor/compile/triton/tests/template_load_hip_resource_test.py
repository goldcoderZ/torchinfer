# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


"""Test that HIP templates contain correct AMD APIs.

This test validates the HIP (AMD) template resources by checking that they
contain the expected HIP-specific APIs and headers. It runs on CPU machines
without requiring GPU hardware since it only reads template files.
"""

import unittest
from unittest.mock import patch

# @dep=//aot_tensor/compile/triton/templates:triton_templates
from aot_tensor.compile.template_utils import TRITON_TEMPLATES
from aot_tensor.compile.triton.tests.template_visibility import (
    assert_generated_regions_are_hidden,
)


class TemplateHipApiTest(unittest.TestCase):
    def setUp(self) -> None:
        # Load the templates as a ROCm build of PyTorch would.
        patcher = patch("aot_tensor.compile.template_utils.is_amd", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_kernel_cpp_has_hip_apis(self) -> None:
        """Test kernel.cpp contains HIP-specific headers and APIs."""
        kernel_cpp = TRITON_TEMPLATES.load("kernel.cpp")

        self.assertIn("torch/csrc/stable/accelerator.h", kernel_cpp)
        self.assertIn("TRITON_AOT_CU_CHECK", kernel_cpp)
        self.assertIn("hipDeviceGetAttribute", kernel_cpp)
        self.assertIn("hipFunction_t", kernel_cpp)

        # Verify CUDA APIs and ATen GPU headers are NOT present.
        self.assertNotIn("cuDeviceGetAttribute", kernel_cpp)
        self.assertNotIn("CUfunction", kernel_cpp)
        self.assertNotIn("cuda/CUDAContext.h", kernel_cpp)
        self.assertNotIn("hip/HIPContext.h", kernel_cpp)
        self.assertNotIn("HIPStreamMasqueradingAsCUDA.h", kernel_cpp)

        # Device/stream uses stable ABI. ROCm builds of PyTorch keep the
        # shim's CUDA name.
        self.assertIn("triton_aot_get_current_stream", kernel_cpp)
        self.assertIn("aoti_torch_get_current_cuda_stream", kernel_cpp)
        self.assertNotIn("c10/cuda/CUDAStream.h", kernel_cpp)
        self.assertNotIn("c10/hip/HIPStream.h", kernel_cpp)
        self.assertNotIn("c10::cuda::current_device", kernel_cpp)

        # Error handling uses TRITON_AOT_CU_CHECK, not ATen macros.
        self.assertNotIn("AT_CUDA_DRIVER_CHECK", kernel_cpp)

        # No ATen headers.
        self.assertNotIn("ATen/Tensor.h", kernel_cpp)
        self.assertNotIn("torch/library.h", kernel_cpp)
        self.assertIn("torch/headeronly/core/ScalarType.h", kernel_cpp)

        # Scratch alloc uses stable Tensor, not AOTI runtime RAII handle.
        self.assertIn("torch/csrc/stable/tensor.h", kernel_cpp)
        self.assertIn("torch/csrc/stable/macros.h", kernel_cpp)
        self.assertNotIn("torch/csrc/inductor/aoti_runtime/utils.h", kernel_cpp)
        self.assertNotIn("RAIIAtenTensorHandle", kernel_cpp)

        # enable_large_smem_or_throw is a no-op on HIP (LDS needs no opt-in),
        # but the symbol must still exist so codegen ``load_*()`` calls link.
        self.assertIn(
            "void enable_large_smem_or_throw(int shared, hipFunction_t func)",
            kernel_cpp,
        )
        self.assertNotIn("void check_errors", kernel_cpp)
        # Uses hipCtxGetDevice for the current device (NOT hardcoded 0).
        self.assertIn("hipCtxGetDevice", kernel_cpp)
        self.assertNotIn("int device = 0", kernel_cpp)

    def test_kernel_h_has_hip_apis(self) -> None:
        """Test kernel.h contains HIP-specific types."""
        kernel_h = TRITON_TEMPLATES.load("kernel.h")

        self.assertIn("hipStream_t", kernel_h)
        self.assertIn("hip/hip_runtime.h", kernel_h)

        self.assertNotIn("cudaStream_t", kernel_h)
        self.assertNotIn("<cuda.h>", kernel_h)
        self.assertNotIn("cuda_runtime.h", kernel_h)

        # Stable ABI headers replace ATen.
        self.assertIn("torch/csrc/stable/tensor.h", kernel_h)
        self.assertNotIn("ATen/Tensor.h", kernel_h)
        self.assertNotIn("torch/types.h", kernel_h)

    def test_generated_regions_are_hidden(self) -> None:
        assert_generated_regions_are_hidden(self, TRITON_TEMPLATES)

    def test_torch_op_cpp_has_stable_abi(self) -> None:
        """Test torch_op.cpp, which HIP shares with CUDA, uses stable ABI
        registration."""
        torch_op = TRITON_TEMPLATES.load("torch_op.cpp")

        self.assertIn("STABLE_TORCH_LIBRARY_FRAGMENT", torch_op)
        self.assertIn("STABLE_TORCH_LIBRARY_IMPL", torch_op)
        self.assertIn("TORCH_BOX", torch_op)
        self.assertIn("torch/csrc/stable/library.h", torch_op)

        self.assertNotIn("ATen/Tensor.h", torch_op)
        self.assertNotIn(
            "TORCH_LIBRARY_FRAGMENT",
            torch_op.replace("STABLE_TORCH_LIBRARY_FRAGMENT", ""),
        )
