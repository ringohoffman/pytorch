import copy
import re
from typing import List, Optional

import torch
from third_party.cutlass.tools.library.scripts import (
    gemm_operation as cutlass_gemm_op,
    library as cutlass_lib,
)

from ...ir import IRNode, Layout
from ..common import IndentedBuffer

from . import cutlass_utils
from .cuda_kernel import CUDATemplateKernel
from .cuda_template import CutlassTemplate

# Only supports alpha * A@B + beta * C now.
# TODO: Support arbitrary epilogue after epilogue visitor is released in cutlass 3.2.
GEMM_TEMPLATE = r"""
{{template.header().getvalue()}}

{{template.globals().getvalue()}}

{{instance_definition}}

// When workspace_size is not a nullptr, populates requested workspace_size and returns.
// Otherwise, compuates the Gemm kernel using the given workspace ptr.
extern "C" {
{{kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=input_reorder)}} {
  {{kernel.check_not_null(X)}}
  {{kernel.check_not_null(W)}}
  {{kernel.check_not_null(Bias)}}
  {{kernel.check_not_null(Y)}}

  int64_t B = {{kernel.size(Y, 0, -3, default_value=1)}};
  int64_t M = {{kernel.size(X, -2)}};
  int64_t K = {{kernel.size(X, -1)}};
  int64_t N = {{kernel.size(W, -1)}};

  using ElementComputeEpilogue = {{instance_type}}::ElementAccumulator;
  using coord_t = cutlass::gemm::GemmCoord::Index;
  {{instance_type}}::Arguments arguments;
  {{gemm_arguments}}

  {{instance_type}} gemm_op;

  if (workspace_size) {
    *workspace_size = gemm_op.get_workspace_size(arguments);
    return;
  }

  {
    auto status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);
  }
  {
    auto status = gemm_op.initialize(arguments, workspace, stream);
    CUTLASS_CHECK(status);
  }
  {
    auto status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }

  return;
}
}

"""


GEMM_ARGS_CUTLASS_2X = r"""
  int64_t batch_stride_x = {{kernel.stride(X, -3)}};
  int64_t row_stride_x = {{kernel.row_stride(X)}};

  int64_t batch_stride_w = {{kernel.stride(W, -3)}};
  int64_t row_stride_w = {{kernel.row_stride(W)}};

  int64_t batch_stride_bias = {{kernel.stride(Bias, -3)}};
  int64_t row_stride_bias = {{kernel.row_stride(Bias)}};

  int64_t batch_stride_y = {{kernel.stride(Y, -3)}};
  int64_t row_stride_y = {{kernel.row_stride(Y)}};

  // Initialize GemmUniversalInstance arguments.
  arguments = {
    {{template.gemm_mode()}},  // GemmUniversalMode mode
    {
      static_cast<coord_t>(M),
      static_cast<coord_t>(N),
      static_cast<coord_t>(K)
    },  // GemmCoord problem_size
    {{split_k if split_k > 1 else 'B'}},  // int batch_count
    {ElementComputeEpilogue({{alpha}}), ElementComputeEpilogue({{beta}})},  // typename EpilogueOutputOp::Params epilogue
    ({{template.dtype(X)}}*)({{kernel.ptr(X)}}),  // void const * ptr_A
    ({{template.dtype(W)}}*)({{kernel.ptr(W)}}),  // void const * ptr_B
    ({{template.dtype(Bias)}}*)({{kernel.ptr(Bias)}}),  // void const * ptr_C
    ({{template.dtype(Y)}}*)({{kernel.ptr(Y)}}),  // void * ptr_D
    batch_stride_x,  // int64_t batch_stride_A
    batch_stride_w,  // int64_t batch_stride_B
    batch_stride_bias,  // int64_t batch_stride_C
    batch_stride_y,  // int64_t batch_stride_D
    row_stride_x,  // typename LayoutA::Stride::LongIndex lda
    row_stride_w,  // typename LayoutB::Stride::LongIndex ldb
    row_stride_bias,  // typename LayoutC::Stride::LongIndex ldc
    row_stride_y,  // typename LayoutC::Stride::LongIndex ldd
  };
"""


GEMM_ARGS_CUTLASS_3X = r"""
  // Initialize GemmUniversal3xInstance arguments.
  arguments = {
    {{template.gemm_mode()}},  // GemmUniversalMode mode
    {
      static_cast<coord_t>(M),
      static_cast<coord_t>(N),
      static_cast<coord_t>(K),
      static_cast<coord_t>(B)
    }, // ProblemShape problem_shape
    {
      ({{template.dtype(X)}}*)({{kernel.ptr(X)}}),  // ElementA const* ptr_A
      { {{template.cute_int(kernel.stride(X, -2), "stride_x0")}}, {{template.cute_int(kernel.stride(X, -1), "stride_x1")}}, {{template.cute_int(kernel.stride(X, -3), "batch_stride_x")}}},  // StrideA dA
      ({{template.dtype(W)}}*)({{kernel.ptr(W)}}),  // ElementB const* ptr_B
      { {{template.cute_int(kernel.stride(W, -1), "stride_w1")}}, {{template.cute_int(kernel.stride(W, -2), "stride_w0")}}, {{template.cute_int(kernel.stride(W, -3), "batch_stride_w")}}},  // StrideB dB
    },  // MainloopArguments mainloop
    {{epilogue_arguments}}
  };
"""


GEMM_ARGS_CUTLASS_3X_EPILOGUE_NO_TMA = r"""
    {
      {ElementComputeEpilogue({{alpha}}), ElementComputeEpilogue({{beta}})},  // typename ThreadEpilogueOp::Params thread
      ({{template.dtype(Bias)}}*)({{kernel.ptr(Bias)}}),  // ElementC const* ptr_C
      { {{template.cute_int(kernel.stride(Bias, -2), "stride_bias0")}}, {{template.cute_int(kernel.stride(Bias, -1), "stride_bias1")}}, {{template.cute_int(kernel.stride(Bias, -3), "batch_stride_bias")}}},  // StrideC dC
      ({{template.dtype(Y)}}*)({{kernel.ptr(Y)}}),  // ElementD const* ptr_D
      { {{template.cute_int(kernel.stride(Y, -2), "stride_y0")}}, {{template.cute_int(kernel.stride(Y, -1), "stride_y1")}}, {{template.cute_int(kernel.stride(Y, -3), "batch_stride_y")}}},  // StrideD dD
    },  // EpilogueArguments epilogue, no TMA
"""


GEMM_ARGS_CUTLASS_3X_EPILOGUE_TMA_BIAS_VECTOR = r"""
    {
      {ElementComputeEpilogue({{alpha}}), ElementComputeEpilogue({{beta}})},  // typename ThreadEpilogueOp::Params thread
      nullptr,  // ElementC const* ptr_C
      {cute::Int<1>{}, cute::Int<0>{}, cute::Int<0>{}},  // StrideC dC
      ({{template.dtype(Y)}}*)({{kernel.ptr(Y)}}),  // ElementD const* ptr_D
      { {{template.cute_int(kernel.stride(Y, -1), "stride_y1")}}, {{template.cute_int(kernel.stride(Y, -2), "stride_y0")}}, {{template.cute_int(kernel.stride(Y, -3), "batch_stride_y")}}},  // StrideD dD
      ({{template.dtype(Bias)}}*)({{kernel.ptr(Bias)}}),  // ElementBias const* ptr_Bias
    },  // EpilogueArguments epilogue, TMA bias vector
"""


GEMM_ARGS_CUTLASS_3X_EPILOGUE_TMA_BIAS_MATRIX = r"""
    {
      {ElementComputeEpilogue({{alpha}}), ElementComputeEpilogue({{beta}})},  // typename ThreadEpilogueOp::Params thread
      ({{template.dtype(Bias)}}*)({{kernel.ptr(Bias)}}),  // ElementC const* ptr_C
      { {{template.cute_int(kernel.stride(Bias, -2), "stride_bias0")}}, {{template.cute_int(kernel.stride(Bias, -1), "stride_bias1")}}, {{template.cute_int(kernel.stride(Bias, -3), "batch_stride_bias")}}},  // StrideC dC
      ({{template.dtype(Y)}}*)({{kernel.ptr(Y)}}),  // ElementD const* ptr_D
      { {{template.cute_int(kernel.stride(Y, -2), "stride_y0")}}, {{template.cute_int(kernel.stride(Y, -1), "stride_y1")}}, {{template.cute_int(kernel.stride(Y, -3), "batch_stride_y")}}},  // StrideD dD
      nullptr,  // ElementBias const* ptr_Bias
    },  // EpilogueArguments epilogue, TMA matrix, no bias vector
"""


class CutlassGemmTemplate(CutlassTemplate):
    # Calculates alpha * X@W + beta * Bias

    def __init__(
        self,
        input_nodes: List[IRNode],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: List[int]=None,
    ):
        super().__init__("cutlass_gemm", input_nodes, layout, input_reorder)
        self.alpha = alpha
        self.beta = beta

    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
                #include "cutlass/gemm/gemm.h"
                #include "cutlass/gemm/device/gemm_universal.h"
                #include "cutlass/gemm/device/gemm_universal_adapter.h"

                #include "cutlass/gemm/kernel/gemm_universal.hpp"
                #include "cutlass/gemm/collective/collective_builder.hpp"
                #include "cutlass/epilogue/collective/collective_builder.hpp"
            """
        )
        return res

    @staticmethod
    def cutlass_layout(torch_layout) -> Optional[cutlass_lib.LayoutType]:
        if torch_layout.stride[-1] == 1:
            return cutlass_lib.LayoutType.RowMajor
        elif torch_layout.stride[-2] == 1:
            return cutlass_lib.LayoutType.ColumnMajor
        else:
            return None

    @staticmethod
    def flip_cutlass_layout(
        cutlass_layout: cutlass_lib.LayoutType,
    ) -> cutlass_lib.LayoutType:
        if cutlass_layout == cutlass_lib.LayoutType.RowMajor:
            return cutlass_lib.LayoutType.ColumnMajor
        else:
            return cutlass_lib.LayoutType.RowMajor

    @staticmethod
    def layout_match(torch_layout, cutlass_layout) -> bool:
        return CutlassGemmTemplate.cutlass_layout(torch_layout) == cutlass_layout

    @staticmethod
    def set_alignment(torch_layout, op_element) -> bool:
        alignment = cutlass_utils.get_alignment(torch_layout)
        if alignment < op_element.alignment:
            return False
        else:
            op_element.alignment = alignment
            return True

    @staticmethod
    def has_tma_epilogue(op) -> bool:
        result = False
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            epilogue_schedule_str = str(op.epilogue_schedule).split(".")[-1]
            result = epilogue_schedule_str.lower().startswith("tma")
        return result

    @staticmethod
    def define_gemm_instance(
        op: cutlass_gemm_op.GemmOperation,
    ) -> str:
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            emitter = cutlass_gemm_op.EmitGemmUniversal3xInstance()
            op_def = emitter.emit(op)
            pattern = re.compile(r"\s*struct\s(.*?)\s:")
            decl = [line for line in op_def.split("\n") if "struct " in line][-1]
        else:
            emitter = cutlass_gemm_op.EmitGemmInstance()
            op_def = emitter.emit(op)
            op_def = op_def.replace(
                "cutlass::gemm::device::Gemm", "cutlass::gemm::device::GemmUniversal"
            )
            op_def = op_def.replace("false,", "")
            pattern = re.compile(r"\s*using\s(.*?)\s=")
            decl = op_def.split("\n")[2]
        match = pattern.match(decl)
        if match is None:
            raise RuntimeError("Invalid Gemm config: \n" + op_def)
        op_type = match.groups()[0]
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            op_def += f"\n  using {op_type}_device_type = cutlass::gemm::device::GemmUniversalAdapter<{op_type}>;\n"
            op_type = f"{op_type}_device_type"
        return op_def, op_type

    @staticmethod
    def should_treat_bias_as_tma_matrix(
        op: cutlass_gemm_op.GemmOperation,
        bias: IRNode,
        beta: float,
    ) -> bool:
        has_tma_epilogue = CutlassGemmTemplate.has_tma_epilogue(op)
        has_bias = bias is not None
        is_bias_matrix = len(bias.get_size()) > 1

        return has_tma_epilogue and has_bias and (beta != 1.0 or is_bias_matrix)

    @staticmethod
    def swap_XW(
        op: cutlass_gemm_op.GemmOperation,
        X: IRNode,
        W: IRNode,
        Y: IRNode,
        Bias: IRNode,
    ) -> (cutlass_gemm_op.GemmOperation, IRNode, IRNode, IRNode, IRNode):
        # Swap X and W in GemmOperation.
        new_op = copy.deepcopy(op)
        new_op.A.layout = CutlassGemmTemplate.flip_cutlass_layout(new_op.A.layout)
        new_op.B.layout = CutlassGemmTemplate.flip_cutlass_layout(new_op.B.layout)
        new_op.A, new_op.B = new_op.B, new_op.A
        new_op.C.layout = CutlassGemmTemplate.flip_cutlass_layout(new_op.C.layout)
        new_op.D.layout = CutlassGemmTemplate.flip_cutlass_layout(new_op.D.layout)

        # Swap
        new_Bias = copy.deepcopy(Bias)
        Bias_layout = new_Bias.get_layout()
        if len(Bias_layout.stride) == 1:
            Bias_layout.stride = [1] + Bias_layout.stride
        Bias_layout.stride[-2], Bias_layout.stride[-1] = (
            Bias_layout.stride[-1],
            Bias_layout.stride[-2],
        )
        new_Y = copy.deepcopy(Y)
        Y_layout = new_Y.get_layout()
        Y_layout.stride[-2], Y_layout.stride[-1] = (
            Y_layout.stride[-1],
            Y_layout.stride[-2],
        )

        return (new_op, W, X, new_Y, new_bias)

    def filter_op(
        self,
        op: cutlass_gemm_op.GemmOperation,
    ) -> cutlass_gemm_op.GemmOperation:
        # Skip GroupedGemmOperation.
        if isinstance(op, cutlass_gemm_op.GroupedGemmOperation):
            return None

        # Skip simt kernels
        if (
            op.tile_description.math_instruction.opcode_class
            == cutlass_lib.OpcodeClass.Simt
        ):
            return None

        # Filter ops by dtypes.
        X = self.input_nodes[0]
        W = self.input_nodes[1]
        if not (
            cutlass_utils.dtype_match(X.get_dtype(), op.A.element)
            and cutlass_utils.dtype_match(W.get_dtype(), op.B.element)
            and cutlass_utils.dtype_match(
                self.output_node.get_layout().dtype, op.C.element
            )
        ):
            return None

        # Filter ops by accumulation type.
        if torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction:
            if (
                cutlass_utils.dtype_match(torch.float16, op.A.element)
                and cutlass_utils.dtype_match(torch.float16, op.B.element)
                and not cutlass_utils.dtype_match(torch.float16, op.accumulator_type())
            ):
                return None

        # Filter ops by input layouts.
        if not (
            self.layout_match(X.get_layout(), op.A.layout)
            and self.layout_match(W.get_layout(), op.B.layout)
        ):
            return None

        # Update op.
        op = copy.deepcopy(op)

        # Set output layout.
        op.D.layout = CutlassGemmTemplate.cutlass_layout(self.output_node.get_layout())

        # Filter ops by alignments and set alignments.
        if not (
            self.set_alignment(X.get_layout(), op.A)
            and self.set_alignment(W.get_layout(), op.B)
            and self.set_alignment(self.output_node.get_layout(), op.D)
        ):
            return None

        # Set epilogue.
        # TODO: update epilogue functor according to epilogues.
        op.element_epilogue = op.accumulator_type()

        # Set bias layout and alignment.
        if len(self.input_nodes) == 3:
            Bias = self.input_nodes[2]
            op.C.layout = CutlassGemmTemplate.cutlass_layout(Bias.get_layout())
            if not self.set_alignment(Bias.get_layout(), op.C):
                return None

        return op

    def gen_ops(self) -> List[cutlass_gemm_op.GemmOperation]:
        ops = cutlass_utils.gen_ops()[cutlass_lib.OperationKind.Gemm]
        res = []
        for key, op_list in ops.items():
            for op in op_list:
                filter_res = self.filter_op(op)
                if filter_res is not None:
                    res.append(filter_res)
        print(f"Got cutlass configs: {len(res)=}")
        return [res[0]]

    def gemm_mode(self) -> str:
        sizes = self.output_node.get_size()
        if len(sizes) > 2:
            return "cutlass::gemm::GemmUniversalMode::kBatched"
        else:
            return "cutlass::gemm::GemmUniversalMode::kGemm"

    def render(
        self,
        kernel: CUDATemplateKernel,
        op: cutlass_gemm_op.GemmOperation,
    ) -> str:
        assert len(self.input_nodes) >= 2 and self.output_node is not None
        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]

        instance_definition, instance_type = self.define_gemm_instance(op)

        options = dict(
            alpha=self.alpha,
            beta=self.beta,
            X=X,
            W=W,
            Y=Y,
            Bias=Bias,
            template=self,
            kernel=kernel,
            instance_definition=instance_definition,
            instance_type=instance_type,
            input_reorder=self.input_reorder,
        )
        arguments = ""
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            epilogue_arguments = ""
            if self.has_tma_epilogue(op):
                if self.should_treat_bias_as_tma_matrix(op, bias, beta):
                    epilogue_arguments = self._template_from_string(
                        GEMM_ARGS_CUTLASS_3X_EPILOGUE_TMA_BIAS_MATRIX
                    ).render(**options)
                else:
                    # TMA epilogue requires bias vector in column major to get best perf.
                    if op.D.layout == cutlass_lib.LayoutType.RowMajor:
                        (
                            options[op],
                            options[X],
                            options[W],
                            options[Y],
                            options[Bias],
                        ) = self.maybe_swap_XW(op, X, W, Y, Bias)
                    epilogue_arguments = self._template_from_string(
                        GEMM_ARGS_CUTLASS_3X_EPILOGUE_TMA_BIAS_VECTOR
                    ).render(**options)
            else:
                assert Bias.get_stride()[-1] == 1, "Only support bias vector with stride=1"
                epilogue_arguments = self._template_from_string(
                    GEMM_ARGS_CUTLASS_3X_EPILOGUE_NO_TMA
                ).render(**options)
            arguments = self._template_from_string(GEMM_ARGS_CUTLASS_3X).render(
                epilogue_arguments=epilogue_arguments, **options
            )
        else:
            # TODO: Support split_k.
            arguments = self._template_from_string(GEMM_ARGS_CUTLASS_2X).render(
                split_k=1, **options
            )
        res = self._template_from_string(GEMM_TEMPLATE).render(
            gemm_arguments=arguments, **options
        )

        return res
