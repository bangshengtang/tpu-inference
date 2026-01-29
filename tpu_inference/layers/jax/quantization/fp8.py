# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from tpu_inference.layers.common.quantization import fp8 as jax_common
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig


class Fp8LinearMethod(QuantizeMethodBase, jax_common.Fp8LinearMethod):
    """Fp8 method for JAX Linear layer."""

    def create_weights_jax(self, layer: JaxModule, *weight_args,
                           **extra_weight_attrs):
        assert isinstance(layer, JaxEinsum)
        # layer.weight is (input_dim, output_dim)
        output_dim = layer.weight.shape[1]
        # We initiate the weight scale parameter.
        # Initialize with ones, will be loaded from checkpoint.
        # Note: The underlying kernel treats this as a multiplier (scale),
        # even though it is named `weight_scale_inv`.
        layer.weight_scale_inv = nnx.Param(
            jnp.ones((output_dim, ), dtype=jnp.float32))

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        assert isinstance(layer, JaxEinsum)

        with jax.named_scope(layer.__class__.__name__):
            if self.linear_config.fuse_matmuls:
                # layer.weight.value is (In, Out).
                # _apply_fused (and the underlying kernel) expects (Out, In).
                # So we must transpose the weight.
                out = self._apply_fused(
                    x, layer.weight.value.T, layer.weight_scale_inv.value,
                    layer.bias.value if layer.bias else None)
            else:
                raise NotImplementedError(
                    "Non-fused matmuls not implemented yet.")

        return out


class Fp8Config(QuantizationConfig):

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, JaxEinsum):
            # QuantLinearConfig requires output_sizes and enable_sp.
            # layer.weight is (In, Out), so output_size is shape[1].
            # We currently default enable_sp to False for JAX layers
            # as it requires global config access not present here.
            linear_config = QuantLinearConfig(
                output_sizes=[layer.weight.shape[1]], enable_sp=False)
            return Fp8LinearMethod(linear_config)
        return None
