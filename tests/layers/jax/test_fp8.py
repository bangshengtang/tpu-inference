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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh
from vllm.config import ModelConfig, VllmConfig

from tpu_inference.layers.jax.linear import JaxLinear
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.layers.jax.quantization.fp8 import (Fp8Config,
                                                       Fp8LinearMethod)


@pytest.fixture(scope="module")
def mesh():
    """
    Creates a mesh with 1 device.
    """
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")

    devices = np.array(jax.local_devices()[:1])
    num_devices = len(devices)
    assert num_devices == 1
    device_mesh = devices.reshape((num_devices, 1, 1, 1))

    with Mesh(device_mesh,
              axis_names=('data', 'attn_dp', 'expert', 'model')) as m:
        yield m


@pytest.fixture
def rng():
    """Provides a reusable JAX PRNGKey."""
    return jax.random.PRNGKey(42)


class TestFp8Linear:

    def test_fp8_linear_init(self, mesh):
        vllm_config = VllmConfig(model_config=ModelConfig(
            model="Qwen/Qwen3-0.6B", quantization="fp8"))

        quant_config = get_tpu_quantization_config(vllm_config)
        assert isinstance(quant_config, Fp8Config)

        input_dim = 16
        output_dim = 32
        batch_size = 1
        layer = JaxLinear(input_size=input_dim,
                          output_size=output_dim,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config)

        assert isinstance(layer.quant_method, Fp8LinearMethod)
        assert hasattr(layer, "weight_scale_inv")
        assert layer.weight_scale_inv.value.shape == (output_dim, )
        assert layer.weight_scale_inv.value.dtype == jnp.float32

        layer.quant_method.linear_config.mesh = mesh

        with mesh:
            hidden_states = jnp.ones((batch_size, input_dim))
            out = layer(hidden_states)
            assert out.shape == (batch_size, output_dim)

    def test_fp8_linear_correctness(self, mesh, rng):
        vllm_config = VllmConfig(model_config=ModelConfig(
            model="Qwen/Qwen3-0.6B", quantization="fp8"))
        quant_config = get_tpu_quantization_config(vllm_config)

        input_dim = 16
        output_dim = 32
        batch_size = 1
        layer = JaxLinear(input_size=input_dim,
                          output_size=output_dim,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config)

        k1, k2, k3 = jax.random.split(rng, 3)
        w_val = jax.random.normal(k1, (input_dim, output_dim),
                                  dtype=jnp.float32)
        s_val = jax.random.uniform(k2, (output_dim, ), dtype=jnp.float32)

        layer.weight.value = w_val
        layer.weight_scale_inv.value = s_val

        hidden_states = jax.random.uniform(k3, (batch_size, input_dim),
                                           dtype=jnp.float32)

        effective_w = w_val * s_val
        expected = jnp.dot(hidden_states, effective_w)

        layer.quant_method.linear_config.mesh = mesh

        with mesh:
            out = layer(hidden_states)
            assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)
