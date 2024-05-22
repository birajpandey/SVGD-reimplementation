import os
import sys
import unittest

import jax
import jax.numpy as jnp
import torch

REQUIRED_PYTHON = "python3"

class TestEnvironment(unittest.TestCase):

    def test_python_version(self):
        """Test if the current Python version matches the required version."""
        required_major = 3 if REQUIRED_PYTHON == "python3" else 2
        system_major = sys.version_info.major
        self.assertEqual(system_major, required_major,
                         f"This project requires Python {required_major}. "
                         f"Found: Pythin {system_major}")
    def test_jax_array_creation_and_sum(self):
        """Test that JAX can create an array and compute its sum."""
        arr = jnp.array([1, 2, 3])
        arr_sum = jnp.sum(arr)
        self.assertEqual(arr_sum, 6, "JAX array sum did not match expected "
                                     "value")

    def test_torch_tensor_creation_and_multiplication(self):
        """Test that PyTorch can create a tensor and compute its
        multiplication."""
        tensor = torch.tensor([1, 2, 3])
        tensor_mul = tensor * 2
        # Convert tensor to list for comparison
        expected_result = [2, 4, 6]
        self.assertListEqual(tensor_mul.tolist(), expected_result, "Pytorch "
                                                                   "tensor "
                                                                   "multiplication did not match expected values.")


    def test_jax_gpu_availability(self):
        # get a list of all available Jax devices
        devices = jax.devices()

        # check if any of the devices is a GPU
        # gpu_available = any(device.device_kind == 'GPU' for device in devices)

        # Assert that GPU is available
        self.assertTrue(len(devices) > 1, "No GPU available for Jax")


if __name__ == '__main__':
    unittest.main()