"""Benchmark test comparing CPU vs MPS (Metal Performance Shaders) performance.

This test is designed to run on macOS systems with Apple Silicon to compare
the performance difference between CPU and MPS acceleration for Boltz models.
"""

import time
from typing import Dict, Optional, Tuple

import pytest
import torch
import torch.nn as nn
import numpy as np


def check_mps_availability() -> bool:
    """Check if MPS is available on the current system."""
    return torch.backends.mps.is_available()


class SimpleTransformerBlock(nn.Module):
    """A simplified transformer block similar to those used in Boltz."""
    
    def __init__(self, dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.mha(normed, normed, normed)
        x = x + attn_out
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class BenchmarkModel(nn.Module):
    """A model that mimics key computational patterns in Boltz."""
    
    def __init__(self, num_blocks: int = 4, dim: int = 256):
        super().__init__()
        self.embedding = nn.Linear(128, dim)
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(dim) for _ in range(num_blocks)
        ])
        self.output = nn.Linear(dim, 3)  # 3D coordinates
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


def run_benchmark(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Dict[str, float]:
    """Run benchmark on specified device.
    
    Parameters
    ----------
    model : nn.Module
        Model to benchmark
    input_tensor : torch.Tensor
        Input tensor for the model
    device : str
        Device to run on ('cpu' or 'mps')
    num_warmup : int
        Number of warmup iterations
    num_runs : int
        Number of benchmark iterations
        
    Returns
    -------
    Dict[str, float]
        Benchmark results including mean, std, min, max times
    """
    device_obj = torch.device(device)
    model = model.to(device_obj)
    input_tensor = input_tensor.to(device_obj)
    
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(input_tensor)
    
    # Synchronize if using MPS
    if device == "mps":
        torch.mps.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # Synchronize to ensure computation is complete
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cpu":
            # Force CPU sync by accessing the result
            _ = output.cpu().numpy()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    times_array = np.array(times)
    
    return {
        "mean": float(np.mean(times_array)),
        "std": float(np.std(times_array)),
        "min": float(np.min(times_array)),
        "max": float(np.max(times_array)),
        "median": float(np.median(times_array)),
        "times": times,
    }


@pytest.mark.skipif(not check_mps_availability(), reason="MPS not available")
class TestCPUvsMPS:
    """Test suite comparing CPU and MPS performance."""
    
    def test_small_model_benchmark(self):
        """Benchmark a small model (similar to confidence head)."""
        model = BenchmarkModel(num_blocks=2, dim=128)
        batch_size = 4
        seq_len = 100
        input_tensor = torch.randn(batch_size, seq_len, 128)
        
        # Run CPU benchmark
        cpu_results = run_benchmark(model, input_tensor, "cpu")
        
        # Run MPS benchmark
        mps_results = run_benchmark(model, input_tensor, "mps")
        
        # Calculate speedup
        speedup = cpu_results["mean"] / mps_results["mean"]
        
        # Report results
        print(f"\n{'='*60}")
        print("SMALL MODEL BENCHMARK (2 blocks, dim=128)")
        print(f"{'='*60}")
        print(f"Input shape: {tuple(input_tensor.shape)}")
        print(f"\nCPU Performance:")
        print(f"  Mean time: {cpu_results['mean']*1000:.2f} ms")
        print(f"  Std dev:   {cpu_results['std']*1000:.2f} ms")
        print(f"  Min time:  {cpu_results['min']*1000:.2f} ms")
        print(f"  Max time:  {cpu_results['max']*1000:.2f} ms")
        
        print(f"\nMPS Performance:")
        print(f"  Mean time: {mps_results['mean']*1000:.2f} ms")
        print(f"  Std dev:   {mps_results['std']*1000:.2f} ms")
        print(f"  Min time:  {mps_results['min']*1000:.2f} ms")
        print(f"  Max time:  {mps_results['max']*1000:.2f} ms")
        
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"{'='*60}\n")
        
        # Store results for assertion
        assert mps_results["mean"] > 0, "MPS benchmark failed"
        assert cpu_results["mean"] > 0, "CPU benchmark failed"
    
    def test_medium_model_benchmark(self):
        """Benchmark a medium model (similar to trunk modules)."""
        model = BenchmarkModel(num_blocks=8, dim=256)
        batch_size = 2
        seq_len = 256
        input_tensor = torch.randn(batch_size, seq_len, 128)
        
        # Run CPU benchmark
        cpu_results = run_benchmark(model, input_tensor, "cpu", num_warmup=2, num_runs=5)
        
        # Run MPS benchmark
        mps_results = run_benchmark(model, input_tensor, "mps", num_warmup=2, num_runs=5)
        
        # Calculate speedup
        speedup = cpu_results["mean"] / mps_results["mean"]
        
        # Report results
        print(f"\n{'='*60}")
        print("MEDIUM MODEL BENCHMARK (8 blocks, dim=256)")
        print(f"{'='*60}")
        print(f"Input shape: {tuple(input_tensor.shape)}")
        print(f"\nCPU Performance:")
        print(f"  Mean time: {cpu_results['mean']*1000:.2f} ms")
        print(f"  Std dev:   {cpu_results['std']*1000:.2f} ms")
        print(f"  Min time:  {cpu_results['min']*1000:.2f} ms")
        print(f"  Max time:  {cpu_results['max']*1000:.2f} ms")
        
        print(f"\nMPS Performance:")
        print(f"  Mean time: {mps_results['mean']*1000:.2f} ms")
        print(f"  Std dev:   {mps_results['std']*1000:.2f} ms")
        print(f"  Min time:  {mps_results['min']*1000:.2f} ms")
        print(f"  Max time:  {mps_results['max']*1000:.2f} ms")
        
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"{'='*60}\n")
        
        assert mps_results["mean"] > 0, "MPS benchmark failed"
        assert cpu_results["mean"] > 0, "CPU benchmark failed"
    
    @pytest.mark.slow
    def test_large_model_benchmark(self):
        """Benchmark a large model (closer to full Boltz model)."""
        model = BenchmarkModel(num_blocks=16, dim=512)
        batch_size = 1
        seq_len = 512
        input_tensor = torch.randn(batch_size, seq_len, 128)
        
        # Run CPU benchmark (fewer runs due to longer execution time)
        cpu_results = run_benchmark(model, input_tensor, "cpu", num_warmup=1, num_runs=3)
        
        # Run MPS benchmark
        mps_results = run_benchmark(model, input_tensor, "mps", num_warmup=1, num_runs=3)
        
        # Calculate speedup
        speedup = cpu_results["mean"] / mps_results["mean"]
        
        # Report results
        print(f"\n{'='*60}")
        print("LARGE MODEL BENCHMARK (16 blocks, dim=512)")
        print(f"{'='*60}")
        print(f"Input shape: {tuple(input_tensor.shape)}")
        print(f"\nCPU Performance:")
        print(f"  Mean time: {cpu_results['mean']*1000:.2f} ms")
        print(f"  Std dev:   {cpu_results['std']*1000:.2f} ms")
        print(f"  Min time:  {cpu_results['min']*1000:.2f} ms")
        print(f"  Max time:  {cpu_results['max']*1000:.2f} ms")
        
        print(f"\nMPS Performance:")
        print(f"  Mean time: {mps_results['mean']*1000:.2f} ms")
        print(f"  Std dev:   {mps_results['std']*1000:.2f} ms")
        print(f"  Min time:  {mps_results['min']*1000:.2f} ms")
        print(f"  Max time:  {mps_results['max']*1000:.2f} ms")
        
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"{'='*60}\n")
        
        assert mps_results["mean"] > 0, "MPS benchmark failed"
        assert cpu_results["mean"] > 0, "CPU benchmark failed"
    
    def test_matrix_operations_benchmark(self):
        """Benchmark common matrix operations used in Boltz."""
        
        def run_matrix_ops(device: str, size: int = 1024) -> Dict[str, float]:
            """Run common matrix operations and measure time."""
            device_obj = torch.device(device)
            
            # Create tensors
            a = torch.randn(size, size, device=device_obj)
            b = torch.randn(size, size, device=device_obj)
            
            operations = {}
            
            # Matrix multiplication
            start = time.perf_counter()
            c = torch.matmul(a, b)
            if device == "mps":
                torch.mps.synchronize()
            operations["matmul"] = time.perf_counter() - start
            
            # Softmax (common in attention)
            start = time.perf_counter()
            d = torch.softmax(a, dim=-1)
            if device == "mps":
                torch.mps.synchronize()
            operations["softmax"] = time.perf_counter() - start
            
            # Layer norm (very common in transformers)
            ln = nn.LayerNorm(size).to(device_obj)
            start = time.perf_counter()
            e = ln(a)
            if device == "mps":
                torch.mps.synchronize()
            operations["layernorm"] = time.perf_counter() - start
            
            # GELU activation
            start = time.perf_counter()
            f = torch.nn.functional.gelu(a)
            if device == "mps":
                torch.mps.synchronize()
            operations["gelu"] = time.perf_counter() - start
            
            return operations
        
        # Run benchmarks
        cpu_ops = run_matrix_ops("cpu")
        mps_ops = run_matrix_ops("mps")
        
        # Report results
        print(f"\n{'='*60}")
        print("MATRIX OPERATIONS BENCHMARK (1024x1024)")
        print(f"{'='*60}")
        
        for op_name in cpu_ops.keys():
            cpu_time = cpu_ops[op_name] * 1000
            mps_time = mps_ops[op_name] * 1000
            speedup = cpu_time / mps_time if mps_time > 0 else 0
            
            print(f"\n{op_name.upper()}:")
            print(f"  CPU: {cpu_time:.3f} ms")
            print(f"  MPS: {mps_time:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
        
        print(f"{'='*60}\n")


@pytest.mark.skipif(check_mps_availability(), reason="Testing CPU-only fallback")
def test_cpu_only_fallback():
    """Test that benchmarks work on CPU-only systems."""
    model = BenchmarkModel(num_blocks=2, dim=64)
    input_tensor = torch.randn(2, 50, 128)
    
    cpu_results = run_benchmark(model, input_tensor, "cpu", num_warmup=1, num_runs=3)
    
    assert cpu_results["mean"] > 0, "CPU benchmark should work"
    assert len(cpu_results["times"]) == 3, "Should have correct number of runs"
    
    print(f"\nCPU-only system benchmark successful")
    print(f"Mean time: {cpu_results['mean']*1000:.2f} ms")


if __name__ == "__main__":
    """Allow running as a standalone script for quick benchmarking."""
    if check_mps_availability():
        print("MPS is available! Running benchmarks...\n")
        test_suite = TestCPUvsMPS()
        test_suite.test_small_model_benchmark()
        test_suite.test_medium_model_benchmark()
        test_suite.test_matrix_operations_benchmark()
        
        # Optionally run the slow test
        # test_suite.test_large_model_benchmark()
    else:
        print("MPS not available. Running CPU-only benchmark...")
        test_cpu_only_fallback()