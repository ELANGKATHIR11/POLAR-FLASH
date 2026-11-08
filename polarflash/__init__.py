"""
POLAR-FLASH: Extreme Neural Network Compression & Decoupled Fine-Tuning Framework
Converting Any Open-Source LLM into High-Performance SLMs for Lower-Spec Edge Hardware.
"""

from polarflash_quant.polar.polar_quant import PolarQuantizer, PolarQuant2D, RecursivePolarQuant
from polarflash_quant.residual.qjl import QJLResidualEstimator, BitPackedQJLEstimator
from polarflash_quant.transforms.hadamard import RandomizedHadamardTransform, fast_walsh_hadamard_transform
from polarflash_quant.lattice.e8_lattice import E8LatticeQuantizer
from polarflash_quant.hardware.roofline import RooflineAnalyzer, MemoryFeasibilityGate, LOW_SPEC_4GB_DEVICE, RTX_5060_LAPTOP

from polarflash_rt.kv_cache.compressed_kv import CompressedKVCache
from polarflash_rt.streaming.flash_streamer import FlashPageCoalescer, DoubleBufferPipeline
from polarflash_rt.cuda.engine import PFlashCUDAEngine
from polarflash_rt.cpu.engine import PFlashCPUEngine, PolarLinearLayer

from polarflash_train.lora.polar_lora import PolarLoRALinear
from polarflash_train.qat.polar_qat import PolarQATLinear
from polarflash_train.gradcodes.pv_gradcode import PVGradCodeOptimizer, TIESWeightMerger

from polarflash_bench.benchmark_suite import run_full_polarflash_benchmark

# High-level friendly aliases
QJLResidual = QJLResidualEstimator
WalshHadamardTransform = RandomizedHadamardTransform
FlashStreamer = DoubleBufferPipeline
PolarCUDAEngine = PFlashCUDAEngine

__version__ = "1.0.0"
__author__ = "Kathir (Zenith AI)"
__license__ = "Apache-2.0"

__all__ = [
    "PolarQuantizer",
    "PolarQuant2D",
    "RecursivePolarQuant",
    "QJLResidualEstimator",
    "QJLResidual",
    "BitPackedQJLEstimator",
    "RandomizedHadamardTransform",
    "WalshHadamardTransform",
    "fast_walsh_hadamard_transform",
    "E8LatticeQuantizer",
    "RooflineAnalyzer",
    "MemoryFeasibilityGate",
    "CompressedKVCache",
    "FlashPageCoalescer",
    "DoubleBufferPipeline",
    "FlashStreamer",
    "PFlashCUDAEngine",
    "PolarCUDAEngine",
    "PFlashCPUEngine",
    "PolarLinearLayer",
    "PolarLoRALinear",
    "PolarQATLinear",
    "PVGradCodeOptimizer",
    "TIESWeightMerger",
    "run_full_polarflash_benchmark",
]
