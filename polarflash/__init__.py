"""
POLAR-FLASH: Extreme Neural Network Compression & Decoupled Fine-Tuning Framework
Converting Any Open-Source LLM into High-Performance SLMs for Lower-Spec Edge Hardware.
"""

from polarflash_quant.polar.polar_quant import (
    PolarQuantizer, PolarQuant2D, RecursivePolarQuant, pack_polar_bits, unpack_polar_bits,
    compare_polar_vs_e8_at_matched_bitrate
)
from polarflash_quant.residual.qjl import (
    QJLResidualEstimator, BitPackedQJLEstimator, XNORPopcountQJLEstimator, verify_qjl_unbiasedness_and_variance
)
from polarflash_quant.transforms.hadamard import (
    RandomizedHadamardTransform, fast_walsh_hadamard_transform, BlockHadamardTransform,
    compute_distribution_diagnostics, verify_gaussian_concentration
)
from polarflash_quant.lattice.e8_lattice import (
    E8LatticeQuantizer, E8ResidualVectorQuantizer, quantize_d8, quantize_e8, generate_e8_roots
)
from polarflash_quant.information_theory.rate_distortion import (
    shannon_entropy, empirical_entropy, gaussian_rate_distortion, vector_source_rate_distortion,
    EntropyConstrainedQuantizer, estimate_empirical_rd_curve, LayerRateAllocator, calculate_effective_bpw
)
from polarflash_quant.calibration.fisher import (
    FisherSensitivityEstimator, SparseTernaryResidual, compare_sparsity_levels
)
from polarflash_quant.hardware.roofline import (
    RooflineAnalyzer, MemoryFeasibilityGate, LOW_SPEC_4GB_DEVICE, RTX_5060_LAPTOP, compute_energy_consumption, HardwareProfile
)
from polarflash_quant.bitpacking import (
    pack_int4, unpack_int4, pack_int2, unpack_int2, pack_int3, unpack_int3
)
from polarflash_quant.unified_quantizer import (
    UnifiedQuantizer, ArchitectureAdapter
)

from polarflash_rt.kv_cache.compressed_kv import CompressedKVCache
from polarflash_rt.streaming.flash_streamer import FlashPageCoalescer, DoubleBufferPipeline, AsyncDoubleBufferQueue
from polarflash_rt.cuda.engine import PFlashCUDAEngine
from polarflash_rt.cpu.engine import PFlashCPUEngine, PolarLinearLayer
from polarflash_rt.sparse.contextual_sparse import ContextualSparseFFN, LookaheadNeuronRouter
from polarflash_rt.auto_engine import AutoEngine, HardwareProbe, ThermalGovernor

from polarflash_train.lora.polar_lora import PolarLoRALinear
from polarflash_train.qat.polar_qat import PolarQATLinear
from polarflash_train.gradcodes.pv_gradcode import PVGradCodeOptimizer, TIESWeightMerger

from polarflash_core.formats.pflash_format import PFlashWriter, PFlashReader, PolarFlashBridge
from polarflash_core.ingestion import ModelIngestionEngine
from polarflash_bench.benchmark_suite import run_full_polarflash_benchmark

# High-level friendly aliases
QJLResidual = QJLResidualEstimator
WalshHadamardTransform = RandomizedHadamardTransform
FlashStreamer = DoubleBufferPipeline
PolarCUDAEngine = PFlashCUDAEngine

__version__ = "1.1.0"
__author__ = "Kathir (Zenith AI)"
__license__ = "Apache-2.0"

__all__ = [
    "PolarQuantizer",
    "PolarQuant2D",
    "RecursivePolarQuant",
    "pack_polar_bits",
    "unpack_polar_bits",
    "QJLResidualEstimator",
    "QJLResidual",
    "BitPackedQJLEstimator",
    "XNORPopcountQJLEstimator",
    "verify_qjl_unbiasedness_and_variance",
    "RandomizedHadamardTransform",
    "WalshHadamardTransform",
    "fast_walsh_hadamard_transform",
    "BlockHadamardTransform",
    "compute_distribution_diagnostics",
    "verify_gaussian_concentration",
    "E8LatticeQuantizer",
    "E8ResidualVectorQuantizer",
    "quantize_d8",
    "quantize_e8",
    "generate_e8_roots",
    "shannon_entropy",
    "empirical_entropy",
    "gaussian_rate_distortion",
    "vector_source_rate_distortion",
    "EntropyConstrainedQuantizer",
    "estimate_empirical_rd_curve",
    "LayerRateAllocator",
    "calculate_effective_bpw",
    "FisherSensitivityEstimator",
    "SparseTernaryResidual",
    "compare_sparsity_levels",
    "RooflineAnalyzer",
    "MemoryFeasibilityGate",
    "LOW_SPEC_4GB_DEVICE",
    "RTX_5060_LAPTOP",
    "compute_energy_consumption",
    "HardwareProfile",
    "pack_int4",
    "unpack_int4",
    "pack_int2",
    "unpack_int2",
    "pack_int3",
    "unpack_int3",
    "UnifiedQuantizer",
    "ArchitectureAdapter",
    "CompressedKVCache",
    "FlashPageCoalescer",
    "DoubleBufferPipeline",
    "AsyncDoubleBufferQueue",
    "FlashStreamer",
    "PFlashCUDAEngine",
    "PolarCUDAEngine",
    "PFlashCPUEngine",
    "PolarLinearLayer",
    "ContextualSparseFFN",
    "LookaheadNeuronRouter",
    "AutoEngine",
    "HardwareProbe",
    "ThermalGovernor",
    "PolarLoRALinear",
    "PolarQATLinear",
    "PVGradCodeOptimizer",
    "TIESWeightMerger",
    "PFlashWriter",
    "PFlashReader",
    "PolarFlashBridge",
    "compare_polar_vs_e8_at_matched_bitrate",
    "ModelIngestionEngine",
    "run_full_polarflash_benchmark",
]
