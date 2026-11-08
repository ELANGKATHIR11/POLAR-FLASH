"""
POLAR-FLASH: Flash-Page Coalescing & Asynchronous Double-Buffering Engine
Implements Part III Sections 10 & 11 of the POLAR-FLASH Specification:
1. Flash-Page Coalescing: N_pages = ceil(((o mod P) + L) / P).
2. Storage Utilization U = Useful / Transferred, Read Amplification A = Transferred / Useful.
3. Asynchronous Double-Buffering Pipeline: T_pipeline ~ max(T_compute, T_IO).
4. Real-time background prefetching queue with double-buffering.
"""

import math
import time
import threading
import queue
from typing import Dict, List, Any, Tuple, Optional, Callable


class FlashPageCoalescer:
    """
    Computes storage page boundaries, alignment penalty, storage utilization U,
    and read amplification A for non-aligned or aligned file access patterns.
    """
    def __init__(self, physical_page_size: int = 4096):
        """
        physical_page_size: Physical SSD NAND / OS page cache block size (default 4KB = 4096 bytes).
        """
        self.P = physical_page_size

    def calculate_page_touches(self, offset: int, length: int) -> Dict[str, Any]:
        """
        Calculates touched pages and read amplification:
        N_pages = ceil(((offset % P) + length) / P)
        Total bytes transferred from storage = N_pages * P
        """
        offset_mod_p = offset % self.P
        num_pages = math.ceil((offset_mod_p + length) / self.P)
        total_transferred = num_pages * self.P
        useful_bytes = length

        storage_utilization = useful_bytes / max(total_transferred, 1)
        read_amplification = total_transferred / max(useful_bytes, 1)

        return {
            "offset": offset,
            "requested_length": length,
            "page_size": self.P,
            "num_pages_touched": num_pages,
            "total_bytes_transferred": total_transferred,
            "storage_utilization_U": storage_utilization,
            "read_amplification_A": read_amplification,
            "is_page_aligned": (offset_mod_p == 0) and (length % self.P == 0)
        }


class DoubleBufferPipeline:
    """
    Simulates / manages asynchronous double-buffered tensor streaming:
    Buffer A supplies active compute while Buffer B prefetches the subsequent layer.
    Ideal pipeline time: T_pipeline ~ max(T_compute, T_IO).
    """
    def __init__(self, io_bandwidth_gb_s: float = 5.0, compute_tflops: float = 110.0):
        self.io_bandwidth_bytes_per_sec = io_bandwidth_gb_s * (1024 ** 3)
        self.compute_flops_per_sec = compute_tflops * 1e12

        self.buffer_a: Optional[str] = None
        self.buffer_b: Optional[str] = None
        self.stats: Dict[str, Any] = {
            "total_layers_processed": 0,
            "total_compute_time_s": 0.0,
            "total_io_time_s": 0.0,
            "total_pipelined_time_s": 0.0,
            "io_stall_time_s": 0.0,
            "io_stall_fraction": 0.0
        }

    def simulate_pipeline(self, layer_workloads: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Simulates end-to-end execution of a sequence of layers:
        layer_workloads = [{'layer_id': 'layer_0', 'tensor_bytes': 65536000, 'flops': 1.5e11}, ...]
        """
        total_compute = 0.0
        total_io = 0.0
        pipelined_elapsed = 0.0
        io_stall = 0.0

        num_layers = len(layer_workloads)
        if num_layers == 0:
            return self.stats

        # Initial prefetch for layer 0
        first_io_time = layer_workloads[0]["tensor_bytes"] / self.io_bandwidth_bytes_per_sec
        pipelined_elapsed += first_io_time
        total_io += first_io_time
        io_stall += first_io_time

        for i in range(num_layers):
            cur = layer_workloads[i]
            t_comp = cur["flops"] / self.compute_flops_per_sec
            total_compute += t_comp

            # Asynchronously prefetch layer i + 1 while computing layer i
            if i + 1 < num_layers:
                nxt = layer_workloads[i + 1]
                t_io = nxt["tensor_bytes"] / self.io_bandwidth_bytes_per_sec
                total_io += t_io

                # Double-buffer overlap: time taken is max(t_compute, t_io)
                cycle_time = max(t_comp, t_io)
                pipelined_elapsed += cycle_time
                if t_io > t_comp:
                    io_stall += (t_io - t_comp)
            else:
                # Last layer compute
                pipelined_elapsed += t_comp

        serial_time = total_compute + total_io
        speedup = serial_time / max(pipelined_elapsed, 1e-12)

        self.stats = {
            "num_layers": num_layers,
            "total_compute_time_s": total_compute,
            "total_io_time_s": total_io,
            "serial_time_s": serial_time,
            "pipelined_time_s": pipelined_elapsed,
            "double_buffer_speedup": speedup,
            "io_stall_time_s": io_stall,
            "io_stall_percentage": (io_stall / max(pipelined_elapsed, 1e-12)) * 100.0
        }
        return self.stats


class AsyncDoubleBufferQueue:
    """
    Live asynchronous double-buffering worker that prefetches layers in background
    threads to overlap SSD reads with PyTorch computation.
    """
    def __init__(self, fetch_fn: Callable[[str], Any], max_queue_depth: int = 2):
        self.fetch_fn = fetch_fn
        self.max_queue_depth = max_queue_depth
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_depth)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start_prefetch(self, layer_names: List[str]):
        """Starts background worker to sequentially load layers into the queue."""
        def _worker():
            for name in layer_names:
                if self._stop_event.is_set():
                    break
                data = self.fetch_fn(name)
                self.queue.put((name, data))

        self._stop_event.clear()
        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def get_next(self, timeout: float = 10.0) -> Tuple[str, Any]:
        """Retrieves prefetched layer from queue, stalling only if I/O is slower than compute."""
        return self.queue.get(timeout=timeout)

    def stop(self):
        """Stops background prefetching thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
