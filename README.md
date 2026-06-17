# Mini LLM Training + Inference Performance Stack

A 15-week project-driven learning plan for building a small end-to-end ML systems stack covering:

* Tiny Transformer training
* Training performance profiling
* CUDA/C++ kernel implementation
* Distributed training basics
* Inference serving
* Dynamic batching
* KV cache simulation
* Multi-worker serving
* Fault tolerance
* Metrics and benchmarking
* Scaling concepts from PyTorch Distributed and the JAX Scaling Book

The goal is not to build a state-of-the-art model.
The goal is to understand how modern ML training and inference systems work from the inside.

---

## Final Project Goal

By the end of this plan, the repository should contain:

```text
mini-llm-systems/
├── training/
│   ├── tiny_gpt.py
│   ├── train.py
│   ├── dataloader.py
│   ├── checkpoint.py
│   └── profile_training.py
│
├── kernels/
│   ├── vector_add.cu
│   ├── matmul_naive.cu
│   ├── matmul_tiled.cu
│   ├── layernorm.cu
│   └── pytorch_extension/
│
├── distributed_training/
│   ├── ddp_train.py
│   ├── fsdp_notes.md
│   └── scaling_notes.md
│
├── serving/
│   ├── api_server.py
│   ├── router.py
│   ├── worker.py
│   ├── scheduler.py
│   ├── kv_cache.py
│   └── metrics.py
│
├── benchmarks/
│   ├── bench_training.py
│   ├── bench_kernels.py
│   └── bench_serving.py
│
├── docs/
│   ├── architecture.md
│   ├── training_perf_report.md
│   ├── cuda_perf_report.md
│   ├── inference_perf_report.md
│   ├── distributed_serving_report.md
│   ├── distributed_training_report.md
│   ├── scaling_report.md
│   └── failure_postmortems.md
│
├── tests/
│   ├── test_model.py
│   ├── test_scheduler.py
│   ├── test_kv_cache.py
│   ├── test_router_failure.py
│   └── test_cuda_correctness.py
│
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## Core Resources

Use these resources non-linearly. Do not read everything cover to cover.
When a project task requires a concept, read the minimum relevant section, implement, benchmark, and write a note.

| Resource                                  | Role                                                                                      |
| ----------------------------------------- | ----------------------------------------------------------------------------------------- |
| CMU 11-485 / 11-785                       | Deep learning and Transformer fundamentals                                                |
| CMU 10-414 / 10-714 Deep Learning Systems | Autodiff, tensor systems, hardware acceleration, DL framework internals                   |
| PyTorch Docs                              | Implementation reference for models, training, profiler, AMP, CUDA memory, DataLoader     |
| PyTorch Distributed Docs                  | DDP, FSDP, distributed launch, collectives, debugging                                     |
| JAX Scaling Book                          | Large-scale training/inference parallelism, communication, sharding, scaling mental model |
| PMPP                                      | CUDA programming model, memory hierarchy, tiling, reductions                              |
| NVIDIA CUDA Guide                         | CUDA semantics, memory management, streams, synchronization                               |
| Nsight Systems / Nsight Compute           | GPU profiling and kernel-level performance analysis                                       |
| vLLM Docs + PagedAttention                | LLM serving, continuous batching, KV cache, PagedAttention                                |
| DDIA                                      | Distributed systems: failure, partitioning, replication, consistency                      |
| CSAPP                                     | Memory hierarchy, linking, I/O, networking, concurrency                                   |
| OSTEP                                     | Processes, threads, locks, scheduling, virtual memory                                     |
| LearnCpp                                  | Minimum C++ syntax, classes, pointers, RAII                                               |
| Effective Modern C++                      | Move semantics, smart pointers, modern C++ resource management                            |
| FastAPI Docs                              | Inference API server                                                                      |
| Docker Docs                               | Containerization and multi-service local deployment                                       |

---

## Learning Method

For every concept:

```text
Concept appears in project
→ read minimum resource section
→ implement toy version
→ benchmark or test it
→ write a short note
→ integrate into project
```

Daily loop:

```text
30–60 min: targeted reading
90–150 min: implementation
30–45 min: debugging / profiling / benchmarking
15–30 min: write notes
```

For unfamiliar concepts:

```text
60–90 min reading
60–90 min toy experiment
30 min explanation note
```

For familiar concepts:

```text
15–30 min reading
2–3 hours implementation/debugging
```

---

# Week 1 — Tiny GPT Baseline

## Goal

Build a small Transformer/GPT model that can run forward, compute loss, train on a tiny dataset, and save/load checkpoints.

## Concepts

* Transformer forward pass
* Token embeddings
* Positional embeddings
* Multi-head causal self-attention
* Tensor shapes
* Autograd basics
* Cross entropy loss
* Optimizer state
* Checkpointing

## Resources

* CMU 11-485 / 11-785 Transformer notes
* nanoGPT code, for structure only
* PyTorch `nn.Module`, `Tensor`, `autograd`, `cross_entropy`
* CMU 10-414 tensor/autodiff intro lectures

## Build Tasks

Create:

```text
training/
├── tiny_gpt.py
├── train.py
├── checkpoint.py
└── tests/test_tiny_gpt.py
```

Implement:

* Token embedding
* Positional embedding
* Causal self-attention
* MLP block
* Residual connections
* LayerNorm
* Cross entropy loss
* Basic training loop
* Checkpoint save/load

## Experiments

* Print every tensor shape inside the forward pass.
* Verify causal masking with a sequence length 4 toy example.
* Compare memory usage with and without `torch.no_grad()`.
* Resume training from checkpoint and verify loss continuity.

## Deliverable

```bash
python training/train.py
```

Expected result:

* Loss decreases on a tiny dataset.
* Checkpoint save/load works.

## Writeup

Create:

```text
docs/week1_model_baseline.md
```

Include:

* Architecture diagram
* Tensor shape table
* Loss curve
* Checkpoint resume test

---

# Week 2 — Training Performance Baseline + Profiling

## Goal

Move from “the model trains” to “I can measure where training time and memory go.”

## Concepts

* Efficient DataLoader
* Tokens/sec
* Step time
* GPU memory
* PyTorch profiler
* CPU vs CUDA time
* Dataloader bottlenecks
* Tensor layout and locality

## Resources

* PyTorch DataLoader docs
* PyTorch Profiler tutorial
* PyTorch Performance Tuning Guide
* CSAPP memory hierarchy chapter, locality sections

## Build Tasks

Create:

```text
training/
├── dataloader.py
├── profile_training.py
└── metrics.py

benchmarks/
└── bench_training.py
```

Implement:

* Dataset/DataLoader split
* Tokens/sec logging
* Step time logging
* GPU memory logging
* Profiler trace export
* Timing breakdown: dataloader, H2D, forward, backward, optimizer

## Experiments

Sweep:

```text
num_workers = 0, 2, 4, 8
pin_memory = False, True
batch_size = 8, 16, 32
seq_len = 64, 128, 256
```

Record:

* Tokens/sec
* Step time
* Peak CUDA memory
* GPU utilization
* Dataloader time
* Forward time
* Backward time
* Optimizer time

## Deliverable

Create:

```text
benchmarks/training_perf.csv
```

Example columns:

```text
config,batch_size,seq_len,num_workers,pin_memory,tokens_per_sec,step_time_ms,max_memory_mb
```

## Writeup

Create:

```text
docs/week2_training_profiler.md
```

Include:

* Profiler trace summary
* Top bottlenecks
* Hypotheses for slow parts

---

# Week 3 — CPU ↔ GPU Transfer, AMP, Gradient Accumulation

## Goal

Understand and apply common training performance knobs.

## Concepts

* CPU to GPU memory transfer
* Host memory vs device memory
* Pinned memory
* Non-blocking transfer
* Mixed precision
* FP32, FP16, BF16
* Gradient accumulation
* Compute-bound vs memory-bound intuition

## Resources

* PyTorch AMP docs
* PyTorch CUDA memory docs
* PyTorch Performance Tuning Guide
* PMPP CUDA memory model intro
* NVIDIA Mixed Precision Guide

## Build Tasks

Add CLI options to `training/train.py`:

```text
--amp
--grad-accum-steps
--num-workers
--pin-memory
--seq-len
--batch-size
```

Create:

```text
benchmarks/bench_training_configs.py
```

## Experiments

Compare:

* FP32 vs AMP
* Batch size 32 direct vs batch size 8 with gradient accumulation 4
* `pin_memory=True/False`
* `non_blocking=True/False`
* Sequence length vs memory/time

Record:

* Tokens/sec
* Peak memory
* Loss curve sanity
* Step time

## Deliverable

Create a performance table with:

```text
baseline
+ dataloader tuning
+ AMP
+ gradient accumulation
```

## Writeup

Create:

```text
docs/week3_training_optimization.md
```

Include:

* AMP effect on memory and throughput
* Gradient accumulation tradeoff
* Whether CPU→GPU transfer is a bottleneck

---

# Week 4 — CUDA Basics + Minimum C++

## Goal

Write basic CUDA kernels and build a minimum C++/CUDA development setup.

## Concepts

* CUDA kernel launch
* Grid/block/thread indexing
* Host memory vs device memory
* `cudaMalloc`
* `cudaMemcpy`
* `cudaFree`
* CUDA events
* C++ pointers/references
* C++ classes
* RAII
* Move semantics

## Resources

* PMPP early CUDA chapters
* NVIDIA CUDA C++ Programming Guide
* LearnCpp: pointers, references, classes, constructors/destructors
* Effective Modern C++: RAII and move semantics, skim only

## Build Tasks

Create:

```text
kernels/
├── CMakeLists.txt
├── vector_add.cu
├── matrix_add.cu
├── matmul_naive.cu
└── common/
    └── gpu_buffer.hpp
```

Implement:

* Vector add kernel
* Matrix add kernel
* Naive matmul kernel
* CUDA error checking macro
* CUDA event timing
* `GpuBuffer` RAII wrapper

`GpuBuffer` requirements:

* `cudaMalloc` in constructor
* `cudaFree` in destructor
* Copy disabled
* Move enabled

## Experiments

* Time H2D, kernel, D2H separately.
* Try different block sizes.
* Verify correctness against CPU implementation.

## Deliverable

Commands should work:

```bash
make cuda_basics
./kernels/vector_add
./kernels/matmul_naive
```

## Writeup

Create:

```text
docs/week4_cuda_basics.md
```

Include:

* Grid/block/thread diagram
* H2D/kernel/D2H timing breakdown
* `GpuBuffer` ownership explanation

---

# Week 5 — CUDA Performance: Coalescing, Tiling, Reduction, LayerNorm

## Goal

Go beyond writing CUDA code and understand why kernels are fast or slow.

## Concepts

* Memory coalescing
* Strided access
* Effective memory bandwidth
* Shared memory
* Tiling
* `__syncthreads()`
* Parallel reduction
* LayerNorm
* Nsight profiling
* Comparing custom kernels with PyTorch ops

## Resources

* PMPP memory coalescing chapter
* PMPP matrix multiplication / tiling chapter
* PMPP reduction chapter
* CUDA Best Practices Guide
* Nsight Systems / Nsight Compute docs
* PyTorch LayerNorm docs
* PyTorch benchmarking utilities

## Build Tasks

Create:

```text
kernels/
├── matmul_tiled.cu
├── reduction.cu
├── layernorm.cu
└── bench_kernels.py
```

Implement:

* Naive matmul vs tiled matmul
* Tile size sweep: 8, 16, 32
* Sum reduction kernel
* Optional LayerNorm forward kernel
* Correctness comparison against PyTorch
* At least one Nsight profile

## Experiments

* Contiguous vs strided access bandwidth
* Naive matmul vs tiled matmul speedup
* Custom LayerNorm vs PyTorch LayerNorm
* Kernel launch overhead for many tiny kernels

## Deliverable

Create:

```text
docs/week5_cuda_perf_report.md
```

Include:

* Before/after speed table
* Nsight summary
* Why PyTorch/cuBLAS is faster than your custom kernel

---

# Week 6 — Distributed Training Intro: DDP + Scaling Mental Model

## Goal

Run tiny distributed training and understand the basic mechanics of data parallelism.

## Concepts

* Data parallelism
* Rank
* World size
* Local rank
* `torchrun`
* DDP
* Gradient all-reduce
* Communication cost
* FSDP high-level
* Tensor parallelism high-level
* Pipeline parallelism high-level

## Resources

* PyTorch Distributed docs
* PyTorch DDP tutorial
* PyTorch FSDP docs, high-level only
* JAX Scaling Book: parallelism and communication sections

## Build Tasks

Create:

```text
distributed_training/
├── ddp_train.py
├── launch.sh
└── notes.md
```

Implement:

* Tiny GPT wrapped with DDP
* Launch with `torchrun`
* Rank/world size/local rank logging
* Per-rank loss logging
* Short explanation of where gradient sync occurs

## Experiments

* Run 2 processes.
* Compare single process vs 2-process DDP if hardware allows.
* Log gradient synchronization timing if possible.
* Intentionally misconfigure world size or port and document the error.

## Deliverable

This should run:

```bash
torchrun --nproc_per_node=2 distributed_training/ddp_train.py
```

## Writeup

Create:

```text
docs/week6_distributed_training.md
```

Include:

* DDP diagram
* Why all-reduce is needed
* What FSDP shards
* Data parallel vs tensor parallel vs pipeline parallel

---

# Week 7 — Inference Baseline: Prefill, Decode, OpenAI-like API

## Goal

Serve the tiny model behind an API and measure inference-specific latency metrics.

## Concepts

* Prefill
* Decode
* TTFT: time to first token
* TPOT: time per output token
* Generation loop
* OpenAI-like API
* FastAPI
* Basic latency metrics
* KV cache intro

## Resources

* vLLM docs: serving concepts and OpenAI-compatible API
* Hugging Face generation docs
* JAX Scaling Book inference sections
* FastAPI docs

## Build Tasks

Create:

```text
serving/
├── api_server.py
├── generation.py
├── metrics.py
└── client.py
```

Implement:

* Minimal `/v1/completions` endpoint
* Prompt input → token generation
* TTFT measurement
* TPOT measurement
* Total latency measurement
* Optional streaming response

## Experiments

Vary:

```text
prompt_length = 16, 128, 512
max_new_tokens = 16, 64, 128
```

Record:

* TTFT
* TPOT
* Total latency
* Output tokens/sec

## Deliverable

These commands should work:

```bash
uvicorn serving.api_server:app --reload
python serving/client.py
```

## Writeup

Create:

```text
docs/week7_inference_baseline.md
```

Include:

* Prefill vs decode explanation
* TTFT/TPOT table
* Generation loop diagram

---

# Week 8 — Static Batching, Dynamic Batching, KV Cache Simulation

## Goal

Implement core LLM inference serving ideas: queueing, batching, scheduling, and KV cache memory management.

## Concepts

* Request queue
* Static batching
* Dynamic batching
* Continuous batching
* Scheduler tick
* Active sequence set
* KV cache
* KV block allocation
* Fragmentation
* Timeout
* Cancellation
* Latency vs throughput

## Resources

* vLLM docs: continuous batching and scheduler concepts
* vLLM PagedAttention docs/paper/blog
* Python `asyncio` docs
* DDIA Chapter 8, timeout and failure sections

## Build Tasks

Create:

```text
serving/
├── scheduler.py
├── kv_cache.py
├── batcher.py
└── load_test.py
```

Implement:

* Async request queue
* Static batching mode
* Dynamic batching mode
* Scheduler tick, for example every 20ms
* Fake KV cache block allocator
* Timeout/cancellation cleanup

A fake worker is acceptable at this stage:

```text
compute_time = base_time + batch_size * factor
```

## Experiments

Vary:

```text
batch_size = 1, 2, 4, 8
scheduler_interval = 5ms, 20ms, 50ms
arrival_rate = low, medium, high
```

Record:

* p50 latency
* p95 latency
* p99 latency
* Requests/sec
* Tokens/sec
* Queue time
* Compute time
* KV cache allocated blocks

## Deliverable

Create:

```text
docs/week8_batching_kv_cache.md
```

Include:

* Static vs dynamic batching table
* KV cache memory formula
* Block allocation simulation
* Timeout cleanup test

---

# Week 9 — Multi-worker Serving: Routing, Health Checks, Retries

## Goal

Turn the inference server into a small distributed serving system.

## Concepts

* Request routing
* Round-robin routing
* Least-loaded routing
* Sticky routing
* Worker registry
* Liveness
* Readiness
* Heartbeat
* Retry
* Idempotency key
* Partial failure
* Worker crash
* Slow worker
* Control plane vs data plane

## Resources

* DDIA Chapter 8: unreliable networks and partial failure
* DDIA Chapter 6: partitioning, routing, skew
* Kubernetes probes docs
* Kubernetes controller/reconciliation mental model
* Optional: Google SRE retry/backoff sections

## Build Tasks

Create:

```text
serving/
├── router.py
├── worker.py
├── registry.py
├── health.py
└── failure_sim.py
```

Implement:

* 3 model workers
* Router with round-robin and least-loaded routing
* Worker heartbeat
* Readiness/liveness
* `request_id`-based idempotency
* Retry once on timeout/failure
* Worker kill/restart simulation
* Optional draining state

## Experiments

* 1 worker vs 2 workers vs 4 workers
* One slow worker
* One crashed worker
* Retry on timeout
* Stale registry state
* Sticky routing vs non-sticky routing

## Deliverable

Example commands:

```bash
python serving/router.py
python serving/worker.py --id worker1
python serving/worker.py --id worker2
python serving/load_test.py
```

Killing one worker should cause the router to stop routing traffic to it.

## Writeup

Create:

```text
docs/week9_distributed_serving.md
```

Include:

* Control plane vs data plane diagram
* Worker state machine
* Retry/idempotency failure case
* Worker failure postmortem

---

# Week 10 — Polish, Tests, Reports, Containerization

## Goal

Turn the project from a learning exercise into a resume-ready engineering artifact.

## Concepts

* Benchmarks
* Correctness tests
* Integration tests
* Failure tests
* Performance report writing
* Docker
* Docker Compose
* Reproducibility
* Design tradeoffs

## Resources

* pytest docs
* Docker docs
* Docker Compose docs
* NVIDIA Container Toolkit docs, optional
* Prometheus client docs, optional
* DDIA Chapter 1 for tradeoff language
* Good GitHub README examples

## Build Tasks

Create:

```text
tests/
├── test_model.py
├── test_scheduler.py
├── test_kv_cache.py
├── test_router_failure.py
└── test_cuda_correctness.py

docker-compose.yml
Dockerfile.router
Dockerfile.worker
Makefile
README.md
```

Add Makefile commands:

```text
make train
make profile-training
make cuda-bench
make serve
make load-test
make test
```

## Final Docs

Create:

```text
docs/architecture.md
docs/training_perf_report.md
docs/cuda_perf_report.md
docs/inference_perf_report.md
docs/distributed_serving_report.md
docs/failure_postmortems.md
```

## Deliverable

A GitHub repo that an interviewer can understand in 3 minutes.

README structure:

```text
1. What this project is
2. Architecture diagram
3. Training stack
4. CUDA kernels
5. Distributed training
6. Inference serving
7. Batching and KV cache
8. Distributed serving features
9. Benchmarks
10. Failure handling
11. What I learned
12. Future work
```

---

# Week 11 — Deep Dive: JAX Scaling Book and Parallelism Taxonomy

## Goal

Build a clear mental model of how large-scale training and inference are parallelized across many accelerators.

## Concepts

* Data parallelism
* Tensor parallelism
* Pipeline parallelism
* Sequence parallelism
* Expert parallelism
* Sharding
* Replication
* Communication collectives
* All-reduce
* All-gather
* Reduce-scatter
* Communication/computation overlap
* Scaling bottlenecks

## Resources

* JAX Scaling Book: parallelism and communication sections
* PyTorch Distributed docs: collectives
* PyTorch FSDP docs
* Megatron-LM or Megatron-Core docs, high-level only

## Build Tasks

Create:

```text
distributed_training/
├── collectives_toy.py
├── parallelism_notes.md
└── scaling_math.md
```

Implement or simulate:

* All-reduce toy example
* All-gather toy example
* Reduce-scatter toy example
* Simple tensor split of a matrix multiply
* Simple pipeline schedule diagram

## Experiments

* Vary tensor size and measure collective time if using multiple GPUs/processes.
* Simulate communication cost with different model sizes.
* Compare data parallel vs tensor parallel communication patterns conceptually.

## Deliverable

Create:

```text
docs/week11_parallelism_taxonomy.md
```

Include:

* Parallelism taxonomy table
* Communication collective diagrams
* When each parallelism strategy is useful
* Why scaling is not linear

---

# Week 12 — FSDP and Memory-Efficient Distributed Training

## Goal

Understand how large models fit in memory using sharding techniques.

## Concepts

* Parameter sharding
* Gradient sharding
* Optimizer state sharding
* FSDP
* ZeRO-style sharding
* Activation checkpointing
* Mixed precision + sharding
* Memory vs communication tradeoff

## Resources

* PyTorch FSDP docs/tutorial
* PyTorch Distributed docs
* JAX Scaling Book memory/sharding sections
* DeepSpeed ZeRO paper/blog, optional
* 10-414 memory/autodiff lectures

## Build Tasks

Create:

```text
distributed_training/
├── fsdp_train.py
├── memory_accounting.py
└── fsdp_notes.md
```

Implement if possible:

* Tiny model with FSDP
* Memory accounting script for params, grads, optimizer states
* Activation checkpointing toggle

If hardware is limited, do a conceptual/simulation version.

## Experiments

Compare:

* DDP memory estimate
* FSDP memory estimate
* Activation checkpointing on/off
* FP32 vs mixed precision memory

## Deliverable

Create:

```text
docs/week12_fsdp_memory_report.md
```

Include:

* What DDP replicates
* What FSDP shards
* Memory formula for params/grads/optimizer state
* Tradeoff: less memory, more communication

---

# Week 13 — Tensor Parallelism and Pipeline Parallelism Simulation

## Goal

Understand how model computation itself is split across devices.

## Concepts

* Column-parallel linear layer
* Row-parallel linear layer
* Tensor parallel matmul
* Pipeline stages
* Pipeline bubble
* Microbatching
* Activation communication
* Sequence length effects
* Communication placement

## Resources

* JAX Scaling Book tensor/pipeline parallelism sections
* Megatron-LM/Megatron-Core docs, high-level
* PyTorch Distributed collectives docs
* 10-414 tensor/matmul/hardware acceleration lectures

## Build Tasks

Create:

```text
distributed_training/
├── tensor_parallel_sim.py
├── pipeline_parallel_sim.py
└── parallel_layers_notes.md
```

Implement or simulate:

* Split matrix multiplication by columns
* Split matrix multiplication by rows
* Simulated communication after partial matmul
* Pipeline schedule with 2 or 4 stages
* Microbatch pipeline bubble visualization

## Experiments

Vary:

* Number of partitions
* Tensor size
* Microbatch count
* Pipeline stage count

Record:

* Simulated compute time
* Simulated communication time
* Bubble overhead
* Total step time

## Deliverable

Create:

```text
docs/week13_tensor_pipeline_parallelism.md
```

Include:

* Row-parallel vs column-parallel diagrams
* Pipeline schedule diagram
* Why pipeline parallelism needs microbatching
* Where communication happens

---

# Week 14 — Inference Scaling: KV Cache, Prefill/Decode Disaggregation, Multi-worker Scheduling

## Goal

Connect the serving system to real LLM inference scaling ideas.

## Concepts

* Prefill/decode disaggregation
* KV cache memory pressure
* Continuous batching
* Chunked prefill
* Prefix caching
* PagedAttention
* Worker specialization
* Routing for cache locality
* Tail latency
* Throughput saturation

## Resources

* vLLM docs
* vLLM PagedAttention paper/blog
* JAX Scaling Book inference sections
* Hugging Face generation docs
* Optional: Ray Serve docs, KServe docs

## Build Tasks

Extend:

```text
serving/
├── scheduler.py
├── kv_cache.py
├── router.py
└── metrics.py
```

Implement or simulate:

* Separate prefill queue and decode queue
* KV cache block usage per request
* Chunked prefill simulation
* Prefix cache simulation
* Cache-aware routing policy
* Tail latency metrics under load

## Experiments

Vary:

* Prompt length distribution
* Output length distribution
* KV cache size limit
* Scheduler interval
* Cache-aware routing on/off

Record:

* TTFT
* TPOT
* p95 latency
* p99 latency
* Tokens/sec
* KV cache utilization
* Rejected requests due to memory pressure

## Deliverable

Create:

```text
docs/week14_inference_scaling_report.md
```

Include:

* Why decode is different from prefill
* KV cache memory formula
* PagedAttention intuition
* Cache-aware routing results
* Tail latency analysis

---

# Week 15 — Final Integration and Capstone Report

## Goal

Integrate all components into a coherent ML systems project and produce a final technical report.

## Concepts

* End-to-end system design
* Training vs inference bottlenecks
* GPU kernel performance
* Distributed training scaling
* Distributed serving failure handling
* Observability
* Reproducibility
* Resume-ready technical communication

## Resources

* Your own previous docs
* PyTorch profiler reports
* Nsight reports
* JAX Scaling notes
* vLLM docs
* DDIA tradeoff language
* Good ML systems project READMEs

## Build Tasks

Finalize:

```text
README.md
docs/final_report.md
docs/architecture.md
docs/results_summary.md
```

Clean up:

* Repo structure
* Makefile commands
* Test suite
* Benchmark scripts
* Docker Compose
* Performance tables
* Architecture diagrams
* Failure postmortems

## Final Benchmark Suite

Run and document:

```bash
make train
make profile-training
make cuda-bench
make ddp-demo
make serve
make load-test
make test
```

## Final Report

Create:

```text
docs/final_report.md
```

Include:

1. Project overview
2. Architecture diagram
3. Tiny GPT training stack
4. Training performance analysis
5. CUDA kernel implementation
6. Distributed training experiments
7. Inference serving system
8. Dynamic batching and KV cache simulation
9. Multi-worker routing and failure handling
10. Scaling concepts from JAX Scaling Book
11. Results table
12. Limitations
13. Future work

## Final Resume Bullets

General version:

```text
Built an end-to-end mini LLM systems stack with PyTorch training, custom CUDA kernels, DDP experiments, and an inference router supporting batching, KV-cache simulation, worker health checks, retries, and latency/throughput profiling.
```

Performance-focused version:

```text
Improved tiny Transformer inference throughput by X% using dynamic batching and KV-cache reuse while measuring p50/p95 latency, queue time, GPU memory, and tokens/sec across multi-worker serving experiments.
```

CUDA-focused version:

```text
Implemented naive and tiled CUDA matrix multiplication and LayerNorm kernels, profiling memory coalescing, shared-memory tiling, and kernel bottlenecks with Nsight and PyTorch profiler.
```

Distributed-systems version:

```text
Designed a multi-worker inference router with worker registry state, health checks, idempotent retries, failure simulation, and control-plane/data-plane separation.
```

Scaling-focused version:

```text
Studied DDP, FSDP, tensor parallelism, pipeline parallelism, and inference scaling tradeoffs using PyTorch Distributed and the JAX Scaling Book, documenting communication and memory bottlenecks.
```

---

# 15-Week Summary

| Week | Focus                       | Main Deliverable                                     |
| ---- | --------------------------- | ---------------------------------------------------- |
| 1    | Tiny GPT baseline           | Model trains and checkpoints                         |
| 2    | Training profiling          | Training profiler report                             |
| 3    | Training optimization       | AMP, gradient accumulation, dataloader benchmarks    |
| 4    | CUDA basics + C++           | Vector add, naive matmul, RAII GPU buffer            |
| 5    | CUDA performance            | Tiled matmul, reduction/LayerNorm, Nsight report     |
| 6    | DDP intro                   | Tiny distributed training demo                       |
| 7    | Inference API               | OpenAI-like API with TTFT/TPOT                       |
| 8    | Batching + KV cache         | Static/dynamic batching and KV cache simulation      |
| 9    | Multi-worker serving        | Router, health checks, retries, failure simulation   |
| 10   | Polish                      | Tests, Docker, README, benchmark suite               |
| 11   | Parallelism taxonomy        | Collectives and scaling notes                        |
| 12   | FSDP/memory                 | FSDP and memory sharding report                      |
| 13   | Tensor/pipeline parallelism | Parallelism simulations                              |
| 14   | Inference scaling           | Prefill/decode, KV cache, cache-aware routing report |
| 15   | Final integration           | Capstone report and resume-ready repo                |

---

# Priority Cuts

If time is limited, cut these:

```text
- Full PyTorch C++ extension
- Full LayerNorm backward kernel
- Real multi-node training
- Full FSDP implementation
- Real OpenTelemetry tracing
- GPU Docker support
- Linux tc network failure simulation
```

Do not cut these:

```text
- Tiny GPT training
- Training profiler report
- AMP / gradient accumulation / dataloader benchmark
- CUDA vector add + naive/tiled matmul
- DDP tiny demo
- Inference API
- Static and dynamic batching
- KV cache simulation
- Multi-worker router
- Health checks
- Retry/idempotency
- Final README with metrics
```

---

# Guiding Principle

This project is not about passively reading CSAPP, OSTEP, PMPP, PyTorch docs, or the JAX Scaling Book.

The loop is:

```text
Build something
→ hit a concept gap
→ read the minimum relevant section
→ implement a toy version
→ benchmark/debug it
→ write what happened
→ integrate it into the system
```

The final goal is to become comfortable reasoning about:

```text
where data moves
where state lives
what fails
what is slow
what can be parallelized
what must be synchronized
what should be measured
what tradeoff each optimization makes
```
