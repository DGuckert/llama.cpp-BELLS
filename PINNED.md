# The host stall, and the 1.80x hiding behind it

Every optimisation in `PREFETCH.md` and `OPTIMIZATION.md` that failed, failed for one reason.
This is that reason, measured.

## The mechanism

`cudaMemcpyAsync` from **pageable** host memory is not asynchronous. The driver stages the
transfer through its own internal pinned buffer and blocks the calling thread for the duration.
Measured in isolation with `bells_h2d.cu` - same bytes, same block size, same stream, only the
source memory differs:

| block | source | bandwidth | host time inside memcpy | |
|---|---|---|---|---|
| 1.09 MB | pageable | 8.55 GB/s | 26.66 ms of 26.73 ms | **blocks** |
| 1.09 MB | pinned | 12.29 GB/s | 0.88 ms of 18.60 ms | async, **1.44x** |
| 2.92 MB | pageable | 11.17 GB/s | 54.75 ms of 54.84 ms | **blocks** |
| 2.92 MB | pinned | 12.57 GB/s | 0.82 ms of 48.75 ms | async, 1.12x |
| 11.3 MB | pageable | 11.05 GB/s | 214.39 ms of 214.46 ms | **blocks** |
| 11.3 MB | pinned | 12.58 GB/s | 0.85 ms of 188.45 ms | async, 1.14x |

**A pageable copy holds the caller for 99.7% of the transfer. A pinned one holds it for 4.7%.**

Block size decides how much bandwidth is on the table, and BELLS lands in the good part of the
curve by accident: it copies `gate`, `up` and `down` as three separate ~1 MB tensors rather than
one 2.92 MB expert. Pageable at 1.09 MB measures 8.55 GB/s against the 8.2 GB/s BELLS actually
achieves - close enough, derived independently, to confirm the model.

## End to end

OLMoE-1B-7B, 3.9 GB, 16 slots, three passes:

| | copy/layer | layer total | decode | perplexity |
|---|---|---|---|---|
| pageable | 1,925 us | 13,219 ms | 31.35 ms | 21.4760 |
| **pinned** | **31 us** | **1,553 ms** | **17.36 ms** | 21.4760 |

**Host time in copy falls 62x. Layer time falls 8.5x. Decode is 1.80x faster.** Bytes moved are
identical at 67.96 GiB and perplexity is identical to four decimals - the same work, the same
output, the host simply stops waiting for it. Variance across passes is under 1%.

## Why this took so long to find

Because it explains the failures rather than announcing itself. From these notes:

| attempt | outcome | actual reason |
|---|---|---|
| pinned staging ring | 37% slower | duplicated the driver's internal copy |
| prefetch, synchronous | slower | serial prologue, host blocked throughout |
| prefetch, background thread | slower | worker thread blocked instead of main |
| second CUDA stream | **no change** | stream ordering was never the constraint |

The second stream is the informative one. It was built correctly, ordered against the graph with
events, and changed nothing - because a second stream cannot help a host that is blocked. That
result was read at the time as "transfers cannot overlap compute"; it actually meant "the host
never gets far enough ahead to overlap anything".

## What blocks it on a 32 GB machine

```
ggml_cuda_host_malloc: failed to allocate 16740.00 MiB of pinned memory: out of memory
```

Windows will not hand out a 16.7 GB pinned block on a 32 GB desktop, and `ggml-cuda.cu:1362`
**silently falls back to a plain CPU buffer**. The diagnostic is `GGML_LOG_DEBUG`, so at normal
verbosity a user asks for pinned memory, receives pageable, and is told nothing.

Every measurement taken inside llama.cpp before this was therefore pageable against pageable,
which is exactly why they all tied. That is worth a warning at normal log level: a silent
downgrade of the thing that turns out to matter most is a bad default.

## Getting at it

`common/arg.cpp` now exposes the pinned host buffer type to `-ot`, which previously could name
every buffer type except the one that matters for tensors deliberately kept on the CPU:

```
llama-bells-profile -m model.gguf -ngl 99 --no-mmap \
    -ot "\.ffn_(gate|up|down)_exps\.=CUDA_Host"
```

`--no-mmap` is required: `llama-model-loader.cpp:1191` replaces a host buffer type with plain CPU
whenever mmap is on, so an mmap'd model can never have pinned weights.

Note this replaces `--cpu-moe` rather than accompanying it - both push a buffer override for the
same tensors, and the first match wins.

## Where this leaves things

The mechanism is confirmed, the integration is proven, and the remaining question is only how
much of a large model can be pinned at once. On 128 GB the answer should be "all of it".
