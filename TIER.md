# A storage tier below the expert cache

BELLS caches experts in VRAM and copies from RAM on a miss. This asks what happens when the
model does not fit in RAM either, so that a miss may have to reach the drive. The short answer
is that it is worth building for models with small experts and worth nothing for the models
people actually want it for.

## The idea, and one correction to it

The proposal was three parts: speculatively pull experts into RAM to keep the drive busy, use
RAM as a warm staging tier between NVMe and GPU, and when the GPU evicts an expert that the
predictor thinks will return, demote it to RAM rather than dropping it.

**The third part already happens.** The GGUF is mmap'd, so expert weights live in RAM
permanently and every VRAM copy reads from there. Evicting from VRAM frees a *slot*; it never
discards data. There is no warm tier to add, because RAM already is the cold tier.

That stops being true the moment the model exceeds RAM. Then the "RAM copy" is a page-cache
entry, and the OS evicts it by a generic LRU that knows nothing about routing. An explicit RAM
tier under BELLS' control is a real design, and it is where prediction would finally earn
something: the oracle gap is 16.4 points at a 2x cache ratio against 2.4 points at 8x, and a
200B model on 32 GB of RAM sits near 2x.

## Queue depth: measured

The load-bearing claim was that mmap page faults, being serial, leave drive bandwidth
unclaimed, so batching expert reads would recover it. Measured on the volume holding the model,
with `FILE_FLAG_NO_BUFFERING` so the page cache cannot answer:

| block | QD1 | QD4 | QD16 | QD32 | QD32/QD1 |
|---|---|---|---|---|---|
| 64 KB *(control)* | 379 MB/s | 1,191 | 1,963 | 1,982 | **5.23x** |
| 1.09 MB *(Qwen3-Next)* | 1,812 MB/s | 2,816 | 2,807 | 2,792 | **1.54x** |
| 2.92 MB *(Qwen3-30B)* | 2,314 MB/s | 2,895 | 2,876 | 2,878 | 1.24x |
| 11.30 MB *(Qwen3-235B)* | 2,604 MB/s | 2,908 | 2,893 | 2,888 | **1.11x** |

The 64 KB row is the control and behaves as small random reads should, so the method is sound.
Then the answer splits along expert size:

- **11.3 MB experts: batching is pointless.** One outstanding read already reaches 90% of the
  drive, because transfer time swamps the ~80 us of latency. This is the model class the tier
  was proposed for, and it is the class the tier cannot help.
- **1.09 MB experts: batching is worth 1.54x.** Serial reads leave a third of the drive on the
  floor.
- **QD4 captures all of it.** Sixteen and thirty-two add nothing, so the implementation is four
  reads in flight, not an async I/O engine.

An earlier claim of 2-3x was a small-random-read figure that does not transfer to megabyte
blocks. The 64 KB row is where that number actually lives.

Caveat: measured on a volume 98% full, where SSDs degrade. The 2.9 GB/s ceiling is below what
the drive should manage, so ratios should hold but absolute figures are pessimistic.

## Where speculative fill pays, and where it does not

Only where the drive has spare capacity, which is narrower than it sounds.

For a 200B model on 32 GB of RAM the drive is not underutilised, it is **saturated** - roughly
3.0 GB/token at ~2.75 GB/s is 1.09 s/token, which is the whole of the ~1 tok/s that
configuration achieves. Speculative reads there displace demand reads, which is precisely the
failure that killed prefetching at the PCIe tier one level up. Same arithmetic, same outcome.

The band where it pays is models **1 to 1.5x the size of RAM**, where the drive is 20-30% busy
and speculation costs spare capacity. Qwen3-Next-80B at ~45 GB against 32 GB of RAM sits in it,
at 1.4x - and it is also the model with the best result in this project, 2.08x measured.

## So

| | small experts (~1 MB) | large experts (~11 MB) |
|---|---|---|
| queue-depth batching | **1.54x on the NVMe tier** | 1.11x, not worth building |
| speculative RAM fill | pays at 1-1.5x RAM | drive already saturated |
| routing-aware RAM eviction | up to 16.4pt over LRU at 2x | same, but the tier is saturated anyway |

The tier is worth building for small-expert models that slightly exceed RAM. It is not a route
to running a 235B on a desktop, because at 11.3 MB per expert the drive is already delivering
everything it has.

## Not built, and what blocks it

- **No local model exceeds RAM.** The largest here is 17.3 GB against 32 GB, so nothing
  exercises the tier at all.
- **No room for one.** Both NVMe volumes are near full - 7.4 GB and 33.2 GB free - and a
  Qwen3-Next-80B at Q4 is ~45 GB. The 1.8 TB volume with room is not one of the SSDs, and
  measuring this on spinning rust would be meaningless.
- A way round both: cap usable RAM with a deliberate allocation so the 17 GB model becomes an
  effective 1.4x-over-RAM workload. Same band, no download, no disk. Intrusive enough on a
  shared machine to want asking first.
