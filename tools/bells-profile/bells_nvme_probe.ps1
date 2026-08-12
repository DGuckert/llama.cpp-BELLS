# Does queue depth buy anything at expert-sized reads?
#
# The proposed NVMe->RAM tier rests on the claim that mmap page faults, being serial, leave
# drive bandwidth unclaimed - so batching expert reads at higher queue depth would recover it.
# That is true for small random reads, where latency dominates. It is not obviously true at
# expert sizes: a 1.09 MB read is ~80 us of latency against ~340 us of transfer, so QD1 should
# already reach ~80% of peak, and an 11.3 MB read should reach ~98%.
#
# This measures it instead of assuming. Reads are issued with FILE_FLAG_NO_BUFFERING so the
# page cache cannot answer them - without that we would be timing RAM, and the 17 GB model file
# fits in 32 GB of it.
#
#   powershell -File bells_nvme_probe.ps1 -Path <big file on the drive under test>

param(
    [string] $Path = "C:\Users\Daniel\llama.cpp-BELLS\models\bells\Qwen3-30B-A3B-Q4_K_M.gguf",
    [int]    $Seconds = 4
)

Add-Type -TypeDefinition @"
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;

public class Nvme {
    const uint GENERIC_READ = 0x80000000;
    const uint FILE_SHARE_READ = 1, FILE_SHARE_WRITE = 2;
    const uint OPEN_EXISTING = 3;
    const uint FILE_FLAG_NO_BUFFERING = 0x20000000;   // page cache cannot serve these
    const uint FILE_FLAG_OVERLAPPED   = 0x40000000;
    const uint MEM_COMMIT = 0x1000, MEM_RESERVE = 0x2000, MEM_RELEASE = 0x8000;
    const uint PAGE_READWRITE = 4;
    const uint INFINITE = 0xFFFFFFFF;

    [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern IntPtr CreateFileW(string p, uint acc, uint share, IntPtr sec,
                                     uint disp, uint flags, IntPtr tmpl);
    [DllImport("kernel32", SetLastError = true)]
    static extern bool ReadFile(IntPtr h, IntPtr buf, uint n, IntPtr read, IntPtr ov);
    [DllImport("kernel32", SetLastError = true)]
    static extern bool GetOverlappedResult(IntPtr h, IntPtr ov, out uint n, bool wait);
    [DllImport("kernel32", SetLastError = true)]
    static extern IntPtr CreateEventW(IntPtr sec, bool manual, bool init, IntPtr name);
    [DllImport("kernel32", SetLastError = true)]
    static extern uint WaitForMultipleObjects(uint n, IntPtr[] h, bool all, uint ms);
    [DllImport("kernel32", SetLastError = true)]
    static extern bool CloseHandle(IntPtr h);
    [DllImport("kernel32", SetLastError = true)]
    static extern IntPtr VirtualAlloc(IntPtr addr, UIntPtr size, uint type, uint prot);
    [DllImport("kernel32", SetLastError = true)]
    static extern bool VirtualFree(IntPtr addr, UIntPtr size, uint type);
    [DllImport("kernel32", SetLastError = true)]
    static extern bool GetFileSizeEx(IntPtr h, out long size);

    // Returns MB/s. Issues `qd` reads of `block` bytes concurrently at random aligned offsets
    // for `seconds`, never letting fewer than `qd` requests be outstanding.
    public static double Run(string path, int block, int qd, int seconds) {
        IntPtr h = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                               IntPtr.Zero, OPEN_EXISTING,
                               FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, IntPtr.Zero);
        if (h == (IntPtr)(-1)) throw new Exception("open failed " + Marshal.GetLastWin32Error());

        long size;
        if (!GetFileSizeEx(h, out size)) throw new Exception("size failed");

        // one page-aligned buffer and one OVERLAPPED per outstanding request
        IntPtr[] bufs = new IntPtr[qd];
        IntPtr[] ovs  = new IntPtr[qd];
        IntPtr[] evs  = new IntPtr[qd];
        for (int i = 0; i < qd; i++) {
            bufs[i] = VirtualAlloc(IntPtr.Zero, (UIntPtr)block, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            ovs[i]  = Marshal.AllocHGlobal(32);
            evs[i]  = CreateEventW(IntPtr.Zero, true, false, IntPtr.Zero);
        }

        Random rnd = new Random(1234);
        long maxBlk = (size - block) / block;
        Stopwatch sw = Stopwatch.StartNew();
        long done = 0;

        Action<int> issue = delegate(int i) {
            long off = (long)(rnd.NextDouble() * maxBlk) * block;   // block-aligned, required
            Marshal.WriteInt64(ovs[i], 0, 0);
            Marshal.WriteInt64(ovs[i], 8, off);
            Marshal.WriteIntPtr(ovs[i], 16, evs[i]);
            ReadFile(h, bufs[i], (uint)block, IntPtr.Zero, ovs[i]);
        };

        for (int i = 0; i < qd; i++) issue(i);

        while (sw.Elapsed.TotalSeconds < seconds) {
            uint idx = WaitForMultipleObjects((uint)qd, evs, false, INFINITE);
            int i = (int)idx;
            if (i < 0 || i >= qd) break;
            uint got;
            GetOverlappedResult(h, ovs[i], out got, true);
            done += got;
            issue(i);          // keep the queue full
        }
        double secs = sw.Elapsed.TotalSeconds;

        for (int i = 0; i < qd; i++) {
            VirtualFree(bufs[i], UIntPtr.Zero, MEM_RELEASE);
            Marshal.FreeHGlobal(ovs[i]);
            CloseHandle(evs[i]);
        }
        CloseHandle(h);
        return (done / 1048576.0) / secs;
    }
}
"@

$file = Get-Item $Path
Write-Output ""
Write-Output ("source: {0} ({1:N1} GB) on {2}" -f $file.Name, ($file.Length/1GB), $file.PSDrive.Name)
Write-Output "FILE_FLAG_NO_BUFFERING, random block-aligned offsets, ${Seconds}s per point"
Write-Output ""
Write-Output ("{0,-14} {1,10} {2,10} {3,10} {4,10}   {5}" -f "block", "QD1", "QD4", "QD16", "QD32", "QD32/QD1")
Write-Output ("-"*70)

# 1.09 MB = Qwen3-Next expert, 2.92 MB = Qwen3-30B expert, 11.3 MB = Qwen3-235B expert
foreach ($kb in @(64, 1116, 2990, 11571)) {
    $block = $kb * 1024
    $r = @()
    foreach ($qd in @(1, 4, 16, 32)) {
        $r += [Nvme]::Run($file.FullName, $block, $qd, $Seconds)
    }
    $label = if ($kb -lt 1024) { "$kb KB" } else { "{0:N2} MB" -f ($kb/1024) }
    Write-Output ("{0,-14} {1,9:N0}M {2,9:N0}M {3,9:N0}M {4,9:N0}M   {5,5:N2}x" -f `
        $label, $r[0], $r[1], $r[2], $r[3], ($r[3]/[math]::Max(1,$r[0])))
}

Write-Output ""
Write-Output "64 KB is the control: queue depth should help most there. If the expert-sized"
Write-Output "rows show little gain, serial mmap faults are not leaving bandwidth unclaimed"
Write-Output "and the batching half of the tier design is not worth building."
