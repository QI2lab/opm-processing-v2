# NVENC comparison: sample001, p000/c000

Acquisition: `F:\ecoli_beads\20260924\20260924_180905_sample001`.

## Timing

17 acquired scan planes ? 5 ms ? 1 stored channel = 85 ms/volume. The combined laser label `488 + 637` is one stored channel, not two. Playback is exactly 200/17 fps (11.7647058824), so 10,000 frames occupy 850 seconds (14:10). Last frame timestamp is 849.915 seconds. This follows the requested exposure-based timing; acquisition overhead and pauses are excluded.

The existing 30 fps H.264 MP4 is 334,527,702 bytes and lasts 333.33 seconds, playing 2.55? faster than this timing model.

## Comparison

The lossless exported TIFFs are the reference, avoiding an extra generation of compression from the existing movie. All trials use the same 360 full-resolution frames: indices 0?119, 4940?5059, and 9880?9999. These three excerpts are concatenated identically, with two scene cuts. Total comparison duration is 30.6 seconds. Resolution is 1868?884; no resizing, frame dropping, or cropping. All use NVENC P7/high-quality, NV12, no B frames, 8,000,000 b/s target and maximum; default GOP is two seconds. Lookahead variants use 32 frames and full-resolution multipass. HEVC uses hvc1; H.264 uses avc1.

| Scheme | Size (decimal MB) | Measured Mbps | PSNR (dB) | Foreground PSNR (dB) |
|---|---:|---:|---:|---:|
| h264_cbr | 32.094 | 8.391 | 38.12 | 33.62 |
| h264_vbr | 32.195 | 8.417 | 38.13 | 33.63 |
| h264_vbr_lookahead | 32.259 | 8.434 | 37.99 | 33.47 |
| h264_vbr_longgop | 32.245 | 8.430 | 38.03 | 33.51 |
| hevc_cbr | 31.427 | 8.216 | 37.85 | 33.33 |
| hevc_vbr | 31.512 | 8.238 | 37.84 | 33.32 |

Every comparison decoded to exactly 360 frames. PSNR compares decoded grayscale intensities against the corresponding TIFFs. Foreground PSNR includes pixels with reference intensity >10, reducing the influence of blank canvas; annotations are included. No perceptual denoising metric or biological-feature accuracy claim is made.

## Finding

H.264 VBR, lookahead/multipass, and longer GOPs did not reduce file size at the same bitrate target. HEVC CBR was only 2.1% smaller in these excerpts and had slightly lower measured PSNR, with narrower playback compatibility. These small differences include rate-control transients and should not be treated as full-movie savings. Retain H.264 CBR for broad compatibility.

At fixed duration, size is average bitrate ? duration / 8 plus container overhead. Material size reductions require a lower actual average bitrate, even if an 8 Mbps ceiling is retained. An 850-second, 8 Mbps movie is approximately 850 MB. Codec changes alone cannot circumvent that relationship.

## Implementation

Both video commands now default to the volume rate in TIFF `volume_interval_ms` metadata. Fractional rates are preserved in NVENC configuration and MP4 muxing. `--fps` remains an explicit override. Standalone encoding offers `--codec h264|hevc` and `--rate-control cbr|vbr`; default remains H.264 CBR at 8 Mbps.

A separate `sample001_decon_deskewed_realtime_8M.mp4` is generated beside the p000/c000 frames; the original movie is preserved. The new file is 851,525,034 bytes (851.53 MB), corresponding to 8.014 Mbps including container overhead at 850 seconds.

Reproduction script: `diagnostics/compare_nvenc_sample001.py`. Detailed measurements and preview movies: `diagnostics/nvenc_comparison_sample001/`. Full movie validation is written to `full_movie.json` there: all 10,000 frames decoded, demuxer frame rate 11.764705882352942 fps, duration 850.0 seconds, H.264/yuv420p at 1868 by 884 pixels. Seven unit/integration checks passed, including the real GPU round trip at the metadata-derived fractional frame rate.

Encoder API reference: [NVIDIA PyNvVideoCodec encoding guide](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/using_pynvvideocodec_apis.html#video-encoding).

## Spatially reduced companion

The revised companion averages only image panels in 4-by-4 blocks. Timestamp
and scale-bar text are redrawn at native 20-pixel size; the bar length uses the
reduced physical pixel spacing. The canvas expands to fit the annotations,
yielding a 534 by 276 encoded frame. The companion bitrate is now 2 Mbps
(configurable with --companion-bitrate), while the full version retains 8 Mbps.

Verified revised output: sample001_decon_deskewed_realtime_8M_4x.mp4,
212,553,617 bytes (212.55 MB), versus 851,525,034 bytes for the full version.
Both decode to 10,000 frames at 200/17 fps and last 850 seconds. The 8M in
the companion filename identifies the full-size source configuration.
Ten unit/integration checks passed; preview and decoded preview were visually
checked for readable micrometer labels and timestamps.

Both versions were regenerated with explicit min:s:ms labels beneath timestamps.
Final file sizes are 851,460,345 bytes (full) and 212,603,227 bytes (4x).
Both again decoded to 10,000 frames at 200/17 fps; validation is recorded in
time_unit_movies.json.
