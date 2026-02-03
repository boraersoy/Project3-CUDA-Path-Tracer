# Depth Timing Implementation

## Overview
This implementation adds per-depth render time measurement to the path tracer, allowing you to see how many milliseconds are spent at each depth level during rendering.

## Changes Made

### 1. utilities.h - GuiDataContainer
Added timing fields to track performance:
- `float DepthTimings[MAX_DEPTH]`: Array storing the time (in milliseconds) spent at each depth level
- `float TotalIterationTime`: Total time for the entire iteration (in milliseconds)
- `MAX_DEPTH = 32`: Maximum depth levels to track

### 2. pathtrace.cu - Timing Measurements
Added high-resolution timing using `std::chrono`:
- **Iteration Timing**: Measures the total time from the start of `pathtrace()` to completion
- **Per-Depth Timing**: Measures time for each depth iteration in the main loop, including:
  - Intersection computations
  - Material sorting (if enabled)
  - Shading
  - Stream compaction
- Uses `cudaDeviceSynchronize()` to ensure accurate GPU timing measurements
- Stores timing data in the `GuiDataContainer` for display

### 3. preview.cpp - ImGui Display
Enhanced the "Path Tracer Analytics" window to show:
- Total iteration time
- Time spent at each depth level
- Cumulative time and percentage for each depth
- Visual breakdown of where time is being spent

## Usage

When you run the path tracer, the ImGui window will now display:

```
Path Tracer Analytics
---------------------
Traced Depth 5
Total Iteration Time: 45.234 ms
-----------------------------------
Time per Depth Level:
  Depth 0: 12.345 ms (27.3% cumulative: 12.345 ms)
  Depth 1: 10.234 ms (49.9% cumulative: 22.579 ms)
  Depth 2: 8.123 ms (68.1% cumulative: 30.702 ms)
  Depth 3: 6.789 ms (83.1% cumulative: 37.491 ms)
  Depth 4: 7.743 ms (100.0% cumulative: 45.234 ms)
-----------------------------------
Application average 16.234 ms/frame (61.6 FPS)
```

## Benefits

1. **Performance Analysis**: Identify which depth levels are most expensive
2. **Optimization Target**: See where to focus optimization efforts (e.g., if depth 0 is slow, optimize first bounce)
3. **Stream Compaction Validation**: Observe how time decreases with depth as paths are terminated
4. **Real-time Monitoring**: Watch timing metrics update live as you render

## Notes

- Timing includes all GPU operations: intersection testing, shading, sorting, and stream compaction
- `cudaDeviceSynchronize()` is called to ensure accurate measurements, which may add slight overhead
- The percentages show cumulative time, helping you understand the cost distribution
- Maximum depth tracking is limited to 32 levels (configurable via `MAX_DEPTH`)
