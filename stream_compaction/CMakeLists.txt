set(headers
    "common.h"
    "cpu.h"
    "naive.h"
    "efficient.h"
    "thrust.h"
    )

set(sources
    "common.cu"
    "cpu.cu"
    "naive.cu"
    "efficient.cu"
    "thrust.cu"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

# Ensure these source files are treated as CUDA files by the generator/IDE
set_source_files_properties(${sources} PROPERTIES LANGUAGE CUDA)

add_library(stream_compaction ${sources} ${headers})
set_target_properties(stream_compaction PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Don't forcibly link cudadevrt; let CMake's CUDA runtime linking handle it via the executable target if available.
