## Parallel Flame Graphs

This repository contains Parallel Flame Graph (PFG) code to jointly visualize the execution parallelism and parallel stack-trace of [OpenMP](https://www.openmp.org/) programs. The work is inspired by Brendan Gregg's [Flame Graphs](http://www.brendangregg.com/flamegraphs.html).

The repository is currently work-in-progress, supporting loop-based OpenMP parallelism. It does not yet support task-based OpenMP parallelism.

### Example

Program code

Standard PFG (no transformation)

CPU-stacked PFG

Folded CPU-time PFG

Inefficiency PFG

### Dependencies

Other than Python3, Matplotlib, and Numpy (tested on `3.6.9`, `3.0.3`, `1.16.4` respectively), the code requires a trace that contains:

**Stack frames**:
- Entry and exit timestamps for each stack frame of interest, with CPU, function symbol, frame identifier, and parent frame identifier
**Work periods**:
- Entry and exit timestamps for each OpenMP parallel for-loop, for each CPU
**Synchronization regions**
- Entry and exit timestamps for periods of OpenMP synchronization on each CPU, e.g. work-stealing periods, waiting at an OpenMP barrier

To generate this tracefile, the [AfterOMPT](My Afterompt fork) tool running on the [Aftermath]() tracing infrastructure has been extended to instrument stack traces along with OpenMP constructs (including the necessary work periods) via the [OMPT]() callback API. Details for the instrumentation can be found in the AfterOMPT repository, which produces an efficient trace with `.ost` extension.

The `.ost` tracefile must be parsed to dump a `.csv` from the tracefile, that can then be ingested by the PFG code. To do this, the Aftermath fork [here]() can be provided the trace and the original binary, together with `--dump-trace" option.

Example traces as `.csv` files have been provided in the examples directory.

### How to use

### Licence








