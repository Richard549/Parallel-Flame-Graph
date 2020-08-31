## Parallel Flame Graphs

This repository contains Parallel Flame Graph (PFG) code to jointly visualize
the execution parallelism and parallel stack-trace of
[OpenMP](https://www.openmp.org/) programs. The work is inspired by Brendan
Gregg's [Flame Graphs](http://www.brendangregg.com/flamegraphs.html).

The repository is currently work-in-progress, supporting loop-based OpenMP
parallelism. It does not yet support task-based OpenMP parallelism.

### Example

PFG examples, visualising a simple parallel program with a single OpenMP
parallel loop executed by two OpenMP threads.

![Image of basic PFG showing all function calls](examples/images/basic.png?raw=true "Basic PFG showing all function calls")

![Image of aggregated PFG merging all similar function calls on the same CPU](examples/images/aggregate.png?raw=true "Aggregated PFG merging all similar function calls on the same CPU")

![Image of stacked PFG where calls from each CPU are stacked vertically](examples/images/stacked.png?raw=true "Stacked PFG where calls from each CPU are stacked vertically")

![Image of folded PFG where function calls are merged across CPUs](examples/images/folded.png?raw=true "Folded PFG where function calls are merged across CPUs")

![Image of folded PFG where width is proportional to wallclock and area is proportional to CPU time](examples/images/cpu_time.png?raw=true "Folded PFG where width is proportional to wallclock and area is proportional to CPU time")

Work-in-progress: PFG where area of each bar is proportional to inefficient
parallel exection (i.e. low parallelism)

### Dependencies

Other than Python3, Matplotlib, and Numpy (tested on `3.6.9`, `3.0.3`, `1.16.4`
respectively), the code requires a trace that contains:

* **Stack frames**: Entry and exit timestamps for each stack frame of interest,
with CPU, function symbol, frame identifier, and parent frame identifier

* **Work periods**: Entry and exit timestamps for each OpenMP parallel
for-loop, for each CPU

* **Synchronization regions**: Entry and exit timestamps for periods of OpenMP
synchronization on each CPU, e.g. work-stealing periods, waiting at an OpenMP
barrier

To generate this tracefile, the
[AfterOMPT](https://github.com/Richard549/Afterompt/tree/fn_instrumentation)
(branch `fn_instrumentation`) tool running on the
[Aftermath](https://www.aftermath-tracing.com/) tracing infrastructure has been
extended to instrument stack traces along with OpenMP constructs (including the
necessary work periods) via the [OMPT](https://www.openmp.org/specifications/)
callback API. Details for the instrumentation can be found in the AfterOMPT
repository, which produces an efficient trace with `.ost` extension.

The `.ost` tracefile must be parsed to dump a CSV from the tracefile, that can
then be ingested by the PFG code. This currently requires my Aftermath fork
[here](https://github.com/Richard549/aftermath/tree/callgraph), which can be
provided the trace and the original binary (to parse its symbol table),
together with the `-o` option to dump the callgraph information to STDOUT.

An example trace as `.csv` files has been provided in the examples directory.

### Usage

Passing `-h` to the runner provides the usage instructions:

		rneill:~/../Parallel-Flame-Graph$ python3 src/PfgRunner.py -h
		usage: PfgRunner.py [-h] -f TRACEFILE [-c HEIGHT_OPTION] [-t TRANSFORM]
												[-o OUTPUT] [-l LOGFILE] [-d LOG_LEVEL] [--tee]

		required arguments:
			-f TRACEFILE, --tracefile TRACEFILE
														Filename to parse for events (as a CSV).

		optional arguments:
			-h, --help            show this help message and exit
			-c HEIGHT_OPTION, --height_option HEIGHT_OPTION
														Calculation method for the bar height when visualising
														the parallel stack trace. Options are:1=CONSTANT,
														2=CPU_TIME, 3=PARALLELISM_INEFFICIENCY.
			-t TRANSFORM, --transform TRANSFORM
														Transformation applied to visualise the parallel stack
														trace. Options are:1=NONE, 2=AGGREGATE_CALLS,
														3=VERTICAL_STACK_CPU, 4=COLLAPSE_GROUPS.
			-o OUTPUT, --output OUTPUT
														Output filename (if set, the PFG will be saved as
														.PNG).
			-l LOGFILE, --logfile LOGFILE
														Filename to output log messages (defaults to log.txt).
			-d LOG_LEVEL, --log_level LOG_LEVEL
														Logging level. Options are:1=INFO, 2=DEBUG, 3=TRACE.
			--tee                 Pipe logging messages to stdout as well as the log
														file.

### Licence

This project is licensed under the [MIT License](LICENSE.txt).
