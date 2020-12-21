from argparse import ArgumentParser, ArgumentTypeError
from collections import defaultdict
import logging

from PfgTree import PFGTree, TransformationOption, ColourMode
from PfgTracefile import process_events, count_function_calls
from PfgPlotting import plot_pfg_tree, HeightDisplayOption
from PfgUtil import initialise_logging, debug_mode, LogLevelOption

def run_pfg(
		tracefile,
		transformation,
		height_display_option,
		output_file,
		x_bounds,
		inclusive
		):
	
	logging.info("Parsing the tracefile.")
	(top_level_entities,
		unique_groups,
		max_depth,
		min_timestamp,
		max_timestamp,
		cpus,
		counters) = process_events(tracefile)

	logging.info("Generating the tree.")
	tree = PFGTree(top_level_entities)
	tree.colour_mode = ColourMode.BY_PARALLELISM

	if debug_mode():
		logging.debug("Printing tree:")	
		tree.print_tree()

	transformed = True
	if transformation == TransformationOption.NONE:
		transformed = False
		logging.info("Applying no tree transformation.")	
	elif transformation == TransformationOption.AGGREGATE_CALLS:
		logging.info("Transforming tree to aggregate repeated function calls into single nodes.")	
		tree.transform_tree_aggregate_stack_frames()
	elif transformation == TransformationOption.MERGE_CALLS_ACROSS_CPUS:
		logging.info("Transforming tree to aggregate sibling function calls executed on different CPUs.")	
		tree.transform_tree_aggregate_siblings_across_cpus()
		tree.generate_exclusive_cpu_times(tree.root_nodes)
		if inclusive:
			tree.generate_inclusive_event_counts(tree.root_nodes, counters)
		else:
			tree.generate_exclusive_parallelism_intervals(tree.root_nodes)
	elif transformation == TransformationOption.VERTICAL_STACK_CPU:
		logging.info("Transforming tree to stack CPU entities vertically.")	
		tree.transform_tree_stack_cpu_vertically()
	elif transformation == TransformationOption.COLLAPSE_GROUPS:
		logging.info("Transforming tree to merge edge-connected nodes across CPUs by function call, and collapsing to fit wallclock.")	
		tree.transform_tree_collapse_groups()
	else:
		transformed = False
		logging.info("The transformation option %s is not yet implemented. Proceeding with no tree transformation.", transformation.name)
	
	tree.assign_colour_indexes_to_nodes(tree.root_nodes, len(cpus))

	if debug_mode() and transformed:
		logging.debug("Printing post-transformation tree:")	
		tree.print_tree()

	logging.info("Plotting the tree.")

	plot_pfg_tree(tree,
		[min_timestamp],
		[max_timestamp],
		cpus,
		height_display_option,
		output_file,
		x_bounds,
		counters,
		tracefile)

def run_pfg_differential(
		tracefile_tar,
		tracefile_ref,
		transformation,
		height_display_option,
		output_file,
		x_bounds,
		inclusive
		):

	reference_tree = None	
	target_tree = None	

	min_timestamps = []
	max_timestamps = []

	cpus_overall = None
	counters_overall = None

	for tracefile in [tracefile_ref, tracefile_tar]:

		id_str = "target"
		if tracefile == tracefile_ref:
			id_str = "reference"

		logging.info("Parsing %s tracefile.", id_str)
		(top_level_entities,
			unique_groups,
			max_depth,
			min_timestamp,
			max_timestamp,
			cpus,
			counters) = process_events(tracefile)

		min_timestamps.append(min_timestamp)
		max_timestamps.append(max_timestamp)

		if cpus_overall is None:
			cpus_overall = cpus
		if counters_overall is None:
			counters_overall = counters

		logging.info("Generating the %s tree.", id_str)
		tree = PFGTree(top_level_entities)
		tree.colour_mode = ColourMode.BY_PARALLELISM

		if debug_mode():
			logging.debug("Printing %s tree:", id_str)	
			tree.print_tree()

		transformed = True
		if transformation == TransformationOption.NONE:
			transformed = False
			logging.info("Applying no tree transformation on %s.", id_str)	
		elif transformation == TransformationOption.AGGREGATE_CALLS:
			logging.info("Transforming %s tree to aggregate repeated function calls into single nodes.", id_str)	
			tree.transform_tree_aggregate_stack_frames()
		elif transformation == TransformationOption.MERGE_CALLS_ACROSS_CPUS:
			logging.info("Transforming %s tree to aggregate sibling function calls executed on different CPUs.", id_str)
			tree.transform_tree_aggregate_siblings_across_cpus()
			tree.generate_exclusive_cpu_times(tree.root_nodes)
			if inclusive:
				tree.generate_inclusive_event_counts(tree.root_nodes, counters)
			else:
				tree.generate_exclusive_parallelism_intervals(tree.root_nodes)
		elif transformation == TransformationOption.VERTICAL_STACK_CPU:
			logging.info("Transforming %s tree to stack CPU entities vertically.", id_str)	
			tree.transform_tree_stack_cpu_vertically()
		elif transformation == TransformationOption.COLLAPSE_GROUPS:
			logging.info("Transforming %s tree to merge edge-connected nodes across CPUs by function call, and collapsing to fit wallclock.", id_str)	
			tree.transform_tree_collapse_groups()
		else:
			transformed = False
			logging.info("The transformation option %s is not yet implemented. Proceeding with no tree transformation on %s.", transformation.name, id_str)

		if debug_mode() and transformed:
			logging.debug("Printing post-transformation %s tree:", id_str)	
			tree.print_tree()
	
		tree.assign_colour_indexes_to_nodes(tree.root_nodes, len(cpus))

		if tracefile == tracefile_ref:
			reference_tree = tree
		else:
			target_tree = tree

	# So I have the two trees, transform one with respect to the other
	reference_tree.calculate_nodes_differential_version2(reference_tree.root_nodes, target_tree.root_nodes, counters_overall)

	# Because I have calculated differential event values, I need to recalibrate the colour mappings
	reference_tree.assign_colour_indexes_to_nodes(reference_tree.root_nodes, len(cpus_overall))

	logging.info("Plotting the differential graph.")

	plot_pfg_tree(reference_tree,
		min_timestamps,
		max_timestamps,
		cpus_overall,
		height_display_option,
		output_file,
		x_bounds,
		counters_overall,
		tracefile_tar,
		tracefile_ref)

def transformation_option(input_string):
	transform_option_int = 0
	try:
		transform_option_int = int(input_string)
		enum_value = TransformationOption(transform_option_int)
	except ValueError:
		raise ArgumentTypeError("Transformation type not recognised. See usage.")
	return enum_value

def height_option(input_string):
	height_option_int = 0
	try:
		height_option_int = int(input_string)
		enum_value = HeightDisplayOption(height_option_int)
	except ValueError:
		raise ArgumentTypeError("Height option not recognised. See usage.")
	return enum_value

def log_level_option(input_string):
	log_level_option_int = 0
	try:
		log_level_option_int = int(input_string)
		enum_value = LogLevelOption(log_level_option_int)
	except ValueError:
		raise ArgumentTypeError("Log level option not recognised. See usage.")
	return enum_value

def parse_args():

	parser = ArgumentParser()
	optional = parser._action_groups.pop() # will return the default 'optional' options-group (including the 'help' option)

	required = parser.add_argument_group('required arguments')
	required.add_argument('-f', '--tracefile', required=True, help="Target filename to parse for events (as a CSV).")
	required.add_argument('-r', '--reference_tracefile', required=False, help="Reference filename to use as reference in differential analysis of the target tracefile.")

	tf_options_str = ", ".join([str(member.value) + "=" + str(name) for name,member in TransformationOption.__members__.items()])
	height_options_str = ", ".join([str(member.value) + "=" + str(name) for name,member in HeightDisplayOption.__members__.items()])
	ll_options_str = ", ".join([str(member.value) + "=" + str(name) for name,member in LogLevelOption.__members__.items()])

	optional.add_argument('-c', '--height_option', type=height_option, default=HeightDisplayOption.CONSTANT, help="Calculation method for the bar height when visualising the parallel stack trace. Options are:" + height_options_str + ".")
	optional.add_argument('-t', '--transform', type=transformation_option, default=TransformationOption.NONE, help="Transformation applied to visualise the parallel stack trace. Options are:" + tf_options_str + ".")
	optional.add_argument('-i', '--inclusive', action='store_true', required=False, help="Sets each node to aggregate data for their entire duration on stack (as opposed to only active duration spent at top of stack).")
	optional.add_argument('-o', '--output', required=False, help="Output filename (if set, the PFG will be saved as .PNG).")

	optional.add_argument('-l', '--logfile', default="log.txt", help="Filename to output log messages (defaults to log.txt).") 
	optional.add_argument('-d', '--log_level', type=log_level_option, default=LogLevelOption.INFO, help="Logging level. Options are:" + ll_options_str + ".")
	optional.add_argument('-w', '--width', required=False, help="X-axis start and end given as a string a_b.")
	optional.add_argument('--function_breakdown', action='store_true', required=False, help="Parse the tracefile and output the number of calls of each function.")
	optional.add_argument('--tee', action='store_true', help="Pipe logging messages to stdout as well as the log file.") 

	parser._action_groups.append(optional)

	args = parser.parse_args()

	return args.tracefile, args.reference_tracefile, args.transform, args.logfile, args.log_level, args.tee, args.height_option, args.output, args.width, args.function_breakdown, args.inclusive

tracefile, reference_tracefile, transformation_type, logfile, log_level, tee_mode, height_option, output_file, xaxis_bounds_str, function_summary, inclusive = parse_args()
initialise_logging(logfile, log_level, tee_mode)

logging.info("Running Parallel Flame Graph for %s", tracefile)
logging.info("Selected transformation option was %s.", transformation_type.name)
logging.info("Selected height display option was %s.", height_option.name)

if xaxis_bounds_str is not None:
	x_start = float(xaxis_bounds_str.split("_")[0])
	x_end = float(xaxis_bounds_str.split("_")[1])
	bounds = [x_start,x_end]
else:
	bounds = None

if function_summary is True:
	function_counts = count_function_calls(tracefile)
	logging.info("Function call breakdown:")
	for function, number in function_counts.items():
		logging.info("%s:%d", function, number)
	logging.info("Done")
else:

	if reference_tracefile is None:
		run_pfg(tracefile, transformation_type, height_option, output_file, bounds, inclusive)
	else:
		run_pfg_differential(tracefile, reference_tracefile, transformation_type, height_option, output_file, bounds, inclusive)

	logging.info("Done")

