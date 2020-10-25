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
		x_bounds
		):
	
	logging.info("Parsing the tracefile.")
	(top_level_entities,
		unique_groups,
		max_depth,
		min_timestamp,
		max_timestamp,
		cpus) = process_events(tracefile)

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
	elif transformation == TransformationOption.VERTICAL_STACK_CPU:
		logging.info("Transforming tree to stack CPU entities vertically.")	
		tree.transform_tree_stack_cpu_vertically()
	elif transformation == TransformationOption.COLLAPSE_GROUPS:
		logging.info("Transforming tree to merge edge-connected nodes across CPUs by function call, and collapsing to fit wallclock.")	
		tree.transform_tree_collapse_groups()
	else:
		transformed = False
		logging.info("The transformation option %s is not yet implemented. Proceeding with no tree transformation.", transformation.name)

	if debug_mode() and transformed:
		logging.debug("Printing post-transformation tree:")	
		tree.print_tree()

	logging.info("Plotting the tree.")

	plot_pfg_tree(tree,
		min_timestamp,
		max_timestamp,
		cpus,
		height_display_option,
		output_file,
		x_bounds)

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
	required.add_argument('-f', '--tracefile', required=True, help="Filename to parse for events (as a CSV).")

	tf_options_str = ", ".join([str(member.value) + "=" + str(name) for name,member in TransformationOption.__members__.items()])
	height_options_str = ", ".join([str(member.value) + "=" + str(name) for name,member in HeightDisplayOption.__members__.items()])
	ll_options_str = ", ".join([str(member.value) + "=" + str(name) for name,member in LogLevelOption.__members__.items()])

	optional.add_argument('-c', '--height_option', type=height_option, default=HeightDisplayOption.CONSTANT, help="Calculation method for the bar height when visualising the parallel stack trace. Options are:" + height_options_str + ".")
	optional.add_argument('-t', '--transform', type=transformation_option, default=TransformationOption.NONE, help="Transformation applied to visualise the parallel stack trace. Options are:" + tf_options_str + ".")
	optional.add_argument('-o', '--output', required=False, help="Output filename (if set, the PFG will be saved as .PNG).")

	optional.add_argument('-l', '--logfile', default="log.txt", help="Filename to output log messages (defaults to log.txt).") 
	optional.add_argument('-d', '--log_level', type=log_level_option, default=LogLevelOption.INFO, help="Logging level. Options are:" + ll_options_str + ".")
	optional.add_argument('-w', '--width', required=False, help="X-axis start and end given as a string a_b.")
	optional.add_argument('--function_breakdown', action='store_true', required=False, help="Parse the tracefile and output the number of calls of each function.")
	optional.add_argument('--tee', action='store_true', help="Pipe logging messages to stdout as well as the log file.") 

	parser._action_groups.append(optional)

	args = parser.parse_args()

	return args.tracefile, args.transform, args.logfile, args.log_level, args.tee, args.height_option, args.output, args.width, args.function_breakdown

tracefile, transformation_type, logfile, log_level, tee_mode, height_option, output_file, xaxis_bounds_str, function_summary = parse_args()
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
	run_pfg(tracefile, transformation_type, height_option, output_file, bounds)
	logging.info("Done")

