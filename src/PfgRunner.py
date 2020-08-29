from argparse import ArgumentParser, ArgumentTypeError
from collections import defaultdict
import logging

from PfgTree import PFGTree, TransformationOption
from PfgTracefile import process_events
from PfgPlotting import plot_pfg_tree, HeightDisplayOption
from PfgUtil import initialise_logging, debug_mode, LogLevelOption

def run_pfg(
		tracefile,
		transformation,
		height_display_option
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

	if debug_mode():
		logging.debug("Printing tree:")	
		tree.print_tree()

	transformed = True
	if transformation == TransformationOption.NONE:
		transformed = False
		logging.info("Applying no tree transformation.")	
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
		height_display_option)

	# TODO save to file (SVG?)

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
	optional.add_argument('-l', '--logfile', default="log.txt", help="Filename to output log messages (defaults to log.txt).") 
	optional.add_argument('-d', '--log_level', type=log_level_option, default=LogLevelOption.INFO, help="Logging level. Options are:" + ll_options_str + ".")
	optional.add_argument('--tee', action='store_true', help="Boolean: include debug_mode and tracing log messages (default False).") 

	parser._action_groups.append(optional)

	args = parser.parse_args()

	return args.tracefile, args.transform, args.logfile, args.log_level, args.tee, args.height_option

tracefile, transformation_type, logfile, log_level, tee_mode, height_option = parse_args()
initialise_logging(logfile, log_level, tee_mode)

logging.info("Running Parallel Flame Graph for %s", tracefile)
logging.info("Selected transformation option was %s.", transformation_type.name)
logging.info("Selected height display option was %s.", height_option.name)

run_pfg(tracefile, transformation_type, height_option)

logging.info("Done")

