import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.transforms import Bbox

import random
import logging
from enum import Enum, auto

from PfgUtil import sizeof_fmt
from PfgTree import ColourMode

import datetime

PUSH_DIFFERENTIAL_EXTENSIONS_TO_END = True

class HeightDisplayOption(Enum):
	CONSTANT = auto()
	CPU_TIME = auto()
	PARALLELISM_INEFFICIENCY = auto()

""""
	This class handles defines the callback to display hover-over text about a tree node
"""
class PFGHoverText:
	def __init__(self, ax):
		self.hover_text = ax.text(1, 1, "", bbox=dict(facecolor='white', alpha=0.7), fontsize=6, zorder=100)
		self.cidmotion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_plot_hover)

	def on_plot_hover(self, event):
		if event.inaxes is not None:
			ax = event.inaxes

			text = ""
			for obj in ax.findobj():
				# Searching which data member corresponds to current mouse position
				if obj.contains(event)[0]:
					if None not in (event.xdata, event.ydata):
						if obj.get_gid() is not None:
							text += obj.get_gid() + "\n"

			if ax.highlighted_rectangle is not None:
				ax.highlighted_rectangle.remove()
				ax.highlighted_rectangle = None

			if text == "":
				self.hover_text.set_text("")
				ax.figure.canvas.draw()
			else:
				self.hover_text.set_text(text)
				self.hover_text.set_position((event.xdata, event.ydata))

				# also highlight the rectangle of the parent!
				# dodgy parsing for now
				parent_identifier = text.split("Parent=[")[1].split("\n")[0].split(":")[-1].split("]")[0]

				if parent_identifier in ax.rectangles:
					rectangle_coords = ax.rectangles[parent_identifier]
					# draw a highlighted rectangle
					rect = patches.Rectangle(
						(rectangle_coords[0],rectangle_coords[2]),
						rectangle_coords[1],
						rectangle_coords[3],
						linewidth=4,
						edgecolor=(0.0,0.0,0.0,1.0),
						facecolor=(0.0,0.0,0.0,0.0),
						gid=None,
						zorder=100,
						clip_on=False)
					ax.add_patch(rect)

					ax.highlighted_rectangle = rect

				ax.figure.canvas.draw()

"""
	This class implements the text on each tree node
	The text position and size changes when the user refines the axis limits or resizes the window
"""
class PFGBarText:

	def __init__(self, ax, x1, x2, y1, y2, text, alpha=1.0):

		# The resize event will be called when the window opens,
		# so we can wait for the call function to run for the actual plotting of the text
		ax.bar_texts.append(self)
		self.ax = ax
		self.bar_x1 = x1
		self.bar_x2 = x2
		self.bar_y1 = y1
		self.bar_y2 = y2
		self.text = text
		self.alpha=alpha

		text_box = ax.text(-20, -20, "", zorder=20, fontsize='small', alpha=alpha)
		self.text_box = text_box

		ax.figure.canvas.mpl_connect('resize_event', self) # when the window is resized
		ax.callbacks.connect('xlim_changed',self)
		ax.callbacks.connect('ylim_changed',self)

	def get_per_char_data_width(self):
		return self.per_char_data_width

	def compute_text_position_info(self, ax, xlims, ylims, text):

		text_position_x = 0
		text_position_y = 0
		processed_text = ""

		# check if the bar actually displayed, or if only part of the bar is displayed

		visible_bar_x1 = None
		visible_bar_x2 = None
		visible_bar_y1 = None
		visible_bar_y2 = None

		# check x axis:

		if self.bar_x1 >= xlims[0] and self.bar_x1 <= xlims[1]:
			# the x start is inside
			visible_bar_x1 = self.bar_x1

		if self.bar_x2 <= xlims[1] and self.bar_x2 >= xlims[0]:
			# the x end is inside
			visible_bar_x2 = self.bar_x2

		if self.bar_x1 >= xlims[0] and self.bar_x2 <= xlims[1]:
			visible_bar_x1 = self.bar_x1
			visible_bar_x2 = self.bar_x2

		if self.bar_x1 <= xlims[0] and self.bar_x2 >= xlims[1]:
			visible_bar_x1 = xlims[0]
			visible_bar_x2 = xlims[1]

		if visible_bar_x1 is not None and visible_bar_x2 is None:
			# This means it is truncated on the right hand side
			visible_bar_x2 = xlims[1]
		
		if visible_bar_x2 is not None and visible_bar_x1 is None:
			# This means it is truncated on the left hand side
			visible_bar_x1 = xlims[0]
		
		# check y axis:

		if self.bar_y1 >= ylims[0] and self.bar_y1 <= ylims[1]:
			# the y start is inside
			visible_bar_y1 = self.bar_y1

		if self.bar_y2 <= ylims[1] and self.bar_y2 >= ylims[0]:
			# the y end is inside
			visible_bar_y2 = self.bar_y2
		
		if self.bar_y1 >= ylims[0] and self.bar_y2 <= ylims[1]:
			visible_bar_y1 = self.bar_y1
			visible_bar_y2 = self.bar_y2

		if self.bar_y1 <= ylims[0] and self.bar_y2 >= ylims[1]:
			visible_bar_y1 = ylims[0]
			visible_bar_y2 = ylims[1]

		if visible_bar_y1 is not None and visible_bar_y2 is None:
			# This means it is truncated on the top
			visible_bar_y2 = ylims[1]
		
		if visible_bar_y2 is not None and visible_bar_y1 is None:
			# This means it is truncated on the bottom
			visible_bar_y1 = ylims[0]

		if visible_bar_x1 is None or visible_bar_y1 is None:
			# then the bar is not visible
			return text_position_y, text_position_y, processed_text

		# How many characters can the visible bar handle?

		width = float(visible_bar_x2) - visible_bar_x1
		height = float(visible_bar_y2) - visible_bar_y1

		num_characters_available = (width / self.per_char_data_width)
		padding_characters = 2
		num_characters_support = num_characters_available - padding_characters

		#logging.debug("Num characters support: %s", num_characters_support)
		#logging.debug("Height: %s", height)
		#logging.debug("Per char data width: %s", self.per_char_data_width)

		# Can we support the number of characters required?
		# Setting a minimum of 2 actual symbol characters
		if num_characters_support < 2.0 or height < self.per_char_data_height:
			return text_position_y, text_position_y, processed_text

		# Check if we need to truncate the text
		if num_characters_support < len(text):
			processed_text = text[:int(num_characters_support-2)] + ".."
		else:
			processed_text = text[:int(num_characters_support)]
		
		text_position_y = visible_bar_y1 + (height/2.0) - (0.5*self.per_char_data_height)
		
		free_character_spaces = float(num_characters_available) - len(processed_text)
		text_position_x = visible_bar_x1 + (((free_character_spaces+1)/2.0))*self.per_char_data_width

		return text_position_x, text_position_y, processed_text

	def __call__(self, event):

		# function is called on each scale change or window resized

		# recalculate the width of each character in data coordinates
		test_box = self.ax.text(5, 5, 'a', fontsize='small',alpha=self.alpha)
		bbox = test_box.get_window_extent(renderer=self.ax.figure.canvas.get_renderer())
		bbox = Bbox(self.ax.transData.inverted().transform(bbox))
		self.per_char_data_width = bbox.width
		self.per_char_data_height = bbox.height
		test_box.remove()

		xlim = self.ax.get_xlim()
		ylim = self.ax.get_ylim()

		x, y, processed_text = self.compute_text_position_info(self.ax, xlim, ylim, self.text)
		if processed_text == "":
			self.text_box.set_text(processed_text)
			return

		self.text_box.set_position((x, y))
		self.text_box.set_text(processed_text)

def onpick(event):
	art = event.artist
	if event.mouseevent.inaxes is not None:
		ax = event.mouseevent.inaxes
		if art in ax.counters_line_map:
			selected_counter = ax.counters_line_map[art]

			if ax.selected_counter == selected_counter:
				return
			
			for handle_idx, handle in enumerate(ax.legend.legendHandles):
				if handle_idx == selected_counter:
					handle.set_color('red')
				else:
					handle.set_color('none')

			for rect in ax.plotted_bars:
				rect.set_facecolor(rect.colour_options[selected_counter])

			ax.selected_counter = selected_counter
			ax.figure.canvas.draw()


"""
	If width is wallclock, height may be relative to CPU-time, or parallel inefficiency, etc
"""
def calculate_node_height(
		node,
		width_to_interval_ratio,
		height_option,
		reference_duration,
		reference_height,
		reference_height_value,
		maximum_parallelism=None
		):

	heights = []

	reference_width = width_to_interval_ratio * reference_duration
	reference_area = reference_width * reference_height

	for part in node.node_partitions:

		if height_option == HeightDisplayOption.CONSTANT:

			heights.append(reference_height)

		elif height_option == HeightDisplayOption.CPU_TIME:

			part_width = width_to_interval_ratio * part.wallclock_duration
			part_height = ((part.cpu_time / reference_height_value) * reference_area) / part_width
			heights.append(part_height)

		elif height_option == HeightDisplayOption.PARALLELISM_INEFFICIENCY:

			total_cpu_cycles_lost = 0
			for parallelism, interval in part.parallelism_intervals.items():
				optimal_cpu_cycles = ((parallelism)/maximum_parallelism) * interval
				total_cpu_cycles_lost += (interval - optimal_cpu_cycles)
			
			logging.debug("Node [%s] total_cpu_cycles_lost %s from parallelism intervals %s",
				",".join([part.name for part in node.node_partitions]),
				sizeof_fmt(total_cpu_cycles_lost),
				",".join([str(list(str(part.parallelism_intervals.values()))) for part in node.node_partitions])
				)

			part_width = width_to_interval_ratio * part.wallclock_duration
			part_height = ((total_cpu_cycles_lost / reference_height_value) * reference_area) / part_width

			if part_height == 0.0:
				part_height = 0.0001 * reference_height # just so that we can zoom in if we want to see what the thing is

			heights.append(part_height)

		else:
			logging.error("Height display option %s not supported.", height_option)
			raise NotImplementedError()

	return heights

def plot_pfg_node(
		ax,
		node,
		x0,
		y0,
		height_option,
		reference_duration,
		reference_height_value,
		reference_height,
		width_to_interval_ratio,
		parent_name,
		colours,
		colour_values,
		cpus,
		node_colour_mapping,
		colour_mode,
		counters
		):

	total_wallclock_duration = max(node.wallclock_durations)
	total_node_width = width_to_interval_ratio * total_wallclock_duration

	heights = calculate_node_height(node, width_to_interval_ratio, height_option, reference_duration, reference_height, reference_height_value, len(cpus))
	total_node_height = max(heights)

	edgecolour = (0.0,0.0,0.0,1.0)

	num = 0
	denom = 0
	for parallelism, interval in node.node_partitions[0].parallelism_intervals.items():
		num += parallelism*interval
		denom += interval

	# denom cannot be 0
	weighted_arithmetic_mean_parallelism = float(num) / denom

	# Key to use to determine the colour of the rectangle
	colour_identifier = "None"
	if colour_mode == ColourMode.BY_PARENT:
		if node.original_parent_node is not None:
			colour_identifier = str(hex(id(node.original_parent_node)))
	elif colour_mode == ColourMode.BY_CPU:
		colour_identifier = ",".join([str(cpu) for cpu in node.cpus])
	elif colour_mode == ColourMode.BY_PARALLELISM:
		colour_identifier = weighted_arithmetic_mean_parallelism
	elif colour_mode == ColourMode.BY_EVENT_VALUE:
		# Default event is the first one
		colour_identifier = node.node_partitions[0].per_event_values[0]
	else:
		logging.error("Colour mode not supported.")
		raise NotImplementedError()
	
	# To enable hover-over to find the correct parent, record the unique ids for the rectangles
	node_identifier = str(hex(id(node)))
	if node.original_parent_node is None:
		parent_identifier = "None"
	else:
		parent_identifier = str(hex(id(node.original_parent_node)))

	total_realised_wallclock_duration = 0.0
	original_x0 = x0
	extended_rectangles = []

	for part_idx, part in enumerate(node.node_partitions):

		if part_idx == 1:
			# TODO review partitioning design
			logging.error("Multiple partitions is not yet implemented properly with the latest code")
			exit(1)
		
		realised_part_duration = part.wallclock_duration
		if node.differential_interval is not None and node.differential_interval < 0:
			realised_part_duration += node.differential_interval

		total_realised_wallclock_duration += realised_part_duration
		
		part_width = width_to_interval_ratio * realised_part_duration
		part_height = heights[part_idx]
	
		#facecolour = (1.0,1.0,1.0,1.0)

		colour_options = []

		if colour_mode == ColourMode.BY_PARALLELISM:
			minimum_colour = colour_values[0]
			maximum_colour = colour_values[1]
			value = minimum_colour + colour_identifier

			# Choose between darker red = poor parallelism (first line) or darker_red = high parallelism (second line)
			#colour_value = maximum_colour - ((float(colour_identifier) / len(cpus)) * (maximum_colour-minimum_colour))
			colour_value = minimum_colour + ((float(colour_identifier) / len(cpus)) * (maximum_colour-minimum_colour))
			facecolour = colours(colour_value)

			colour_options.append(facecolour)

		elif colour_mode == ColourMode.BY_EVENT_VALUE:
			
			for event_idx in list(counters.keys()):

				minimum_colour = colour_values[0]
				maximum_colour = colour_values[1]

				colour_identifier = part.per_event_values[event_idx]

				min_colour_identifier = min(list(node_colour_mapping[event_idx].keys()))
				max_colour_identifier = max(list(node_colour_mapping[event_idx].keys()))
				if max_colour_identifier == min_colour_identifier:
					colour_value = minimum_colour
				else:
					colour_value = minimum_colour + ((float(colour_identifier) / (max_colour_identifier-min_colour_identifier)) * (maximum_colour-minimum_colour))

				'''
				logging.trace("Minimum value %s, maximum value %s, and this node (%s) value %s, gives colour_value %s",
					sizeof_fmt(min_colour_identifier,suffix=""),
					sizeof_fmt(max_colour_identifier,suffix=""),
					part.name,
					sizeof_fmt(colour_identifier,suffix=""),
					colour_value)
				'''

				facecolour = colours(colour_value)

				colour_options.append(facecolour)

			# Integrate the PARALLELISM metric as an 'event'
			# TODO integrate this properly, to be a single approach for all metrics
			minimum_colour = colour_values[0]
			maximum_colour = colour_values[1]

			colour_identifier = weighted_arithmetic_mean_parallelism

			#min_colour_identifier = min(list(node_colour_mapping[-1].keys()))
			min_colour_identifier = 1.0
			max_colour_identifier = float(len(cpus))

			colour_value = None
			if max_colour_identifier == min_colour_identifier:
				colour_value = maximum_colour
			else:
				colour_value = minimum_colour + ((float(colour_identifier) / (max_colour_identifier-min_colour_identifier)) * (maximum_colour-minimum_colour))

			'''
			logging.trace("Minimum value %s, maximum value %s, and this node (%s) value %s, gives colour_value %s",
				sizeof_fmt(min_colour_identifier,suffix=""),
				sizeof_fmt(max_colour_identifier,suffix=""),
				part.name,
				sizeof_fmt(colour_identifier,suffix=""),
				colour_value)
			'''

			facecolour = colours(colour_value)
			colour_options.append(facecolour)

		else:

			facecolour = colours(colour_values[node_colour_mapping[colour_identifier]])
			colour_options.append(facecolour)

		logging.trace("Plotting %s on cpus %s with width %f at x=%f,y=%f", part.name, node.cpus, part_width, x0, y0)

		# If there is a differential, then I need to split the rectangle into two
		if node.differential_interval is not None:
			if node.differential_interval < 0:
				# I need to divide the rectangle into an inner fast-interval and a low-alpha reference remainder

				#inner_part_width = width_to_interval_ratio * (part.wallclock_duration + node.differential_interval)
				inner_rect = patches.Rectangle(
					(x0,y0),
					part_width,
					part_height,
					linewidth=1,
					edgecolor=edgecolour,
					facecolor=colour_options[0],
					gid=None,
					zorder=10,
					picker=True)
				
				inner_rect.colour_options = colour_options
				ax.add_patch(inner_rect)
				ax.plotted_bars.append(inner_rect)
				text = PFGBarText(ax, x0, x0+part_width, y0, y0+part_height, part.name)
			
				# TODO The extended part should be moved to the end of the siblings!
				extended_part_width = width_to_interval_ratio * abs(node.differential_interval)
				extended_edgecolour = list(edgecolour)
				extended_facecolour = list(colour_options[0])
				extended_edgecolour[3] = 0.05
				extended_facecolour[3] = 0.05
				extended_edgecolour = tuple(extended_edgecolour)
				extended_facecolour = tuple(extended_facecolour)

				if PUSH_DIFFERENTIAL_EXTENSIONS_TO_END == False:
					extended_rect = patches.Rectangle(
						(x0+part_width,y0),
						extended_part_width,
						part_height,
						linewidth=1,
						edgecolor=extended_edgecolour,
						facecolor=extended_facecolour,
						gid=None,
						zorder=10,
						picker=True)

					extended_rect.colour_options = colour_options
					ax.add_patch(extended_rect)
					ax.plotted_bars.append(extended_rect)
					text = PFGBarText(ax, x0+part_width, x0+part_width+extended_part_width, y0, y0+part_height, part.name, 0.05)

					part_width += extended_part_width
					total_realised_wallclock_duration += abs(node.differential_interval)

				else:
					extended_rect_info = [
						extended_part_width,
						part_height,
						extended_edgecolour,
						extended_facecolour,
						part.name,
						colour_options
					]
					extended_rectangles.append(extended_rect_info)

			else:
				# it is longer in the target so I need to extend!
				# TODO TODO TODO
				rect = patches.Rectangle(
					(x0,y0),
					part_width,
					part_height,
					linewidth=1,
					edgecolor=edgecolour,
					facecolor=colour_options[0],
					#facecolor=colours(colour_values[0]) if node.differential_interval < 0 else colours(colour_values[1]),
					gid=None,
					zorder=10,
					picker=True)

				rect.colour_options = colour_options

				ax.add_patch(rect)
				ax.plotted_bars.append(rect)

				text = PFGBarText(ax, x0, x0+part_width, y0, y0+part_height, part.name)

		else:
			rect = patches.Rectangle(
				(x0,y0),
				part_width,
				part_height,
				linewidth=1,
				edgecolor=edgecolour,
				facecolor=colour_options[0],
				#facecolor=colours(colour_values[0]) if node.differential_interval < 0 else colours(colour_values[1]),
				gid=None,
				zorder=10,
				picker=True)

			rect.colour_options = colour_options

			ax.add_patch(rect)
			ax.plotted_bars.append(rect)
		
			text = PFGBarText(ax, x0, x0+part_width, y0, y0+part_height, part.name)

		x0 += part_width
	
	total_node_width = width_to_interval_ratio * total_realised_wallclock_duration
	
	info_text = "Total wallclock duration: " + sizeof_fmt(total_wallclock_duration) + "\n"
	info_text += "Total CPU time: " + sizeof_fmt(sum([part.cpu_time for part in node.node_partitions])) + "\n"
	info_text += "Avg parallelism: " + str(weighted_arithmetic_mean_parallelism) + "\n"
	info_text += "".join([part.name + ": " + sizeof_fmt(part.wallclock_duration) + "\n" for part in node.node_partitions]) + "\n"
	info_text += "Parent=[" + str(parent_name) + ":" + str(parent_identifier) + "]\n\n"
	for event_idx, event_name in counters.items():
		info_text += event_name + ": " + sizeof_fmt(node.node_partitions[0].per_event_values[event_idx], suffix="") + "\n"
		if event_idx + 1 == len(counters):
			info_text += "\n"
	wallclock_durations_by_cpu = node.get_per_cpu_wallclock_durations()
	for cpu, duration in wallclock_durations_by_cpu.items():
		info_text += str(cpu) + ": " + str(sizeof_fmt(duration)) + "\n"

	# Invisible rectangle just for the hover-over text
	rect = patches.Rectangle(
		(original_x0,y0),
		total_node_width,
		total_node_height,
		linewidth=0,
		edgecolor=(0.0,0.0,0.0,0.0),
		facecolor=(0.0,0.0,0.0,0.0),
		gid=info_text,
		zorder=0)
	
	ax.add_patch(rect)

	if ax.rectangles is None:
		ax.rectangles = {node_identifier: [original_x0, total_node_width, y0, total_node_height]}
	else:
		ax.rectangles[node_identifier] = [original_x0, total_node_width, y0, total_node_height]

	return total_node_width, total_node_height, extended_rectangles

def plot_pfg_tree(tree,
		min_timestamp,
		max_timestamp,
		cpus,
		height_option,
		output_file=None,
		x_bounds=None,
		counters={}
		):

	if len(tree.root_nodes) == 0:
		logging.warn("There are no root nodes in the tree.")
		return

	colours = cm.get_cmap("Reds")

	# There should be a colour for each 'original parent'
	node_colour_mapping = tree.node_colour_mapping

	maximum_colour = 0.75;
	minimum_colour = 0.00;
	colour_step = (maximum_colour-minimum_colour)/len(node_colour_mapping)
	colour_values = [(i+1)*colour_step + minimum_colour for i in range(len(node_colour_mapping))]

	if tree.colour_mode == ColourMode.BY_PARALLELISM:
		colour_values = [minimum_colour, maximum_colour]
	elif tree.colour_mode == ColourMode.BY_EVENT_VALUE:
		colour_values = [minimum_colour, maximum_colour]
	else:
		random.shuffle(colour_values)

	fig = plt.figure()
	fig.set_size_inches(14, 8)
	ax = fig.add_subplot(111, picker=True)

	ax.rectangles = {}
	ax.plotted_bars = []
	ax.highlighted_rectangle = None
	ax.bar_texts = []
	
	ax.counters_line_map = {}

	default_counter_idx = 0
	if len(counters) > 0:

		counter_patches = []
		# make a list of counters using the legend, that can be clicked
		for counter_idx, counter_name in counters.items():
			color = "Red" if counter_idx == default_counter_idx else "none"
			counter_patch = patches.Patch(color=color, label=counter_name, picker=True)
			counter_patches.append(counter_patch)

	color = "Red" if len(counters) == 0 else "none"
	counter_patch = patches.Patch(color=color, label="PARALLELISM", picker=True)
	counter_patches.append(counter_patch)

	leg = plt.legend(handles=counter_patches, ncol=2)

	for leg_idx, leg_label in enumerate(leg.get_texts()):
		leg_label.set_picker(5)
		ax.counters_line_map[leg_label] = leg_idx

	ax.legend = leg
	ax.selected_counter = default_counter_idx
	fig.canvas.mpl_connect('pick_event', onpick)

	top_level_width = 100.0

	maximum_x = 0
	maximum_y = 0

	# Calculate the wallclock interval to width ratio
	total_top_level_wallclock_duration = 0.0
	for node in tree.root_nodes:
		max_wallclock_duration = max(node.wallclock_durations)
		total_top_level_wallclock_duration += max_wallclock_duration
	width_to_interval_ratio = top_level_width / total_top_level_wallclock_duration 

	# The reference duration (i.e. width), actual height, and height-value, allow other nodes to calculate their actual heights using their height values
	reference_height = 10.0
	reference_height_value = tree.root_nodes[0].node_partitions[0].cpu_time
	reference_duration = tree.root_nodes[0].wallclock_durations[0]

	if height_option == HeightDisplayOption.PARALLELISM_INEFFICIENCY:
		total_cpu_cycles_lost = 0
		for parallelism, interval in tree.root_nodes[0].node_partitions[0].parallelism_intervals.items():
			optimal_cpu_cycles = ((parallelism)/len(cpus)) * interval
			total_cpu_cycles_lost += (interval - optimal_cpu_cycles)

		reference_height_value = total_cpu_cycles_lost

	# The root nodes are considered siblings
	sibling_node_sets = [tree.root_nodes]


	# Each sibling set may produce some extension rectangles
	# The extensions should be plotted starting from the end of the final true rectangle, at each level
	final_x_position_by_y = {}
	extension_rectangles_by_y = {}
	
	# Processes each set of siblings, using the alignments given by their parent
	while len(sibling_node_sets) > 0:

		next_sibling_node_sets = []

		for sibling_node_set in sibling_node_sets:

			extended_rectangles = []
			siblings_y_value = None

			accumulated_sibling_width = 0.0
			for sibling_idx, node in enumerate(sibling_node_set):
				
				# What is my x position?
				base_x_position = 0.0
				if node.ancestor_alignment_node is not None:
					base_x_position = node.ancestor_alignment_node.start_x

				x_position = base_x_position + accumulated_sibling_width

				# What is my y position?
				y_position = 0.0
				parent_name = "None"
				if node.parent_node is not None:
					y_position = node.parent_node.start_y + node.parent_node.height

				if siblings_y_value is None:
					siblings_y_value = y_position

				if node.original_parent_node is not None:
					parent_name = " and ".join(["("+part.name+")" for part in node.original_parent_node.node_partitions])

				# Plot the node
				width, height, node_extended_rectangles = plot_pfg_node(ax,
					node,
					x_position,
					y_position,
					height_option,
					reference_duration,
					reference_height_value,
					reference_height,
					width_to_interval_ratio,
					parent_name,
					colours,
					colour_values,
					cpus,
					node_colour_mapping,
					tree.colour_mode,
					counters
					)
				logging.trace("Plotting a realised rectangle (%s with parent %s) at y=%s and x=%s to %s",
					parent_name, node.node_partitions[0].name,
					y_position, x_position, x_position+width)

				extended_rectangles.extend(node_extended_rectangles)

				# write the positions of this node for my children/siblings
				node.start_x = x_position
				node.start_y = y_position
				node.width = width
				node.height = height
				accumulated_sibling_width += width

				if x_position + width > maximum_x:
					maximum_x = x_position + width
				if y_position + height > maximum_y:
					maximum_y = y_position + height

				if y_position in final_x_position_by_y:
					if final_x_position_by_y[y_position] < (x_position + width):
						final_x_position_by_y[y_position] = x_position + width
				else:
					final_x_position_by_y[y_position] = x_position + width

				# get child nodes for the next plotting pass
				next_sibling_node_sets.append(node.child_nodes)

			if len(extended_rectangles) > 0:
				if siblings_y_value in extension_rectangles_by_y:
					extension_rectangles_by_y[siblings_y_value].extend(extended_rectangles)
				else:
					extension_rectangles_by_y[siblings_y_value] = extended_rectangles

		# finished plotting the current sets of siblings
		# set the next ones to plot
		sibling_node_sets = next_sibling_node_sets
		
		# Now plot the extensions at the end of the siblings
		for y_value, extension_rectangles in extension_rectangles_by_y.items():

			x_value = final_x_position_by_y[y_value]
		
			accumulated_extension_width = 0.0
			for extended_rect in extension_rectangles:
			
				extended_x_start = x_value + accumulated_extension_width
				accumulated_extension_width += extended_rect[0]

				logging.trace("Plotting an extension rectangle (%s with parent %s) at y=%s and x=%s to %s",
					parent_name, node.node_partitions[0].name,
					y_value, extended_x_start, extended_x_start+extended_rect[0])

				extended_rect_patch = patches.Rectangle(
					(extended_x_start,y_value),
					extended_rect[0],
					extended_rect[1],
					linewidth=1,
					edgecolor=extended_rect[2],
					facecolor=extended_rect[3],
					gid=None,
					zorder=10,
					picker=True)
				
				extended_rect_patch.colour_options = extended_rect[5]
				ax.add_patch(extended_rect_patch)
				ax.plotted_bars.append(extended_rect_patch)
				text = PFGBarText(ax, extended_x_start, extended_x_start+extended_rect[0], y_value, y_value+extended_rect[1], extended_rect[4], 0.05)

	# Now display	
	ax.set_facecolor((0.9, 0.9, 0.9))

	if x_bounds is None:
		ax.set_xlim([0,maximum_x])
	else:
		maximum_x_cycles = x_bounds[1]
		coefficient = maximum_x_cycles/max_timestamp

		ax.set_xlim([0, coefficient*maximum_x])

	ax.set_ylim([0,maximum_y*1.5])

	# Create the hover-over
	hover_text = PFGHoverText(ax)

	wallclock_duration = sizeof_fmt(max_timestamp - min_timestamp)
	ax.set_title("OpenMP Parallel FlameGraph")

	ax.set_yticks([])
	ax.set_xticks([0,top_level_width])
	ax.set_xticklabels((str(sizeof_fmt(0)), str(sizeof_fmt(max_timestamp - min_timestamp))))

	if output_file is None:
		logging.info("Displaying interactive plot.")
		plt.show()
	else:

		# If we aren't displaying a window, then the resize event doesn't trigger on each of the bar texts
		# Therefore manually trigger the calls to make sure they are positioned correctly before saving
		for bar_text in ax.bar_texts:
			bar_text.__call__(None)

		logging.info("Saving plot to %s.", output_file)
		fig.savefig(output_file, format="png", dpi=400, bbox_inches="tight")


