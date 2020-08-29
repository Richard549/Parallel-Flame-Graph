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

			if text == "":
				self.hover_text.set_text("")
				ax.figure.canvas.draw()
			else:
				self.hover_text.set_text(text)
				self.hover_text.set_position((event.xdata, event.ydata))
				ax.figure.canvas.draw()

"""
	This class implements the text on each tree node
	The text position and size changes when the user refines the axis limits or resizes the window
"""
class PFGBarText:

	def __init__(self, ax, x1, x2, y1, y2, text):

		self.ax = ax

		# Calculate the per character width (in data-coords)
		test_box = ax.text(5, 5, 'a', fontsize='small')
		bbox = test_box.get_window_extent(renderer=ax.figure.canvas.get_renderer())
		bbox = Bbox(ax.transData.inverted().transform(bbox))
		self.per_char_data_width = bbox.width
		self.per_char_data_height = bbox.height
		test_box.remove()

		self.bar_x1 = x1
		self.bar_x2 = x2
		self.bar_y1 = y1
		self.bar_y2 = y2
		self.text = text

		xlims = ax.get_xlim()
		ylims = ax.get_ylim()

		# determine the initial position of the text
		x, y, processed_text = self.compute_text_position_info(ax, xlims, ylims, text)
		
		text_box = ax.text(x, y, processed_text, zorder=20, fontsize='small')

		self.cid = ax.callbacks.connect('resize_event', self)
		self.cid2 = ax.callbacks.connect('xlim_changed', self)
		self.cid3 = ax.callbacks.connect('ylim_changed', self)

		self.text_box = text_box

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
		test_box = self.ax.text(5, 5, 'a', fontsize='small')
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

		self.text_box.figure.canvas.draw()

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
			for parallelism, interval in enumerate(part.parallelism_intervals):
				optimal_cpu_cycles = ((parallelism+1.0)/maximum_parallelism) * interval
				total_cpu_cycles_lost += (interval - optimal_cpu_cycles)
			
			logging.debug("Node [%s] total_cpu_cycles_lost %s from parallelism intervals %s",
				",".join([part.name for part in node.node_partitions]),
				sizeof_fmt(total_cpu_cycles_lost),
				",".join([str(part.parallelism_intervals) for part in node.node_partitions])
				)

			part_width = width_to_interval_ratio * part.wallclock_duration
			part_height = ((total_cpu_cycles_lost / reference_height_value) * reference_area) / part_width
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
		cpus
		):

	total_wallclock_duration = max(node.wallclock_durations)
	total_node_width = width_to_interval_ratio * total_wallclock_duration

	heights = calculate_node_height(node, width_to_interval_ratio, height_option, reference_duration, reference_height, reference_height_value, len(cpus))
	total_node_height = max(heights)

	edgecolour = (0.0,0.0,0.0,1.0)

	for part_idx, part in enumerate(node.node_partitions):
		
		part_width = width_to_interval_ratio * part.wallclock_duration
		part_height = heights[part_idx]
	
		#facecolour = (1.0,1.0,1.0,1.0)
		facecolour = colours(colour_values[cpus.index(node.cpus[0])]) # what colour for multiple CPUs?

		logging.trace("Plotting %s on cpus %s with width %f at x=%f,y=%f", part.name, node.cpus, part_width, x0, y0)

		info_text = part.name + "\n"
		info_text += "Parent=[" + str(parent_name) + "]\n"
		wallclock_durations_by_cpu = node.get_per_cpu_wallclock_durations()
		for cpu, duration in wallclock_durations_by_cpu.items():
			info_text += str(cpu) + ": " + str(sizeof_fmt(duration)) + "\n"

		rect = patches.Rectangle(
			(x0,y0),
			part_width,
			part_height,
			linewidth=1,
			edgecolor=edgecolour,
			facecolor=facecolour,
			gid=info_text,
			zorder=10)

		ax.add_patch(rect)
		
		text = PFGBarText(ax, x0, x0+part_width, y0, y0+part_height, part.name)

		x0 += part_width

	return total_node_width, total_node_height

def plot_pfg_tree(tree,
		min_timestamp,
		max_timestamp,
		cpus,
		height_option,
		output_file=None
		):

	if len(tree.root_nodes) == 0:
		logging.warn("There are no root nodes in the tree.")
		return

	colours = cm.get_cmap("Reds")

	# TODO make better use of colours (and also, have a graph class that has the colour-mappings as a member
	maximum_colour = 0.3;
	minimum_colour = 0.05;
	colour_step = (maximum_colour-minimum_colour)/len(cpus)
	colour_values = [(i+1)*colour_step + minimum_colour for i in range(len(cpus))]
	random.shuffle(colour_values)

	fig = plt.figure()
	fig.set_size_inches(14, 8)
	ax = fig.add_subplot(111)

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
		for parallelism, interval in enumerate(tree.root_nodes[0].node_partitions[0].parallelism_intervals):
			optimal_cpu_cycles = ((parallelism+1.0)/len(cpus)) * interval
			total_cpu_cycles_lost += (interval - optimal_cpu_cycles)

		reference_height_value = total_cpu_cycles_lost

	# The root nodes are considered siblings
	sibling_node_sets = [tree.root_nodes]

	# Processes each set of siblings, using the alignments given by their parent
	while len(sibling_node_sets) > 0:

		next_sibling_node_sets = []

		for sibling_node_set in sibling_node_sets:

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
					parent_name = " and ".join(["("+part.name+")" for part in node.parent_node.node_partitions])

				# Plot the node
				width, height = plot_pfg_node(ax,
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
					cpus
					)

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

				# get child nodes for the next plotting pass
				next_sibling_node_sets.append(node.child_nodes)

		# finished plotting the current sets of siblings
		# set the next ones to plot
		sibling_node_sets = next_sibling_node_sets
		
	# Now display	
	ax.set_facecolor((0.9, 0.9, 0.9))

	ax.set_xlim([0,maximum_x])
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
		logging.info("Saving plot to %s.", output_file)
		fig.savefig(output_file, format="png", dpi=400, bbox_inches="tight")






