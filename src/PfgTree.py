from collections import defaultdict
from enum import Enum, auto
import logging
import numpy as np

from PfgUtil import debug_mode

class TransformationOption(Enum):
	NONE = auto()
	AGGREGATE_CALLS = auto()
	MERGE_CALLS_ACROSS_CPUS = auto()
	VERTICAL_STACK_CPU = auto()
	COLLAPSE_GROUPS = auto()

class ColourMode(Enum):
	BY_PARENT = auto()
	BY_CPU = auto()
	BY_PARALLELISM = auto()

def get_durations_for_entities(
		entities
		):

	wallclock_duration_by_cpu = defaultdict(int)
	parallelism_intervals = defaultdict(int)
	for entity in entities:

		for cpu, cpu_interval in entity.per_cpu_intervals.items():
			wallclock_duration_by_cpu[cpu] += cpu_interval

		for cpu, extra_cpu_interval in entity.extra_duration_by_cpu.items():
			wallclock_duration_by_cpu[cpu] += extra_cpu_interval

		# Consider top of stack or whole stack?
		# TODO make this a configurable option 
		#for parallelism, interval in entity.top_of_stack_parallelism_intervals.items():
		for parallelism, interval in entity.parallelism_intervals.items():
			parallelism_intervals[parallelism] += interval
	
	cpus = []
	wallclock_durations = []
	for cpu, duration in wallclock_duration_by_cpu.items():
		cpus.append(cpu)
		wallclock_durations.append(duration)

	if len(cpus) > 1:
		sorted_indices = np.argsort(wallclock_durations)[::-1]
		wallclock_durations = np.array(wallclock_durations)[sorted_indices].tolist()
		cpus = np.array(cpus)[sorted_indices].tolist()

	return cpus, wallclock_durations, parallelism_intervals

"""
	This is what will actually be plotted as a sub-bar
	These are built from the execution intervals of the node
"""
class PFGTreeNodeElement:

	def __init__(self, 
			name,
			wallclock_duration,
			parallelism_intervals,
			cpu_time=None
			):

		self.name = name
		self.wallclock_duration = wallclock_duration # this is truly wallclock duration, must be careful to choose the correct one when merging
		self.parallelism_intervals = parallelism_intervals # total CPU time spent in each parallelism state, this should sum to CPU time

		if cpu_time is None:
			self.cpu_time = wallclock_duration
		else:
			self.cpu_time = cpu_time

"""
	This is a specific execution interval
	Adding these up should give total CPU time of the node in each function
"""
class PFGTreeNodeExecutionInterval:

	def __init__(self, 
			name,
			cpu,
			duration,
			parallelism_intervals
			):

		self.cpu = cpu
		self.name = name
		self.duration = duration
		self.parallelism_intervals = parallelism_intervals # these are how long this CPU spent in each parallelism state, should sum to duration

class PFGTreeNode:

	def __init__(self,
			parent_node,
			name,
			cpu,
			wallclock_duration,
			parallelism_intervals,
			depth,
			alignment_node=None
			):

		# TODO a node should maintain an aggregate name from its partitions

		self.original_parent_node = parent_node
		self.original_depth = depth
		self.cpus = [cpu]
		self.wallclock_durations = [wallclock_duration] # currently this is only ever of length 1 (corresponding to the total wallclock duration of the node partitions)

		self.parent_node = parent_node
		if alignment_node is None:
			self.ancestor_alignment_node = parent_node
		else:
			self.ancestor_alignment_node = alignment_node

		self.child_nodes = []

		first_element = PFGTreeNodeElement(name, wallclock_duration, parallelism_intervals)
		self.node_partitions = [first_element]
		
		first_interval = PFGTreeNodeExecutionInterval(name, cpu, wallclock_duration, parallelism_intervals)
		self.execution_intervals = [first_interval]

		# for plotting
		self.start_x = None
		self.width = None
		self.start_y = None
		self.height = None

	"""
		If cpu_time is True, the wallclock times for different CPUs will be added so that the result is a node with CPU time
	"""
	def build_node_partitions_from_intervals(self, cpu_time=False):
		self.node_partitions.clear()

		wallclock_durations_by_group = defaultdict(int)
		cpu_time_by_group = defaultdict(int)
		parallelism_intervals_by_group = {}
		for execution_interval in self.execution_intervals:

			if cpu_time == True or execution_interval.cpu == self.cpus[0]:
				wallclock_durations_by_group[execution_interval.name] += execution_interval.duration
			
			if execution_interval.cpu not in self.cpus:
				self.cpus.append(execution_interval.cpu)

				for parallelism, interval in execution_interval.parallelism_intervals.items():
					parallelism_intervals_by_group[execution_interval.name][parallelism] += interval 
			else:
				parallelism_intervals_by_group[execution_interval.name] = execution_interval.parallelism_intervals

			cpu_time_by_group[execution_interval.name] += execution_interval.duration

		for group, duration in cpu_time_by_group.items():
			if group not in wallclock_durations_by_group:
				# this means it is an appended node part from a different CPU
				wallclock_durations_by_group[group] += duration

		total_wallclock_duration = 0
		for group, duration in wallclock_durations_by_group.items():
			cpu_time = cpu_time_by_group[group]
			parallelism_intervals_for_group = parallelism_intervals_by_group[group]

			part = PFGTreeNodeElement(group, duration, parallelism_intervals_for_group, cpu_time)

			total_wallclock_duration += duration
			self.node_partitions.append(part)

		self.wallclock_durations[0] = total_wallclock_duration
		
	# Return the durations over all intervals. For each CPU, we assume the intervals occur sequentially is wallclock.
	# TODO fix what happens in a recursive function, where one CPU calls a function many times - the wallclocks of these functions will be summed!
	def get_per_cpu_wallclock_durations(self):

		wallclock_durations_by_cpu = defaultdict(int)
		for execution_interval in self.execution_intervals:
			wallclock_durations_by_cpu[execution_interval.cpu] += execution_interval.duration

		return wallclock_durations_by_cpu
			

class PFGTree:

	def __init__(self, root_entities, aggregate_stack_frames_by_default=False):

		self.merged_siblings_by_cpu = False
		self.root_nodes = []

		self.stack_frames_aggregated = aggregate_stack_frames_by_default

		self.colour_mode = ColourMode.BY_CPU

		# As part of the constructor, build the tree
		self.gen_tree_process_entities_into_child_nodes(None, root_entities, self.root_nodes, 0, aggregate_stack_frames_by_default)

	"""
		The basic tree has each node representing an entity, i.e., a single stack
		frame on a single CPU. A child node is thus a single function call from the
		parent. Child nodes are therefore of the same CPU (unless there was a fork)

		Optionally, the tree can be built by default with the entitites aggregated
		such that each node represents all entities of one function called by one
		parent on one CPU

		Only upon transformation are nodes potentially composed of entities from
		multiple groups or CPUs, through merging
	"""
	def gen_tree_process_entities_into_child_nodes(self, parent_node, entities, parent_node_list, depth, aggregate_stack_frames_by_default):

		# split the entities by CPU
		entities_by_cpu = defaultdict(list)
		for entity in entities:
			entities_by_cpu[entity.cpu].append(entity)

		# for each CPU's entities, split them into their groups
		for cpu, entities_for_cpu in entities_by_cpu.items():

			entities_by_group = defaultdict(list)
			for entity in entities_for_cpu:
				entities_by_group[entity.group].append(entity)

			# for each group, create a node
			for group, entities_for_group in entities_by_group.items():

				if aggregate_stack_frames_by_default == False:

					for entity in entities_for_group:

						entity_as_list = [entity]
						cpus, wallclock_durations, parallelism_intervals = get_durations_for_entities(entity_as_list)

						node = PFGTreeNode(
							parent_node=parent_node,
							name=group,
							cpu=cpu,
							wallclock_duration=wallclock_durations[0],
							parallelism_intervals=parallelism_intervals,
							depth=depth
							)

						parent_node_list.append(node)

						if len(entity.child_entities) > 0:
							self.gen_tree_process_entities_into_child_nodes(node, entity.child_entities, node.child_nodes, depth+1, aggregate_stack_frames_by_default)		

				else:
					cpus, wallclock_durations, parallelism_intervals = get_durations_for_entities(entities_for_group)

					node = PFGTreeNode(
						parent_node=parent_node,
						name=group,
						cpu=cpu,
						wallclock_duration=wallclock_durations[0],
						parallelism_intervals=parallelism_intervals,
						depth=depth
						)

					parent_node_list.append(node)
		
					# collect all children of these entities, and convert them to nodes
					child_entities = []
					for entity in entities_for_group:
						for child_entity in entity.child_entities:
							child_entities.append(child_entity)
			
					if len(child_entities) > 0:
						self.gen_tree_process_entities_into_child_nodes(node, child_entities, node.child_nodes, depth+1, aggregate_stack_frames_by_default)		
	
	"""
		Transform the tree such that each node represents a set of entities,
		grouped by symbol and CPU, which were executed with a common parent node
		(set of function calls)

		Child nodes are then the set of entities (again grouped by symbol/CPU) that
		any entity of the parent node called.
	"""
	def transform_tree_aggregate_stack_frames(self):

		self.stack_frames_aggregated = True

		node_sets_queue = [self.root_nodes]

		while len(node_sets_queue) > 0:

			sibling_nodes = node_sets_queue[0]

			sibling_nodes_by_group_by_cpu = {}
			for node in sibling_nodes:
				if node.cpus[0] in sibling_nodes_by_group_by_cpu:
					sibling_nodes_by_group_by_cpu[node.cpus[0]][node.node_partitions[0].name].append(node)
				else:
					sibling_nodes_by_group_by_cpu[node.cpus[0]] = defaultdict(list)
					sibling_nodes_by_group_by_cpu[node.cpus[0]][node.node_partitions[0].name].append(node)

			for cpu, nodes_by_group in sibling_nodes_by_group_by_cpu.items():

				for group, nodes in nodes_by_group.items():

					# Because we are merging intra-CPU, there is no need to sort them to ensure we retain maximum wallclock
					merged_node = self.merge_node_set(nodes)
					node_sets_queue.append(merged_node.child_nodes)

			node_sets_queue.pop(0)

	"""
		This function goes through all siblings node sets, and collects nodes (of any group) by CPU, and merges each collection
		If a CPU executes 2 function calls sequentially, this merges them as one node with 2 node partitions

		The reason for this is to ensure that their sequential execution is not not broken when transforming the tree
		Where they become considered a single atomic node in the graph
	"""
	def transform_tree_merge_sibling_nodes_by_cpu(self):

		self.merged_siblings_by_cpu = True

		node_sets_queue = [self.root_nodes]

		while len(node_sets_queue) > 0:

			sibling_nodes = node_sets_queue[0]

			sibling_nodes_by_cpu = defaultdict(list)
			for node in sibling_nodes:
				sibling_nodes_by_cpu[node.cpus[0]].append(node)

			# Now merge all of the siblings into one node, regardless of group
			for cpu, nodes in sibling_nodes_by_cpu.items():

				# Because we are merging intra-CPU, there is no need to sort them to ensure we retain maximum wallclock
				merged_node = self.merge_node_set(nodes)
				node_sets_queue.append(merged_node.child_nodes)

			node_sets_queue.pop(0)

	def transform_tree_aggregate_siblings_across_cpus(self):

		self.colour_mode = ColourMode.BY_PARALLELISM

		if self.stack_frames_aggregated == False:
			self.transform_tree_aggregate_stack_frames()

		# Assuemes that each node currently only represents execution on one CPU
		node_sets_queue = [self.root_nodes]

		while len(node_sets_queue) > 0:

			sibling_nodes = node_sets_queue[0]

			sibling_nodes_by_group = defaultdict(list)
			for node in sibling_nodes:
				sibling_nodes_by_group[node.node_partitions[0].name].append(node)

			# Now merge all of the siblings into one node, regardless of cpu
			for group, nodes in sibling_nodes_by_group.items():

				# We do need to sort so that the CPU that took the longest to execute the function defines the wallclock
				merged_node = self.merge_node_set(nodes, True)
				node_sets_queue.append(merged_node.child_nodes)

			node_sets_queue.pop(0)
	
	"""
		Take a set of nodes and merge them into a single node with the union of all execution intervals
		If sort is False, the *first* node in the list is the base node that others will be merged into
			- meaning its wallclock will be used for node width when the tree is plot)
		If sort is True, the function will find the node with the maximum wallclock on its CPU to be the base node
		if cpu_time is True, the wallclocks of nodes merged across CPUs will be *added* so that the result is CPU time
	"""
	def merge_node_set(self, nodes_to_merge, sort=True, cpu_time=False):

		if len(nodes_to_merge) == 1:
			return nodes_to_merge[0]
		
		if len(nodes_to_merge) == 0:
			logging.error("Cannot merge a node set of length 0.")
			raise ValueError()

		# If sort, then have the nodes merged into the longest wallclock_duration node
		if sort:
			longest_wallclock_duration = 0
			longest_wallclock_duration_node_idx = 0
			for node_idx, node in enumerate(nodes_to_merge):
				if node.wallclock_durations[0] > longest_wallclock_duration:
					longest_wallclock_duration = node.wallclock_durations[0]
					longest_wallclock_duration_node_idx = node_idx

			new_nodes_to_merge = [nodes_to_merge[longest_wallclock_duration_node_idx]]
			new_nodes_to_merge.extend([n for i,n in enumerate(nodes_to_merge) if i != longest_wallclock_duration_node_idx])
			nodes_to_merge = new_nodes_to_merge

		merged_node = nodes_to_merge[0]

		if len(nodes_to_merge) > 1:
			while len(nodes_to_merge) > 1:
				node_to_merge = nodes_to_merge[1]

				node_to_merge.original_parent_node = merged_node.original_parent_node

				merged_node.execution_intervals.extend(node_to_merge.execution_intervals)
				merged_node.child_nodes.extend(node_to_merge.child_nodes)

				if node_to_merge.parent_node:
					node_to_merge.parent_node.child_nodes.remove(node_to_merge)
				else:
					self.root_nodes.remove(node_to_merge)

				for child_node in node_to_merge.child_nodes:
					child_node.parent_node = merged_node
					child_node.ancestor_alignment_node = merged_node
					child_node.original_parent_node = merged_node

				nodes_to_merge.remove(node_to_merge)

		merged_node.build_node_partitions_from_intervals(cpu_time)

		return merged_node

	"""
		This function aims to aggregate the tree nodes, such that each child node
		is accurately displayed within the wallclock of its parent, even if it is
		executed in parallel

		The nodes are merged into single nodes with multiple CPUs if possible, with
		those sequential function calls that do not fit side-by-side, instead
		stacked vertically

		To do this, we process outgoing-edge nodes (starting at the root):
		- We find the set of nodes with the longest wallclock on a single CPU
		- These nodes are merged into one base node, with multiple partitions
		- We then try to merge any of the other CPU's node sets into this base
			node.
		- The condition is, for the nodes of each CPU independently:
			-- Are all corresponding groups of the CPU < wallclock of the base groups
			-- AND can all of the non-corresponding groups be appended sequentially
				 with the base node (without extending past the parent)
		- If yes, then add the intervals of the CPU to the base node
		- If not, then change the parent node of these CPU's nodes to be the base
			node, to be processed as children of the base node (i.e. above the base
			node, rather than adjacent)
		- After all CPUs' nodes have been considered, process the outgoing-edges
			of the base node
	"""
	def transform_tree_collapse_groups(self):
		
		self.colour_mode = ColourMode.BY_PARENT

		# Assumes each node has one cpu and one group, and that node.cpus and node.wallclock_durations is correct
		parent_nodes_to_process = [node for node in self.root_nodes]

		while len(parent_nodes_to_process) > 0:
			parent_node = parent_nodes_to_process.pop(0)

			if len(parent_node.child_nodes) == 0:
				continue

			child_nodes_by_group_by_cpu = {}
			for child_node in parent_node.child_nodes:
				if child_node.cpus[0] not in child_nodes_by_group_by_cpu:
					child_nodes_by_group_by_cpu[child_node.cpus[0]] = defaultdict(list)
				child_nodes_by_group_by_cpu[child_node.cpus[0]][child_node.node_partitions[0].name].append(child_node)

			longest_wallclock_duration = 0
			longest_wallclock_duration_cpu = 0
			longest_wallclock_duration_duration_by_group = {} # i.e. for the set of nodes with the total longest duration, what is the duration of each group?
			for cpu, child_nodes_by_group in child_nodes_by_group_by_cpu.items():
				total_wallclock_duration = 0
				wallclock_duration_by_group = {}
				for group, child_nodes in child_nodes_by_group.items():
					wallclock_duration_for_group = 0
					for child_node in child_nodes:
						wallclock_duration_for_group += child_node.wallclock_durations[child_node.cpus.index(cpu)]

					wallclock_duration_by_group[group] = wallclock_duration_for_group
					total_wallclock_duration += wallclock_duration_for_group

				if total_wallclock_duration > longest_wallclock_duration:
					longest_wallclock_duration = total_wallclock_duration
					longest_wallclock_duration_cpu = cpu
					longest_wallclock_duration_duration_by_group = wallclock_duration_by_group

			# We now have the nodes (of multiple groups) with the longest wallclock duration on one CPU
			# This becomes our single merged base node
			sibling_nodes_to_merge = []
			for group, nodes in child_nodes_by_group_by_cpu[longest_wallclock_duration_cpu].items():
				sibling_nodes_to_merge.extend(nodes)

			# Can I avoid the above somewhat by having the merge function handling the sort?
			base_node = self.merge_node_set(sibling_nodes_to_merge, sort=False)
			
			logging.trace("Merging into the base node: %s", str(base_node.cpus) + ":" + str(base_node.wallclock_durations))

			nodes_to_merge = [] # (which ones can we merge into the base node)
			nodes_unable_to_be_merged = [] # (which ones will need to be pushed up to the next level)

			# now check the nodes of the other cpus and see if I can merge them into this base node
			for cpu, child_nodes_by_group in child_nodes_by_group_by_cpu.items():
				if cpu == longest_wallclock_duration_cpu:
					continue

				wallclock_duration_by_group = {}
				all_child_nodes = []
				for group, child_nodes_for_group in child_nodes_by_group.items():

					wallclock_duration_for_group = 0
					for child_node in child_nodes_for_group:
						wallclock_duration_for_group += child_node.wallclock_durations[child_node.cpus.index(cpu)]
						all_child_nodes.append(child_node)

					wallclock_duration_by_group[group] = wallclock_duration_for_group

				can_merge = True
				extra_duration = 0
				for group, duration in wallclock_duration_by_group.items():

					if group not in longest_wallclock_duration_duration_by_group:
						# This group isn't in the base node
						# Check if I can append it to the base node, as an adjacent partition
						extra_duration += duration
						if (longest_wallclock_duration + extra_duration) > parent_node.wallclock_durations[0]:
							can_merge = False
							break

					if duration > longest_wallclock_duration_duration_by_group[group]:
						can_merge = False
						break

				if can_merge:
					nodes_to_merge.extend(all_child_nodes)
				else:
					nodes_unable_to_be_merged.extend(all_child_nodes)

			# Should I try to merge the current siblings with the next level's siblings? This might push the bar way up vertically (if it keeps getting pushed)
			nodes_to_merge.insert(0, base_node)
			new_parent_node = self.merge_node_set(nodes_to_merge, sort=False)
			
			logging.trace("After merging, the resulting node is on cpus %s with wallclock durations %s", str(new_parent_node.cpus), str(new_parent_node.wallclock_durations))

			for node in nodes_unable_to_be_merged:
				if node.parent_node:
					node.parent_node.child_nodes.remove(node)
				node.parent_node = new_parent_node
				node.ancestor_alignment_node = new_parent_node

			# The nodes that we couldn't merge get added as outgoing-edges from the node we just merged, i.e. will get processed in the next loop
			new_parent_node.child_nodes.extend(nodes_unable_to_be_merged)

			parent_nodes_to_process.append(new_parent_node)

	"""
		This function transforms the tree so that children cannot extend past wallclock (by always stacking children vertically)

		Siblings on the same CPU are first merged so that sequential relationships are preserved
		Then, each CPU's nodes are displayed on its own level

		It is assumed that all nodes are associated to a single CPU (i.e. the tree is in its basic state before transformations)
	"""
	def transform_tree_stack_cpu_vertically(self):

		if self.merged_siblings_by_cpu is False:
			self.transform_tree_merge_sibling_nodes_by_cpu()

		for root_node in self.root_nodes:

			parent_node = root_node

			# Because the parent is a different node on a different CPU, the node that we should align to might be several edges up the tree
			# So track the alignment nodes of each CPU
			current_alignment_node_by_cpu = {}

			while parent_node is not None:
				
				if len(parent_node.child_nodes) <= 1:
					break

				cpu = parent_node.cpus[0]
				if cpu in current_alignment_node_by_cpu:
					parent_node.ancestor_alignment_node = current_alignment_node_by_cpu[cpu]

				current_alignment_node_by_cpu[cpu] = parent_node

				original_depths = []
				wallclock_durations = []
				cpus = []
				node_indices = []
				for child_node_idx, child_node in enumerate(parent_node.child_nodes):
					wallclock_durations.append(max(child_node.wallclock_durations))
					original_depths.append(child_node.original_depth)
					node_indices.append(child_node_idx)
					cpus.append(child_node.cpus[0])

				# Before displaying the next 'level' of child nodes, make sure we have first displayed all of the CPUs on the previous level
				same_original_depth = (original_depths.count(original_depths[0]) == len(original_depths))

				filtered_cpus = cpus
				filtered_node_indices = node_indices
				if same_original_depth == False:
					# filter the lists to only those nodes which have the minimum depths
					minimum_depth_indices = [idx for idx, d in enumerate(original_depths) if d == min(original_depths)]
					filtered_cpus = [cpu for idx, cpu in enumerate(cpus) if idx in minimum_depth_indices]
					filtered_node_indices = [idx for idx, cpu in enumerate(cpus) if idx in minimum_depth_indices]

				minimum_filtered_cpu = min(filtered_cpus)
				child_node_index = filtered_node_indices[filtered_cpus.index(minimum_filtered_cpu)]
				
				new_child_node = parent_node.child_nodes[child_node_index]
				other_child_nodes = [node for node_idx, node in enumerate(parent_node.child_nodes) if node_idx != child_node_index]

				parent_node.child_nodes.clear()
				parent_node.child_nodes.append(new_child_node)
				
				new_child_node.parent_node = parent_node
				new_child_node.child_nodes.extend(other_child_nodes)

				for other_child_node in other_child_nodes:
					other_child_node.parent_node = new_child_node

				parent_node = new_child_node

	def print_nodes(self, nodes, depth):

		for node_idx, node in enumerate(nodes):
			name = str(hex(id(node))) + ":" + " and ".join([part.name for part in node.node_partitions])
			cpus = "[" + ",".join([str(cpu) for cpu in node.cpus]) + "]"
			durations = "[" + ",".join([str(part.wallclock_duration) for part in node.node_partitions]) + "]"
			cpu_times = "[" + ",".join([str(part.cpu_time) for part in node.node_partitions]) + "]"
			parallelism_intervals = "[" + ",".join([str(list(part.parallelism_intervals.values())) for part in node.node_partitions]) + "]"
			parent_node_name = str(hex(id(node.original_parent_node))) + ":" + " and ".join([part.name for part in node.original_parent_node.node_partitions])

			logging.debug("Depth %d node %d/%d on CPUs %s for durations %s with parallelism intervals %s and cpu times %s: %s with parent node [%s]",
				depth,
				node_idx+1,
				len(nodes),
				cpus,
				durations,
				parallelism_intervals,
				cpu_times,
				name,
				parent_node_name)
			
			self.print_nodes(node.child_nodes, depth+1)

	def print_tree(self):

		for node_idx, root_node in enumerate(self.root_nodes):

			name = str(hex(id(root_node))) + ":" + " and ".join([part.name for part in root_node.node_partitions])
			cpus = "[" + ",".join([str(cpu) for cpu in root_node.cpus]) + "]"
			durations = "[" + ",".join([str(part.wallclock_duration) for part in root_node.node_partitions]) + "]"
			logging.debug("Descending root node %d/%d on CPUs %s for durations %s: %s", node_idx+1, len(self.root_nodes), cpus, durations, name)

			child_nodes = root_node.child_nodes
			self.print_nodes(child_nodes, 1)

	def assign_colour_indexes_to_nodes(self, nodes, num_cpus, mapped_nodes=None):

		if mapped_nodes is None:
			mapped_nodes = {}

		for node in nodes:

			if self.colour_mode == ColourMode.BY_PARENT:
				if node.original_parent_node is None:
					colour_identifier = "None"
				else:
					colour_identifier = str(hex(id(node.original_parent_node)))
			elif self.colour_mode == ColourMode.BY_CPU:
				colour_identifier = ",".join([str(cpu) for cpu in node.cpus])
			elif self.colour_mode == ColourMode.BY_PARALLELISM:
				# identifier is the number of cycles lost due to inefficiency, so the colours are ordered
				total_cpu_cycles_lost = 0
				for parallelism, interval in node.node_partitions[0].parallelism_intervals.items():
					optimal_cpu_cycles = ((parallelism)/num_cpus) * interval
					total_cpu_cycles_lost += (interval - optimal_cpu_cycles)
				colour_identifier = total_cpu_cycles_lost

				# Instead, it should be proportion of interval in parallelism, and must be independent of the interval!
				# Or how about, average parallelism, weighted by time spent in that parallelism!
				
				num = 0
				denom = 0
				for parallelism, interval in node.node_partitions[0].parallelism_intervals.items():
					num += parallelism*interval
					denom += interval

				# denom cannot be 0
				weighted_arithmetic_mean_parallelism = float(num) / denom

				colour_identifier = weighted_arithmetic_mean_parallelism

			else:
				logging.error("Colour mode not supported.")
				raise NotImplementedError()

			if colour_identifier not in mapped_nodes:
				# assign a new index
				next_index = len(mapped_nodes)
				mapped_nodes[colour_identifier] = next_index
			
			self.assign_colour_indexes_to_nodes(node.child_nodes, num_cpus, mapped_nodes)

		return mapped_nodes

