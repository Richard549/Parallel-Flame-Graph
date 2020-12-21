from collections import defaultdict
from enum import Enum, auto
import logging
import numpy as np

from PfgUtil import debug_mode, sizeof_fmt

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
	BY_EVENT_VALUE = auto() 

def get_aggregate_event_counts_for_entities(
		entities
		):

	per_event_values_across_cpus = defaultdict(list)

	for entity in entities:
		for cpu, value_per_event in entity.per_cpu_event_values.items():
			for event_idx, value in value_per_event.items():
				per_event_values_across_cpus[event_idx].append(value)

	summed_per_event_values = {}
	average_per_event_values = {}

	for event_idx, values in per_event_values_across_cpus.items():
		summed_per_event_values[event_idx] = sum(values)
		average_per_event_values[event_idx] = np.mean(values)

	return summed_per_event_values, average_per_event_values

def get_durations_for_entities(
		entities
		):

	wallclock_duration_by_cpu = defaultdict(int)
	active_wallclock_duration_by_cpu = defaultdict(int)
	parallelism_intervals = defaultdict(int)
	for entity in entities:

		for cpu, cpu_interval in entity.per_cpu_intervals.items():
			wallclock_duration_by_cpu[cpu] += cpu_interval
		for cpu, cpu_interval in entity.per_cpu_active_intervals.items():
			active_wallclock_duration_by_cpu[cpu] += cpu_interval

		#for cpu, extra_cpu_interval in entity.extra_duration_by_cpu.items():
		#	wallclock_duration_by_cpu[cpu] += extra_cpu_interval

		# Consider top of stack or whole stack?
		# TODO make this a configurable option 
		#for parallelism, interval in entity.top_of_stack_parallelism_intervals.items():
		for parallelism, interval in entity.parallelism_intervals.items():
			parallelism_intervals[parallelism] += interval
	
	cpus = []
	wallclock_durations = []
	active_wallclock_durations = []
	for cpu, duration in wallclock_duration_by_cpu.items():
		cpus.append(cpu)
		wallclock_durations.append(duration)
		active_wallclock_durations.append(active_wallclock_duration_by_cpu[cpu])

	if len(cpus) > 1:
		sorted_indices = np.argsort(wallclock_durations)[::-1]
		wallclock_durations = np.array(wallclock_durations)[sorted_indices].tolist()
		active_wallclock_durations = np.array(active_wallclock_durations)[sorted_indices].tolist()
		cpus = np.array(cpus)[sorted_indices].tolist()

	return cpus, wallclock_durations, parallelism_intervals, active_wallclock_durations

"""
	This is what will actually be plotted as a sub-bar
	These are built from the execution intervals of the node
"""
class PFGTreeNodeElement:

	def __init__(self, 
			name,
			wallclock_duration,
			parallelism_intervals,
			per_event_values,
			cpu_time=None
			):

		self.name = name
		self.wallclock_duration = wallclock_duration # this is truly wallclock duration, must be careful to choose the correct one when merging
		self.parallelism_intervals = parallelism_intervals # total CPU time spent in each parallelism state, this should sum to CPU time
		self.per_event_values = per_event_values

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
			parallelism_intervals,
			per_event_values
			):

		self.cpu = cpu
		self.name = name
		self.duration = duration
		self.parallelism_intervals = parallelism_intervals # these are how long this CPU spent in each parallelism state, should sum to duration

		self.per_event_values = per_event_values

class PFGTreeNode:

	def __init__(self,
			parent_node,
			name,
			cpu,
			wallclock_duration,
			parallelism_intervals,
			depth,
			per_event_values,
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

		first_element = PFGTreeNodeElement(name, wallclock_duration, parallelism_intervals, per_event_values)
		self.node_partitions = [first_element]
		
		first_interval = PFGTreeNodeExecutionInterval(name, cpu, wallclock_duration, parallelism_intervals, per_event_values)
		self.execution_intervals = [first_interval]

		# Always differential as target - reference (meaning a negative interval is a speedup from reference)
		self.differential_interval = None # i.e. wallclock
		self.differential_parallelism = None
		self.differential_cpu_time = None
		self.antiisomorphic = None

		self.antiisomorphic_parallelism_intervals = None # A hacky accumulation of child parallelism intervals, to allow for calculation of differential parallelisms of antiisomorphic nodes

		self.corresponding_node = None

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
		parallelism_intervals_by_group = defaultdict(lambda: defaultdict(int))
		per_event_values_by_group = defaultdict(lambda: defaultdict(int))
		for execution_interval in self.execution_intervals:

			if cpu_time == True or execution_interval.cpu == self.cpus[0]:
				wallclock_durations_by_group[execution_interval.name] += execution_interval.duration

				for event_idx, value in execution_interval.per_event_values.items():
					per_event_values_by_group[execution_interval.name][event_idx] += value

				for parallelism, interval in execution_interval.parallelism_intervals.items():
					parallelism_intervals_by_group[execution_interval.name][parallelism] += interval 
			
			# We always add the duration of all intervals (across CPUs) to the total CPU time
			cpu_time_by_group[execution_interval.name] += execution_interval.duration
			
			# could put more information here about the collapsed parallel information (currently only retaining the profiling information for the node with the longest wallclock from the group)
			if execution_interval.cpu not in self.cpus:
				self.cpus.append(execution_interval.cpu)

		for group, duration in cpu_time_by_group.items():
			if group not in wallclock_durations_by_group:
				# this means it is an appended node part from a different CPU
				# TODO is this still relevant?
				wallclock_durations_by_group[group] += duration

		total_wallclock_duration = 0
		for group, duration in wallclock_durations_by_group.items():
			cpu_time_for_group = cpu_time_by_group[group]
			parallelism_intervals_for_group = parallelism_intervals_by_group[group]
			per_event_values_for_group = per_event_values_by_group[group]

			part = PFGTreeNodeElement(group, duration, parallelism_intervals_for_group, per_event_values_for_group, cpu_time_for_group)

			total_wallclock_duration += duration
			self.node_partitions.append(part)
		
			#for parallelism, cpu_time in parallelism_intervals_for_group.items():
			#	self.parallelism_intervals[parallelism] += cpu_time

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
		self.inclusive = True

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
						cpus, wallclock_durations, parallelism_intervals, active_wallclock_durations = get_durations_for_entities(entity_as_list)
						summed_per_event_values, average_per_event_values = get_aggregate_event_counts_for_entities(entity_as_list)

						logging.info("Got %s from entity %s of group %s with aggregate_per_event_values %s", wallclock_durations, entity.identifier, group, summed_per_event_values)

						node = PFGTreeNode(
							parent_node=parent_node,
							name=group,
							cpu=cpu,
							wallclock_duration=wallclock_durations[0],
							parallelism_intervals=parallelism_intervals,
							depth=depth,
							per_event_values=summed_per_event_values
							)

						node.active_wallclock_durations = active_wallclock_durations
						parent_node_list.append(node)

						if len(entity.child_entities) > 0:
							self.gen_tree_process_entities_into_child_nodes(node, entity.child_entities, node.child_nodes, depth+1, aggregate_stack_frames_by_default)		

				else:
					cpus, wallclock_durations, parallelism_intervals, active_wallclock_durations = get_durations_for_entities(entities_for_group)
					summed_per_event_values, average_per_event_values = get_aggregate_event_counts_for_entities(entity_as_list)

					node = PFGTreeNode(
						parent_node=parent_node,
						name=group,
						cpu=cpu,
						wallclock_duration=wallclock_durations[0],
						parallelism_intervals=parallelism_intervals,
						depth=depth,
						per_event_values=summed_per_event_values
						)

					node.active_wallclock_durations = active_wallclock_durations
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

					node = merged_node
					if node.wallclock_durations[0] != sum(node.node_partitions[0].parallelism_intervals.values()):
						logging.error("%s name is bad.", node.node_partitions[0].name)
						exit(0)

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

				if node.wallclock_durations[0] != sum(node.node_partitions[0].parallelism_intervals.values()):
					logging.error("%s name is bad.", node.node_partitions[0].name)
					exit(0)

				node_sets_queue.append(merged_node.child_nodes)

			node_sets_queue.pop(0)

	def transform_tree_aggregate_siblings_across_cpus(self):

		self.colour_mode = ColourMode.BY_PARALLELISM
		self.colour_mode = ColourMode.BY_EVENT_VALUE

		if self.stack_frames_aggregated == False:
			self.transform_tree_aggregate_stack_frames()

		# Assuemes that each node currently only represents execution on one CPU
		node_sets_queue = [self.root_nodes]

		while len(node_sets_queue) > 0:

			sibling_nodes = node_sets_queue[0]

			sibling_nodes_by_group = defaultdict(list)
			unique_cpus = set()
			for node in sibling_nodes:
				for interval in node.execution_intervals:
					unique_cpus.add(interval.cpu)
				sibling_nodes_by_group[node.node_partitions[0].name].append(node)

			# Now merge all of the siblings into one node, regardless of cpu

			total_duration_by_cpu = defaultdict(int)
			for group, nodes in sibling_nodes_by_group.items():
				for node in nodes:
					# if the node is sequential, then we should add its duration to all CPUs
					if len(node.execution_intervals) == 1:
						for cpu in unique_cpus:
							total_duration_by_cpu[cpu] += node.wallclock_durations[0]
					else:
						total_duration_by_cpu[node.cpus[0]] += node.wallclock_durations[0]
			
			highest_total_duration_cpu = 0
			highest_total_duration = 0
			for cpu, total_duration in total_duration_by_cpu.items():
				if total_duration > highest_total_duration:
					highest_total_duration = total_duration
					highest_total_duration_cpu = cpu

			for group, nodes in sibling_nodes_by_group.items():
				
				# ensure the first node in nodes is the one from highest_total_duration_cpu
				nodes_reordered = [node for node in nodes if node.cpus[0] == highest_total_duration_cpu]
				nodes_reordered.extend([node for node in nodes if node.cpus[0] != highest_total_duration_cpu])				

				# We can't sort here, because if there are multiple sibling groups, we might end up taking the longest version
				# of each child, which combined may be longer than the parent!
				# So, we need to find the cpu with the longest combined duration and use that
				# The problem is: that CPU might have a sibling split that is non-representative
				
				merged_node = self.merge_node_set(nodes_reordered, False)

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
			parallelism_intervals = "[" + ",".join([str(list(part.parallelism_intervals.items())) for part in node.node_partitions]) + "]"
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
			cpu_times = "[" + ",".join([str(part.cpu_time) for part in root_node.node_partitions]) + "]"
			parallelism_intervals = "[" + ",".join([str(list(part.parallelism_intervals.items())) for part in root_node.node_partitions]) + "]"

			logging.debug("Descending root node %d/%d on CPUs %s for durations %s with parallelism intervals %s and cpu times %s: %s",
				node_idx+1,
				len(self.root_nodes),
				cpus,
				durations,
				parallelism_intervals,
				cpu_times,
				name)

			child_nodes = root_node.child_nodes
			self.print_nodes(child_nodes, 1)

	def assign_colour_indexes_to_nodes(self, nodes, num_cpus, mapped_nodes=None):

		# TODO so very hacky
		if mapped_nodes is None:
			if self.colour_mode == ColourMode.BY_EVENT_VALUE:
				mapped_nodes = []
			else:
				mapped_nodes = {}

		for node in nodes:

			if self.colour_mode == ColourMode.BY_EVENT_VALUE:

				for event_idx, value in node.node_partitions[0].per_event_values.items():

					if event_idx >= len(mapped_nodes):
						mapped_nodes.append({})

					colour_identifier = value
			
					if colour_identifier not in mapped_nodes[event_idx]:
						# assign a new index
						next_index = len(mapped_nodes[event_idx])
						mapped_nodes[event_idx][colour_identifier] = next_index

				# create a parallelism one
				parallelism_event_idx = len(node.node_partitions[0].per_event_values.items())
				if parallelism_event_idx == len(mapped_nodes):
					mapped_nodes.append({})
					
				num = 0
				denom = 0
				for parallelism, interval in node.node_partitions[0].parallelism_intervals.items():
					num += parallelism*interval
					denom += interval

				# denom cannot be 0
				weighted_arithmetic_mean_parallelism = float(num) / denom
				colour_identifier = weighted_arithmetic_mean_parallelism
			
				if colour_identifier not in mapped_nodes[parallelism_event_idx]:
					# assign a new index
					next_index = len(mapped_nodes[parallelism_event_idx])
					mapped_nodes[parallelism_event_idx][colour_identifier] = next_index
			
				self.assign_colour_indexes_to_nodes(node.child_nodes, num_cpus, mapped_nodes)

			else:

				if self.colour_mode == ColourMode.BY_PARENT:
					if node.original_parent_node is None:
						colour_identifier = "None"
					else:
						colour_identifier = str(hex(id(node.original_parent_node)))
				elif self.colour_mode == ColourMode.BY_CPU:
					colour_identifier = ",".join([str(cpu) for cpu in node.cpus])
				elif self.colour_mode == ColourMode.BY_PARALLELISM:
					num = 0
					denom = 0
					for parallelism, interval in node.node_partitions[0].parallelism_intervals.items():
						num += parallelism*interval
						denom += interval

					# denom cannot be 0
					weighted_arithmetic_mean_parallelism = float(num) / denom
					colour_identifier = weighted_arithmetic_mean_parallelism

				elif self.colour_mode == ColourMode.BY_EVENT_VALUE:
					pass

				else:
					logging.error("Colour mode not supported.")
					raise NotImplementedError()

				if colour_identifier not in mapped_nodes:
					# assign a new index
					next_index = len(mapped_nodes)
					mapped_nodes[colour_identifier] = next_index
				
				self.assign_colour_indexes_to_nodes(node.child_nodes, num_cpus, mapped_nodes)

		self.node_colour_mapping = mapped_nodes

		return mapped_nodes

	def generate_exclusive_cpu_times(self, nodes):

		self.inclusive = False
		for node in nodes:

			if node.parent_node is not None:

				for interval_idx, interval in enumerate(node.execution_intervals):
					# Find the corresponding interval of the parent
					if interval.cpu in node.parent_node.cpus:

						# Remove it from the summed cpu_time

						logging.info("Removing an interval of duration %s of %s on cpu %d from the node %s with duration %s",
							sizeof_fmt(interval.duration),
							node.node_partitions[0].name,
							interval.cpu,
							node.parent_node.node_partitions[0].name,
							sizeof_fmt(node.parent_node.node_partitions[0].cpu_time))

						node.parent_node.node_partitions[0].cpu_time -= interval.duration

						# Also remove from the parent's interval to keep consistent
						parent_interval_idx = node.parent_node.cpus.index(interval.cpu)
						node.parent_node.execution_intervals[parent_interval_idx].duration -= interval.duration
			
			# descend to this node's children and do the same
			self.generate_exclusive_cpu_times(node.child_nodes)

	def generate_exclusive_parallelism_intervals(self, nodes):

		self.inclusive = False

		for node in nodes:
			if node.parent_node is not None:
				for parallelism, interval in node.node_partitions[0].parallelism_intervals.items():
					# remove the interval from the parent
					node.parent_node.node_partitions[0].parallelism_intervals[parallelism] -= interval

			# descend to this node's children and do the same
			self.generate_exclusive_parallelism_intervals(node.child_nodes)

	# This function needs to do depth first search, and accumulate upwards!
	# Parallelism intervals are inclusive by default, and CPU times are inclusive after merging across CPUs
	def generate_inclusive_event_counts(self, nodes, counters):

		self.inclusive = True

		accumulated_counts = defaultdict(int)
		accumulated_cpu_time = 0

		for node in nodes:
			node_event_counts = node.node_partitions[0].per_event_values

			accumulated_child_counts, accumulated_child_cpu_time = self.generate_inclusive_event_counts(node.child_nodes, counters)
			
			# add accumulated child counts to this node
			for event, value in accumulated_child_counts.items():
				node_event_counts[event] += value

			# process the rates, which should be be an average not a sum
			rate_events = [event for event in list(counters.values()) if "_RATE" in event]
			for rate_event in rate_events:
				event_idx = list(counters.values()).index(rate_event.replace("_RATE",""))
				rate_event_idx = list(counters.values()).index(rate_event)
				duration_event_idx = list(counters.values()).index("PAPI_TOT_CYC")

				accumulated_event_count = node_event_counts[event_idx]
				duration = node_event_counts[duration_event_idx]
				node_event_counts[rate_event_idx] = float(accumulated_event_count) / duration

			# add to the accumulated counts for the siblings
			for event, value in node_event_counts.items():
				accumulated_counts[event] += value
				
			for interval_idx, interval in enumerate(node.execution_intervals):
				# Find the corresponding interval of the parent
				if node.parent_node is not None and interval.cpu in node.parent_node.cpus:
					# Also add it to the parent's corresponding interval to keep consistent
					parent_interval_idx = node.parent_node.cpus.index(interval.cpu)
					node.parent_node.execution_intervals[parent_interval_idx].duration += interval.duration

			# add child cpu time to the accumulated cpu time for siblings
			node.node_partitions[0].cpu_time += accumulated_child_cpu_time
			accumulated_cpu_time += node.node_partitions[0].cpu_time

		return accumulated_counts, accumulated_cpu_time

	def match_corresponding_nodes(self, reference_nodes, target_nodes, reference_depth, target_depth):

		matched_nodes = [] # each element is an array of 2 nodes
		reference_antiisomorphic_nodes = []
		target_antiisomorphic_nodes = []

		# For each reference node:
		for reference_idx in range(len(reference_nodes)):

			reference_node = reference_nodes[reference_idx]
			if reference_node.corresponding_node is not None:
				continue

			# find a target for this node
			found_pair = False
			for target_node in target_nodes:

				# If same name, and target_node isn't already associated, then we have a match
				if (reference_node.node_partitions[0].name == target_node.node_partitions[0].name and target_node.corresponding_node is None):

					reference_node.corresponding_node = target_node
					target_node.corresponding_node = reference_node
					matched_nodes.append([reference_node, target_node])
					found_pair = True
					break

			if found_pair:
				# descend into the matched nodes respective children, and try to match them
				child_matches, child_ref_anti_nodes, child_tar_anti_nodes  = self.match_corresponding_nodes(reference_node.child_nodes, reference_node.corresponding_node.child_nodes, reference_depth+1, target_depth+1)
				matched_nodes.extend(child_matches)
				reference_antiisomorphic_nodes.extend(child_ref_anti_nodes)
				target_antiisomorphic_nodes.extend(child_tar_anti_nodes)
				continue

			# Check if both reference nodes and target nodes contain parallel loop
			reference_nodes_contain_parallel_loop = any([True if node.node_partitions[0].name == "omp_parallel_loop" else False for node in reference_nodes])
			target_nodes_contain_parallel_loop = any([True if node.node_partitions[0].name == "omp_parallel_loop" else False for node in target_nodes])
			if reference_nodes_contain_parallel_loop and target_nodes_contain_parallel_loop:
				# Failed
				pass
			else:
				# If reference is omp_parallel_loop:
				if reference_node.node_partitions[0].name == "omp_parallel_loop":

					# Compare target_nodes with reference_node.child_nodes
					# If I find a match, then add it as a node pair, and set this reference_node to be anti-isomorphic
					cross_depth_matches, cross_depth_ref_anti_nodes, cross_depth_tar_anti_nodes = self.match_corresponding_nodes(reference_node.child_nodes, target_nodes, reference_depth+1, target_depth)

					# Can't get here if no match found
					reference_node.antiisomorphic = True
					if reference_node not in reference_antiisomorphic_nodes:
						reference_antiisomorphic_nodes.append(reference_node)

					matched_nodes.extend(cross_depth_matches)
					reference_antiisomorphic_nodes.extend(cross_depth_ref_anti_nodes)
					target_antiisomorphic_nodes.extend(cross_depth_tar_anti_nodes)
					found_pair = True

				elif target_nodes_contain_parallel_loop:
					# Compare reference_node with the child nodes of the target_node that is an omp_parallel_loop
					target_loop_idx = [idx for idx, node in enumerate(target_nodes) if node.node_partitions[0].name == "omp_parallel_loop"][0]
					target_loop_node = target_nodes[target_loop_idx]

					# Compare this node with target_loop_node's children
					# If I find a match, then add it as a node pair, and set that target_node to be anti-isomorphic
					cross_depth_matches, cross_depth_ref_anti_nodes, cross_depth_tar_anti_nodes = self.match_corresponding_nodes([reference_node], target_loop_node.child_nodes, reference_depth, target_depth+1)
					
					# Can't get here if no match found
					target_loop_node.antiisomorphic = True
					if target_loop_node not in target_antiisomorphic_nodes:
						target_antiisomorphic_nodes.append(target_loop_node)

					matched_nodes.extend(cross_depth_matches)
					reference_antiisomorphic_nodes.extend(cross_depth_ref_anti_nodes)
					target_antiisomorphic_nodes.extend(cross_depth_tar_anti_nodes)
					found_pair = True

				else:
					# Failed
					pass

			if found_pair == False:
				# Failed to find a match
				logging.error("Failed to find a match for reference node %s at depth %d from targets at depth %d: %s",
					reference_node.node_partitions[0].name,
					reference_depth,
					target_depth,
					",".join([node.node_partitions[0].name for node in target_nodes]))
				exit(1)
				
		return matched_nodes, reference_antiisomorphic_nodes, target_antiisomorphic_nodes

	def calculate_nodes_differential(self, reference_root_nodes, target_root_nodes, counters):

		matched_nodes, reference_antiisomorphic_nodes, target_antiisomorphic_nodes = self.match_corresponding_nodes(reference_root_nodes, target_root_nodes, 0, 0)

		# Antiisomorphic target nodes: this is an extra omp_parallel_loop in the target.
		# If there is a loop, then it will be fully contained in the parent, so the parent's wallclock will account for the wallclock differential.
			# The target node might have event counts that are exclusive to it, however, unless the tree has been trasnformed for inclusive counts
			# I need to add those event counts to its parent, so that the differential works out

		if self.inclusive == False:

			# Each construct has it's own counts. But we won't be using the antiisomorphic nodes for differential calculations.
			# Therfore I need to give their exclusive counts to their parents to make the differential correct
			for node_set in [target_antiisomorphic_nodes, reference_antiisomorphic_nodes]:
				for anti_node in node_set:

					# TODO
					# What if I have multiple levels of anti-isomorphism?
					# I need to recurse to get the first set of non-antiisomorphic values!

					event_counts = anti_node.node_partitions[0].per_event_values
					parent_event_counts = anti_node.parent_node.node_partitions[0].per_event_values # guaranteed to have a parent (main cannot be omp_parallel_loop)
					
					# add accumulated child counts to this node
					for event, value in event_counts.items():
						parent_event_counts[event] += value

					# process the rates, which should be be an average not a sum
					rate_events = [event for event in list(counters.values()) if "_RATE" in event]
					for rate_event in rate_events:
						event_idx = list(counters.values()).index(rate_event.replace("_RATE",""))
						rate_event_idx = list(counters.values()).index(rate_event)
						duration_event_idx = list(counters.values()).index("PAPI_TOT_CYC")

						parent_event_count = parent_event_counts[event_idx] # i.e. the one we have accumulated
						duration = parent_event_counts[duration_event_idx]
						parent_event_counts[rate_event_idx] = float(parent_event_count) / duration

					# add the cpu time to the accumulated cpu time for siblings
					anti_node.parent_node.node_partitions[0].cpu_time += anti_node.node_partitions[0].cpu_time 

					# add the parallelism intervals of the child to the parent
					for parallelism, interval in anti_node.node_partitions[0].parallelism_intervals.items():
						anti_node.parent_node.node_partitions[0].parallelism_intervals[parallelism] += interval

		# Unmatched reference nodes: there is an extra omp_parallel_loop in the reference, and this will be displayed, meaning it needs to have differential values (e.g. wallclock interval)
		# The reference will always be shown, so what wallclock should it have?
		# It should have an differential interval that is the sum of its children
		# If counts are inclusive, it simply has exactly the same as the sum of its children
		# If counts are exclusive, all differential values are 0
		# This must be done *after* the differential intervals are calculated

		for ref_node, tar_node in matched_nodes:

			ref_node_wallclock = max(ref_node.wallclock_durations)
			tar_node_wallclock = max(tar_node.wallclock_durations)

			ref_node.differential_interval = tar_node_wallclock - ref_node_wallclock

			per_event_value_differential = {}
			for event_idx, tar_value in tar_node.node_partitions[0].per_event_values.items():
				ref_value = ref_node.node_partitions[0].per_event_values[event_idx]
				per_event_value_differential[event_idx] = tar_value - ref_value
				logging.debug("Target node %s at depth %s had %s more %s than the reference", ref_node.node_partitions[0].name, ref_node.original_depth, tar_value-ref_value, counters[event_idx])

			ref_node.node_partitions[0].per_event_values = per_event_value_differential

			# add differential value for parallelism
			num = 0
			denom = 0
			for parallelism, interval in tar_node.node_partitions[0].parallelism_intervals.items():
				num += parallelism*interval
				denom += interval
			target_weighted_arithmetic_mean_parallelism = float(num) / denom
			
			num = 0
			denom = 0
			for parallelism, interval in ref_node.node_partitions[0].parallelism_intervals.items():
				num += parallelism*interval
				denom += interval
			reference_weighted_arithmetic_mean_parallelism = float(num) / denom

			ref_node.differential_parallelism = target_weighted_arithmetic_mean_parallelism - reference_weighted_arithmetic_mean_parallelism

			# add differential value for total cpu time
			ref_node.differential_cpu_time = tar_node.node_partitions[0].cpu_time - ref_node.node_partitions[0].cpu_time

			if ref_node.differential_interval < 0:
				logging.trace("Target node %s at depth %s was faster than the reference by %s", ref_node.node_partitions[0].name, ref_node.original_depth, abs(ref_node.differential_interval))
			else:
				logging.trace("Target node %s at depth %s was slower than the reference by %s", ref_node.node_partitions[0].name, ref_node.original_depth, ref_node.differential_interval)

			# The antiisomorphic parents need to have the different interval taken from their non-antiisomorphic children
			# And if the tree is inclusive, also the event counts and so on
			antiisomorphic_parents = []

			isomorphic_parent_node = None
			if ref_node.parent_node is not None and ref_node.parent_node.antiisomorphic:
				# search for first non-antiisomorphic parent
				parent_node = ref_node.parent_node
				while isomorphic_parent_node is None:
					antiisomorphic_parents.append(parent_node)
					if parent_node.parent_node is not None:
						if parent_node.parent_node.antiisomorphic is None or parent_node.parent_node.antiisomorphic == False:
							isomorphic_parent_node = parent_node.parent_node
						else:
							parent_node = parent_node.parent_node
					else:
						logging.error("A node (%s) couldn't find a parent node that was isomorphic!", ref_node.node_partitions[0].name)
						raise ValueError()

			# I must append my values to all antiisomorphic parents, so that they have the sum differential of their children
			for parent_node in antiisomorphic_parents:

				if parent_node.differential_interval is None: 
					parent_node.differential_interval = ref_node.differential_interval

					if self.inclusive:
						parent_node.differential_cpu_time = ref_node.differential_cpu_time
						parent_node.node_partitions[0].per_event_values = ref_node.node_partitions[0].per_event_values
						parent_node.antiisomorphic_parallelism_intervals = ref_node.node_partitions[0].parallelism_intervals

				else:
					parent_node.differential_interval += ref_node.differential_interval

					if self.inclusive:
						parent_node.differential_cpu_time += ref_node.differential_cpu_time

						for event_idx, ref_value_differential in ref_node.node_partitions[0].per_event_values.items():
							ref_value = ref_node.node_partitions[0].per_event_values[event_idx]
							parent_node.node_partitions[0].per_event_values[event_idx] += ref_value_differential

						# I need a many-to-one computation between the reference node and the set of children, in light of there being no corresponding node
						for parallelism, interval in ref_node.node_partitions[0].parallelism_intervals.items():
							parent_node.antiisomorphic_parallelism_intervals[parallelism] += interval
			
			# ALTERNATIVELY, I COULD JUST REMOVE THE ANTI_ISOMORPHIC REFERENCE
			# IN FACT I REALLY SHOULD ONLY SEE THE ANTI_ISOMORPHIC TARGET
			# TODO now that I do this, can I just remove all the other stuff?

			# TODO I now need to ADD the target antiisomorphic nodes to the reference graph!!

			if isomorphic_parent_node is not None:
				for parent_node in antiisomorphic_parents:
					if parent_node in isomorphic_parent_node.child_nodes:
						isomorphic_parent_node.child_nodes.remove(parent_node)

				isomorphic_parent_node.child_nodes.append(ref_node)
				
				# set the parent to be the ismorphic one
				ref_node.parent_node = isomorphic_parent_node
				ref_node.ancestor_alignment_node = isomorphic_parent_node
		
		for ref_anti_node in reference_antiisomorphic_nodes:
			if self.inclusive:
				# Recompute the averaged event counts for each reference
				rate_events = [event for event in list(counters.values()) if "_RATE" in event]
				for rate_event in rate_events:
					event_idx = list(counters.values()).index(rate_event.replace("_RATE",""))
					rate_event_idx = list(counters.values()).index(rate_event)
					duration_event_idx = list(counters.values()).index("PAPI_TOT_CYC")

					event_count = ref_anti_node.node_partitions[0].per_event_values[event_idx]
					duration = ref_anti_node.node_partitions[0].per_event_values[duration_event_idx]
					ref_anti_node.node_partitions[0].per_event_values[rate_event_idx] = float(event_count) / duration
			
				# Compute the differential parallelism, after having accumulated all of the child parallelism_intervals
				num = 0
				denom = 0
				for parallelism, interval in ref_anti_node.antiisomorphic_parallelism_intervals.items():
					num += parallelism*interval
					denom += interval
				target_weighted_arithmetic_mean_parallelism = float(num) / denom
				
				num = 0
				denom = 0
				for parallelism, interval in ref_anti_node.node_partitions[0].parallelism_intervals.items():
					num += parallelism*interval
					denom += interval
				reference_weighted_arithmetic_mean_parallelism = float(num) / denom

				ref_anti_node.differential_parallelism = target_weighted_arithmetic_mean_parallelism - reference_weighted_arithmetic_mean_parallelism

				# The differential interval will have already been computed during the processing of the matched nodes				

			else:
				ref_anti_node.differential_parallelism = 0.0
				ref_anti_node.differential_cpu_time = 0.0

				for event_idx in range(len(ref_anti_node.node_partitions[0].per_event_values)):
					ref_anti_node.node_partitions[0].per_event_values[event_idx] = 0.0 # no differential, because it doesn't exist in the other one
				
				# The differential interval will have already been computed during the processing of the matched nodes				

	def calculate_nodes_differential_version2(self, reference_root_nodes, target_root_nodes, counters):

		matched_nodes, reference_antiisomorphic_nodes, target_antiisomorphic_nodes = self.match_corresponding_nodes(reference_root_nodes, target_root_nodes, 0, 0)

		if self.inclusive == False:
			# Each construct has it's own counts. But we won't be using the antiisomorphic nodes for differential calculations.
			# Therfore I need to give their exclusive counts to their parents to make the differential correct
			for node_set in [target_antiisomorphic_nodes, reference_antiisomorphic_nodes]:
				for anti_node in node_set:
					
					# Find the first non-antiisomorphic parent of this node
					# In order to give it my exclusive event counts

					event_counts = anti_node.node_partitions[0].per_event_values

					isomorphic_parent_node = None
					if anti_node.parent_node is not None and anti_node.parent_node.antiisomorphic:
						# search for first non-antiisomorphic parent
						parent_node = anti_node.parent_node
						while isomorphic_parent_node is None:
							if parent_node.parent_node is not None:
								if parent_node.parent_node.antiisomorphic is None or parent_node.parent_node.antiisomorphic == False:
									isomorphic_parent_node = parent_node.parent_node
								else:
									parent_node = parent_node.parent_node
							else:
								logging.error("A node (%s) couldn't find a parent node that was isomorphic!", anti_node.node_partitions[0].name)
								raise ValueError()
	
					# If I have already removed it, don't worry about the anti_node
					if isomorphic_parent_node is None:
						continue

					parent_event_counts = isomorphic_parent_node.node_partitions[0].per_event_values
					
					# add accumulated child counts to this node
					for event, value in event_counts.items():
						parent_event_counts[event] += value

					# process the rates, which should be be an average not a sum
					rate_events = [event for event in list(counters.values()) if "_RATE" in event]
					for rate_event in rate_events:
						event_idx = list(counters.values()).index(rate_event.replace("_RATE",""))
						rate_event_idx = list(counters.values()).index(rate_event)
						duration_event_idx = list(counters.values()).index("PAPI_TOT_CYC")

						parent_event_count = parent_event_counts[event_idx] # i.e. the one we have accumulated
						duration = parent_event_counts[duration_event_idx]
						parent_event_counts[rate_event_idx] = float(parent_event_count) / duration

					isomorphic_parent_node.node_partitions[0].per_event_values = parent_event_counts
					
					"""
					logging.info("Adding CPU time %s of %s to the node %s with duration %s",
						sizeof_fmt(anti_node.node_partitions[0].cpu_time),
						anti_node.node_partitions[0].name,
						isomorphic_parent_node.node_partitions[0].name,
						sizeof_fmt(isomorphic_parent_node.node_partitions[0].cpu_time))
					exit(0)
					"""

					# add the cpu time to the accumulated cpu time for siblings
					isomorphic_parent_node.node_partitions[0].cpu_time += anti_node.node_partitions[0].cpu_time 
					
					# add the parallelism intervals of the child to the parent
					for parallelism, interval in anti_node.node_partitions[0].parallelism_intervals.items():
						isomorphic_parent_node.node_partitions[0].parallelism_intervals[parallelism] += interval

		# My goal is to remove all reference antiisomorphic nodes
		# Do this by connecting the parent and children of all the reference nodes, obviously
		for ref_anti_node in reference_antiisomorphic_nodes:
			for child_node in ref_anti_node.child_nodes:
				child_node.parent_node = ref_anti_node.parent_node
				child_node.ancestor_alignment_node = ref_anti_node.ancestor_alignment_node
			ref_anti_node.parent_node.child_nodes.remove(ref_anti_node)
			ref_anti_node.parent_node.child_nodes.extend(ref_anti_node.child_nodes)
			ref_anti_node.child_nodes.clear()

		# Now, add all target antiisomorphic nodes to the reference
		for tar_anti_node in target_antiisomorphic_nodes:
			logging.debug("%s exists in the target tracefile but does not have a counterpart in the reference!", tar_anti_node.node_partitions[0].name)

			# remove the target node from its parent
			tar_anti_node.parent_node.child_nodes.remove(tar_anti_node)

			# set the target node's parent to be the reference version of its parent
			tar_anti_node.parent_node = tar_anti_node.parent_node.corresponding_node
			tar_anti_node.ancestor_alignment_node = tar_anti_node.parent_node
			
			# find the reference matches for the children of the target node
			child_nodes_to_move = [child_node for child_node in tar_anti_node.parent_node.child_nodes if child_node.corresponding_node in tar_anti_node.child_nodes]

			# remove the matching references from the reference parent
			for child_node in child_nodes_to_move:
				child_node.parent_node.child_nodes.remove(child_node)
				child_node.parent_node = tar_anti_node
				child_node.ancestor_alignment_node = tar_anti_node

			# remove the children of the target node
			tar_anti_node.child_nodes.clear()
			
			# add the matching reference children to the target node
			tar_anti_node.child_nodes = child_nodes_to_move
			tar_anti_node.node_partitions[0].wallclock_duration = sum([child_node.node_partitions[0].wallclock_duration for child_node in tar_anti_node.child_nodes])

			# finally, add the target node to the reference parent
			tar_anti_node.parent_node.child_nodes.append(tar_anti_node)

		# I now have the correct tree to display

		# Now, I need to compute the differentials!
		for ref_node, tar_node in matched_nodes:

			ref_node_wallclock = max(ref_node.wallclock_durations)
			tar_node_wallclock = max(tar_node.wallclock_durations)

			ref_node.differential_interval = tar_node_wallclock - ref_node_wallclock

			per_event_value_differential = {}
			for event_idx, tar_value in tar_node.node_partitions[0].per_event_values.items():
				ref_value = ref_node.node_partitions[0].per_event_values[event_idx]
				per_event_value_differential[event_idx] = tar_value - ref_value
				logging.debug("Target node %s at depth %s had %s more %s than the reference", ref_node.node_partitions[0].name, ref_node.original_depth, tar_value-ref_value, counters[event_idx])

			ref_node.node_partitions[0].per_event_values = per_event_value_differential

			# add differential value for parallelism
			num = 0
			denom = 0
			for parallelism, interval in tar_node.node_partitions[0].parallelism_intervals.items():
				num += parallelism*interval
				denom += interval
			target_weighted_arithmetic_mean_parallelism = float(num) / denom
			
			num = 0
			denom = 0
			for parallelism, interval in ref_node.node_partitions[0].parallelism_intervals.items():
				num += parallelism*interval
				denom += interval
			reference_weighted_arithmetic_mean_parallelism = float(num) / denom

			ref_node.differential_parallelism = target_weighted_arithmetic_mean_parallelism - reference_weighted_arithmetic_mean_parallelism

			# add differential value for total cpu time
			ref_node.differential_cpu_time = tar_node.node_partitions[0].cpu_time - ref_node.node_partitions[0].cpu_time
			
			if ref_node.differential_interval < 0:
				logging.trace("Target node %s at depth %s was faster than the reference by %s", ref_node.node_partitions[0].name, ref_node.original_depth, abs(ref_node.differential_interval))
			else:
				logging.trace("Target node %s at depth %s was slower than the reference by %s", ref_node.node_partitions[0].name, ref_node.original_depth, ref_node.differential_interval)

			# Now, I need to compute the differential counts for any antiisomorphic reference nodes

			# The antiisomorphic parents need to have the different interval taken from their non-antiisomorphic children
			# And if the tree is inclusive, also the event counts and so on
			antiisomorphic_parents = []

			isomorphic_parent_node = None
			if ref_node.parent_node is not None and ref_node.parent_node.antiisomorphic:
				# search for first non-antiisomorphic parent
				parent_node = ref_node.parent_node
				while isomorphic_parent_node is None:
					antiisomorphic_parents.append(parent_node)
					if parent_node.parent_node is not None:
						if parent_node.parent_node.antiisomorphic is None or parent_node.parent_node.antiisomorphic == False:
							isomorphic_parent_node = parent_node.parent_node
						else:
							parent_node = parent_node.parent_node
					else:
						logging.error("A node (%s) couldn't find a parent node that was isomorphic!", ref_node.node_partitions[0].name)
						raise ValueError()

			for parent_node in antiisomorphic_parents:

				if parent_node.differential_interval is None: 
					parent_node.differential_interval = ref_node.differential_interval

					if self.inclusive:
						parent_node.differential_cpu_time = ref_node.differential_cpu_time
						parent_node.node_partitions[0].per_event_values = ref_node.node_partitions[0].per_event_values
						parent_node.antiisomorphic_parallelism_intervals = ref_node.node_partitions[0].parallelism_intervals
					else:
						parent_node.differential_cpu_time = 0.0
						for event_idx, ref_value_differential in ref_node.node_partitions[0].per_event_values.items():
							parent_node.node_partitions[0].per_event_values[event_idx] = 0.0
						parent_node.antiisomorphic_parallelism_intervals = {}
						for parallelism, interval in ref_node.node_partitions[0].parallelism_intervals.items():
							parent_node.antiisomorphic_parallelism_intervals[parallelism] = 0.0

				else:
					parent_node.differential_interval += ref_node.differential_interval

					if self.inclusive:
						parent_node.differential_cpu_time += ref_node.differential_cpu_time

						logging.info("Differential_cpu_time on parent_node %s was %s after adding the ref_node's %s differential_cpu_time %s",
							parent_node.node_partitions[0].name,
							sizeof_fmt(parent_node.differential_cpu_time),
							ref_node.node_partitions[0].name,
							sizeof_fmt(ref_node.differential_cpu_time))

						for event_idx, ref_value_differential in ref_node.node_partitions[0].per_event_values.items():
							ref_value = ref_node.node_partitions[0].per_event_values[event_idx]
							parent_node.node_partitions[0].per_event_values[event_idx] += ref_value_differential

						# I need a many-to-one computation between the reference node and the set of children, in light of there being no corresponding node
						for parallelism, interval in ref_node.node_partitions[0].parallelism_intervals.items():
							parent_node.antiisomorphic_parallelism_intervals[parallelism] += interval

		for tar_anti_node in target_antiisomorphic_nodes:
			if self.inclusive:
				# Recompute the averaged event counts for each reference
				rate_events = [event for event in list(counters.values()) if "_RATE" in event]
				for rate_event in rate_events:
					event_idx = list(counters.values()).index(rate_event.replace("_RATE",""))
					rate_event_idx = list(counters.values()).index(rate_event)
					duration_event_idx = list(counters.values()).index("PAPI_TOT_CYC")

					event_count = tar_anti_node.node_partitions[0].per_event_values[event_idx]
					duration = tar_anti_node.node_partitions[0].per_event_values[duration_event_idx]
					tar_anti_node.node_partitions[0].per_event_values[rate_event_idx] = float(event_count) / duration
			
				# Compute the differential parallelism, after having accumulated all of the child parallelism_intervals
				num = 0
				denom = 0
				for parallelism, interval in tar_anti_node.antiisomorphic_parallelism_intervals.items():
					num += parallelism*interval
					denom += interval
				reference_weighted_arithmetic_mean_parallelism = float(num) / denom
				
				num = 0
				denom = 0
				for parallelism, interval in tar_anti_node.node_partitions[0].parallelism_intervals.items():
					num += parallelism*interval
					denom += interval
				target_weighted_arithmetic_mean_parallelism = float(num) / denom

				tar_anti_node.differential_parallelism = target_weighted_arithmetic_mean_parallelism - reference_weighted_arithmetic_mean_parallelism

				# The differential interval will have already been computed during the processing of the matched nodes				

			else:
				tar_anti_node.differential_parallelism = 0.0
				tar_anti_node.differential_cpu_time = 0.0

				for event_idx in range(len(tar_anti_node.node_partitions[0].per_event_values)):
					tar_anti_node.node_partitions[0].per_event_values[event_idx] = 0.0 # no differential, because it doesn't exist in the other one
				
				# The differential interval will have already been computed during the processing of the matched nodes				

