from enum import Enum, auto
from copy import copy
from collections import defaultdict

from PfgUtil import debug_mode

import logging
import time
import subprocess

class EntityType(Enum):
	STACKFRAME = auto()
	SYNC_REGION = auto()
	TASK_REGION = auto()
	IMPLICIT_TASK_REGION = auto()
	THREAD = auto()
	TASK_PERIOD = auto()
	PARALLEL_REGION = auto()
	TASK_CREATION = auto()
	WORK = auto()

"""
	Each construct that is extracted from the tracefile is an Entity
	An entity has a type, a CPU, a start and an end time, along with other type-specific information
	TODO this could perhaps be merged with the PFGTree data structures (e.g. PFGTreeNodeExecutionInterval) with some effort
"""
class Entity:

	def __init__(self, entity_type, identifier, cpu, start, end):

		self.entity_type = entity_type

		self.identifier = identifier # this is used differently depending on the type
		self.cpu = cpu
		self.start = start
		self.end = end
		self.depth = None
		self.group = None

		self.task_id = None
		self.parent_id = None

		self.pregion_num_threads = 0 # only used for parallel region type

		self.parallelism_intervals = defaultdict(int)
		self.per_cpu_intervals = defaultdict(int)
		self.per_cpu_top_of_stack_intervals = defaultdict(int)
		self.top_of_stack_parallelism_intervals = defaultdict(int)
		self.per_cpu_top_of_stack_parallelism_intervals = {}

		self.parent_entity = None
		self.child_entities = []

		# i.e. duration that has been merged into this one
		self.extra_duration_by_cpu = defaultdict(int)

	def add_parallelism_interval(self, parallelism, interval, cpu):
		self.per_cpu_intervals[cpu] += interval
		self.parallelism_intervals[parallelism] += interval
	
	def add_top_of_stack_interval(self, interval, cpu):
		self.per_cpu_top_of_stack_intervals[cpu] += interval
	
	def add_top_of_stack_parallelism_interval(self, parallelism, interval):
		self.top_of_stack_parallelism_intervals[parallelism+1] += interval
	
	# TODO what does this mean? each CPU has it's own top_of_stack parallelism interval?
	def add_per_cpu_top_of_stack_parallelism_interval(self, parallelism, interval, cpu):
		if cpu not in self.per_cpu_top_of_stack_parallelism_intervals:
			self.per_cpu_top_of_stack_parallelism_intervals[cpu] = defaultdict(int)
				
		self.per_cpu_top_of_stack_parallelism_intervals[cpu][parallelism+1] += interval

def file_len(fname):
		p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
																							stderr=subprocess.PIPE)
		result, err = p.communicate()
		if p.returncode != 0:
				raise IOError(err)
		return int(result.strip().split()[0])

from intervaltree import Interval, IntervalTree

def parse_trace_preprocessing(filename):

	min_timestamp = float("inf")
	max_timestamp = -1.0
	num_lines = file_len(filename)

	max_cpu = -1
	main_cpu = None
	cpus = set()

	state_transitions = []
	unique_parent_frames = {} # parent_id -> tuple of (includes parallel, cpu, start, end, child frames)
	
	logging.debug("Loading worker state-transitions and leaf stack frames from trace for preprocessing")
	
	# We parse the file to get the state transitions, and build an interval tree per CPU that describes the parallelism states of that CPU
	# On each CPU, the intervals of parallelism are guaranteed to be fully ordered
	# Therefore, to know if a stack frame had some parallel constructs inside of it, we just test if there is > 1 intervals in the map within the stack frame

	# We also collect all of the leaf stack frames, as stack frames that have no children and also do not express parallelism can immediately be merged

	with open(filename, 'r') as f:
		line_idx = -1
		for line in f:

			line_idx += 1

			if line.strip() == "":
				continue

			if line_idx % int(num_lines/10.0) == 0:
				logging.debug("Parsed line %d of %d (%d%%).", line_idx+1, num_lines, ((line_idx+1)/num_lines)*100.0)
			
			split_line = line.strip().split(",")

			cpu = int(split_line[1])
			start = int(split_line[2])
			end = int(split_line[3])
			
			if cpu > max_cpu:
				max_cpu = cpu

			cpus.add(cpu)
			if "main" in line:
				main_cpu = cpu

			if start < min_timestamp:
				min_timestamp = start
			if end > max_timestamp:
				max_timestamp = end

			if split_line[0] == "sync_region":
				state_transitions.append((start, cpu, False)) # At this time, I enter non-parallel execution on this cpu
				state_transitions.append((end, cpu, None)) # At the end time, I don't know what state I go back into

			elif split_line[0] == "work":
				state_transitions.append((start, cpu, True)) # At this time, I enter non-parallel execution on this cpu
				state_transitions.append((end, cpu, None)) # At the end time, I don't know what state I go back into

			elif split_line[0] == "stack_frame":

				frame_id = split_line[4]
				parent_frame_id = split_line[5]
				symbol = ",".join(split_line[7:])

				if parent_frame_id in unique_parent_frames:
					unique_parent_frames[parent_frame_id][4][symbol].add(frame_id)
				else:
					frames_by_symbol = defaultdict(set)
					frames_by_symbol[symbol].add(frame_id)
					unique_parent_frames[parent_frame_id] = [None, cpu, start, end, frames_by_symbol]

	# Now iterate over the state transitions and build the interval trees

	state_transitions.sort(key=lambda x: x[0])

	previous_time_per_cpu = {}
	parallelism_stack_per_cpu = defaultdict(list)
	tree_per_cpu = defaultdict(IntervalTree)

	for cpu in cpus:
		parallelism_stack_per_cpu[cpu].append(False)
		previous_time_per_cpu[cpu] = min_timestamp

	parallelism_stack_per_cpu[main_cpu][-1] = True # the main cpu starts as 'active'

	for (timestamp, cpu, transition_type) in state_transitions:

		# record from the previous time until now as whatever parallelism state the cpu was in (cpu value if executing, False if not executing)
		previous_state = parallelism_stack_per_cpu[cpu][-1]
		previous_time = previous_time_per_cpu[cpu]

		if transition_type is True:

			parallelism_stack_per_cpu[cpu].append(True)

			if previous_state is False:
				# going into active parallel execution, so I only need to update if the previous state was False
				tree_per_cpu[cpu][previous_time:timestamp] = cpu if previous_state is True else False
				previous_time_per_cpu[cpu] = timestamp

		elif transition_type is False:

			parallelism_stack_per_cpu[cpu].append(False)

			if previous_state is True:
				# going into a state of non-active parallel execution, so I only need to update if the previous state was True
				tree_per_cpu[cpu][previous_time:timestamp] = cpu if previous_state is True else False
				previous_time_per_cpu[cpu] = timestamp

		elif transition_type is None:

			parallelism_stack_per_cpu[cpu].pop()
			new_state = parallelism_stack_per_cpu[cpu][-1]
			if previous_state != new_state:
				tree_per_cpu[cpu][previous_time:timestamp] = cpu if previous_state is True else False
				previous_time_per_cpu[cpu] = timestamp
	
	del state_transitions

	combined_tree = None
	for cpu in cpus:
		if combined_tree is None:
			combined_tree = tree_per_cpu[cpu]
		else:
			combined_tree = combined_tree.union(tree_per_cpu[cpu])
	
	# Now iterate over the parents, and any of the child frames that are not themselves in the parent dict can be merged
	
	frames_to_merge = {} # parent_node_id + "_" + symbol -> set_of_frames

	for parent_id, info_tuple in unique_parent_frames.items():

		parent_contains_parallel_constructs = info_tuple[0]
		cpu = info_tuple[1]
		start = info_tuple[2]
		end = info_tuple[3]
		children_by_symbol = info_tuple[4]

		# If it is not known, then determine it
		if parent_contains_parallel_constructs is None:
			parent_contains_parallel_constructs = True if len(tree_per_cpu[cpu].overlap(start, end)) > 1 else False
			info_tuple[0] = parent_contains_parallel_constructs

		for symbol, children in children_by_symbol.items():

			identifier = str(parent_id) + "_" + symbol
			set_of_frames = None

			merge_all = True
			for child_id in children:

				if child_id in unique_parent_frames:
					# it is not a leaf
					merge_all = False

					# to save rechecking the intervaltree:
					if parent_contains_parallel_constructs is False:
						# This means that its children also do not contain parallel constructs
						unique_parent_frames[child_id][0] = parent_contains_parallel_constructs

				elif parent_contains_parallel_constructs == False:
					# this child is a leaf node! I should merge it with something else

					if set_of_frames is None:
						set_of_frames = set(child_id)
					else:
						set_of_frames.add(child_id)

			if set_of_frames is not None and len(set_of_frames) > 1:
				if merge_all:
					frames_to_merge[identifier] = None # None meaning there is no list of frames, just merge all of them
				else:
					frames_to_merge[identifier] = set_of_frames

	for identifier, frames in frames_to_merge.items():
		logging.debug("The parent %s will have its %s frames merged. This will be %s of them.", identifier.split("_")[0], "_".join(identifier.split("_")[1:]), "all" if frames is None else str(len(frames)))

	return combined_tree, frames_to_merge, max_cpu, main_cpu, min_timestamp, max_timestamp

def parse_trace(filename,
		work_tree,
		merged_leaf_parents):

	logging.debug("Parsing the tracefile %s.", filename)

	entries = []
	exits = []
	
	tasks = {}
	unique_groups = set()
	merged_base_frames = {} # merge_identifier -> base frame

	num_lines = file_len(filename)

	with open(filename, 'r') as f:
		line_idx = -1
		for line in f:

			line_idx += 1

			if line.strip() == "":
				continue

			if line_idx % int(num_lines/10.0) == 0:
				logging.debug("Parsed line %d of %d (%d%%).", line_idx+1, num_lines, ((line_idx+1)/num_lines)*100.0)

			split_line = line.strip().split(",")
			cpu = int(split_line[1])

			if split_line[0] == "stack_frame":

				frame_id = split_line[4]
				parent_frame_id = split_line[5]

				frame_start = int(split_line[2])
				frame_end = int(split_line[3])

				depth = int(split_line[6])
				symbol = ",".join(split_line[7:])

				# ignoring openmp outlining function calls
				if "outlined" in symbol:
					continue
				
				if symbol not in unique_groups:
					unique_groups.add(symbol)

				# check if this is a merged frame
				should_merge = False
				identifier = str(parent_frame_id) + "_" + symbol
				if identifier in merged_leaf_parents:
					frames_to_merge = merged_leaf_parents[identifier]

					if frames_to_merge is None:
						should_merge = True
					else:
						if frame_id in frames_to_merge:
							should_merge = True

				# I need to create the entity if I am not being merged,
				# or I am the first one to merge (i.e. the base frame)
				if (should_merge is False) or (should_merge is True and identifier not in merged_base_frames): 

					stack_frame = Entity(EntityType.STACKFRAME, frame_id, cpu, frame_start, frame_end)
					stack_frame.depth = depth
					stack_frame.group = symbol
					stack_frame.parent_id = parent_frame_id

					entries.append(stack_frame)
					exits.append(stack_frame)

					if should_merge is True:
						merged_base_frames[identifier] = stack_frame

				else:

					base_frame = merged_base_frames[identifier]
					# I need to add the duration of the new frame to the one point at by base_frame_id

					base_frame.extra_duration_by_cpu[cpu] += (frame_end - frame_start)

					# I also need to add the correct parallelism interval to the base frame by parsing the combined tree...

			elif split_line[0] == "task_creation":
				continue # currently ignoring
				
				symbol_addr  = int(split_line[3])
				parent_task_id = int(split_line[4])
				new_task_id = int(split_line[5])
				timestamp = int(split_line[2])

				tcreate = Entity(EntityType.TASK_CREATION, new_task_id, cpu, timestamp, 0)
				tcreate.group = symbol_addr
				tcreate.parent_id = parent_task_id

				entries.append(tcreate)
			
			elif split_line[0] == "parallel_region":
				continue # currently ignoring
				
				if split_line[0] not in unique_groups:
					unique_groups.add(split_line[0])
				
				start = int(split_line[2])
				end = int(split_line[3])
				num_threads = int(split_line[4])
				
				pregion = Entity(EntityType.PARALLEL_REGION, None, cpu, start, end)
				pregion.pregion_num_threads = num_threads
				
				entries.append(pregion)
				exits.append(pregion)

			elif split_line[0] == "task_period":
				continue # currently ignoring
				
				if split_line[0] not in unique_groups:
					unique_groups.add(split_line[0])
				
				task_id = split_line[2]

				period_start = int(split_line[5])
				period_end = int(split_line[6])
				start = int(split_line[3])
				end = int(split_line[4])
				prior_task_id = int(split_line[7]) # currently unused
				openmp_task_id = int(split_line[8])

				if task_id in tasks:
					# This is a period of a frame we've seen before
					task_region = tasks[task_id]
					
					# create entry and exits
					task_period = Entity(EntityType.TASK_PERIOD, openmp_task_id, cpu, period_start, period_end)
					task_period.group = split_line[0]
					task_period.task_id = task_id

					entries.append(task_period)
					exits.append(task_period)

				else:
					task_region = Entity(EntityType.TASK_REGION, openmp_task_id, cpu, start, end+1) # add one cycle so we delete the task after we process its final period
					task_region.group = split_line[0]
					task_region.task_id = task_id
					
					tasks[task_id] = task_region
					entries.append(task_region)
					exits.append(task_region)

					# also make a period start/end for the first period of this task region
					task_period = Entity(EntityType.TASK_PERIOD, openmp_task_id, cpu, period_start, period_end)
					task_period.group = split_line[0]
					task_period.task_id = task_id
					task_period.parent_id = prior_task_id

					entries.append(task_period)
					exits.append(task_period)
			
			elif split_line[0] == "implicit_task":
				continue # currently ignoring

				if split_line[0] not in unique_groups:
					unique_groups.add(split_line[0])

				if  "omp_forked_execution" not in unique_groups:
					unique_groups.add("omp_forked_execution")

				start = int(split_line[2])
				end = int(split_line[3])
				task_region = Entity(EntityType.IMPLICIT_TASK_REGION, None, cpu, start, end)
				task_region.group = split_line[0]
				entries.append(task_region)
				exits.append(task_region)

			elif split_line[0] == "sync_region":
				
				if split_line[0] not in unique_groups:
					unique_groups.add(split_line[0])

				start = int(split_line[2])
				end = int(split_line[3])
				sync_region = Entity(EntityType.SYNC_REGION, None, cpu, start, end)
				sync_region.group = split_line[0]
				entries.append(sync_region)
				exits.append(sync_region)

			elif split_line[0] == "thread":
				continue # currently ignoring

				if split_line[0] not in unique_groups:
					unique_groups.add(split_line[0])

				start = int(split_line[2])
				end = int(split_line[3])
				thread = Entity(EntityType.THREAD, None, cpu, start, end)
				thread.group = split_line[0]
				entries.append(thread)
				exits.append(thread)
			
			elif split_line[0] == "work":

				if "omp_parallel_loop" not in unique_groups:
					unique_groups.add("omp_parallel_loop")

				start = int(split_line[2])
				end = int(split_line[3])
				work = Entity(EntityType.WORK, None, cpu, start, end)
				work.group = split_line[0]
				work.identifier = int(split_line[4]) # if 1 then it means it's a LOOP work callback
				work.count = int(split_line[5]) # number of threads in the work team
				entries.append(work)
				exits.append(work)

			else:
				logging.error("Cannot parse line: %s", line);
				raise ValueError()
	
	logging.debug("Sorting the entries and exits")

	entries.sort(key=lambda x: x.start)
	exits.sort(key=lambda x: x.end)

	logging.debug("Finished parsing the tracefile %s.", filename)

	return entries, exits, unique_groups

def get_next_entity(entries, exits, entry_idx, exit_idx):

	is_exit = None
	next_entity = None

	if entry_idx >= len(entries) and exit_idx >= len(exits):
		return None, None, entry_idx, exit_idx
	elif entry_idx >= len(entries):
		is_exit = True # i.e. it is an exit
		next_entity = exits[exit_idx]
		exit_idx += 1
	elif exit_idx >= len(exits):
		is_exit = False # i.e. it is an entry
		next_entity = entries[entry_idx]
		entry_idx += 1
	else:
		# which one is first?
		next_entry = entries[entry_idx]
		next_exit = exits[exit_idx]
		if next_entry.start < next_exit.end:
			is_exit = False
			next_entity = next_entry
			entry_idx += 1
		else:
			is_exit = True
			next_entity = exits[exit_idx]
			exit_idx += 1

	return is_exit, next_entity, entry_idx, exit_idx

def process_entry(entity, saved_call_stacks, current_call_stack_per_cpu, top_level_entities, max_depth):

	if entity.entity_type == EntityType.STACKFRAME:
		# find the correct callstack and push this frame to it
		# once I have the correct callstack, I simply add this new one as a child of the top of the callstack

		logging.trace("%d:wanting to push:%s:%s:%s", entity.cpu, entity.group, entity.entity_type, entity)

		call_stack = saved_call_stacks[current_call_stack_per_cpu[entity.cpu][-1]]
		if len(call_stack) == 0:
			top_level_entities.append(entity)
		else:
			entity.parent_entity = call_stack[-1]
			call_stack[-1].child_entities.append(entity)

		call_stack.append(entity)
		if len(call_stack) > max_depth[0]:
			max_depth[0] = len(call_stack)
		
		logging.trace("%d:pushed to stack:%s:%s:%s:%s", entity.cpu, current_call_stack_per_cpu[entity.cpu][-1], entity.group,entity.entity_type, entity)

	elif entity.entity_type == EntityType.SYNC_REGION:
		# (swap this CPU's callstack to the runtime callstack?)
		# TODO runtime-time (or synchronisation-time) should be recorded separately to work-time for each entity
		pass
	
	elif entity.entity_type == EntityType.TASK_REGION:
		# do nothing, let the changes to CPU state happen on task period
		pass
		
	elif entity.entity_type == EntityType.TASK_PERIOD:
		# I need to swap this CPU's state to the callstack corresponding to the task instance
		
		entered_task_id = entity.identifier
		if entered_task_id not in saved_call_stacks:
			logging.error("Could not find the task id %d in the saved call stacks when I entered it.", entered_task_id)
			raise ValueError()

		current_call_stack_per_cpu[entity.cpu].append(entered_task_id)
		
		logging.trace("%d:created new stack:%s:%s:%s:%s", entity.cpu, current_call_stack_per_cpu[entity.cpu][-1], entity.group, entity.entity_type, entity)
	
	elif entity.entity_type == EntityType.WORK:
		
		if entity.identifier != 1:
			logging.error("Currently only support loop work-callbacks. Cannot handle the work type: %s", entity.identifier)
			raise NotImplementedError()

		# if this CPU is not in parallel execution mode (i.e. the CPU's callstack is 'init')
		# then this should create a separate callstack for this CPU

		call_stack = None
		if len(current_call_stack_per_cpu[entity.cpu]) == 0:
			call_stack = saved_call_stacks["init"]
		else:
			call_stack = saved_call_stacks[current_call_stack_per_cpu[entity.cpu][-1]]

		# copy the current sequential call stack
		duplicate_call_stack = copy(call_stack)

		# create a CPU specific one so that CPU-specific callstacks are correctly handled
		nested_level = 0
		for stack_name in current_call_stack_per_cpu[entity.cpu]:
			if "omp_thread_" in stack_name:
				nested_level += 1

		saved_call_stacks["omp_thread_" + str(entity.cpu) + "_" + str(nested_level)] = duplicate_call_stack
		current_call_stack_per_cpu[entity.cpu].append("omp_thread_" + str(entity.cpu) + "_" + str(nested_level))
		
		logging.trace("%d:created new stack:%s:%s:%s:%s", entity.cpu, current_call_stack_per_cpu[entity.cpu][-1], entity.group, entity.entity_type, entity)

		if nested_level == 0:
			# append this entity as a pseudo call
			entity.group = "omp_parallel_loop"
			
			stack = saved_call_stacks[current_call_stack_per_cpu[entity.cpu][-1]]
			entity.parent_entity = stack[-1]
			stack[-1].child_entities.append(entity)

			stack.append(entity)
			if len(stack) > max_depth[0]:
				max_depth[0] = len(stack)

			logging.trace("%d:pushed new loop work to stack:%s:%s:%s:%s", entity.cpu, current_call_stack_per_cpu[entity.cpu][-1], entity.group, entity.entity_type, entity)

		# if the CPU is already in parallel execution, then this implicit task region is sequential so does not matter
		# assuming no nested parallel regions!
	
	elif entity.entity_type == EntityType.IMPLICIT_TASK_REGION:
		# currently doing nothing, for for-loops I'm conisdering the work itself to announce parallel execution
		# if the CPU is already in parallel execution, then this implicit task region is sequential so does not matter
		pass
	
	elif entity.entity_type == EntityType.THREAD:
		logging.trace("%d:new thread:%s:%s:%s:%s", entity.cpu, entity.group, entity.entity_type, entity)
		pass
	
	elif entity.entity_type == EntityType.PARALLEL_REGION:
		pass
	
	elif entity.entity_type == EntityType.TASK_CREATION:
		# create a new call stack within saved_call_stacks

		call_stack = saved_call_stacks[current_call_stack_per_cpu[entity.cpu][-1]]
		duplicate_call_stack = copy(call_stack) # these should refer to the same entities
		
		created_task_id = entity.identifier
		saved_call_stacks[created_task_id] = duplicate_call_stack
	
	else:
		logging.error("No parsing support for %s", entity.entity_type)
		raise NotImplementedError()

def process_exit(entity, saved_call_stacks, current_call_stack_per_cpu, top_level_entities, exits, exit_idx):

	if entity.entity_type == EntityType.STACKFRAME:
		# find the correct callstack and remove the top frame (they should be the same frame!)

		call_stack = saved_call_stacks[current_call_stack_per_cpu[entity.cpu][-1]]
		if len(call_stack) == 0:
			logging.error("Processing stack frame exit and there was no entity on top of stack,")
			raise ValueError()

		top_of_stack = call_stack[-1]
		if top_of_stack is not entity:
			logging.error("Processing stack frame exit and it was not top of stack.")
			raise ValueError()
	
		# each of these stacks has a separate roll back to replace the disconnected entity
		saved_rolled_back_entity_stacks = {} # first in last out

		# so we don't duplicate the same thing multiple times, save each one that we duplicate common across the stacks
		unique_rolled_back_entities = {}

		rolled_back_entities = False
		if not (entity.group == top_of_stack.group and entity.end == top_of_stack.end):

			rolled_back_entities = True
			# now find all the stacks that have the entity we have just left
			for stack_name in current_call_stack_per_cpu[entity.cpu]:

				temp_call_stack = saved_call_stacks[stack_name]
				if entity in temp_call_stack:

					rolled_back_entities = []

					roll_back_idx = -1
					while(True):
						if temp_call_stack[roll_back_idx] is entity:
							# this is gauranteed to happen at some point before roll_back_idx goes out of bounds
							# if it is already the top of this stack, do nothing
							# if we are processing replacements, we will pop it later
							break
						else:

							dup_identifier = temp_call_stack[roll_back_idx].group + "_" + str(temp_call_stack[roll_back_idx].end)

							if dup_identifier in unique_rolled_back_entities:
								rolled_back_entities.append(unique_rolled_back_entities[dup_identifier])
							else:
								# duplicate the entity and change it's end time to now
								
								duplicated_entity = copy(temp_call_stack[roll_back_idx])
								duplicated_entity.child_entities = []
								duplicated_entity.parent_entity = []
								duplicated_entity.parallelism_intervals = []
								duplicated_entity.start = entity.end

								temp_call_stack[roll_back_idx].end = entity.end

								rolled_back_entities.append(duplicated_entity)
								unique_rolled_back_entities[dup_identifier] = duplicated_entity

								logging.trace("Processing disconnect: looking for exit associated with %s:%s:%s",
									temp_call_stack[roll_back_idx].group,
									temp_call_stack[roll_back_idx].start,
									temp_call_stack[roll_back_idx].identifier)

								if temp_call_stack[roll_back_idx].group == "omp_forked_execution":
									# there is no exit to replace
									roll_back_idx -= 1
									continue
								
								roll_back_idx -= 1

								# go forward in the exits array and replace the original entity's exit with the duplicated one
								# TODO this is really inefficient, I should simply have a reference to the exit in the entry data structure
								tmp_exit_idx = exit_idx

								while(True):
									if exits[tmp_exit_idx] is temp_call_stack[-1]:
										exits[tmp_exit_idx] = duplicated_entity
										break
									else:
										tmp_exit_idx += 1

					saved_rolled_back_entity_stacks[stack_name] = rolled_back_entities

		# either the top of the stack was the entity, or we have rolled back some entities

		# first, we need to pop the entity
		if rolled_back_entities == False:
			call_stack.pop()
		else:
			# now remove the top of each stack where the entity is the top, then roll out the duplicate entities to replace the subtree
			for stack_name, duplicate_entities in saved_rolled_back_entity_stacks.items():
				stack = saved_call_stacks[stack_name]

				# remove the ended entity
				stack.pop() # this happens even if nothing was rolled back in this stack (i.e. the entity was already at the top of the call stack)
				
				# replace the subtree
				for duplicate_entity in reversed(duplicate_entities):
					
					if len(stack) == 0:
						top_level_entities.append(duplicate_entity)
					else:
						duplicate_entity.parent_entity = stack[-1]
						stack[-1].child_entities.append(duplicate_entity)

					stack.append(duplicate_entity)
					duplicate_entity.depth = len(stack)

		logging.trace("%d:popped from stack:%s:%s:%s:%s", entity.cpu, current_call_stack_per_cpu[entity.cpu][-1], entity.group, entity.entity_type, entity)

	elif entity.entity_type == EntityType.SYNC_REGION:

		pass
	
	elif entity.entity_type == EntityType.TASK_REGION:
		# if the region is over, I can remove the instance from the saved call stacks

		task_id = entity.identifier
		if task_id not in saved_call_stacks:
			logging.error("Could not find the task id %d in the saved call stacks when I exited the task.", task_id)
			raise ValueError()

		del saved_call_stacks[task_id]
		
	elif entity.entity_type == EntityType.TASK_PERIOD:
		# stop executing on this call stack by popping it (it still exists in saved_call_stacks to come back to later)

		left_task_id = entity.identifier
		if left_task_id not in saved_call_stacks:
			logging.error("Could not find the task id %d in the saved call stacks when I exited one of its task periods.", left_task_id)
			raise ValueError()

		if current_call_stack_per_cpu[entity.cpu][-1] != left_task_id:
			logging.error("Processing task exit and it was not top of stack.")
			raise ValueError()

		if len(current_call_stack_per_cpu[entity.cpu]) > 1:
			logging.trace("%d:deleted stack:%s:returning to stack:%s:%s:%s:%s",
			entity.cpu,
			current_call_stack_per_cpu[entity.cpu][-1],
			current_call_stack_per_cpu[entity.cpu][-2],
			entity.group,
			entity.entity_type,
			entity)
		else:
			logging.trace("%d:deleted stack:%s:returning to no active stack:%s:%s:%s",
			entity.cpu,
			current_call_stack_per_cpu[entity.cpu][-1],
			entity.group,
			entity.entity_type,
			entity)

		current_call_stack_per_cpu[entity.cpu].pop()
	
	elif entity.entity_type == EntityType.IMPLICIT_TASK_REGION:
		pass
	
	elif entity.entity_type == EntityType.WORK:

		logging.trace("%d:leaving work:%s:%s:%s:%s", entity.cpu, current_call_stack_per_cpu[entity.cpu][-1], entity.group, entity.entity_type, entity)

		# remove the omp_parallel_loop from the current stack
		call_stack = saved_call_stacks[current_call_stack_per_cpu[entity.cpu][-1]]
		if len(call_stack) == 0:
			logging.error("Exiting work entity, and there is no entity on top of stack,")
			raise ValueError()

		top_of_stack = call_stack[-1]

		if top_of_stack is not entity:
			logging.warning("Processing work entity exit, and it was not top of stack, so moving forward one position.")

			# if it is not top of the stack, just move this exit forward and go again
			exits[exit_idx-1] = exits[exit_idx]
			logging.debug("Set exits[%d] to %s", exit_idx-1, exits[exit_idx])
			exits[exit_idx] = entity
			logging.debug("Set exits[%d] to %s", exit_idx, entity)
			return 1
		
		# for OpenMP for-loops, once the work region is finished, we can kill the call stack

		nested_level = 0
		for stack_name in current_call_stack_per_cpu[entity.cpu]:
			if "omp_thread_" in stack_name:
				nested_level += 1

		nested_level -= 1 # we are removing the nested level before
		
		if nested_level == 0:
			call_stack.pop() # remove the parallel loop entity
		
		del saved_call_stacks["omp_thread_" + str(entity.cpu) + "_" + str(nested_level)]
		current_call_stack_per_cpu[entity.cpu].pop()
	
	elif entity.entity_type == EntityType.THREAD:
		pass
	
	elif entity.entity_type == EntityType.PARALLEL_REGION:
		pass
	
	else:
		logging.error("No parsing support for %s", entity.entity_type)
		raise NotImplementedError()

	return 0

def update_parallelism_intervals_for_cpu(
		is_start,
		entity,
		saved_call_stacks,
		current_call_stack_per_cpu,
		prior_parallelism,
		previous_processed_time_per_cpu,
		main_cpu):

	logging.trace("Updating intervals for a %s event.", ("start" if is_start else "end"))

	updated_processed_time = (entity.start if is_start else entity.end)
	previous_processed_time = previous_processed_time_per_cpu[entity.cpu]

	if previous_processed_time != -1:

		stack_ids = current_call_stack_per_cpu[entity.cpu]

		if len(stack_ids) > 0:
			call_stack_id = stack_ids[-1]
			call_stack = saved_call_stacks[call_stack_id]

			# Each CPU should have all of its entities assigned the interval
			interval = updated_processed_time - previous_processed_time

			top_of_stack = True
			for stack_entity in reversed(call_stack):
				stack_entity.add_parallelism_interval(prior_parallelism, interval, entity.cpu)

				if top_of_stack:
					#stack_entity.add_per_cpu_top_of_stack_parallelism_interval(prior_parallelism, interval, entity.cpu)
					stack_entity.add_top_of_stack_parallelism_interval(prior_parallelism, interval)
					#stack_entity.add_top_of_stack_interval(interval, entity.cpu)
					top_of_stack = False

				if "omp_parallel_loop" in stack_entity.group and entity.cpu != main_cpu:
					break

	updated_processed_times_per_cpu = previous_processed_time_per_cpu
	updated_processed_times_per_cpu[entity.cpu] = updated_processed_time

	return updated_processed_times_per_cpu

def update_parallelism_intervals_on_entry(
		entity,
		saved_call_stacks,
		current_call_stack_per_cpu,
		work_state_stack_per_cpu,
		prior_parallelism,
		previous_processed_times_per_cpu,
		main_cpu):

	# We don't need to update the parallelism intervals if the parallelism isn't changing

	updated_processed_times_per_cpu = previous_processed_times_per_cpu
	updated_parallelism = prior_parallelism

	if entity.entity_type == EntityType.STACKFRAME:

		# update the parallelism interval before we change the top of the stack
		updated_processed_times_per_cpu = update_parallelism_intervals_for_cpu(
			True,
			entity,
			saved_call_stacks,
			current_call_stack_per_cpu,
			prior_parallelism,
			previous_processed_times_per_cpu,
			main_cpu)

	elif entity.entity_type == EntityType.SYNC_REGION:

		# update the parallelism interval before we got to this region
		updated_processed_times_per_cpu = update_parallelism_intervals_for_cpu(
			True,
			entity,
			saved_call_stacks,
			current_call_stack_per_cpu,
			prior_parallelism,
			previous_processed_times_per_cpu,
			main_cpu)

		current_work_state = work_state_stack_per_cpu[entity.cpu][-1]
		work_state_stack_per_cpu[entity.cpu].append(False)
		
		if current_work_state == True:
			# going from work to non work
			logging.trace("%s:swap to not active parallelism:%s:%s:%s", entity.cpu, entity.group, entity.entity_type, entity)
			updated_parallelism -= 1
	
	elif entity.entity_type == EntityType.TASK_REGION:
		pass

	elif entity.entity_type == EntityType.TASK_PERIOD:
		# TODO
		pass
	
	elif entity.entity_type == EntityType.IMPLICIT_TASK_REGION:
		pass
		
	elif entity.entity_type == EntityType.WORK:

		# update the parallelism interval before we got to this region
		updated_processed_times_per_cpu = update_parallelism_intervals_for_cpu(
			True,
			entity,
			saved_call_stacks,
			current_call_stack_per_cpu,
			prior_parallelism,
			previous_processed_times_per_cpu,
			main_cpu)

		current_work_state = work_state_stack_per_cpu[entity.cpu][-1]
		work_state_stack_per_cpu[entity.cpu].append(True)
		
		if current_work_state == False:
			# going from work to non work
			logging.trace("%s:swap to active parallelism:%s:%s:%s", entity.cpu, entity.group, entity.entity_type, entity)
			updated_parallelism += 1

	elif entity.entity_type == EntityType.THREAD:
		pass

	elif entity.entity_type == EntityType.PARALLEL_REGION:
		pass
	
	elif entity.entity_type == EntityType.TASK_CREATION:
		pass
	
	else:
		logging.error("No parsing support for %s", entity.entity_type)
		raise NotImplementedError()

	return updated_parallelism, updated_processed_times_per_cpu

def update_parallelism_intervals_on_exit(
		entity,
		saved_call_stacks,
		current_call_stack_per_cpu,
		work_state_stack_per_cpu,
		prior_parallelism,
		previous_processed_times_per_cpu,
		main_cpu):

	updated_processed_time = previous_processed_times_per_cpu
	updated_parallelism = prior_parallelism

	if entity.entity_type == EntityType.STACKFRAME:

		# update the parallelism interval before we change the top of the stack
		updated_processed_times_per_cpu = update_parallelism_intervals_for_cpu(
			False,
			entity,
			saved_call_stacks,
			current_call_stack_per_cpu,
			prior_parallelism,
			previous_processed_times_per_cpu,
			main_cpu)
	
	elif entity.entity_type == EntityType.IMPLICIT_TASK_REGION:
		pass

	elif entity.entity_type == EntityType.WORK:
		
		#if it's not at top of stack, don't do anything
		call_stack = saved_call_stacks[current_call_stack_per_cpu[entity.cpu][-1]]
		if len(call_stack) == 0:
			logging.error("Processing exit of work, and there is no entity on top of stack")
			raise ValueError()

		top_of_stack = call_stack[-1]
		if top_of_stack is not entity:
			# Don't do anything because we aren't going to process this exit yet
			# This is to fix the weird thing where a WORK callback can finish BEFORE the stack frame exit that it encloses
			return prior_parallelism, previous_processed_times_per_cpu

		updated_processed_times_per_cpu = update_parallelism_intervals_for_cpu(
			False,
			entity,
			saved_call_stacks,
			current_call_stack_per_cpu,
			prior_parallelism,
			previous_processed_times_per_cpu,
			main_cpu)
		
		current_work_state = work_state_stack_per_cpu[entity.cpu].pop()
		if current_work_state == True:
			if work_state_stack_per_cpu[entity.cpu][-1] == True:
				pass # no change
			else:
				# going from work to non work
				logging.trace("%s:swap to not active parallelism:%s:%s:%s", entity.cpu, entity.group, entity.entity_type, entity)
				updated_parallelism -= 1
		else:
			if work_state_stack_per_cpu[entity.cpu][-1] == True:
				# going from non-work to work
				logging.trace("%s:swap to active parallelism:%s:%s:%s", entity.cpu, entity.group, entity.entity_type, entity)
				updated_parallelism += 1
			else:
				pass # no change

	elif entity.entity_type == EntityType.SYNC_REGION:
		
		updated_processed_times_per_cpu = update_parallelism_intervals_for_cpu(
			False,
			entity,
			saved_call_stacks,
			current_call_stack_per_cpu,
			prior_parallelism,
			previous_processed_times_per_cpu,
			main_cpu)
		
		current_work_state = work_state_stack_per_cpu[entity.cpu].pop()
		if current_work_state == True:
			if work_state_stack_per_cpu[entity.cpu][-1] == True:
				pass # no change
			else:
				# going from work to non work
				logging.trace("%s:swap to not active parallelism:%s:%s:%s", entity.cpu, entity.group, entity.entity_type, entity)
				updated_parallelism -= 1
		else:
			if work_state_stack_per_cpu[entity.cpu][-1] == True:
				# going from non-work to work
				logging.trace("%s:swap to active parallelism:%s:%s:%s", entity.cpu, entity.group, entity.entity_type, entity)
				updated_parallelism += 1
			else:
				pass # no change
	
	elif entity.entity_type == EntityType.TASK_REGION:
		pass
		
	elif entity.entity_type == EntityType.TASK_PERIOD:

		# if we are leaving a task period, we are stopping 
		updated_processed_times_per_cpu = update_parallelism_intervals_for_cpu(
			False,
			entity,
			saved_call_stacks,
			current_call_stack_per_cpu,
			prior_parallelism,
			previous_processed_times_per_cpu,
			main_cpu)

		current_work_state = work_state_stack_per_cpu[entity.cpu].pop()
		if current_work_state == True:
			if work_state_stack_per_cpu[entity.cpu][-1] == True:
				pass # no change
			else:
				# going from work to non work
				logging.trace("%s:swap to not active parallelism:%s:%s:%s", entity.cpu, entity.group, entity.entity_type, entity)
				updated_parallelism -= 1
		else:
			# TODO should this happen?? how can I be leaving a task period yet be in non-work??
			if work_state_stack_per_cpu[entity.cpu][-1] == True:
				# going from non-work to work
				logging.trace("%s:swap to active parallelism:%s:%s:%s", entity.cpu, entity.group, entity.entity_type, entity)
				updated_parallelism += 1
			else:
				pass # no change
	
	elif entity.entity_type == EntityType.IMPLICIT_TASK_REGION:
		
		updated_processed_times_per_cpu = update_parallelism_intervals_for_cpu(
			False,
			entity,
			saved_call_stacks,
			current_call_stack_per_cpu,
			prior_parallelism,
			previous_processed_times_per_cpu,
			main_cpu)
		
		current_work_state = work_state_stack_per_cpu[entity.cpu].pop()
		if current_work_state == True:
			if work_state_stack_per_cpu[entity.cpu][-1] == True:
				pass # no change
			else:
				# going from work to non work
				logging.trace("%s:swap to not active parallelism:%s:%s:%s", entity.cpu, entity.group, entity.entity_type, entity)
				updated_parallelism -= 1
		else:
			if work_state_stack_per_cpu[entity.cpu][-1] == True:
				# going from non-work to work
				logging.trace("%s:swap to active parallelism:%s:%s:%s", entity.cpu, entity.group, entity.entity_type, entity)
				updated_parallelism += 1
			else:
				pass # no change

	elif entity.entity_type == EntityType.THREAD:
		pass

	elif entity.entity_type == EntityType.PARALLEL_REGION:
		pass

	elif entity.entity_type == EntityType.TASK_CREATION:
		pass
	
	else:
		logging.error("No parsing support for %s", entity.entity_type)
		raise NotImplementedError()

	return updated_parallelism, updated_processed_times_per_cpu

"""
	This function reads the constructs from the trace file, and parses them to build a tree of related entities
	Each entitiy has intervals for its time-on-stack, time-on-top-of-stack, parallelism, and so on
"""
def process_events(filename):

	# parse all of the events into one list of entries/exits
	# sort the events by time
	# then iterate over each event, processing them according to their type

	work_tree, merged_leaf_parents, max_cpu, main_cpu, min_timestamp, max_timestamp = parse_trace_preprocessing(filename)

	entries, exits, unique_groups = parse_trace(filename, work_tree, merged_leaf_parents)

	# Every task must have its own callstack that it works on
	# When it creates a task, it copies its callstack over to the new task
	# All call stacks should just hold references to the same entities

	# If I have created a task which I am not executing, then I do some other stuff
	# And another CPU picks up that task, that other CPU should execute the task as a child of the function that created it

	logging.debug("Processing the constructs found in the tracefile.")
	logging.trace("The main CPU was: %d", main_cpu)

	saved_call_stacks = {}
	saved_call_stacks["init"] = []

	current_call_stack_per_cpu = {}
	work_state_stack_per_cpu = {}
	for i in range(max_cpu+1):
		if i == main_cpu:
			work_state_stack_per_cpu[i] = [True]
			current_call_stack_per_cpu[i] = ["init"] # only the main cpu has an initial call stack!
		else:
			work_state_stack_per_cpu[i] = [False]
			current_call_stack_per_cpu[i] = []

	current_parallelism = 1
	current_processed_times_per_cpu = [-1 for _ in range(max_cpu+1)]

	total_parallelism_entry_time = 0.0
	total_parallelism_exit_time = 0.0
	total_process_entry_time = 0.0
	total_process_exit_time = 0.0

	max_depth = [0]
	unique_cpus = []
	top_level_entities = []

	entry_idx = 0
	exit_idx = 0
	while(True):
	
		next_entity_is_exit, next_entity, entry_idx, exit_idx = get_next_entity(entries, exits, entry_idx, exit_idx)

		if next_entity is None:
			break

		if next_entity_is_exit == False:

			if next_entity.cpu not in unique_cpus:
				unique_cpus.append(next_entity.cpu)

			logging.trace("%d:%d:ENTRY:%s:%s:%s:task_identifier was %s and parent identifier was %s",
				next_entity.start,
				next_entity.cpu,
				next_entity.group,
				next_entity.entity_type,
				next_entity,
				next_entity.identifier,
				next_entity.parent_id)

			if debug_mode():
				t0 = time.time()

			current_parallelism, current_processed_time = update_parallelism_intervals_on_entry(next_entity,
				saved_call_stacks,
				current_call_stack_per_cpu,
				work_state_stack_per_cpu,
				current_parallelism,
				current_processed_times_per_cpu,
				main_cpu)

			if debug_mode():
				t1 = time.time()
				total_parallelism_entry_time += (t1 - t0)

			process_entry(next_entity,
				saved_call_stacks,
				current_call_stack_per_cpu,
				top_level_entities,
				max_depth)

			if debug_mode():
				t2 = time.time()
				total_process_entry_time += (t2 - t1)

		elif next_entity_is_exit == True:

			logging.trace("%d:%d:EXIT:%s:%s:%s:task_identifier was %s and parent identifier was %s",
				next_entity.start,
				next_entity.cpu,
				next_entity.group,
				next_entity.entity_type,
				next_entity,
				next_entity.identifier,
				next_entity.parent_id)

			if debug_mode():
				t0 = time.time()

			current_parallelism, current_processed_time = update_parallelism_intervals_on_exit(next_entity,
				saved_call_stacks,
				current_call_stack_per_cpu,
				work_state_stack_per_cpu,
				current_parallelism,
				current_processed_times_per_cpu,
				main_cpu)

			if debug_mode():
				t1 = time.time()
				total_parallelism_exit_time += (t1 - t0)

			error = process_exit(next_entity,
				saved_call_stacks,
				current_call_stack_per_cpu,
				top_level_entities,
				exits,
				exit_idx)

			if error == 1:
				exit_idx -= 1

			if debug_mode():
				t2 = time.time()
				total_process_exit_time += (t2 - t1)

	logging.debug("Finished processing the constructs found in the tracefile.")

	if debug_mode():
		logging.debug("Timings:")
		logging.debug("Total parallelism entry processing time: %f", total_parallelism_entry_time)
		logging.debug("Total parallelism exit processing time: %f", total_parallelism_exit_time)
		logging.debug("Total entry processing time: %f", total_process_entry_time)
		logging.debug("Total exit processing time: %f", total_process_exit_time)

	return top_level_entities, unique_groups, max_depth[0], min_timestamp, max_timestamp, unique_cpus

def count_function_calls(filename):

	function_counts = defaultdict(int)

	with open(filename, 'r') as f:
		for line in f:

			if line.strip() == "":
				continue

			split_line = line.strip().split(",")
			cpu = int(split_line[1])

			if split_line[0] == "stack_frame":

				frame_id = split_line[4]
				symbol = ",".join(split_line[7:])

				function_counts[symbol] += 1

	return function_counts
