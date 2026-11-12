#=
 * ==================================================================================
 * torcjulia — An MPI-based Runtime System for Asynchronous Task Parallelism in Julia
 * ==================================================================================
 * This file is part of torcjulia, an MPI-based runtime system for asynchronous
 * many task (AMT) parallelism in Julia across both shared and distributed memory
 * platforms, developed at the High Performance Computing Laboratory (HPCLAB)
 * at the Computer Engineering & Informatics Department (CEID),
 * University of Patras, Greece.
 *
 * Repository: https://github.com/CEID-HPCLAB/torcjulia
 *
 * Authors: Argiris Sofotasios (CEID-HPCLAB)
 * License: MIT License — see the LICENSE file in the repository root.
 * ==================================================================================
=#

module runtime

include("config.jl")    
using .config

include("logger.jl")    
import .logger:init_logger, set_logger_level

using MPI

using ThreadPinning
using ConcurrentCollections
using DataStructures

using Logging

using Match
using Printf
using Crayons

mutable struct MPIContext{T}
    comm::T
    rank::UInt
    num_procs::UInt
end

const ctx = Ref{MPIContext}()

# Number of Julia worker threads per MPI process
const TORCJULIA_NUM_WORKERS = Ref{UInt}(0)

const TORCJULIA_USE_MANAGER = Ref{Bool}(false)
const TORCJULIA_INIT_NODES = Ref{Int}(0)

const TORCJULIA_STEALING_ENABLED = Ref{Bool}(false)

# Polling interval between consecutive checks for incoming MPI messages by the manager thread
const TORCJULIA_MANAGER_YIELDTIME = Ref{Float64}(0.01)

# Delay before a worker thread retries task retrieval in the dispatch loop when no tasks were available in the previous iteration
const TORCJULIA_WORKER_YIELDTIME = Ref{Float64}(0.01)

@enum DistributionPolicy ROUND_ROBIN LOCAL RANDOM WRR

const TORCJULIA_DISTRIBUTION_POLICY = Ref{DistributionPolicy}()

const node_weights = Ref(Int[]); node_weights_cumsum =  Ref(Int[])

"""
    _set_policy(policy::DistributionPolicy, weights::Union{Vector{Int}, Nothing})

Sets the task scheduling policy used by the torcjulia runtime engine.

The `policy` argument specifies the scheduling policy and must be one of the following symbols:
- `:ROUND_ROBIN`: Distributes tasks across MPI nodes in a round-robin (cyclic) fashion.
- `:WRR`: Distributes tasks according to the specified weights, with one weight per MPI node.
- `:LOCAL`: Assigns each task to the MPI node that submitted it.
- `:RANDOM`: Assigns each task to a randomly selected MPI node.

The `weights` argument is used only when `policy` is `:WRR`. It must be a vector of integers, with each element specifying the weight of the
corresponding MPI node. For example, with four MPI nodes and weights `[1, 1, 2, 3]`, node 0 and node 1 will receive tasks in a 1:1 ratio, 
node 2 will receive tasks in a 2:1 ratio compared to nodes 0 and 1, and node 3 will receive tasks in a 3:1 ratio compared to nodes 0 and 1. 

For all other policies, `weights` is ignored.
"""
function _set_policy(policy::DistributionPolicy, weights::Union{Vector{Int}, Nothing})
    TORCJULIA_DISTRIBUTION_POLICY[] = policy
    
    if policy == WRR
        node_weights[] = weights
        node_weights_cumsum[] = cumsum(weights)
    end
end

"""
    _parse_policy(policy::Union{String, Symbol})::DistributionPolicy

Parses the specified task distribution `policy` and returns the corresponding `DistributionPolicy` enum value. If the provided policy is invalid,
the function falls back to `ROUND_ROBIN` and emits a warning.
"""
function _parse_policy(policy::Union{String, Symbol})::DistributionPolicy
    _policy = policy isa String ? uppercase(strip(policy)) : string(policy)
    
    for _cand_policy in instances(DistributionPolicy)
        if string(_cand_policy) == _policy
            return _cand_policy
        end
    end
    
    if ctx[].rank == 0
        _supported = join([join(uppercasefirst.(split(lowercase(replace(string(inst), "_" => "-")), "-")), "-") for inst in instances(DistributionPolicy)], ", ")
        @warn "Invalid task distribution policy: $policy -> falling back to Round-Robin (supported policies: $_supported)"
    end

    DistributionPolicy(0)
end

# Maps each Julia worker thread ID to its corresponding local ID within the torcjulia runtime system (1-based)
const torcjulia_tls_id = Int[]

const torcjulia_tls_curr_task = Vector{Ref{Dict{Symbol, Any}}}()

# shw_ptr => (shared array, MPI.Win object)
const torcjulia_shw = Dict{UInt, Tuple{Any, Any}}()

# local_shw_ptr => master_shw_ptr (pointer of the shared array on the node that created it)
const torcjulia_local_to_master = Dict{UInt, UInt}()

const torcjulia_queue = Vector{ConcurrentQueue{Union{Dict{Symbol, Any}, Nothing}}}()

const torcjulia_executed = Threads.Atomic{UInt}(0) 
const torcjulia_created = Threads.Atomic{UInt}(0)
const torcjulia_stole = Threads.Atomic{UInt}(0)
const torcjulia_steal_attempts = Threads.Atomic{UInt}(0)
const torcjulia_last_node = Threads.Atomic{UInt}(0)
const torcjulia_max_queue_depth = Threads.Atomic{UInt}(0)

const torcjulia_stolen = Ref{UInt}(0) 

# Event ensures that a notification is not lost if notify is called before wait
# Auto-reset ensures that each notification releases at most one waiting task
# This is required because the event is waited on twice (initialization and termination)
const manager_thread_lock = Threads.Event(true) 

const completed_tasks_lock = Threads.SpinLock()
const dependent_tasks_lock = Threads.SpinLock()

const shutdown_node_flag = Ref{Bool}(false)

function __init__()
    # Default logging level is `Logging.Debug` (users can change it via `set_logger_level(level::LogLevel)`)
    logger.init_logger(Logging.Debug)
 
    @debug "MPI module was successfully imported"
    MPI.Init_thread(MPI.THREAD_MULTIPLE)
    
    comm = MPI.Comm_dup(MPI.COMM_WORLD)
    ctx[] = MPIContext(comm, UInt(MPI.Comm_rank(comm)), UInt(MPI.Comm_size(comm)))
    
    # Configure work stealing via the `TORCJULIA_STEALING` env variable (default: disabled)
    TORCJULIA_STEALING_ENABLED[] = lowercase(Base.get(ENV, "TORCJULIA_STEALING", "false")) == "true"
    
    TORCJULIA_MANAGER_YIELDTIME[] = (arg = tryparse(Float64, Base.get(ENV, "TORCJULIA_MANAGER_YIELDTIME", "0.01"))) === nothing ? 0.01 : arg
    TORCJULIA_WORKER_YIELDTIME[] = (arg = tryparse(Float64, Base.get(ENV, "TORCJULIA_WORKER_YIELDTIME", "0.01"))) === nothing ? 0.01 : arg
    
    # Configure the task distribution policy via the `TORCJULIA_DISTRIBUTION_POLICY` env variable (default: ROUND_ROBIN)
    TORCJULIA_DISTRIBUTION_POLICY[] = _parse_policy(Base.get(ENV, "TORCJULIA_DISTRIBUTION_POLICY", "ROUND_ROBIN"))
 
    nthreads = Threads.nthreads()
    
    multiprocess = ctx[].num_procs > 1
    
    if multiprocess && nthreads < 2 && ctx[].rank == 0
        @error "torcjulia requires at least 2 threads (1 manager, 1 worker) per MPI process when running with multiple MPI processes"
        MPI.Abort(ctx[].comm, 1)
    end
    
    # If running with multiple MPI processes, reserve one thread for the manager and the rest for workers 
    # Otherwise, all threads are used as workers (signle-node mode)
    num_workers = multiprocess ? (nthreads - 1) : nthreads
    
    TORCJULIA_NUM_WORKERS[] = UInt(num_workers)
    TORCJULIA_USE_MANAGER[]  = multiprocess
    TORCJULIA_INIT_NODES[]  = Int(ctx[].num_procs)
 
    resize!(torcjulia_tls_id, nthreads); fill!(torcjulia_tls_id, -1)
 
    resize!(torcjulia_tls_curr_task, num_workers)
    
    for i in 1:num_workers
        torcjulia_tls_curr_task[i] = Ref{Dict{Symbol, Any}}()
    end
    
    # Thread-safe, multi-level (hierarchical) queue for low-priority (general-purpose) tasks
    for _ in 1:TORCJULIA_QUEUE_LEVELS
        push!(torcjulia_queue, ConcurrentQueue{Union{Dict{Symbol, Any}, Nothing}}())
    end
end

"""
    ConcurrentPriorityQueue{T}

Thread-safe priority queue. 

Wraps `DataStructures.PriorityQueue`.
"""
mutable struct ConcurrentPriorityQueue{T}
    queue::PriorityQueue{T, UInt}
    lock::Threads.SpinLock
    ConcurrentPriorityQueue{T}() where T = new(PriorityQueue{T, UInt}(Base.Order.Reverse), Threads.SpinLock())
end

"""
    Base.push!(priority_queue::ConcurrentPriorityQueue{T}, item::T, priority::UInt) where T

Adds a task `item` with the specified `priority` to the `priority_queue`.
"""
function Base.push!(priority_queue::ConcurrentPriorityQueue{T}, item::T, priority::UInt) where T
    lock(priority_queue.lock)
    try
        enqueue!(priority_queue.queue, item, priority)
    finally
        unlock(priority_queue.lock)
    end
end

"""
    _maybepopfirst!(priority_queue::ConcurrentPriorityQueue{T}) where T

Remove and return the highest-priority item, or nothing if `priority_queue` is empty.
"""
function _maybepopfirst!(priority_queue::ConcurrentPriorityQueue{T}) where T
    lock(priority_queue.lock)
    try
        return isempty(priority_queue.queue) ? nothing : Some(dequeue!(priority_queue.queue))
    finally
        unlock(priority_queue.lock)
    end
end

"""
    Base.isempty(priority_queue::ConcurrentPriorityQueue)

Returns `true` if the `priority_queue` is empty, `false` otherwise.
"""
Base.isempty(priority_queue::ConcurrentPriorityQueue) = begin
    lock(priority_queue.lock)
    
    try
        return isempty(priority_queue.queue)
    finally
        unlock(priority_queue.lock)
    end
end

const torcjulia_priority_queue = ConcurrentPriorityQueue{Dict{Symbol, Any}}()

"""
    _torcjulia_tls_get_id()::Int

Returns the local ID of the calling Julia worker thread within its MPI process (1-based).
"""
@inline function _torcjulia_tls_get_id()
    torcjulia_tls_id[Threads.threadid()]
end

"""
    _torcjulia_tls_set_id(id::Int)

Sets the local ID of the calling Julia worker thread within its MPI process (1-based).
"""
@inline function _torcjulia_tls_set_id(id::Int)
    torcjulia_tls_id[Threads.threadid()] = id
end

"""
    _torcjulia_tls_get_curr_task()::Dict{Symbol, Any}

Returns the descriptor of the current task of the calling Julia worker thread.
"""
@inline function _torcjulia_tls_get_curr_task()::Dict{Symbol, Any}
    torcjulia_tls_curr_task[Threads.threadid()][]
end

"""
    _torcjulia_tls_set_curr_task(task::Dict{Symbol, Any})

Sets the current task descriptor for the calling Julia worker thread.
"""
@inline function _torcjulia_tls_set_curr_task(task::Dict{Symbol, Any})
    torcjulia_tls_curr_task[Threads.threadid()][] = task
end

mutable struct Future
    desc::Dict{Symbol,Any}   
end

"""
    input(task::Future)

Return the positional arguments of the task function stored in the descriptor of `task`.
"""
@inline input(task::Future) = task.desc[:args]

"""
    kw_input(task::Future)

Return the keyword arguments of the task function stored in the descriptor of `task`.
"""
@inline kw_input(task::Future) = task.desc[:kwargs]

"""
    result(task::Future)

Return the output of the task function stored in the descriptor of `task`.
"""
@inline result(task::Future) = task.desc[:out]

@enum TaskState pending running completed

"""
    TaskPriorityLevel

Maps a task priority level to a corresponding integer value.
"""
struct TaskPriorityLevel
    level::Symbol
    val::UInt
end

const TaskPriority = (
    low = TaskPriorityLevel(:low, UInt(1)),
    medium = TaskPriorityLevel(:medium, UInt(2)),
    high = TaskPriorityLevel(:high, UInt(3))
)

const node_created_tasks = ConcurrentQueue{Future}()
const node_tasks_pending_time = ConcurrentQueue{UInt}()
const node_tasks_execution_time = ConcurrentQueue{UInt}()

"""
    init()

Initializes the torcjulia AMT runtime system.

If launching with multiple MPI processes, the last thread of each MPI process is reserved for the manager thread.
"""
function init()
    if MPI.Comm_get_parent() != MPI.COMM_NULL
        parent_comm = MPI.Comm_get_parent()
        
        ctx[].comm = MPI.Intercomm_merge(parent_comm, true)
        
        ctx[].rank = MPI.Comm_rank(ctx[].comm) 
        ctx[].num_procs = MPI.Comm_size(ctx[].comm) 
        
        # Handle the case where only one new MPI process is requested via `add_worker`
        # in `__init__()`, `ctx[].num_procs == 1`, so TORCJULIA_USE_MANAGER[] is set to `false` 
        # and the manager thread is not spawned => deadlock
        if !TORCJULIA_USE_MANAGER[]
            TORCJULIA_NUM_WORKERS[] -= 1
            TORCJULIA_USE_MANAGER[] = true
        end
    end

    if ctx[].rank == 0
        provided = MPI.Query_thread()
        
        @warn "MPI.Query_thread returns $(parse(Int, match(r"\d+", string(provided)).match))"

        if provided < MPI.THREAD_MULTIPLE
            @warn "Warning: MPI.Query_thread returns $provided < $(MPI.THREAD_MULTIPLE)"
        else
            @warn "Info: MPI.Query_thread returns MPI.THREAD_MULTIPLE"
        end

        # Task descriptor for the main (i.e., top-level) task (serves as the primary application task)
        main_task = Dict{Symbol, Any}(
            :deps   => Threads.Atomic{UInt}(0),
            :parent => UInt(0),
            :level  => UInt(0),
            :completed => Vector{Dict{Symbol, Any}}(),
            :state => running
        )   

        main_task[:mytask] = UInt(pointer_from_objref(main_task))
        
        _torcjulia_tls_set_id(1); _torcjulia_tls_set_curr_task(main_task)

        global task_execution_stats = Float64[0, 0, 0, 0]
    end

    if TORCJULIA_USE_MANAGER[]
        ThreadPinning.@spawnat Threads.nthreads() manager_thread()
        
        # Wait for the manager thread to signal that it has started before proceeding with the initialization of the worker threads
        Base.wait(manager_thread_lock)
    end

    Threads.atomic_xchg!(torcjulia_last_node, ctx[].rank * TORCJULIA_NUM_WORKERS[])
end

"""
    launch(func::Function)

Launches the application's main function `func` (i.e., the top-level task) on the main thread of the 
master node (MPI rank 0) and starts worker threads to handle incoming tasks.
"""
function launch(func::Function)
    roles = Vector{Function}(undef, TORCJULIA_NUM_WORKERS[])
    
    # For the master node (MPI rank 0), roles[1] corresponds to the main thread on the master node (rank 0) 
    # and executes the application's main function `func` (i.e., the top-level task) 
    roles[1] = () -> (ctx[].rank == 0 ? (func(); _finalize()) : worker(1))
    
    for i in 2:Int(TORCJULIA_NUM_WORKERS[])
        roles[i] = () -> (worker(i))
    end

    ctx[].rank != 0 || println(Crayon(foreground = :magenta)("torcjulia: main starts"))

    if !shutdown_node_flag[]
        MPI.Barrier(ctx[].comm)
    end

    # Launch the worker threads
    Threads.@threads :static for i in 1:TORCJULIA_NUM_WORKERS[]
        roles[i]()
    end

    # If the node is shutting down via an explicit call to remote_worker(), it must leave the communicator before exiting
    shutdown_node_flag[] ? MPI.Comm_split(ctx[].comm, 1, ctx[].rank) : _remote_workers_sync()
    
    # All worker threads have completed execution, so the main thread on the master node 
    # can now initiate the termination of the manager threads.
    ctx[].rank == 0 && (task = _torcjulia_tls_get_curr_task(); task[:state] = completed; _terminate_manager_threads())
end

"""
    submit(func::Function, args...; node::Int = -1, callback::Union{Function, Nothing} = nothing, async_callback::Bool = true, 
           counted::Bool = true, forced_node::Bool = false, priority::TaskPriorityLevel = TaskPriority.low, 
           distribution_policy::Union{Function, Tuple, Symbol, Nothing} = nothing, kwargs...)
    
Creates and submits a task for execution. 
    
Returns a `Future` object that can be used to retrieve the result of the task once it has completed.

- `func`: The task function to be executed.
- `args...`: The positional arguments to be passed to the function.
- `node`: The MPI node (0-based) to which the task should be assigned. If not specified, the task is assigned based on the current distribution policy.
- `callback`: A function to be called when the task completes.
- `async_callback`: If true, the callback is executed asynchronously (Default: true).
- `counted`: If true, the task is counted in the total number of tasks (Default: true).
- `forced_node`: If true, the task is pinned to the specified node (Default: false). 
                 If false, the task can be stolen by worker threads on other nodes if work stealing is enabled.
- `priority`: The priority level of the task. Can be `TaskPriority.low`, `TaskPriority.medium`, or `TaskPriority.high` (Default: `TaskPriority.low`). 
- `distribution_policy`: If specified, this overrides both the `node` argument and the default task distribution policy for this specific task. 
                         It can be a function, a tuple (function, args...), or a symbol representing the desired distribution policy.
"""
function submit(
    func::Function, 
    args...; 
    node::Int = -1, 
    callback::Union{Function, Nothing} = nothing, 
    async_callback::Bool = true, 
    counted::Bool = true, 
    forced_node::Bool = false, 
    priority::TaskPriorityLevel = TaskPriority.low, 
    distribution_policy::Union{Function, Tuple, Symbol, Nothing} = nothing, 
    kwargs...)
    
    # If the user specifies a node value, it must be within the valid range of MPI ranks (0 to num_procs - 1)
    if node != -1 && (node < 0 || node >= ctx[].num_procs)
        @error "submit: invalid node value ($node) - node must be between 0 and $(ctx[].num_procs - 1)"
        throw(ArgumentError("submit: invalid node value ($node)"))
    end

    node = !isnothing(distribution_policy) ? _assign_task(distribution_policy) : node
    
    # If MPI shared-memory windows are passed as arguments, replace them with their corresponding pointers
    _args, _kwargs = !isempty(torcjulia_shw) ? (_parse_shw_args(args...; kwargs...)) : (args, NamedTuple(kwargs))

    task = create_task(func, callback, async_callback, counted, forced_node, priority, _args...; _kwargs...)

    task[:mytask] = UInt(pointer_from_objref(task))
    
    task[:counted] ? Threads.atomic_max!(torcjulia_max_queue_depth, task[:level]) : nothing

    # If the user did not specify a node value, assign the task to a node based on the current distribution policy
    if node == -1
       node = _assign_task(TORCJULIA_DISTRIBUTION_POLICY[]); task[:forced_node] = false
    end
    
    task[:cbtask] = !isnothing(callback) ? create_cb_task(callback, counted, priority, task[:level]) : nothing

    counted ? Threads.atomic_add!(torcjulia_created, UInt(1 + (!isnothing(callback) ? 1 : 0))) : nothing

    task_f = Future(task); task[:myfuture] = UInt(pointer_from_objref(task_f))
    
    push!(node_created_tasks, task_f)

    # Identify whether the task has data dependencies to enforce execution order
    _has_dependencies = _capture_args_deps(task)

    # If the task has no dependencies, enqueue it for execution on the assigned node
    # Otherwise, it will be enqueued for execution once its dependencies are resolved
    if !_has_dependencies
        ctx[].rank == node ? enqueue(task[:level], task) : begin
            task[:type] = "enqueue"
            MPI.send(task, ctx[].comm; dest = node, tag = TORCJULIA_MANAGER_TAG)
        end
    else
        task[:node] = node
    end

    task_f
end

"""
    create_task(f::Function, callback::Union{Function, Nothing}, async_callback::Bool, counted::Bool, 
                forced_node::Bool, priority::TaskPriorityLevel, args...; kwargs...)::Dict{Symbol, Any}

Creates a task descriptor for the specified function `f` with the given parameters and arguments.
"""
function create_task(
    f::Function, 
    callback::Union{Function, Nothing}, 
    async_callback::Bool, 
    counted::Bool, 
    forced_node::Bool, 
    priority::TaskPriorityLevel, 
    args...; 
    kwargs...)::Dict{Symbol, Any}
    
    parent = _torcjulia_tls_get_curr_task()
    Threads.atomic_add!(parent[:deps], UInt(1)) 

    Dict(
        :f => f,
        :callback => callback,
        :async_callback => async_callback,
        :counted => counted,
        :deps => Threads.Atomic{UInt}(0),
        :completed => Vector{Dict{Symbol, Any}}(),
        :dependent => Vector{Dict{Symbol, Any}}(),
        :homenode => ctx[].rank,
        :parent => UInt(pointer_from_objref(parent))  ,
        :level => min(parent[:level] + 1, TORCJULIA_QUEUE_LEVELS),
        :state => pending,
        :forced_node => forced_node ? true : false,
        :priority => priority,
        :submit_t => time_ns(),
        :out => nothing,
        :args => args,
        :kwargs => kwargs,
    )
end

"""
    create_cb_task(f::Function, counted::Bool, priority::TaskPriorityLevel, level::UInt)::Dict{Symbol, Any}

Creates a descriptor for the callback of the specified function `f` with the given parameters.
"""
function create_cb_task(
    f::Function, 
    counted::Bool, 
    priority::TaskPriorityLevel, 
    level::UInt)::Dict{Symbol, Any}

    parent = _torcjulia_tls_get_curr_task()
    Threads.atomic_add!(parent[:deps], UInt(1)) 

    Dict(
        :f => f,
        :callback => nothing,
        :counted => counted,
        :deps => Threads.Atomic{UInt}(0),
        :parent => UInt(pointer_from_objref(parent)),
        :completed => Vector{Dict{Symbol, Any}}(),
        :dependent => Vector{Dict{Symbol, Any}}(),
        :homenode => ctx[].rank,
        :level => level,
        :state => pending,
        :forced_node => true,
        :priority => priority,
        :out => nothing,
        :args => (),
        :kwargs => NamedTuple()
    )   
end

"""
    _assign_task(policy::DistributionPolicy)::Int

Assigns a task to an MPI node according to the specified `policy`.

Supported policies are: ROUND_ROBIN, LOCAL, WRR, and RANDOM.
"""
function _assign_task(policy::DistributionPolicy)::Int
    if policy == ROUND_ROBIN
        _node = Threads.atomic_add!(torcjulia_last_node, UInt(1)) % (ctx[].num_procs * TORCJULIA_NUM_WORKERS[])
        
        return trunc(UInt, _node / TORCJULIA_NUM_WORKERS[])
    end

    if policy == WRR
        if isempty(node_weights[])
            @warn "Weighted Round-Robin (WRR) requires weights to be set beforehand -> falling back to Round-Robin"
            
            _set_scheduling_policy(:ROUND_ROBIN)
            
            _node = Threads.atomic_add!(torcjulia_last_node, UInt(1)) % (ctx[].num_procs * TORCJULIA_NUM_WORKERS[])
            
            return trunc(UInt, _node / TORCJULIA_NUM_WORKERS[])
        end

        _node = Threads.atomic_add!(torcjulia_last_node, UInt(1)) % (node_weights_cumsum[][end])
        _node_idx = searchsortedfirst(node_weights_cumsum[], _node + 1)

        return UInt(_node_idx - 1)
    end
    
    if policy == LOCAL
        return ctx[].rank
    end

    # policy == RANDOM
    rand(0:ctx[].num_procs - 1)
end

"""
    _assign_task(policy::Symbol)::Int

Wraps `_assign_task(policy::DistributionPolicy)` to allow users to specify the task distribution policy 
as a symbol (e.g., `:ROUND_ROBIN`, `:LOCAL`, `:WRR`, or `:RANDOM`).
"""
function _assign_task(policy::Symbol)::Int
    policy = _parse_policy(policy)
    
    _assign_task(policy)
end

"""
    _assign_task(policy::Function)::Int

`policy`: A zero-argument function that returns an integer representing the MPI node (0-based) to which the task should be assigned.
"""
function _assign_task(policy::Function)::Int
    node = policy()
    
    if node isa Integer && 0 <= node < ctx[].num_procs
        return node
    end
    
    @warn "submit: task assignment failed - provided distribution policy returned invalid value: $node (must be an integer between 0 and $(ctx[].num_procs - 1) -> task will be reassigned"
    
    # Falls back to the default Round-Robin policy if the returned value from the user-defined function is invalid
    _assign_task(ROUND_ROBIN)
end

"""
    _assign_task(policy::Tuple)::Int

`policy`: A tuple containing a scheduling function and its arguments. 
          The function should return an integer representing the MPI node (0-based) to which the task should be assigned.
"""
function _assign_task(policy::Tuple)::Int
    sched, args = policy

    if !isa(sched, Function)
        @warn "submit: task assignment failed - provided distribution policy is not a function: $sched -> task will be reassigned"
        return _assign_task(ROUND_ROBIN)
    end

    node = try
        sched(args...)
    catch err
        @warn "submit: task assignment failed - distribution policy threw an exception: $err -> task will be reassigned"

        return _assign_task(ROUND_ROBIN)
    end
    
    # Falls back to the default Round-Robin policy if the returned value from the user-defined function is invalid
    if node isa Integer && 0 <= node < ctx[].num_procs
        return node
    end

    @warn "submit: task assignment failed - provided distribution policy returned invalid value: $node (must be an integer between 0 and $(ctx[].num_procs - 1)) -> task will be reassigned"

    _assign_task(ROUND_ROBIN)
end

"""
    worker(torcjulia_id::Int)

Dispatch loop of a worker thread (work-first, help-then policy).
    
The worker thread continuously retrieves and executes tasks from the local 
queues (i.e., queues containing tasks assigned to its MPI process) (work sharing). 

If no tasks are available locally and work stealing is enabled, 
it attempts to steal tasks from the workpile of other MPI processes (work stealing).
"""
function worker(torcjulia_id::Int)
    _torcjulia_tls_set_id(torcjulia_id)
    
    while true
        task = dequeue()
        
        # If `task` is `nothing`, the worker thread should terminate,
        # typically in response to a shutdown signal from the manager thread
        isnothing(task) ? break : nothing

        if isempty(task)
            if ctx[].num_procs > 1 && TORCJULIA_STEALING_ENABLED[]
                task = steal()
                Threads.atomic_add!(torcjulia_steal_attempts, UInt(1))
                
                if task[:type] == "nowork"
                    sleep(2 * TORCJULIA_WORKER_YIELDTIME[])
                    continue
                else
                    task[:counted] ? Threads.atomic_add!(torcjulia_stole, UInt(1)) : nothing
                end
            
            else
                sleep(TORCJULIA_WORKER_YIELDTIME[])
                continue
            end
        end

        ctx[].num_procs > 1 ? sleep(TORCJULIA_TASK_YIELDTIME) : nothing
        
        task[:start_t] = time_ns()
        
        work(task)
    end
end

"""
    work(task::Dict{Symbol, Any})

Executes the specified `task` and handles its completion.
"""
function work(task::Dict{Symbol, Any})
    _torcjulia_tls_set_curr_task(task)

    task[:state] = running; func = task[:f]; args = task[:args]; kwargs = task[:kwargs]
    
    # Restore arguments containing pointers to MPI shared-memory windows
    # to their original shared-array values
    _args, _kwargs = !isempty(torcjulia_shw) ? (_restore_shw_args(args...; kwargs...)) : (args, NamedTuple(kwargs))

    y = func(_args...;  _kwargs...)

    task[:out] = y
    
    task[:counted] ? Threads.atomic_add!(torcjulia_executed, UInt(1)) : nothing
    task[:end_t] = time_ns()

    handle_completed_task(task)
end

"""
    handle_completed_task(task::Dict{Symbol, Any})

Handles the completion of the specified `task`.

If the task was executed on a different MPI process than the one where it was submitted, 
forward its result and execution metadata to the manager thread of the original process.

Update the parent task and its dependent tasks. Decrement each dependent task's dependency counter and, 
when its dependency counter reaches zero, enqueue it for execution on its designated MPI process.

If the `task` has a callback, initiate its execution on the submitting MPI process, 
according to the `async_callback` flag.
"""
function handle_completed_task(task::Dict{Symbol, Any})
    if ctx[].rank == task[:homenode]
        is_stolen = (Base.get(task, :type, nothing) == "stolen")

        # Handle the case where a task submitted by the current MPI process is assigned
        # to another process but is subsequently stolen back by its home process
        if is_stolen
            native_task = unsafe_pointer_to_objref(Ptr{UInt}(task[:mytask]))
            
            native_task[:out] = task[:out]
            native_task[:start_t] = task[:start_t]; native_task[:end_t] = task[:end_t]
            
            parent = unsafe_pointer_to_objref(Ptr{UInt}(native_task[:parent]))

            Threads.atomic_sub!(parent[:deps], UInt(1))  
            lock(completed_tasks_lock); push!(parent[:completed], native_task); unlock(completed_tasks_lock)

            _capture_time_stats(native_task[:submit_t], native_task[:start_t], native_task[:end_t])
            
            !isnothing(native_task[:callback]) ? exec_task_callback(native_task) : nothing

            native_task[:state] = completed

            for task in native_task[:dependent]
                Threads.atomic_sub!(task[:deps], UInt(1))

                if task[:deps][] == 0
                    ctx[].rank == task[:node] ? enqueue(task[:level], task) : begin
                                task[:type] = "enqueue"; MPI.send(task, ctx[].comm; dest = task[:node], tag = TORCJULIA_MANAGER_TAG)
                            end
                end
            end
        
        else
            parent = unsafe_pointer_to_objref(Ptr{UInt}(task[:parent])) 

            Threads.atomic_sub!(parent[:deps], UInt(1))
            lock(completed_tasks_lock); push!(parent[:completed], task); unlock(completed_tasks_lock)

            _capture_time_stats(task[:submit_t], task[:start_t], task[:end_t])
            
            !isnothing(task[:callback]) ? exec_task_callback(task) : nothing

            lock(completed_tasks_lock); task[:state] = completed; unlock(completed_tasks_lock)

            for task in task[:dependent]
                Threads.atomic_sub!(task[:deps], UInt(1))

                if task[:deps][] == 0
                    ctx[].rank == task[:node] ? enqueue(task[:level], task) : begin
                                task[:type] = "enqueue"; MPI.send(task, ctx[].comm; dest = task[:node], tag = TORCJULIA_MANAGER_TAG)
                            end
                end
            end
        end

    else
        task[:type] = "answer"; dest = task[:homenode]
        
        Base.filter!(((k,_),) -> k in (:type, :mytask, :out, :start_t, :end_t), task)
        
        MPI.send(task, ctx[].comm; dest = dest, tag = TORCJULIA_MANAGER_TAG)
    end
end

"""
    exec_task_callback(task::Dict{Symbol, Any})

Executes the callback function associated with the specified `task` if synchronous, 
or enqueues it for later execution if asynchronous.
"""
function exec_task_callback(task::Dict{Symbol, Any})
    cb_task = task[:cbtask]
    
    cb_task[:args] = task[:out]
    cb_task[:submit_t] = time_ns()

    task[:async_callback] ? enqueue(cb_task[:level], cb_task) : begin
        cb_task[:start_t] = time_ns()
        cb_task[:f](cb_task[:args])
        cb_task[:end_t] = time_ns()
        
        parent = unsafe_pointer_to_objref(Ptr{UInt}(cb_task[:parent]))
        Threads.atomic_sub!(parent[:deps], UInt(1)) 
        
        task[:counted] ? Threads.atomic_add!(torcjulia_executed, UInt(1)) : nothing
        
        _capture_time_stats(cb_task[:submit_t], cb_task[:start_t], cb_task[:end_t])
    end

    nothing
end

"""
    steal()::Dict{Symbol, Any}

Attempts to steal a task from remote queues (i.e., queues containing tasks assigned to other MPI processes).

The worker thread sends a task stealing request to the manager thread of the next MPI rank. 

If the request yields no work, the worker thread proceeds to the next rank and repeats the 
process until a pending task is obtained, or all ranks have been queried.
"""
function steal()::Dict{Symbol, Any}
    task = Dict(:type => "nowork"); request = Dict(:type => "steal")
    
    for node in Base.filter(x -> x != ctx[].rank, 0:ctx[].num_procs - 1)
        torcjulia_terminate_flag && return task

        MPI.send(request, ctx[].comm; dest = node, tag = TORCJULIA_MANAGER_TAG)

        flag = false
        
        while !flag
            flag = MPI.Iprobe(ctx[].comm; source = node, tag = TORCJULIA_STEAL_RESPONSE_TAG)
            flag || sleep(TORCJULIA_WORKER_YIELDTIME[])
        end

        response = MPI.recv(ctx[].comm; source = node, tag = TORCJULIA_STEAL_RESPONSE_TAG)

        response[:type] == "nowork" || return response
    end
    
    task
end

"""
    manager_thread()

Main loop of the manager thread, processing incoming MPI requests from other nodes.

Handles all remote task-coordination operations, such as serving task-stealing requests,
enqueuing incoming tasks, processing task-completion messages, and managing node shutdown.
"""
function manager_thread()
    _torcjulia_tls_set_id(Threads.nthreads())
    
    global torcjulia_terminate_flag = false
    
    notify(manager_thread_lock)
    
    while true
        flag, status = false, nothing
        
        while !flag
            flag, status = MPI.Iprobe(ctx[].comm, MPI.Status; source = MPI.ANY_SOURCE, tag = TORCJULIA_MANAGER_TAG)
            flag || sleep(TORCJULIA_MANAGER_YIELDTIME[])
        end
        
        task = MPI.recv(ctx[].comm; source = status.source, tag = TORCJULIA_MANAGER_TAG)

        @match task[:type] begin 
            "exit" => begin
                # Notify the master's node manager thread to terminate
                ctx[].rank == 1 && _notify_master_manager_thread()
                
                break
            end

            "terminate" => begin
                global torcjulia_terminate_flag = true

                # Notify all worker threads to terminate, enqueueing TORCJULIA_NUM_WORKERS[] termination signals
                foreach(_ -> enqueue(UInt(1), nothing), 1:TORCJULIA_NUM_WORKERS[])
            end

            "enqueue" => begin
                enqueue(task[:level], task)
            end

            "answer" => begin
                native_task = unsafe_pointer_to_objref(Ptr{UInt}(task[:mytask]))
                
                native_task[:out] = task[:out]
                native_task[:state] = completed
                
                native_task[:start_t] = task[:start_t]; native_task[:end_t] = task[:end_t]
                
                parent = unsafe_pointer_to_objref(Ptr{UInt}(native_task[:parent]))
                
                Threads.atomic_sub!(parent[:deps], UInt(1))  
                lock(completed_tasks_lock); push!(parent[:completed], native_task); unlock(completed_tasks_lock)

                _capture_time_stats(native_task[:submit_t], native_task[:start_t], native_task[:end_t])

                !isnothing(native_task[:callback]) ? exec_task_callback(native_task) : nothing

                # If the task has dependent tasks, decrement their dependency counters and enqueue them for execution if their counters reach zero
                for task in native_task[:dependent]
                    Threads.atomic_sub!(task[:deps], UInt(1))
                    
                    if task[:deps][] == 0
                        ctx[].rank == task[:node] ? enqueue(task[:level], task) : begin
                                    task[:type] = "enqueue"; MPI.send(task, ctx[].comm; dest = task[:node], tag = TORCJULIA_MANAGER_TAG)
                                end
                    end
                end
            end

            "steal" => begin
                task = dequeue_steal()
                
                # If no task is available for stealing, send a "nowork" response to the requesting MPI process
                # `task` === nothing indicates that a termination sentinel has been dequeued from the local workpile
                # This can occur during shutdown if a worker thread on a remote MPI process issues a task stealing request
                # after this process has begun shutting down and enqueued termination signals, 
                # but before the remote worker has received its own termination sentinel
                # The dequeued sentinel must be re-enqueued to ensure that the local worker thread for which it was intended
                # can eventually retrieve it 
                # Otherwise, that worker thread may block indefinitely, resulting in a deadlock during shutdown
                t = isnothing(task) ? (enqueue(UInt(1), nothing); Dict(:type => "nowork")) : begin
                    task[:type] = isempty(task) ? "nowork" : "stolen"
                    
                    if task[:type] == "stolen" && task[:counted]
                        torcjulia_stolen[] += 1
                    end
                    
                    task
                end

                MPI.send(t, ctx[].comm; dest = status.source, tag = TORCJULIA_STEAL_RESPONSE_TAG)
            end

            "enable_stealing" => begin
                TORCJULIA_STEALING_ENABLED[] = true
            end

            "disable_stealing" => begin
                TORCJULIA_STEALING_ENABLED[] = false
            end
            
            # Handle shutdown initiated by an explicit call to remove_worker()
            "remove" => begin
                shutdown_node_flag[] = true
                
                foreach(_ -> enqueue(UInt(1), nothing), 1:TORCJULIA_NUM_WORKERS[])
                
                break
            end

            _ => begin
                @warn "Unknown task type: $(task[:type])"
            end
        end
    end

    notify(manager_thread_lock)
end

"""
    enqueue(level::UInt, task::Union{Dict{Symbol, Any}, Nothing})

Enqueues a task into the appropriate queue based on its priority.

`level` specifies the task's parallelism level, i.e., its depth in the task dependency tree.

Tasks with low priority are enqueued in the corresponding `level` of `torcjulia_queue`, whereas tasks with 
medium or high priority are enqueued in `torcjulia_priority_queue`.

Tasks are assigned low priority by default (i.e., general-purpose tasks), 
unless an explicit priority is specified when the task is submitted.
"""
@inline function enqueue(level::UInt, task::Union{Dict{Symbol, Any}, Nothing})
    if isnothing(task) || (haskey(task, :priority) && task[:priority] == TaskPriority.low)
        push!(torcjulia_queue[level], task)
    
    else
        push!(torcjulia_priority_queue, task, task[:priority].val)
    end
    
    nothing
end

"""
    dequeue()::Union{Dict{Symbol, Any}, Nothing}

Dequeues the next available task from the local queues, i.e., the queues containing tasks assigned to the current MPI process.

Tasks are dequeued according to the following scheduling policy:
    1. Medium and high-priority tasks are dequeued from `torcjulia_priority_queue`, with higher-priority tasks served first.
    2. If the priority queue is empty, general-purpose tasks, i.e., those assigned the default priority, are dequeued from `torcjulia_queue`, 
       Among these tasks, tasks from higher spawning levels (i.e., coarse-grained/outermost tasks) are prioritized, 
       while tasks at the same level are retrieved in FIFO order. 
"""
@inline function dequeue()
    # Check tasks in the priority queue first
    task = _maybepopfirst!(torcjulia_priority_queue)
    
    if !isnothing(task)
        return something(task) 
    end

    # From outermost to innermost levels
    for level in 1:TORCJULIA_QUEUE_LEVELS
        task = maybepopfirst!(torcjulia_queue[level]) # -> returns (Some(T), nothing)
        
        if !isnothing(task)
            return something(task) 
        end
    end

    Dict{Symbol, Any}()
end

"""
    dequeue_steal()::Union{Dict{Symbol, Any}, Nothing}

Dequeues the next available task that can be stolen from the local queues, 
i.e., the queues containing tasks assigned to the current MPI process, skipping tasks that are pinned to a specific node.

Tasks are dequeued according to the following scheduling policy:
    1. Medium and high-priority tasks are dequeued from `torcjulia_priority_queue`, with higher-priority tasks served first.
    2. If the priority queue is empty, general-purpose tasks, i.e., those assigned the default priority, are dequeued from `torcjulia_queue`, 
       Among these tasks, tasks from lower spawning levels (i.e., fine-grained/innermost tasks) are prioritized, 
       while tasks at the same level are retrieved in FIFO order. 
"""
@inline function dequeue_steal()
    # Check tasks in the priority queue first
    task_opt = _maybepopfirst!(torcjulia_priority_queue)
    
    if !isnothing(task_opt)
        task = something(task_opt)

        # If the task is pinned to a specific node, it cannot be stolen by worker threads on other nodes
        if !task[:forced_node]
            return task
        end

        # Push back because this task is pinned to a specific node and must not be stolen; reinserting prevents it from being lost
        push!(torcjulia_priority_queue, task, task[:priority].val)
    end

    # From innermost to outermost levels
    for level in TORCJULIA_QUEUE_LEVELS:-1:1
        task_opt = maybepopfirst!(torcjulia_queue[level])
        
        if !isnothing(task_opt)
            task = something(task_opt)
            
            if !(task isa Dict) || !task[:forced_node]
                return task   
            end
            
            # Push back because this task is pinned to a specific node and must not be stolen; reinserting prevents it from being lost
            enqueue(level, task) 
        end
    end

    Dict{Symbol, Any}()
end

"""
    wait_tasks_dispatch_loop()

Dispatch loop for the `wait()` function.
"""
function wait_tasks_dispatch_loop()
    while true
        task = dequeue()
        
        isnothing(task) ? break : nothing

        if isempty(task)
            if ctx[].num_procs > 1 && TORCJULIA_STEALING_ENABLED[]
                task = steal()
                
                Threads.atomic_add!(torcjulia_steal_attempts, UInt(1))
                
                if task[:type] == "nowork"
                    # Avoid flooding the MPI comm with task stealing requests 
                    sleep(TORCJULIA_WORKER_YIELDTIME[])
                    break
                end
                
                task[:counted] ? Threads.atomic_add!(torcjulia_stole, UInt(1)) : nothing
            
            else
                sleep(TORCJULIA_WORKER_YIELDTIME[])
                break
            end
        end

        ctx[].num_procs > 1 ? sleep(TORCJULIA_TASK_YIELDTIME) : nothing
        
        task[:start_t] = time_ns()
        work(task)
    end

    nothing
end

"""
    wait(tasks::Union{Vector{Future}, Future, Nothing} = nothing; return_completed::Bool = false)

Waits for the specified tasks to complete.

If `tasks` is `nothing`, waits for all tasks spawned by the current task to complete.

If `return_completed` is `true`, returns the `Future` objects for the completed tasks.
"""
function wait(tasks::Union{Vector{Future}, Future, Nothing} = nothing; return_completed::Bool = false)
    isnothing(tasks) && return _waitall(return_completed)

    tasks isa Future && return _waitspec([tasks], return_completed)

    _waitspec(tasks, return_completed)
end

"""
    _waitall(return_completed::Bool)

Waits for all tasks spawned by the current task (child tasks) to complete. 

While waiting, the current worker thread participates in executing pending tasks rather than remaining idle.

If `return_completed` is `true`, returns the `Future` objects for the completed tasks.
"""
function _waitall(return_completed::Bool)
    main_task = _torcjulia_tls_get_curr_task()

    while main_task[:deps][] > 0
        # The thread executing the task that called `wait()` becomes a worker thread until all child tasks have completed
        wait_tasks_dispatch_loop()
    end

    _torcjulia_tls_set_curr_task(main_task)

    return_completed ? [unsafe_pointer_to_objref(Ptr{UInt}(task[:myfuture])) for task in main_task[:completed]] : nothing
end

"""
    _waitspec(tasks::Vector{Future}, return_completed::Bool; ref_task::Union{UInt, Nothing} = nothing)

Waits for the specified tasks (`tasks`) to complete.

While waiting, the current worker thread participates in executing pending tasks rather than remaining idle.

If `return_completed` is `true`, returns the `Future` objects for the completed tasks.

If `ref_task` is provided, it is used to identify the reference task for which the specified tasks are being waited on.
"""
function _waitspec(tasks::Vector{Future}, return_completed::Bool; ref_task::Union{UInt, Nothing} = nothing)
    main_task = isnothing(ref_task) ? _torcjulia_tls_get_curr_task() : unsafe_pointer_to_objref(Ptr{UInt}(ref_task))
    
    rec_tasks = Dict{UInt, Bool}(); pending_tasks = UInt[]

    for task in tasks
        t = task.desc[:mytask]
        
        if !haskey(rec_tasks, t)
            rec_tasks[t] = true; push!(pending_tasks, t)
        
        else
            println("Duplicate task detected: $t — will be ignored")
        end
    end

    curr_completed_tasks = lock(completed_tasks_lock) do
        [task[:mytask] for task in main_task[:completed]]
    end

    while !all(t -> t in curr_completed_tasks, pending_tasks)
        # The thread executing the task that called `wait()` becomes a worker thread until all specified tasks have completed
        wait_tasks_dispatch_loop()

        curr_completed_tasks = lock(completed_tasks_lock) do
            [task[:mytask] for task in main_task[:completed]]
        end
    end

    _torcjulia_tls_set_curr_task(main_task)

    return_completed ? [task for task in tasks if task.desc[:mytask] in curr_completed_tasks] : nothing
end

"""
    _remote_workers_sync()

Blocks the calling thread until the main thread of all MPI nodes reaches the barrier.
"""
function _remote_workers_sync()
    req = MPI.Ibarrier(ctx[].comm)
    
    flag = false
    
    while !flag
        flag = MPI.Test(req)
        flag || sleep(TORCJULIA_WORKER_YIELDTIME[])  
    end

    nothing
end

"""
    _terminate_nodes()

Sends a termination signal to the manager thread of each remote MPI node, initating the shutdown of its worker threads.

This function is called from the main thread of the master node (MPI rank 0).
"""
function _terminate_nodes() 
    foreach(node -> MPI.send(Dict(:type => "terminate"), ctx[].comm; dest = node, tag = TORCJULIA_MANAGER_TAG), 1:(ctx[].num_procs - 1))

    nothing
end

"""
    _terminate_manager_threads()

Sends a termination signal to the manager thread of each remote MPI process.

This function is called from the main thread of the master node (MPI rank 0).
"""
function _terminate_manager_threads()
    foreach(node -> MPI.send(Dict(:type => "exit"), ctx[].comm; dest = node, tag = TORCJULIA_MANAGER_TAG), 1:(ctx[].num_procs - 1))

    nothing
end

"""
    _notify_master_manager_thread()

Sends a termination signal to the manager thread of the master node (MPI rank 0).

This function is called by the manager thread of MPI rank 1 immediately after receiving a termination signal from the master node.
"""
function _notify_master_manager_thread()
    MPI.send(Dict(:type => "exit"), ctx[].comm; dest = 0, tag = TORCJULIA_MANAGER_TAG)

    nothing
end

"""
    finalize(MPI_finalize::Bool)

Finalizes the torcjulia runtime environment.

If `MPI_finalize` is `true`, it also finalizes the MPI environment.
"""
function finalize(MPI_finalize::Bool)
    TORCJULIA_USE_MANAGER[] ? Base.wait(manager_thread_lock) : nothing
   
    !shutdown_node_flag[] ? _torcjulia_stats() : nothing
    
    # Free all MPI shared-memory windows
    torcjulia_stolen[] = 0; _free_shw() 

    # If shutdown_node_flag[] is true, shutdown was explicitly initiated by a call to remove_worker() on the current MPI process
    # Skip the barrier to avoid deadlock
    !shutdown_node_flag[] ? MPI.Barrier(ctx[].comm) : nothing
    
    (MPI_finalize || shutdown_node_flag[]) ? MPI.Finalize() : nothing
end

"""
    _finalize()

Called from the main thread of the master node (MPI rank 0) immediately after 
the primary application task passed to `torcjulia.init()` completes.

Initiates the shutdown of the master node's worker threads and sends termination signals to the manager threads 
of all remote MPI nodes to shutdown their worker threads.
"""
function _finalize()
    global torcjulia_terminate_flag = true
    
    foreach(_ -> enqueue(UInt(1), nothing), 1:TORCJULIA_NUM_WORKERS[]) 
    
    ctx[].num_procs > 1 ? _terminate_nodes() : nothing
end

"""
    _capture_args_deps(task::Dict{Symbol, Any})::Bool

Scans the arguments of the specified `task` to identify any data dependencies on other pending tasks.

If a dependency is found, the function increments the dependency counter of the current task 
and adds it to the dependent list of the task it depends on.

Returns `true` if any dependencies were found, otherwise returns `false`.
"""
function _capture_args_deps(task::Dict{Symbol, Any})::Bool
    deps = UInt(0)
    
    args = task[:args]; kwargs = task[:kwargs]

    """
        _scan(x)

    Recursively scans `x` to identify any data dependencies on other pending tasks.
    """
    _scan(x) = begin
        if x isa Number || x isa String || x isa Type || x === nothing
            return
        end
        
        if x isa Future
            lock(completed_tasks_lock) do
                if x.desc[:state] != completed
                    deps += 1

                    lock(dependent_tasks_lock) do
                        push!(x.desc[:dependent], task)
                    end 

                    Threads.atomic_add!(task[:deps], UInt(1))
                end
            end
        
        elseif x isa AbstractArray || x isa Tuple
            foreach(_scan, x)  
        
        elseif x isa AbstractDict
            foreach(_scan, values(x))
        
        elseif isstructtype(typeof(x))
            foreach(_scan, (getfield(x, f) for f in fieldnames(typeof(x))))
        end
    end

    _scan(args); _scan(values(kwargs))
    
    deps > 0
end

""" 
    _parse_shw_args(args...; kwargs...)

Recursively scans the provided arguments to identify any MPI shared-memory windows 
and replaces them with their corresponding pointers for transmission to remote MPI processes.
"""
function _parse_shw_args(args...; kwargs...)
    """
        _replace(x)

    Recursively replaces any MPI shared-memory windows in `x` with their corresponding pointers.
    """
    function _replace(x)
        if x isa Number || x isa String || x isa Type || x === nothing
            return x
        end

        if x isa AbstractArray
            ptr = UInt(pointer_from_objref(x))
            return haskey(torcjulia_local_to_master, ptr) ? torcjulia_local_to_master[ptr] : x
        
        elseif x isa Tuple
            return Base.map(_replace, x)
        
        elseif x isa NamedTuple
            return Base.map(_replace, x)
        
        elseif x isa Dict
            return Dict(k => _replace(v) for (k,v) in x)
        
        elseif isstructtype(typeof(x))
            fvals = Base.map(f -> _replace(getfield(x, f)), fieldnames(typeof(x)))
            return (typeof(x))(fvals...)
        
        else
            return x
        end
    end

    Base.map(_replace, args), Base.map(_replace, NamedTuple(kwargs))
end

"""
    _restore_shw_args(args...; kwargs...)

Recursively scans the provided arguments to identify any pointers to MPI shared-memory windows
and replaces them with their corresponding shared-array values for task execution on the current MPI process.
"""
function _restore_shw_args(args...; kwargs...)
    """
        _restore(x)
    
    Recursively restores any pointers to MPI shared-memory windows in `x` to their corresponding shared-array values.
    """
    function _restore(x)
        if x isa UInt && haskey(torcjulia_shw, x)
            return _get_shared_memwin(x)

        elseif x isa UInt && haskey(torcjulia_local_to_master, x)
            return _get_shared_memwin(torcjulia_local_to_master[x])

        elseif x isa Number || x isa String || x isa Type || x === nothing
            return x

        elseif x isa Tuple
            return tuple(Base.map(_restore, x)...)

        elseif x isa NamedTuple
            return NamedTuple{keys(x)}(Base.map(_restore, values(x)))

        elseif x isa Dict
            return Dict(k => _restore(v) for (k,v) in x)

        elseif isstructtype(typeof(x))
            fvals = Base.map(f -> _restore(getfield(x, f)), fieldnames(typeof(x)))
            return (typeof(x))(fvals...)

        else
            return x
        end
    end

    Base.map(_restore, args), Base.map(_restore, NamedTuple(kwargs))
end

"""
    shw_alloc(dim::Tuple{Vararg{Int}}, T::Type)

Allocates an MPI shared-memory window of the specified dimensions (`dim`) and type (`T`).
"""
function shw_alloc(dim::Tuple{Vararg{Int}}, T::Type)
    for rank in Base.filter(r -> r != Int(ctx[].rank), 0:(Int(ctx[].num_procs) - 1))
        submit(_alloc_shared_memwin, dim, T, ctx[].rank; node = rank, forced_node = true, counted = false)
    end
    
    win, arr = MPI.Win_allocate_shared(Array{T}, dim, ctx[].comm)
    
    ptr = MPI.bcast(UInt(pointer_from_objref(arr)), ctx[].comm; root = ctx[].rank)
    
    _reg_shared_memwin(ptr, arr, win)

    _waitall(false)
    
    arr
end

"""
    _free_shw()

Frees all MPI shared-memory windows allocated by the torcjulia runtime environment.
"""
function _free_shw()
    isempty(torcjulia_shw) && return
    
    for (_, (_, win)) in torcjulia_shw
        MPI.free(win)
    end
    
    empty!(torcjulia_shw)
end

"""
    _alloc_shared_memwin(size::Tuple{Vararg{Int}}, type::Type, master::UInt)

Allocate an MPI shared-memory window for the specified `size` and `type`, with the specified `master` MPI process as the owner.

This function is called by a worker thread on each remote MPI process, excluding the owner process.
"""
function _alloc_shared_memwin(size::Tuple{Vararg{Int}}, type::Type, master::UInt)
    win, arr = MPI.Win_allocate_shared(Array{type}, (0,), ctx[].comm)
    
    arr = MPI.Win_shared_query(Array{type}, size, win; rank = master)
    
    ptr = MPI.bcast(UInt(0), ctx[].comm; root = master)
    
    _reg_shared_memwin(ptr, arr, win)
    
    nothing
end

"""
    _reg_shared_memwin(ptr::UInt, buffer, win)

Registers the specified `buffer` and its associated MPI shared-memory window (`win`) under the corresponding pointer (`ptr`).

Also, maps the buffer's local pointer to its corresponding pointer on the owner process.
"""
function _reg_shared_memwin(ptr::UInt, buffer, win) 
    torcjulia_shw[ptr] = (buffer, win)

    torcjulia_local_to_master[UInt(pointer_from_objref(buffer))] = ptr
end

"""
    _get_shared_memwin(ptr::UInt)

Retrieves the MPI shared-memory window associated with the specified pointer (`ptr`).

If the pointer is not registered, returns `nothing`.
"""
_get_shared_memwin(ptr::UInt) = haskey(torcjulia_shw, ptr) ? torcjulia_shw[ptr][1] : nothing

"""
    _agg_time_stats(pending_q::ConcurrentQueue{UInt}, exec_q::ConcurrentQueue{UInt})::Tuple{Tuple{UInt, UInt}, Tuple{UInt, UInt}}

Aggregates the total pending and execution times for tasks submitted by the current MPI node.

This function is called by the master thread of each MPI node.
"""
function _agg_time_stats(pending_q::ConcurrentQueue{UInt}, exec_q::ConcurrentQueue{UInt})::Tuple{Tuple{UInt, UInt}, Tuple{UInt, UInt}}
    agg(q)::Tuple{UInt, UInt} = begin
        _sum::UInt = 0
        _cnt::UInt = 0
        
        while true
            t = maybepopfirst!(q)
            
            isnothing(t) && break
            
            _sum += something(t)
            _cnt += 1
        end
        
        return _sum, _cnt
    end

    t1 = Threads.@spawn agg(pending_q)
    t2 = Threads.@spawn agg(exec_q)

    res1 = fetch(t1)::Tuple{UInt, UInt}
    res2 = fetch(t2)::Tuple{UInt, UInt}

    res1, res2
end

"""
    _torcjulia_stats()

Collects and displays runtime statistics for the torcjulia environment across all MPI nodes, including the number of tasks created, executed, 
stolen, and stolen attempts,the maximum queue depth and the average task pending and execution times.

This function is called by the master thread of each MPI node right before the torcjulia runtime environment is finalized.

WARNING: If an MPI node is shutdown by a call to `remove_worker()`, this function is also called immediately before that node terminates. 
"""
function _torcjulia_stats()
    created = Threads.atomic_xchg!(torcjulia_created, UInt(0))
    executed = Threads.atomic_xchg!(torcjulia_executed, UInt(0))
    
    stole = Threads.atomic_xchg!(torcjulia_stole, UInt(0))
    steal_attempts = Threads.atomic_xchg!(torcjulia_steal_attempts, UInt(0))
    
    max_queue_depth = Threads.atomic_xchg!(torcjulia_max_queue_depth, UInt(0))

    (pending_sum, pending_cnt), (exec_sum, exec_cnt) = _agg_time_stats(node_tasks_pending_time, node_tasks_execution_time)

    println(Crayon(foreground = :green)(
        @sprintf("torcjulia [node %d]: created = %d, executed = %d, stole = %d, stolen = %d, steal_attempts = %d, max_queue_depth = %d",
                    ctx[].rank, created, executed, stole, torcjulia_stolen[], steal_attempts, max_queue_depth)
    ))

    flush(stdout)

    _local = Float64[pending_sum, pending_cnt, exec_sum, exec_cnt]
    _global = ctx[].num_procs > 1 ? MPI.Reduce(_local, +, 0, ctx[].comm) : _local

    if ctx[].rank == 0
        global task_execution_stats
        
        _global .+= task_execution_stats
        
        total_pending_sum, total_pending_cnt, total_exec_sum, total_exec_cnt = _global
        
        avg_pending = total_pending_cnt == 0 ? 0.0 : total_pending_sum / total_pending_cnt / 1e9
        avg_exec = total_exec_cnt == 0 ? 0.0 : total_exec_sum / total_exec_cnt / 1e9

        println(Crayon(foreground = :magenta)(
            @sprintf("torcjulia [tasks]: average pending time = %.3fs, average execution time = %.3fs",
                    avg_pending, avg_exec)
        ))
    end

    flush(stdout)
    
    nothing
end

"""
    _capture_time_stats(submit_t::UInt64, start_t::UInt64, end_t::UInt64)

Records the pending and execution times of the completed task.
"""
function _capture_time_stats(submit_t::UInt64, start_t::UInt64, end_t::UInt64)
    push!(node_tasks_pending_time, start_t - submit_t)
    push!(node_tasks_execution_time, end_t - start_t)
    
    nothing
end

"""
    spmd(func, args...; counted::Bool = true, return_completed::Bool = true, kwargs...)

Dynamically switches the execution model of the torcjulia runtime environment from AMT to Single Program Multiple Data (SPMD) mode.

Executing the specified `func` on all MPI nodes, passing the provided `args` and `kwargs` to each invocation.

If `return_completed` is `true`, returns the `Future` objects for the completed tasks.
"""
function spmd(
    func, 
    args...; 
    counted = true, 
    return_completed = true, 
    kwargs...)
    
    for node in 0:Int(ctx[].num_procs) - 1
        submit(func, args...; node = node, counted = counted, kwargs...)
    end

    _waitall(return_completed)
end

"""
    map(func, iterables...; chunksize::Union{Int, Nothing} = nothing, node::Int = -1, counted::Bool = true, 
        forced_node::Bool = false, priority::TaskPriorityLevel = TaskPriority.low, args::Tuple = (), kwargs...)

Parallel version of the built-in `Base.map()` function, applying the specified `func` to each element of the provided `iterables`.

When multiple iterables are provided, `func` is applied to the corresponding elements of each iterable, similar to `Base.map()`.

The `chunksize` parameter specifies the number of elements processed by each task. If `chunksize` is not specified, 
the AMT runtime automatically determines an appropriate chunk size based on the number of available workers.

The `args` and `kwargs` parameters specify additional positional and keyword arguments, respectively, to pass to `func` on each invocation.

Returns an array containing the results of applying `func` to the corresponding elements of the input iterables.

Examples: 
- `map(f, [1, 2, 3])` → `[f(1), f(2), f(3)]`
- `map(f, [1,2], [3,4])` → `[f(1,3), f(2,4)]`
"""
function map(
    func::Function, 
    iterables...; 
    chunksize::Union{Int, Nothing} = nothing, 
    node::Int = -1, 
    counted::Bool = true, 
    forced_node::Bool = false, 
    priority::TaskPriorityLevel = TaskPriority.low, 
    args::Tuple = (), 
    kwargs...)

    items = zip(iterables...)

    chunksize = something(chunksize, max(cld(length(iterables[1]), ctx[].num_procs * TORCJULIA_NUM_WORKERS[]), 1))

    _wrapper = x -> func(x..., args...; kwargs...)
    
    futures = if chunksize == 1
        [submit(_wrapper, x; node = node, counted = counted, forced_node = forced_node, priority = priority) for x in items]
    
    else
        _ch_wrapper = ch -> [_wrapper(x) for x in ch]
        
        [submit(_ch_wrapper, ch; node = node, counted = counted, forced_node = forced_node, priority = priority) 
                for ch in Iterators.partition(items, chunksize)]
    end
    
    _waitall(false)
    
    results = [result(t) for t in futures]
    
    flat = chunksize == 1 ? results : vcat(results...)
    reshape(flat, size(iterables[1])...)    
end

"""
    starmap(func, iterable; chunksize::Union{Int, Nothing} = nothing, node::Int = -1, counted::Bool = true, 
            forced_node::Bool = false, priority::TaskPriorityLevel = TaskPriority.low, args::Tuple = (), kwargs...)

Parallel version of Python's `itertools.starmap()`, applying the specified `func` to each element of the 
provided `iterable`, unpacking the elements of the iterable as separate arguments to `func`.

Key difference from `map`:
- `map(f, [(1,2), (3,4)])` → `[f((1,2)), f((3,4))]` (passes tuple as single arg)
- `starmap(f, [(1,2), (3,4)])` → `[f(1,2), f(3,4)]` (unpacks tuple elements as separate args)

The `chunksize` parameter specifies the number of elements processed by each task. If `chunksize` is not specified, 
the AMT runtime automatically determines an appropriate chunk size based on the number of available workers.

The `args` and `kwargs` parameters specify additional positional and keyword arguments, respectively, to pass to `func` on each invocation.

Example:
- `starmap(f, [(1,2), (3,4)])` → `[f(1,2), f(3,4)]`
"""
function starmap(
    func::Function, 
    iterable; 
    chunksize::Union{Int, Nothing} = nothing, 
    node::Int = -1, 
    counted::Bool = true, 
    forced_node::Bool = false, 
    priority::TaskPriorityLevel = TaskPriority.low, 
    args::Tuple = (), 
    kwargs...)

    T = Base.promote_typeof(func(iterable[1]..., args...; kwargs...)) 

   chunksize = something(chunksize, max(cld(length(iterable), ctx[].num_procs * TORCJULIA_NUM_WORKERS[]), 1))

    _wrapper = x -> T(func(x..., args...; kwargs...))
    
   futures = if chunksize == 1
        [submit(_wrapper, x; node = node, counted = counted, forced_node = forced_node, priority = priority) for x in iterable]
    
    else
        _ch_wrapper = ch -> begin 
            out = Vector{T}(undef, length(ch))
            
            @inbounds for (i,x) in enumerate(ch)
                out[i] = func(x..., args...; kwargs...)
            end
            
            out
        end
        
        [submit(_ch_wrapper, ch; node = node, counted = counted, forced_node = forced_node, priority = priority) 
        for ch in Iterators.partition(iterable, chunksize)]
    end

    _waitall(false)
    
    results = [result(t) for t in futures]
    
    chunksize == 1 ? results : foldl(vcat, results, init = T[])
end

"""
    AsyncResult

Wraps the result of an algorithmic skeleton execution, providing access to its completion status and results.
"""
mutable struct AsyncResult
    tasks::Vector{Future}
end

"""
    ready(ar::AsyncResult)

Check whether all tasks associated with the corresponding asynchronous algorithmic skeleton (`ar`) have completed.
"""
@inline function ready(ar::AsyncResult)::Bool
    all(t -> t.desc[:state] == completed, ar.tasks)
end

"""
    wait(ar::AsyncResult)

Waits for all tasks associated with the corresponding asynchronous algorithmic skeleton (`ar`) to complete.
"""
function wait(ar::AsyncResult)
    _waitspec(ar.tasks, false)
end

"""
    get(ar::AsyncResult)

Waits for all tasks associated with `ar` to complete and returns the resulting array.
"""
function get(ar::AsyncResult)
    _waitspec(ar.tasks, false)
    
    results = [result(t) for t in ar.tasks]
    all(x -> isa(x, AbstractArray), results) ? collect(Iterators.flatten(results)) : results
end

"""
    map_async(func, iterables...; chunksize::Union{Int, Nothing} = nothing, node::Int = -1, counted::Bool = true, 
              forced_node::Bool = false, priority::TaskPriorityLevel = TaskPriority.low, callback = nothing, 
              args::Tuple = (), kwargs...)

Asynchronous version of `map()`, applying the specified `func` to each element of the provided `iterables` in parallel.

Unlike `map` which blocks execution until all tasks complete, `map_async` returns immediately with an AsyncResult handle, 
allowing the caller to continue execution while tasks are processed in the background.
    
The `chunksize` parameter specifies the number of elements processed by each task. If `chunksize` is not specified, 
the AMT runtime automatically determines an appropriate chunk size based on the number of available workers.

The `args` and `kwargs` parameters specify additional positional and keyword arguments, respectively, to pass to `func` on each invocation.

`map_async` supports also an optional `callback` function that is invoked once all tasks have completed, 
receiving the collected results as its argument.

Returns an `AsyncResult` object that can be used to check the completion status of the tasks and retrieve their results.
"""
function map_async(
    func::Function, 
    iterables...; 
    chunksize::Union{Int, Nothing} = nothing, 
    node::Int = -1, 
    counted::Bool = true, 
    forced_node::Bool = false, 
    priority::TaskPriorityLevel = TaskPriority.low, 
    callback = nothing, 
    args::Tuple = (), 
    kwargs...)

    items = zip(iterables...) 

    chunksize = something(chunksize, max(cld(length(iterables[1]), ctx[].num_procs * TORCJULIA_NUM_WORKERS[]), 1))

    _wrapper = x -> func(x..., args...; kwargs...)

    futures = if chunksize == 1
        [submit(_wrapper, x; node = node, counted = counted, forced_node = forced_node, priority = priority) for x in items]
    
    else
        _ch_wrapper = ch -> [_wrapper(x) for x in ch]
        
        [submit(_ch_wrapper, ch; node = node, counted = counted, forced_node = forced_node, priority = priority) 
                for ch in Iterators.partition(items, chunksize)]
    end

    if !isnothing(callback)
        function _exec_cb(ref_task, tasks, cb)
            _waitspec(tasks, false; ref_task = ref_task)
            
            res = [result(t) for t in tasks]
            res = all(x -> isa(x, AbstractArray), res) ? vcat(res...) : res
            
            cb(res)
        end
        
        ref_task = _torcjulia_tls_get_curr_task()[:mytask]; _node = ctx[].rank
        
        submit(_exec_cb, ref_task, futures, callback; node = Int(_node), forced_node = true)
    end

    AsyncResult(futures)
end

"""
    starmap_async(func, iterable; chunksize::Union{Int, Nothing} = nothing, node::Int = -1, counted::Bool = true, 
                  forced_node::Bool = false, priority::TaskPriorityLevel = TaskPriority.low, callback = nothing, 
                  args::Tuple = (), kwargs...)

Asynchronous version of `starmap()`, applying the specified `func` to each element of the provided `iterable` in parallel,
unpacking the elements of the iterable as separate arguments to `func`.

Unlike `starmap` which blocks execution until all tasks complete, `starmap_async` returns immediately with an AsyncResult handle,
allowing the caller to continue execution while tasks are processed in the background.

The `chunksize` parameter specifies the number of elements processed by each task. If `chunksize` is not specified, 
the AMT runtime automatically determines an appropriate chunk size based on the number of available workers.

The `args` and `kwargs` parameters specify additional positional and keyword arguments, respectively, to pass to `func` on each invocation.

An optional `callback` function can be provided, which is invoked once all tasks have completed, receiving the collected results as its argument.

Returns an `AsyncResult` object that can be used to check the completion status of the tasks and retrieve their results.
"""
function starmap_async(
    func::Function, 
    iterable; 
    chunksize::Union{Int, Nothing} = nothing, 
    node::Int = -1, 
    counted::Bool = true, 
    forced_node::Bool = false, 
    priority::TaskPriorityLevel = TaskPriority.low, 
    callback = nothing, 
    args::Tuple = (), 
    kwargs...)

    chunksize = something(chunksize, max(cld(length(iterable), ctx[].num_procs * TORCJULIA_NUM_WORKERS[]), 1))

    _wrapper = x -> func(x..., args...; kwargs...)

    futures = if chunksize == 1
        [submit(_wrapper, x; node = node, counted = counted, forced_node = forced_node, priority = priority) for x in iterable]
    
    else
        _ch_wrapper = ch -> [_wrapper(x) for x in ch]
       
        [submit(_ch_wrapper, ch; node = node, counted = counted, forced_node = forced_node, priority = priority) 
                for ch in Iterators.partition(iterable, chunksize)]
    end

    if !isnothing(callback)
        function _exec_cb(ref_task, tasks, cb)
            _waitspec(tasks, false; ref_task = ref_task)
            res = [result(t) for t in tasks]
            res = all(x -> isa(x, AbstractArray), res) ? vcat(res...) : res
            cb(res)
        end
        
        ref_task = _torcjulia_tls_get_curr_task()[:mytask]; _node = ctx[].rank
        
        submit(_exec_cb, ref_task, futures, callback; node = Int(_node), forced_node = true)
    end

    AsyncResult(futures)
end

"""
    reduce(func, data; chunksize::Union{Int, Nothing} = nothing, direction::Symbol = :column, mode::String = "1d", 
           node::Int = -1, counted::Bool = true, forced_node::Bool = false, priority::TaskPriorityLevel = TaskPriority.low, 
           args::Tuple = (), kwargs...)

Parallel version of the built-in `Base.reduce()` function, applying the binary associative operation `func`
to reduce the provided `data` along the specified `direction`.

`data` can be a vector or a matrix. 

For vectors, the operation is applied across all elements. 

For matrices, `direction` determines the reduction direction:
- `:column`: reduces along each column, combining elements across rows.
- `:row`: reduces along each row, combining elements across columns.

The `mode` parameter determines the reduction output:
- "1d": reduce each row or column to a single scalar, resulting in a vector of reduced values.
- "2d": progressively reduce the matrix to a single scalar by first reducing along the specified direction and then reducing the resulting vector.

The `direction` parameter is ignored when `data` is a vector. The "2d" mode requires `data` to be a matrix.

The `chunksize` parameter specifies the number of elements processed per task. 
If omitted, the runtime determines the chunk size automatically based on the number of available workers.

The `args` and `kwargs` parameters specify additional positional and keyword arguments, respectively, passed to `func` on each invocation.

Returns the reduction result: a scalar or a vector, depending on `data`, and `mode`.

Examples:
- `A = [1, 2, 3, 4, 5]`
- `B = [1 2 3; 4 5 6; 7 8 9]`
- `reduce(+, A)` → `15`
- `reduce(+, B, direction = :column, mode = "1d")` → `[12, 15, 18]`
- `reduce(+, B, direction = :row, mode = "1d")` → `[6, 15, 24]`
- `reduce(+, B; mode = "2d")` → `45`
"""
function reduce(
    func::Function, 
    data::AbstractArray; 
    chunksize::Union{Int, Nothing} = nothing, 
    direction::Symbol = :column, 
    mode::String = "1d", 
    node::Int = -1, 
    counted::Bool = true, 
    forced_node::Bool = false, 
    priority::TaskPriorityLevel = TaskPriority.low, 
    args::Tuple = (), 
    kwargs...)

    if !(direction in (:column, :row))
        @error "direction must be :column or :row"
        MPI.Abort(ctx[].comm, 1)
    end

    if !(mode in ("1d", "2d"))
        @error "mode must be \"1d\" or \"2d\""
        MPI.Abort(ctx[].comm, 1)
    end

    if mode == "2d" && isa(data, AbstractVector) 
        @error "2D reduction requires a matrix."
        MPI.Abort(ctx[].comm, 1)
    end

    isa(chunksize, Int) && chunksize == 1 && (@warn "Reduction chunksize cannot be 1. Setting to 2."; chunksize = 2)
    
    ndim = isa(data, AbstractVector) ? length(data) : (direction === :column ? size(data, 1) : size(data, 2))
    
    if ndim == 1
        return (data isa AbstractVector ? data[1] : (mode === "1d" ? (direction === :column ? data[1, :] : data[:, 1]) :
                (direction === :column ? reduce(func, data[1, :]; chunksize = chunksize, node = node, 
                forced_node = forced_node, priority = priority, args = args, kwargs...) 
                : reduce(func, data[:, 1]; chunksize = chunksize, node = node, 
                forced_node = forced_node, priority = priority, args = args, kwargs...))))
    end
    
    chunksize = something(chunksize, max(cld(ndim, ctx[].num_procs * TORCJULIA_NUM_WORKERS[]), 2))

    local_reduce(x) = begin
        if isa(x, AbstractVector)
            acc = x[1]
            
            @inbounds for i in 2:length(x)
                acc = func(acc, x[i], args...; kwargs...)
            end
            
            return acc
        
        else
            rows, cols = size(x)
            
            if direction === :column
                acc = [x[1, j] for j in 1:cols]
                
                @inbounds for j in 1:cols
                    @simd for i in 2:rows
                        acc[j] = func(acc[j], x[i,j], args...; kwargs...)
                    end
                end
            
            else
                acc = [x[i, 1] for i in 1:rows]
                
                @inbounds for i in 1:rows
                    @simd for j in 2:cols
                        acc[i] = func(acc[i], x[i,j], args...; kwargs...)
                    end
                end
            end
            
            return acc
        end
    end

    current = data
    
    while true
        len = isa(current, AbstractVector) ? length(current) :
              (direction === :column ? size(current, 1) : size(current, 2))

        if len ≤ chunksize
            res = len == 1 ? (current isa AbstractVector ? current[1] :
                               direction === :column ? current[1, :] : current[:, 1]) :
                  local_reduce(current)
            break
        end

        if current isa AbstractVector
            @views chunks = [current[i:min(i+chunksize-1, len)] for i in 1:chunksize:len]
        
        elseif direction === :column
            @views chunks = [current[i:min(i+chunksize-1, len), :] for i in 1:chunksize:len]
        
        else
            @views chunks = [current[:, i:min(i+chunksize-1, len)] for i in 1:chunksize:len]
        end

        futures = submit.(local_reduce, chunks; node = node, counted = counted, forced_node = forced_node, priority = priority)
        
        _waitall(false)
        
        partials = result.(futures)

        if all(p -> p isa AbstractVector, partials)
            current = direction === :column ? vcat(reshape.(partials, 1, :)...) :
                                             hcat(reshape.(partials, :, 1)...)
        
                                            else
            current = partials  
        end
    end

    if mode == "2d"
        while length(res) > chunksize
            chunks = [@view res[i:min(i+chunksize-1, length(res))] for i in 1:chunksize:length(res)]
            
            futures = submit.(local_reduce, chunks; node = node, counted = counted, forced_node = forced_node, priority = priority)
            
            _waitall(false)
            
            res = result.(futures)
        end
        
        res = res isa AbstractVector ? (length(res) == 1 ? res[1] : local_reduce(res)) : res
    end

    res
end

"""
    mapReduce(map_func, reduce_func, iterables...; chunksize::Union{Int, Nothing} = nothing, direction::Symbol = :column, 
              mode::String = "1d", node::Int = -1, counted::Bool = true, forced_node::Bool = false, 
              priority::TaskPriorityLevel = TaskPriority.low, args::Tuple = (), kwargs...)

Combines the parallel `map()` and `reduce()` operations into a single skeleton, applying the specified `map_func` 
to each element of the provided `iterables` and then reducing the results using the specified `reduce_func`.

The `chunksize` parameter specifies the number of elements processed by each task. If `chunksize` is not specified,
the AMT runtime automatically determines an appropriate chunk size based on the number of available workers.

The `direction` and `mode` parameters are passed to `reduce_func`, determining the reduction direction (:column or :row) and mode ("1d" or "2d").

The `args` and `kwargs` parameters specify additional positional and keyword arguments, respectively, 
to pass to both `map_func` and `reduce_func` on each invocation.

Returns the result of applying `reduce_func` to the results of `map_func`.

Examples:
- `mapReduce(f, g, [1, 2, 3])` → `g(f(1), f(2), f(3))`
- `mapReduce(f, g, [1, 2], [3, 4])` → `g(f(1,3), f(2,4))`
"""
function mapReduce(
    map_func::Function, 
    reduce_func::Function, 
    iterables...; 
    chunksize::Union{Int, Nothing} = nothing, 
    direction::Symbol = :column, 
    mode::String = "1d", 
    node::Int = -1, 
    counted::Bool = true, 
    forced_node::Bool = false, 
    priority::TaskPriorityLevel = TaskPriority.low, 
    args::Tuple = (),
    kwargs...)

    chunksize = something(chunksize, max(cld(length(iterables[1]), ctx[].num_procs * TORCJULIA_NUM_WORKERS[]), 1))

    map_items = map(map_func, iterables...; chunksize = chunksize, node = node, counted = counted, forced_node = forced_node,
                    priority = priority, args = args, kwargs...)

    reduce(reduce_func, map_items; chunksize = chunksize, direction = direction, mode = mode, node = node, counted = counted,
           forced_node = forced_node, priority = priority, args = args, kwargs...)
end

"""
    filter(func, iterables...; chunksize::Union{Int, Nothing} = nothing, node::Int = -1, counted::Bool = true, 
           forced_node::Bool = false, priority::TaskPriorityLevel = TaskPriority.low, args::Tuple = (), kwargs...)

Parallel version of the built-in `Base.filter()` function, applying the specified `func` to each element of the provided `iterables`
and returning an array containing only the elements for which `func` returns `true`.

When multiple iterables are provided, `func` is applied to the corresponding elements of each iterable.

The `chunksize` parameter specifies the number of elements processed by each task. If `chunksize` is not specified,
the AMT runtime automatically determines an appropriate chunk size based on the number of available workers.

The `args` and `kwargs` parameters specify additional positional and keyword arguments, respectively, to pass to `func` on each invocation.

Returns an array containing the elements of the input iterables for which `func` returns `true`.

Examples:
- `workA(x::Int) = x % 2 == 0`
- `workB(data::Tuple{Int, Int}) = data[1] + data[2] > 10`
- `filter(workA, [1, 2, 3, 4])` → `[2, 4]`
- `filter(workB, [(1, 2), (3, 4), (5, 6), (7, 8)])` → `[(5, 6), (7, 8)]` 
"""
function filter(
    func::Function, 
    iterables...; 
    chunksize::Union{Int, Nothing} = nothing,  
    node::Int = -1, 
    counted::Bool = true, 
    forced_node::Bool = false, 
    priority::TaskPriorityLevel = TaskPriority.low, 
    args::Tuple = (), 
    kwargs...)

    items = zip(iterables...)
    T = eltype(iterables[1])
   
    _constr = typeof(iterables[1]) <: AbstractArray ? collect : typeof(iterables[1])

    chunksize = something(chunksize, max(cld(length(iterables[1]), ctx[].num_procs * TORCJULIA_NUM_WORKERS[]), 1))

    _wrapper = chunksize == 1 ? (x -> func(x..., args...; kwargs...) ? Some(T(x[1])) : nothing) :
        (ch -> begin out = Vector{T}(); for x in ch func(x..., args...; kwargs...) && push!(out, T(x[1])); end; out end)

    futures = chunksize == 1 ? [submit(_wrapper, x; node = node, counted = counted, forced_node = forced_node, priority = priority) for x in items] :
                               [submit(_wrapper, ch; node = node, counted = counted, forced_node = forced_node, priority = priority) 
                                        for ch in Iterators.partition(items, chunksize)]
                
    _waitall(false)

    results = [result(t) for t in futures]
    
    _constr(chunksize == 1 ? T[something(x) for x in results if x !== nothing] : Base.reduce(vcat, results; init = T[]))
end

"""
    mapPairs(func, xs, ys; chunksize::Union{Int, Nothing} = nothing, node::Int = -1, counted::Bool = true, 
             forced_node::Bool = false, priority::TaskPriorityLevel = TaskPriority.low, args::Tuple = (), kwargs...)

Applies `func` in parallel to every pair `(x, y)` in the Cartesian product of `xs` and `ys`.

The `chunksize` parameter specifies the number of pairs processed by each task. If omitted,
the AMT runtime automatically determines an appropriate chunk size based on the number of available workers.

The `args` and `kwargs` parameters specify additional positional and keyword arguments, 
respectively, to pass to `func` on each invocation.

Returns a matrix of size `length(xs)×length(ys)`, where each entry contains 
the result of `func(x, y)` for the corresponding pair `(x, y)`.

Example:
- `mapPairs(+, [1, 2, 6], [10, 20, 12])` → `[11 21 13; 12 22 14; 16 26 18]`
"""
function mapPairs(
    func::Function, 
    xs::AbstractVector, 
    ys::AbstractVector; 
    chunksize::Union{Int, Nothing} = nothing, 
    node::Int = -1, 
    counted::Bool = true, 
    forced_node::Bool = false, 
    priority::TaskPriorityLevel = TaskPriority.low, 
    args::Tuple = (), 
    kwargs...)

    nx, ny = length(xs), length(ys)

    chunksize = something(chunksize, max(cld(length(xs) * length(ys), ctx[].num_procs * TORCJULIA_NUM_WORKERS[]), 1))
    
    idx_pairs = Iterators.product(1:nx, 1:ny)
    chunks = chunksize == 1 ? [(i,) for i in idx_pairs] : Iterators.partition(idx_pairs, chunksize)

    futures = [submit(ch -> [func(xs[i], ys[j], args...; kwargs...) for (i,j) in ch], ch; node = node, 
                      counted = counted, forced_node = forced_node, priority = priority) for ch in chunks]

    _waitall(false)

    partials = Iterators.flatten(result.(futures))
    T = typeof(first(partials)); results = Array{T}(undef, nx, ny)

    for (val, (i,j)) in zip(Iterators.flatten(result.(futures)), idx_pairs)
        results[i,j] = val
    end

    results
end

"""
    mapPairsReduce(map_func, reduce_func, xs, ys; chunksize::Union{Int, Nothing} = nothing, direction::Symbol = :column, 
                   mode::String = "1d", node::Int = -1, counted::Bool = true, forced_node::Bool = false, 
                   priority::TaskPriorityLevel = TaskPriority.low, args::Tuple = (), kwargs...)


Combines the parallel `mapPairs()` and `reduce()` operations into a single skeleton, applying the specified `map_func`
to every pair `(x, y)` in the Cartesian product of `xs` and `ys, and then reducing the results using the specified `reduce_func`.

The `chunksize` parameter specifies the number of pairs processed by each task. If omitted,
the AMT runtime automatically determines an appropriate chunk size based on the number of available workers.

The `direction` and `mode` parameters are passed to `reduce_func`, determining the reduction direction (:column or :row) and mode ("1d" or "2d").

The `args` and `kwargs` parameters specify additional positional and keyword arguments, respectively, 
to pass to both `map_func` and `reduce_func` on each invocation.

Returns the result of applying `reduce_func` to the results of `map_func`.

Examples:
- `mapPairsReduce(f, g, [1, 2], [3, 4])` → `g(f(1, 3), f(1, 4), f(2, 3), f(2, 4))`
"""
function mapPairsReduce(
    map_func::Function, 
    reduce_func::Function, 
    xs::AbstractVector, 
    ys::AbstractVector; 
    chunksize::Union{Int, Nothing} = nothing, 
    direction::Symbol = :column, 
    mode::String = "1d", 
    node::Int = -1, 
    counted::Bool = true, 
    forced_node::Bool = false, 
    priority::TaskPriorityLevel = TaskPriority.low, 
    args::Tuple = (), 
    kwargs...)

    chunksize = something(chunksize, max(cld(length(xs) * length(ys), ctx[].num_procs * TORCJULIA_NUM_WORKERS[]), 1))
    
    mapPair_mat = mapPairs(map_func, xs, ys; chunksize = chunksize, node = node, counted = counted,
                           forced_node = forced_node, priority = priority, args = args, kwargs...)

    reduce(reduce_func, mapPair_mat; chunksize = chunksize, direction = direction, mode = mode, node = node, counted = counted,
           forced_node = forced_node, priority = priority, args = args, kwargs...)
end

"""
    scan(func, data; chunksize::Union{Int, Nothing} = nothing, mode::String = "inclusive", identity::Union{Any, Nothing} = nothing, 
         node::Int = -1, counted::Bool = true, forced_node::Bool = false, priority::TaskPriorityLevel = TaskPriority.low, 
         args::Tuple = (), kwargs...)

Computes the prefix scan of `data` in parallel using the specified binary associative operation `func`.

The `mode` parameter determines the type of scan:
- `"inclusive"`: each output element includes the corresponding input element (Default).
- `"exclusive"`: each output element excludes the corresponding input element, starting with the specified `identity` value.

The `identity` parameter specifies the identity element of `func` (required for exclusive scans).
It defaults to zero(T), where T is the element type of input data.

The `chunksize` parameter specifies the number of elements processed per task. 
If omitted, the runtime determines the chunk size automatically based on the number of available workers.

The `args` and `kwargs` parameters specify additional positional and keyword arguments, respectively, to pass to `func` on each invocation.

Returns an array containing the prefix scan results, with the same length as `data`.

Examples:
- `scan(+, [1, 2, 3, 4], mode = "inclusive")` → `[1, 3, 6, 10]`
- `scan(+, [1, 2, 3, 4], mode = "exclusive", identity = 0)` → `[0, 1, 3, 6]`
"""
function scan(
    func::Function, 
    data::AbstractVector; 
    chunksize::Union{Int, Nothing} = nothing, 
    mode::String = "inclusive", 
    identity::Union{Any, Nothing} = nothing, 
    node::Int = -1, 
    counted::Bool = true, 
    forced_node::Bool = false, 
    priority::TaskPriorityLevel = TaskPriority.low, 
    args::Tuple = (), 
    kwargs...)
    
     if !(mode in ("inclusive", "exclusive"))
        @error "mode must be \"inclusive\" or \"exclusive\""
        MPI.Abort(ctx[].comm, 1)
    end
    
    T = eltype(data)

    if identity !== nothing && !(identity isa T)
        @error "identity type must match data element type"
        MPI.Abort(ctx[].comm, 1)
    end

    ndim = length(data)
    chunksize = something(chunksize, max(cld(ndim, ctx[].num_procs * TORCJULIA_NUM_WORKERS[]), 1))

    prefix_chunk = (x, args...; kwargs...) -> begin
        _x = collect(x)
        res = similar(_x)
        
        res[1] = _x[1]
        
        for i in 2:length(_x)
            res[i] = func(res[i-1], _x[i], args...; kwargs...)
        end
        
        res
    end
    
    _wrapper_prefix = x -> prefix_chunk(x, args...; kwargs...)
    
    futures = [submit(_wrapper_prefix, ch; node = node, counted = counted, forced_node = forced_node, priority = priority) 
               for ch in Iterators.partition(data, chunksize)]
    
    _waitall(false)
    
    chunk_pref = [result(f) for f in futures]

    _chunk_acc = [c[end] for c in chunk_pref]
    
    _total_acc = similar(_chunk_acc); _total_acc[1] = _chunk_acc[1]
    
    for i in 2:length(_chunk_acc)
        _total_acc[i] = func(_total_acc[i-1], _chunk_acc[i], args...; kwargs...)
    end

    _offsets = [zero(T); _total_acc[1:end-1]]

    _reduce_chunks = (c, offset, args...; kwargs...) -> begin
        _x = collect(c); [func(x, offset, args...; kwargs...) for x in _x]
    end

    _wrapper_reduce = (x, offset) -> _reduce_chunks(x, offset, args...; kwargs...)
    
    futures = [submit(_wrapper_reduce, ch, off; node = node, counted = counted, forced_node = forced_node, priority = priority)
               for (ch, off) in zip(chunk_pref, _offsets)]

    _waitall(false)
    
    results = [result(f) for f in futures]

    if mode == "inclusive"
        return vcat(results...)
    end
    
    identity = identity === nothing ? zero(T) : identity
        
    [identity; vcat(results...)[1:end-1]]
end

"""
    add_worker(; num_procs::Int = 1, weights::Union{Vector{Int}, Nothing} = nothing)

Launches additional MPI nodes (workers) in the torcjulia runtime environment.

The `num_procs` parameter specifies the number of new MPI nodes to launch.

The `weights` parameter is an optional vector of integers specifying the weight for each new node, which is required 
if the selected distribution policy is Weighted Round-Robin (WRR). The length of `weights` must match `num_procs`.
"""
function add_worker(; num_procs::Int = 1, weights::Union{Vector{Int}, Nothing} = nothing)

    if TORCJULIA_DISTRIBUTION_POLICY[] == WRR
        if isnothing(weights)
            @error "Node launch aborted: Weighted Round-Robin (WRR) policy requires node weights"
            return 
            
        elseif length(weights) != num_procs
            @error "Node launch aborted: expected $(num_procs) weights for WRR, got $(length(weights))"
            return
        end
    end

    function _launch_mpi_workers(; num_procs::Int, weights::Union{Vector{Int}, Nothing})
        threads = TORCJULIA_NUM_WORKERS[] + (TORCJULIA_USE_MANAGER[] ? 1 : 0)

        if VERSION >= v"1.12"
            threads = "$threads,0"
        end

        MPI.Barrier(ctx[].comm)

        intercomm = MPI.Comm_spawn(
                    joinpath(Sys.BINDIR, "julia"),
                    [
                        "--threads=$threads",
                        PROGRAM_FILE
                    ],
                    num_procs,
                    ctx[].comm
                )
        
        # Merge the intercommunicator into a new intracommunicator that includes both the parent (initial) and child processes
        ctx[].comm = MPI.Intercomm_merge(intercomm, false)
        ctx[].num_procs = MPI.Comm_size(ctx[].comm)

        MPI.Barrier(ctx[].comm)
        
        if TORCJULIA_DISTRIBUTION_POLICY[] == WRR
            append!(node_weights[], weights)
            node_weights_cumsum[] = cumsum(node_weights[])
        end
        
        node_weights_cumsum[] = TORCJULIA_DISTRIBUTION_POLICY[] == WRR ? cumsum(node_weights[]) : node_weights_cumsum[]

        if ctx[].rank == 0
            message = TORCJULIA_STEALING_ENABLED[] ? "enable_stealing" : "disable_stealing"
            
            msg_dict = Dict(:type => message)

            # Set the state (enabled/disabled) of the stealing mechanism on the newly added nodes 
            foreach(node -> MPI.send(msg_dict, ctx[].comm; dest = Int(node), tag = TORCJULIA_MANAGER_TAG), (ctx[].num_procs - num_procs):(ctx[].num_procs - 1))
            
            # Configure the scheduling policy on the newly added nodes
            tasks = [submit(_set_policy, TORCJULIA_DISTRIBUTION_POLICY[], node_weights[]; node = Int(node), counted = false, 
                            forced_node = true) for node in (ctx[].num_procs - num_procs):(ctx[].num_procs - 1)]
            
            _waitspec(tasks, false)
        end
    end

    spmd(_launch_mpi_workers; num_procs = num_procs, weights = weights, counted = false, forced_node = true) 
end

"""
    remove_worker(; node_id::Int)

Removes the specified MPI node from the torcjulia runtime environment.

The `node_id` parameter specifies the ID of the MPI node to remove (0-based index).

Warning: Only MPI nodes that were not part of the initial torcjulia runtime configuration can be removed. 
"""
function remove_worker(; node_id::Int)
    if node_id < 0 || node_id >= ctx[].num_procs
        @error "Node removal aborted: invalid node ID ($node_id)"
        return
    end

    if node_id < TORCJULIA_INIT_NODES[]
        @error "Node removal aborted: initial (parent) nodes cannot be removed"
        return
    end

    """
        _split_comm(node::Int) [Helper]
    
    Splits the communicator to exclude the specified node (`node`).
    """
    function _split_comm(node::Int)
        updated_comm = MPI.Comm_split(ctx[].comm, 0, ctx[].rank)
        
        ctx[].comm = updated_comm
        ctx[].rank = MPI.Comm_rank(ctx[].comm)
        ctx[].num_procs = MPI.Comm_size(ctx[].comm)

        if TORCJULIA_DISTRIBUTION_POLICY[] == WRR
            deleteat!(node_weights[], node + 1)
            node_weights_cumsum[] = cumsum(node_weights[])
        end

        MPI.Barrier(ctx[].comm)

        nothing
    end

    """
        _get_stats()::Vector{Float64} [Helper]

    Displays and returns task execution statistics for the calling MPI process.

    This function is called by a worker thread on the MPI process being removed.
    """
    function _get_stats()::Vector{Float64}
        created = Threads.atomic_xchg!(torcjulia_created, UInt(0))
        executed = Threads.atomic_xchg!(torcjulia_executed, UInt(0))
        
        stole = Threads.atomic_xchg!(torcjulia_stole, UInt(0))
        steal_attempts = Threads.atomic_xchg!(torcjulia_steal_attempts, UInt(0))
        
        max_queue_depth = Threads.atomic_xchg!(torcjulia_max_queue_depth, UInt(0))

        (pending_sum, pending_cnt), (exec_sum, exec_cnt) = _agg_time_stats(node_tasks_pending_time, node_tasks_execution_time)

        _local = Float64[pending_sum, pending_cnt, exec_sum, exec_cnt]

        println(Crayon(foreground = :green)(
            @sprintf("torcjulia: [node %d]: created = %d, executed = %d, stole = %d, stolen = %d, steal_attempts = %d, max_queue_depth = %d",
                        ctx[].rank, created, executed, stole, torcjulia_stolen[], steal_attempts, max_queue_depth)
        ))

        flush(stdout)
        
        _local
    end

    task = submit(_get_stats; node = node_id, priority = TaskPriority.high, counted = false, forced_node = true)

    wait(task)

    global task_execution_stats

    # Record the task execution statistics from the removed node
    task_execution_stats .+= result(task)

    # Trigger the specified node to initiate its shutdown
    MPI.send(Dict(:type => "remove"), ctx[].comm; dest = node_id, tag = TORCJULIA_MANAGER_TAG)

    tasks = [submit(_split_comm, node_id; node = Int(node), priority = TaskPriority.high, counted = false, forced_node = true) 
                    for node in 0:ctx[].num_procs - 1 if node != node_id]
  
    wait(tasks)
    
    @warn "MPI node $node_id shut down successfully"
    
    nothing
end

"""
    _set_scheduling_policy(policy::Symbol, weights::Union{Vector{Int}, Nothing})

Sets the task scheduling policy used by the torcjulia runtime environment.

The `policy` argument specifies the scheduling policy and must be one of the following symbols:
- `:ROUND_ROBIN`: Distributes tasks across MPI nodes in a round-robin (cyclic) fashion.
- `:WRR`: Distributes tasks according to the specified weights, with one weight per MPI node.
- `:LOCAL`: Assigns each task to the MPI node that submitted it.
- `:RANDOM`: Assigns each task to a randomly selected MPI node.

The `weights` argument is used only when `policy` is `:WRR`. It must be a vector of integers, with each element specifying the weight of the
corresponding MPI node. For example, with four MPI nodes and weights `[1, 1, 2, 3]`, node 0 and node 1 will receive tasks in a 1:1 ratio, 
node 2 will receive tasks in a 2:1 ratio compared to nodes 0 and 1, and node 3 will receive tasks in a 3:1 ratio compared to nodes 0 and 1. 

For all other policies, `weights` is ignored.
"""
function _set_scheduling_policy(policy::Symbol, weights::Union{Vector{Int}, Nothing})
    if policy == :WRR
        if isnothing(weights)
            @warn "Node weights not provided for Weighted Round-Robin (WRR) -> falling back to Round-Robin"
            policy = :ROUND_ROBIN
        
        elseif length(weights) != ctx[].num_procs
            @warn "Weights size ($(length(weights))) doesn't match the number of nodes ($(ctx[].num_procs)) -> falling back to Round-Robin"
            policy = :ROUND_ROBIN
        end
    end

    spmd(_set_policy, _parse_policy(policy), weights; counted = false, forced_node = true)
end

"""
    _get_scheduling_policy()::DistributionPolicy [Helper]

Returns the current task scheduling policy used by the torcjulia runtime environment.
"""
_get_scheduling_policy()::DistributionPolicy = TORCJULIA_DISTRIBUTION_POLICY[]

"""
    _enable_stealing() [Helper]

Enables the work stealing mechanism.
"""
function _enable_stealing()
    # By default, `forced_node` is set to `false` when a task is submitted
    # As a result, a task assigned to one MPI process may be stolen by a worker
    # thread on another process where work stealing has already been enabled
    # In this case, work stealing may remain enabled on the MPI process the task was stolen from
    # To prevent this, set `forced_node` to `true`
    spmd(() -> (TORCJULIA_STEALING_ENABLED[] = true), priority = TaskPriority.high, counted = false, forced_node = true)
end

"""
    _disable_stealing() [Helper]

Disables the work stealing mechanism.
"""
function _disable_stealing()
    # By default, `forced_node` is set to `false` when a task is submitted
    # As a result, a task assigned to one MPI process may be stolen by a worker
    # thread on another process where work stealing has not yet been disabled
    # In this case, work stealing may remain enabled on the MPI process the task was stolen from
    # To prevent this, set `forced_node` to `true`
    spmd(() -> (TORCJULIA_STEALING_ENABLED[] = false), priority = TaskPriority.high, counted = false, forced_node = true)
end

"""
    _get_stealing_status() [Helper]

Returns the current status of the work stealing mechanism.

If work stealing is enabled, returns `true`; otherwise, returns `false`.
"""
function _get_stealing_status()::Bool
    TORCJULIA_STEALING_ENABLED[]
end

"""
    gettime()::Float64 [Helper]

Returns the current time in seconds.
"""
function gettime()::Float64
    time_ns() * 1e-9
end

end