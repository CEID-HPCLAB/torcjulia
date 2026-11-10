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
 * Authors (CEID-HPCLAB): Argiris Sofotasios
 * License:               MIT - See LICENSE file in the project root
 * ==================================================================================
=#

"""
Example: Task Prioritazion and Work Stealing in torcjulia

Description: This example demonstrates torcjulia's priority-based task scheduling when work stealing is enabled. The runtime maintains 
             separate queues for different priority levels, and work stealing prioritizes high-priority tasks when dynamically 
             redistributing work among idle workers.

Priority levels:
- `TaskPriority.high`: urgent tasks (green output)
- `TaskPriority.medium`: medium priority tasks (yellow output)
- `TaskPriority.low`: low priority tasks (cyan output, default)

Notes:
- Work stealing is enabled, allowing idle workers to steal pending tasks.
- When stealing, high-priority tasks are preferentially stolen over lower-priority ones.

Workflow:
- 15 general-purpose tasks (low priority) are submitted first.
- 25 priority tasks with random priorities are submitted second.

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3 priority_work_stealing.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3,0 priority_work_stealing.jl
"""

using torcjulia

using Crayons, Random

function priority_task_func(i::Int, p::torcjulia.TaskPriorityLevel)
    color = p == torcjulia.TaskPriority.high ? :green :
            p == torcjulia.TaskPriority.medium ? :yellow : :cyan
    
    println(Crayon(foreground = color)("Priority task $i [$(p.level)] started."))
    
    sleep(0.6 + rand() * 0.8)
    
    println(Crayon(foreground = color)("Priority task $i [$(p.level)] finished."))
end

function generic_task_func(i::Int)
    println(Crayon(foreground = :cyan)("General-purpose task $i [low] started."))
    
    sleep(0.8 + rand() * 0.8)
    
    println(Crayon(foreground = :cyan)("General-purpose task $i [low] finished."))
end

function main()
    Random.seed!(69)
    
    num_generic_tasks = 15; num_priority_tasks = 25

    torcjulia.enable_stealing()

    priorities = [torcjulia.TaskPriority.low, torcjulia.TaskPriority.medium, torcjulia.TaskPriority.high]

    tasks = torcjulia.Future[]

    println(Crayon(foreground = :magenta)("Submitting $num_generic_tasks general-purpose tasks (default low priority)."))
    
    for i in 1:num_generic_tasks
        push!(tasks, torcjulia.submit(generic_task_func, i))
    end

    println(Crayon(foreground = :magenta)("Submitting $num_priority_tasks priority tasks with random priorities."))
   
    for (i, p) in enumerate(rand(priorities, num_priority_tasks))
        push!(tasks, torcjulia.submit(priority_task_func, i, p; priority = p))
    end

    torcjulia.wait(tasks)
    
    println(Crayon(foreground = :magenta)("All tasks completed successfully."))
end

torcjulia.init(main)