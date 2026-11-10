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

"""
Example: Multi-Level Task Dependencies in torcjulia

Description: This example demonstrates how multi-level task dependencies can be expressed in torcjulia.

Notes: In this example, Task C depends on Task A, Task D depends on Task C, and Task E depends on Task B. Finally, Task F depends on both Tasks D and E.
       The runtime system automatically enforces these dependencies and schedules tasks for execution once their dependencies have been satisfied. 
       Dependencies can be specified using either positional or keyword arguments. For clarity, the standard output (stdout) of tasks at the same 
       level of the dependency graph is displayed in the same color.

Generated Task Dependency Directed Acyclic Graph (DAG):

    A (2s) ──> C (1s) ──> D (1s) ──┐
                                   ├──> F (1s)
    B (1.5s) ──────> E (0.5s) ─────┘

Workflow:
    1. Task A and Task B start immediately (independent, run in parallel).
    2. Task C waits for Task A to complete.
    3. Task D waits for Task C to complete.
    4. Task E waits only for Task B. Tasks D and E are independent of each other, so they can run in parallel. The same holds for E and C, 
       and in general for any pair of tasks with no dependency path between them (e.g. A and B, or A and E).
    5. Task F waits for both Task D and Task E to complete.

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2 multilevel_dependencies.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2,0 multilevel_dependencies.jl
"""

using torcjulia

using Crayons

function workA(x::Int)::Int
    println(Crayon(foreground = :yellow)("Task A started → Input: $x."))
    
    sleep(2)
    
    result = 2 * x
    
    println(Crayon(foreground = :yellow)("Task A completed → Result: $result."))
    
    result
end

function workB(x::Int)::Int
    println(Crayon(foreground = :yellow)("Task B started → Input: $x."))
    
    sleep(1.5)
    
    result = x + 3
    
    println(Crayon(foreground = :yellow)("Task B completed → Result: $result."))
    
    result
end

function workC(x::torcjulia.Future)::Int
    x = torcjulia.result(x)
    
    println(Crayon(foreground = :cyan)("Task C started → Input: Task A = $x."))

    sleep(1)

    result = 3 * x
    
    println(Crayon(foreground = :cyan)("Task C completed → Result: $result."))
    
    result
end

function workD(; x::torcjulia.Future)::Int
    x = torcjulia.result(x)
    
    println(Crayon(foreground = :green)("Task D started → Input: Task C = $x."))
    
    sleep(1)

    result = x - 5

    println(Crayon(foreground = :green)("Task D completed → Result: $result."))

    result
end

function workE(; x::torcjulia.Future)::Int
    x = torcjulia.result(x)
    
    println(Crayon(foreground = :cyan)("Task E started → Input: Task B = $x."))
    
    sleep(0.5)
    
    result = 5 * x - 10

    println(Crayon(foreground = :cyan)("Task E completed → Result: $result."))

    result
end

function workF(; futures::Tuple{torcjulia.Future, torcjulia.Future})::Int
    x = torcjulia.result(futures[1])
    
    y = torcjulia.result(futures[2])
    
    println(Crayon(foreground = :red)("Task F started → Inputs: Task D = $x, Task E = $y."))
    
    sleep(1)
    
    result = x + y

    println(Crayon(foreground = :red)("Task F completed → Result: $result."))
    
    result
end

function main()
    x = 4; y = 7

    # Comment the following line to disable work stealing 
    torcjulia.enable_stealing()

    taskA = torcjulia.submit(workA, x) # 2 * 4 = 8     
    taskB = torcjulia.submit(workB, y) # 7 + 3 = 10        
    
    taskC = torcjulia.submit(workC, taskA) # 3 * 8 = 24
    
    taskD = torcjulia.submit(workD; x = taskC) # 24 - 5 = 19
    taskE = torcjulia.submit(workE; x = taskB) # 10 * 5 - 10 = 40

    taskF = torcjulia.submit(workF; futures = (taskD, taskE)) # 19 + 40 = 59

    torcjulia.wait(taskF)

    @assert torcjulia.result(taskF) == 59
end

torcjulia.init(main)