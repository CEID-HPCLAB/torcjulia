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
Example: Nested Task Dependencies in torcjulia

Description: This example illustrates how nested task dependencies can be expressed in torcjulia, enabling dependency-aware dynamic 
             task scheduling with nested parallelism.

Notes: In this example, Task B depends on Task A, and Task D depends on both Tasks B and C. Task D creates its own sub-DAG of tasks (E, F, G, H) 
       that are executed in parallel (nested parallelism). For clarity, the standard output (stdout) of tasks at the same 
       level of the dependency graph is displayed in the same color.

Generated Task Dependency Directed Acyclic Graph (DAG):

    A (2s) ──> B (1s) ──┐
                        ├──> D ──> [E (1.2s) ──┐
    C (1.5s) ───────────┘                      ├──> G (1s) ──> H (0.7s)]
                                    F (0.5s) ──┘

Workflow:
    1. Task A and Task C start immediately (independent, run in parallel).
    2. Task B waits for Task A to complete.
    3. Task D waits for both Task B and Task C to complete.
    4. Task D creates its own sub-DAG of tasks (E, F, G, H) that are executed in parallel.
    5. Task D waits for all its child (nested) tasks to complete before returning its result.

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2 nested_dependencies.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2,0 nested_dependencies.jl
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

function workB(x::torcjulia.Future)::Int
    x = torcjulia.result(x)
    
    println(Crayon(foreground = :cyan)("Task B started → Input: Task A = $x."))

    sleep(1)
    
    result = 4 * x
    
    println(Crayon(foreground = :cyan)("Task B completed → Result: $result."))
    
    result
end

function workC(x::Int)::Int
    println(Crayon(foreground = :yellow)("Task C started → Input: $x."))
    
    sleep(1.5)
    
    result = x - 2
    
    println(Crayon(foreground = :yellow)("Task C completed → Result: $result."))
    
    result
end

function workD(x::torcjulia.Future, y::torcjulia.Future)::Int
    x = torcjulia.result(x) # 40
    y = torcjulia.result(y) # 8
    
    println(Crayon(foreground = :green)("Task D started → Inputs: Task B = $x, Task C = $y."))
 
    taskE = torcjulia.submit(workE, x) # 40 + 1 = 41     
    taskF = torcjulia.submit(workF, y) # 8 * 3 = 24    
    taskG = torcjulia.submit(workG, taskE, taskF) # 41 + 24 = 65
    taskH = torcjulia.submit(workH, taskG) # 65 - 10 = 55

    torcjulia.wait(taskH)

    result = 4 * torcjulia.result(taskH) # 4 * 55 = 220
    
    println(Crayon(foreground = :green)("Task D completed → Result: $result."))

    result
end

function workE(x::Int)::Int
    println(Crayon(foreground = :red)("Task E started → Input: $x."))
    
    sleep(1.2)
    
    result = x + 1
    
    println(Crayon(foreground = :red)("Task E completed → Result: $result."))
    
    result
end

function workF(y::Int)::Int
    println(Crayon(foreground = :red)("Task F started → Input: $y."))
    
    sleep(0.5)
    
    result = y * 3
    
    println(Crayon(foreground = :red)("Task F completed → Result: $result."))
    
    result
end

function workG(x::torcjulia.Future, y::torcjulia.Future)::Int
    x = torcjulia.result(x)
    y = torcjulia.result(y)

    println(Crayon(foreground = :magenta)("Task G started → Inputs: Task E = $x, Task F = $y."))

    sleep(1)
    
    result = x + y
    
    println(Crayon(foreground = :magenta)("Task G completed → Result: $result."))
    
    result
end

function workH(x::torcjulia.Future)::Int
    x = torcjulia.result(x)

    println(Crayon(foreground = (255, 100, 0))("Task H started → Input: Task G = $x."))
    
    sleep(0.7)
    
    result = x - 10
    
    println(Crayon(foreground = (255, 100, 0))("Task H completed → Result: $result."))
    
    result
end

function main()
    x = 5; y = 10
    
    # Comment the following line to disable work stealing 
    torcjulia.enable_stealing()

    taskA = torcjulia.submit(workA, x) # 2 * 5 = 10
    taskB = torcjulia.submit(workB, taskA) # 4 * 10 = 40 
    taskC = torcjulia.submit(workC, y) # 10 - 2 = 8
    taskD = torcjulia.submit(workD, taskB, taskC) # 220

    torcjulia.wait(taskD)

    result = 6 * torcjulia.result(taskD) # 6 * 220 = 1320

    @assert result == 1320
end

torcjulia.init(main)