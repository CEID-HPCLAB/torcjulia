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
Example: Nested Parallelism in torcjulia (Parallel Recursive Fibonacci Computation)

Description: This example demonstrates torcjulia's support for nested parallelism through the recursive computation of the n-th Fibonacci number.  
             Multiple levels of parallelism are exposed by recursively submitting two tasks within each task function and waiting for their results, 
             forming a hierarchical dependency graph. To avoid task oversubscription, the depth of recursive task spawning is bounded. 

Note: Since this example is intended to show torcjulia's support for nested parallelism, identical function calls are recomputed rather than memoized. 

Julia < v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 3 nested_parallelism.jl

Julia >= v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 3,0 nested_parallelism.jl
"""

using torcjulia

function fibonacci(n::Int)::Int
    n < 2 && return n
    n < 30 && return fibonacci(n - 1) + fibonacci(n - 2)

    taskA = torcjulia.submit(fibonacci, n - 1)
    taskB = torcjulia.submit(fibonacci, n - 2)

    torcjulia.wait([taskA, taskB])
    torcjulia.result(taskA) + torcjulia.result(taskB)
end

function main()
    N = 45

    t0 = torcjulia.gettime()
   
    value = fibonacci(N)

    elapsed = torcjulia.gettime() - t0
   
    @assert value == 1_134_903_170
    
    println("Fibonacci($N) = $value.")
    println("All tasks completed successfully.")
    println("Total elapsed time: $(round(elapsed, digits = 3)) seconds.")
end

torcjulia.init(main)