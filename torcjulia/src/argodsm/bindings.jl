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

module ArgoDSMBindings

export argo_init, argo_finalize, argo_barrier, collective_alloc, ArgoDSM_INITIALIZED

using Base: Ptr, Csize_t, Cint, Cvoid
using Libdl

const libargo_path   = joinpath(@__DIR__, "lib", "libargo.so")
const libbackend_path = joinpath(@__DIR__, "lib", "libargobackend-mpi.so")

const _libargo_loaded = Ref(false)

# Flag to indicate whether ArgoDSM has been initialized
const ArgoDSM_INITIALIZED = Ref(false)

function ld_libargo()
    if !_libargo_loaded[]
        Libdl.dlopen(libbackend_path, Libdl.RTLD_GLOBAL)
        Libdl.dlopen(libargo_path, Libdl.RTLD_GLOBAL)
        _libargo_loaded[] = true
    end
end

"""
    argo_init(mem::Int, cache::Int)

Initializes the ArgoDSM runtime environment with the specified memory and cache sizes.

- `mem`: The total memory size (in bytes) to be allocated for the ArgoDSM runtime system.
- `cache`: The cache size (in bytes) to be used by the ArgoDSM runtime system.
"""
function argo_init(mem::Int, cache::Int)
    ld_libargo()
    if !ArgoDSM_INITIALIZED[]
        ccall((:argo_init, libargo_path), Cvoid, (Csize_t, Csize_t), mem, cache)
         ArgoDSM_INITIALIZED[] = true
    end
end

"""
    argo_finalize()

Finalizes the ArgoDSM runtime environment, releasing any resources allocated during initialization.
"""
function argo_finalize()
    if ArgoDSM_INITIALIZED[]
        ccall((:argo_finalize, libargo_path), Cvoid, ())
       ArgoDSM_INITIALIZED[]  = false
    end
end
atexit(argo_finalize)

"""
    argo_node_id()::Int

Returns the ID of the calling MPI process within the ArgoDSM runtime environment.
"""
function argo_node_id()::Int
    ld_libargo()
    ccall((:argo_node_id, libargo_path), Cint, ())
end

"""
    argo_number_of_nodes()::Int

Returns the total number of nodes in the ArgoDSM runtime environment.
"""
function argo_number_of_nodes()::Int
    ld_libargo()
    ccall((:argo_number_of_nodes, libargo_path), Cint, ())
end

"""
    argo_barrier(tag::Int)

Synchronizes all MPI processes in the ArgoDSM runtime environment. 

- `tag`: An integer tag used to identify the barrier operation.
"""
function argo_barrier(tag::Int)
    ld_libargo()
    ccall((:argo_barrier, libargo_path), Cvoid, (Cint,), tag)
end

"""
    collective_alloc(bytes::Int)::Ptr{Cvoid}

Allocates a block of memory of the specified size (in bytes) across all MPI processes in the ArgoDSM runtime environment.

Returns a pointer to the allocated memory block.

- `bytes`: The size of the memory block to allocate (in bytes).
"""
function collective_alloc(bytes::Int)::Ptr{Cvoid}
    ld_libargo()
    ptr = ccall((:collective_alloc, libargo_path), Ptr{Cvoid}, (Csize_t,), bytes)
    if ptr == C_NULL
        error("ArgoDSM collective allocation failed for $bytes bytes")
    end
    
    ptr
end

end