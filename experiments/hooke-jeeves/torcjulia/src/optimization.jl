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

function explore(x::Vector{Float64}, delta::Vector{Float64}, fbefore::Float64, obj_fn)::Tuple{Float64, Int}
    fmin = fbefore; funevals = 0
    
    @inbounds for i in 1:length(x)
        orig = x[i]; x[i] = orig + delta[i]
        fplus = obj_fn(x); funevals += 1
        
        if fplus < fmin
            fmin = fplus
        
        else
            delta[i] = -delta[i]; x[i] = orig + delta[i]
            fminus = obj_fn(x); funevals += 1
            
            if fminus < fmin
                fmin = fminus
            else
                x[i] = orig
            end
        end
    end
    
    fmin, funevals
end

function run_trial(x0::Vector{Float64}, rho_begin::Float64, max_iter::Int, 
                  epsmin::Float64, obj_fn)::Tuple{Float64, Vector{Float64}, Int, Int}
    n = length(x0)
    
    xbefore = copy(x0); xnew = Vector{Float64}(undef, n); delta = Vector{Float64}(undef, n)  

    @inbounds for i in 1:n
        delta[i] = abs(xbefore[i] * rho_begin) > 0 ? abs(xbefore[i] * rho_begin) : rho_begin
    end
    
    steplength = rho_begin; fbefore = obj_fn(xbefore)
    
    iters = 0; funevals = 1

    while steplength > epsmin && iters < max_iter
        iters += 1; copyto!(xnew, xbefore)  
        
        fnew, evals = explore(xnew, delta, fbefore, obj_fn)
        
        funevals += evals

        discard = true
        
        while fnew < fbefore && discard
            @inbounds for i in 1:n
                delta[i] = xnew[i] <= xbefore[i] ? -abs(delta[i]) : abs(delta[i])
                tmp = xbefore[i]; xbefore[i] = xnew[i]; xnew[i] = 2 * xnew[i] - tmp
            end
            
            fbefore = fnew; fnew, evals = explore(xnew, delta, fbefore, obj_fn)
            
            funevals += evals
            
            discard = false
            
            @inbounds for i in 1:n
                if abs(xnew[i] - xbefore[i]) > 0.5 * abs(delta[i])
                    (discard = true) && break
                end
            end
        end

        if fnew >= fbefore
            steplength *= rho_begin
            
            @inbounds for i in 1:n
                delta[i] *= rho_begin
            end
        end
    end

    f_min = obj_fn(xbefore); funevals += 1
    
    f_min, xbefore, iters, funevals
end