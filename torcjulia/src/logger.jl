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

module logger

export init_logger, set_logger_level

using Logging, Dates

const ORANGE    = "\x1b[38;5;214m"
const BOLD_GRAY = "\x1b[1;90m"
const CYAN      = "\x1b[36m"
const GREEN     = "\x1b[32m"
const YELLOW    = "\x1b[33m"
const BLUE      = "\x1b[38;5;33m"
const RED       = "\x1b[31m"
const RESET     = "\x1b[0m"

color_message(level, msg) = level == Logging.Debug ? GREEN*msg*RESET  :
                            level == Logging.Info  ? BLUE*msg*RESET   :
                            level == Logging.Warn  ? YELLOW*msg*RESET :
                            level == Logging.Error ? RED*msg*RESET    : msg

level_string(level) = level == Logging.Debug ? "DEBUG" :
                      level == Logging.Info  ? "INFO"  :
                      level == Logging.Warn  ? "WARNING" :
                      level == Logging.Error ? "ERROR" : string(level)

mutable struct TorcLogger <: AbstractLogger
    name::String
    min_level::LogLevel
end

Logging.min_enabled_level(l::TorcLogger) = l.min_level

function Logging.shouldlog(l::TorcLogger, level, _m, g, id)
    if level < l.min_level
        return false
    end
    
    if level == Logging.Debug || level == Logging.Info
        return occursin("torcjulia", string(_m))
    end

    true
end

function Logging.handle_message(l::TorcLogger, level, msg, _m, g, id, file, line; kwargs...)
    t = Dates.format(now(), "yyyy-mm-dd HH:MM:SS.sss")
    
    h = string(ORANGE*"["*ORANGE*l.name*" "*BOLD_GRAY*level_string(level)*BOLD_GRAY*"]"*RESET)
    
    println(CYAN*t*RESET, " ", h, " ", color_message(level, msg))
end

function init_logger(min_level::LogLevel)
    name = join(split(string(@__MODULE__), ".")[2:3], ".") |> lowercase
    
    global_logger(TorcLogger(name, min_level))
end

"""
    set_logger_level(level::LogLevel)

Sets the minimum logging level for the torcjulia runtime environment.

The `level` argument can be `Logging.Debug`, `Logging.Info`, `Logging.Warn`, or `Logging.Error`.
"""
set_logger_level(level::LogLevel) = (current_logger().min_level = level)

end