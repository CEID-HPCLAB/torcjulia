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
Example: Parallel Flux Measurement of Radio Sources using torcjulia

Description: This example demonstrates parallel radio-source detection and flux measurement in FITS images using torcjulia.
             
             FITS images are extracted from a tar archive and processed to identify radio sources. The processing
             pipeline applies kappa-sigma clipping, performs threshold-based source detection using connected-component labeling,
             and computes integrated source fluxed and intensity-weighted centers of mass. 

Notes:
    - Dataset: The FITS images are part of the LOFAR Two-metre Sky Survey (LoTSS) Data Release 2 dataset for the P9Hetdex01 field. 
               The dataset is available at SURF Data Repository (https://repository.surfsara.nl/datasets/lotss-dr2/P9Hetdex01#).
    
    - Images: The FITS images contain radio-source intensity measurements in Jy/beam, obtained with the LOFAR telescope for the 
              P9Hetdex01 field, at a reference frequency of approximately 146.255 MHz.

    - MPI Shared-Memory Windows: Images and corresponding labels are stored in MPI shared-memory windows allocated with `torcjulia.shw_alloc`,
                                 allowing MPI ranks within the same memory island to directly access the corresponding data.
    
    - Nested Parallelism: Image processing tasks spawn additional tasks to process image chunks concurrently. Partial flux sums 
                          and weighted-coordinate accumulations are reduced to obtain the final measurements for each source.

Warning: This example requires a tar archive (`images.tar`, ~20 GiB) containing the FITS image files. The archive can be downloaded from the SURF Data 
         Repository (https://repository.surfsara.nl/datasets/lotss-dr2/P9Hetdex01#). By default, the example looks for the archive in 
         `examples/flux_measurement/data/`. To use a different location, set the global variable `IMAGES_DIR` accordingly.

Requirements: 8 MPI processes, 1 worker thread per process (shared-memory computing platform).

Julia < v1.12:
Run: mpiexecjl -n 8 julia --project=/path/to/torcjulia/project --threads 2 pipeline.jl

Julia >= v1.12:
Run: mpiexecjl -n 8 julia --project=/path/to/torcjulia/project --threads 2,0 pipeline.jl
"""

using torcjulia

using AstroImages
using ImageMorphology

using Tar

using Statistics

const IMAGES_DIR = joinpath(@__DIR__, "data")

const IMAGE_TAR_FILE = joinpath(IMAGES_DIR, "images.tar")

const IMAGE_FILES = [
    "image_full_ampphase_di_m.NS_Band0_shift.int.facetRestored.fits",
    "image_full_ampphase_di_m.NS_Band1_shift.int.facetRestored.fits",
    "image_full_ampphase_di_m.NS_Band2_shift.int.facetRestored.fits",
    "image_full_ampphase_di_m.NS_shift.int.facetRestored.fits"
]

function get_spatial_dims(tar_file::String, image_file::String)::Vector{Int}
    naxis = 0; dims_dict = Dict{Int, Int}()

    try
        open(ignorestatus(`tar -xOf $tar_file $image_file`)) do io
            while !eof(io)
                rec = String(read(io, 80))
                
                if startswith(rec, "END     ")
                    break
                end

                m = match(r"^NAXIS(\d*)\s*=\s*(\d+)", rec)

                if m !== nothing
                    if isempty(m.captures[1])
                        naxis = parse(Int, m.captures[2])
                    else
                        dim_idx = parse(Int, m.captures[1])
                        dim_val = parse(Int, m.captures[2])
                        dims_dict[dim_idx] = dim_val
                    end
                end
            end
        end
    
    catch e
        if !(e isa Base.IOError)
            rethrow(e)
        end
    end

    collect(Base.filter(x -> x != 1, ntuple(i -> dims_dict[i], naxis)))
end

function read_image(tar_file::String, image_file::String)::AstroImage
    ram_dir = mktempdir("/dev/shm"; prefix = "fits_")

    try
        Tar.extract(hdr -> endswith(hdr.path, image_file), tar_file, ram_dir)

        extracted_files = readdir(ram_dir; join = true)

        isempty(extracted_files) && error("Could not extract $image_file from $tar_file.")

        return AstroImage(only(extracted_files))
    
    finally
        rm(ram_dir; force = true, recursive = true)
    end
end

function kappa_sigma_clip(data::AbstractArray{T}, kappa::Real; max_iter::Int = 5)::AbstractArray{T} where T<:Real
    data = Base.filter(!isnan, vec(data))

    for _ in 1:max_iter
        μ = mean(data)
        σ = std(data)

        mask = (data .>= μ - kappa * σ) .& (data .<= μ + kappa * σ)

        if all(mask) break end

        data = data[mask]
    end

    data
end

function compute_flux_chunk(shw_images::Array{Float32}, shw_labels::Array{Int64}, index::Int, 
                            start_idx::Int, end_idx::Int, num_labels::Int)::Tuple{Vector{Float64}, Matrix{Float64}}
    
    image_data = view(shw_images[:, :, index], :)
    labels = view(shw_labels[:, :, index], :)

    num_dims = ndims(image_data)

    pixels_sum = zeros(Float64, num_labels)
    weighted_coords = zeros(Float64, ndims(image_data), num_labels)

    ranges = ntuple(d -> d == num_dims ? (start_idx:end_idx) : (1:size(image_data, d)), num_dims)

    # Compute the flux and weighted coordinates for each source label in the corresponding image's chunk 
    @inbounds @fastmath for idx in CartesianIndices(ranges)
        label = labels[idx]

        if label > 0
            pixel_value = image_data[idx]
            
            if !isnan(pixel_value)
                pixels_sum[label] += pixel_value
            end

            for d in 1:num_dims
                weighted_coords[d, label] += pixel_value * idx[d]
            end
        end
    end

    (pixels_sum, weighted_coords)
end

function process_image(index::Int, image_name::String, tar_file::String, shw_images::Array{Float32}, shw_labels::Array{Int64})
    image = read_image(tar_file, image_name)

    image_data = image.data
    
    singl_dims = findall(==(1), size(image_data))

    if !isempty(singl_dims)
        image_data = dropdims(image_data; dims = Tuple(singl_dims))
    end

    k = 3.0

    # Kappa-sigma clipping (kappa value -> 3.0)
    clipped = kappa_sigma_clip(image_data, k)

    # Source detection threshold 
    boundary = std(clipped, corrected = false) * 5.0
    
    mask = Base.map(x -> !isnan(x) && x > boundary, image_data)

    labels = label_components(mask)
    num_labels = maximum(labels)

    if num_labels == 0
        return (image_name, 0, Float64[], Float64[])
    end

    # Store the image data and labels in the shared-memory windows
    shw_images[:, :, index] = image_data
    shw_labels[:, :, index] = labels

    num_dims = ndims(image_data)

    chunk_size = cld(size(image_data)[end], 2)

    node = torcjulia.node_id()

    # Second level of parallelism -> process the image in chunks to compute the flux and weighted coordinates for each source label
    tasks = torcjulia.wait([torcjulia.submit(compute_flux_chunk, shw_images, shw_labels, index, start_idx, 
                              min(start_idx + chunk_size - 1, size(image_data)[end]), num_labels; node = mod(node + size(image_data)[end], torcjulia.num_nodes()))
                              for start_idx in 1:chunk_size:size(image_data)[end]]; return_completed = true)

    pixels_sum = zeros(Float64, num_labels)
    weighted_coords = zeros(Float64, num_dims, num_labels)

    for task in tasks
        chunk_sum, chunk_weighted_coords = torcjulia.result(task)
        
        # Reduce the partial results to obtain the final measurements for each source
        pixels_sum .+= chunk_sum
        weighted_coords .+= chunk_weighted_coords
    end

    coms = Vector{NTuple{num_dims, Float64}}(undef, num_labels)

    @fastmath @inbounds for label in 1:num_labels
        s = pixels_sum[label]
        
        coms[label] = s != 0 ? ntuple(d -> weighted_coords[d, label] / s, Val(num_dims)) : ntuple(d -> NaN, Val(num_dims))
    end
    
    (image_name, num_labels, pixels_sum, coms)
end

function main()
    # Get the spatial dimensions of the FITS images (all images have the same dimensions)
    spatial_dims = get_spatial_dims(IMAGE_TAR_FILE, IMAGE_FILES[1])

    shw_dims = (spatial_dims[1], spatial_dims[2], length(IMAGE_FILES))
    
    # Allocate MPI shared memory windows for the image data and the corresponding labels
    shw_images = torcjulia.shw_alloc(shw_dims, Float32)
    shw_labels = torcjulia.shw_alloc(shw_dims, Int64)

    data = [(index, file) for (index, file) in enumerate(IMAGE_FILES)]

    num_nodes = torcjulia.num_nodes()

    t0 = torcjulia.gettime()
    results = torcjulia.wait([torcjulia.submit(process_image, image[1], image[2], IMAGE_TAR_FILE, shw_images, 
                              shw_labels; node = mod(i - 1, num_nodes)) for (i, image) in enumerate(data)]; return_completed = true)
    
    println("Total Processing Time: $(round(torcjulia.gettime() - t0, digits = 3)) seconds")

    for res in results
        image_name, num_sources, pixels_sum, coms = torcjulia.result(res)
        brightest = length(pixels_sum) > 0 ? maximum(pixels_sum) : 0.0
        
        println("Image: $image_name, Number of Sources: $num_sources, Brightest Source Flux: $(round(brightest, digits = 3)).")
    end
end

torcjulia.init(main)