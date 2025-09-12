using ITensors
using ITensorMPS
using ITensorGaussianMPS
using LinearAlgebra
using LayeredLayouts
using Graphs
using GLM
using DataFrames
using CSV
using StatsModels
using HDF5
using Printf
using Plots
using LightGraphs
using IterTools
using LaTeXStrings

#dmrg() set up 
#######
N = 20 #Number of qubits 
n = 4 #Number of states you want DMRG to search
maxbdim = 256 #maxdim in dmrg()
cutoff = 1.0e-16 #cutoff in dmrg()
sweep = 10 #nsweeps in dmrg()
kdim = 128 #eigsolve_krylovdim in dmrg()
weight = 0.1 #weight in dmrg()
sites = siteinds("S=1/2", N) #Build the sites of two-level system
str = "efimov_e^x"
include(joinpath(@__DIR__,"functions.jl"))
include(joinpath(@__DIR__,str*".jl"))
#######

maxerr = Matrix{Float64}(undef, n, sweep)
maxlinkdim = Matrix{Int}(undef, n, sweep)
energy = Matrix{Float64}(undef, n, sweep)
y = Matrix{Float64}(undef, n, 2^N)
tmps = Vector{MPS}()
inmps = Vector{MPS}(undef, n)


y = h5read(joinpath(@__DIR__,str,str*"_function(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "function")


@time begin
	for i in 1:n
		open(joinpath(@__DIR__,str,"info.txt",str*"_info($i)(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).txt"), "w") do file
			# Redirect standard output to the file
			redirect_stdout(file) do
				energy, mps = dmrg(H, tmps, INMPS; nsweeps = sweep, mindim = 2, maxdim = maxbdim, eigsolve_krylovdim = kdim, cutoff = cutoff, weight = weight)
				push!(tmps, mps)
				x, y[i,:], nor = mps2f(mps)
				h5write(joinpath(@__DIR__,str,"mps.h5",str*"_MPS($i)(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "MPS", mps)
			end
		end
		output_str = read(joinpath(@__DIR__,str,"info.txt",str*"_info($i)(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).txt"), String)
		lines = split(output_str, '\n')
		maxerr[i, :] = [parse(Float64, match(r"maxerr=([0-9.eE+-]+)", line).captures[1])
						for line in lines if occursin("maxerr=", line)]
		maxlinkdim[i, :] = [parse(Int, match(r"maxlinkdim=([0-9]+)", line).captures[1])
							for line in lines if occursin("maxerr=", line)]
		energy[i, :] = [parse(Float64, match(r"energy=([0-9.eE+-]+)", line).captures[1])
						for line in lines if occursin("maxerr=", line)]/factor
	end
end
@show energy

h5write(joinpath(@__DIR__,str,str*"_energy(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "energy", energy)
h5write(joinpath(@__DIR__,str,str*"_error(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "error", maxerr)
h5write(joinpath(@__DIR__,str,str*"_bond_dim(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "bond_dim", maxlinkdim)
h5write(joinpath(@__DIR__,str,str*"_function(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "function", y)



#=Plot
#######
slab = Matrix{String}(undef, 1, n)
flab = Matrix{String}(undef, 1, n)
elab = Matrix{String}(undef, 1, n)
blab = Matrix{String}(undef, 1, n)
markers = Matrix{Symbol}(undef, 1, n)

lab = ["Ground~state", "First~excited~state", "Second~excited~state", "Third~excited~state", "Fourth~excited~state"]
markers[1, :] = [:circle, :cross, :rect, :diamond]
for i ∈ 1:n
	slab[1, i] = latexstring("$(lab[i])")
	elab[1, i] = latexstring("$(lab[i])~(energy=$(@sprintf("%.4e", energy[i,sweep])))")
	#flab[1, i] = "$(lab[i])~(error=$(@sprintf("%.4e", err[i,sweep+1])))"
	#blab[1, i] = "$(lab[i])~(total elements=$(ele[i]))"
end
#######

plot(
	1:sweep-1,
	[replace(abs.((energy[i, 1:end-1] .- energy[i, end]) ./ energy[i, end]), 0.0 => 1e-18) for i ∈ 1:4],
	legend = true,
	seriestype = :path, marker = markers,
	xlabel = L"Sweep",
	ylabel = latexstring("|(E-E_{$sweep})/E_{$sweep}|"),
	label = elab,
	yticks = [1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16],
	xticks = [2i for i in 1:sweep],
	xlim = (1-0.1, sweep+0.1),
	ylim = (1.0e-16, 1.0e0),
	yscale = :log10,
	xguidefontsize = 14,
	yguidefontsize = 14,
	markersize = 3.0,
	linewidth = 2,
)

#savefig("/workspaces/dif/"*str*"_energy(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).svg")
=#

