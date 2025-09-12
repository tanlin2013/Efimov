using Distributed
cpu = 7
if nworkers() < cpu
	addprocs(cpu - nworkers())
elseif nworkers() > cpu
	rmprocs(workers()[cpu+1:end])  # Remove excess workers
end

@everywhere begin
	using ITensors
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
	using SharedArrays
	using Optim
	using LaTeXStrings

	#dmrg() set up 
	#######
	N = 20 #Number of qubits 
	n = 6 #Number of states you want DMRG to search
	maxbdim = 128 #maxdim in dmrg()
	cutoff = 1.0e-16 #cutoff in dmrg()
	sweep = 20 #nsweeps in dmrg()
	kdim = 128 #eigsolve_krylovdim in dmrg()
	#weight = 0.0 #weightin dmrg()
	sites = siteinds("S=1/2", N) #Build the sites of two-level system
	include(joinpath(@__DIR__, "functions.jl"))
	include(joinpath(@__DIR__, "x^-1.jl"))
	#######

	energy = SharedArray{Float64}(n, sweep)
	maxerr = SharedArray{Float64}(n, sweep)
	maxlinkdim = SharedArray{Int}(n, sweep)
	y = SharedArray{Float64}(n, 2^N)
	err = SharedArray{Float64}(n, sweep)
end


y = h5read(joinpath(@__DIR__, str, str * "_function(N,sweep,shift,kdim,cutoff)=($(N),$(sweep),$(shift),$(kdim),$(cutoff)).h5"), "function")
bd = [h5read(joinpath(@__DIR__, str, str * "_maxlinkdim(N,sweep,shift,kdim,cutoff)=($(N),$(sweep),$(shift),$(kdim),$(cutoff)).h5"), "bond_dim")[i,end] for i in 1:n, k in 1:1]
err0 = [ERROR(y[i, :], [1.0 / ((j - 1.0) / 2.0^N + shi[i]) for j in 1:2^N]) for i in 1:n, k in 1:1]
@show err0

#=

@sync begin
	@distributed for i ∈ 1:n
		XIMPO = X(sites, shi[i])
		H = FD2F(XIMPO)
		INMPS = IMPS
		@time begin
			open("/workspaces/dif/x^-1_info($i)(N,sweep,shift,kdim,cutoff)=($(N),$(sweep),$(shift[i]),$(kdim),$(cutoff)).txt", "w") do file
				# Redirect standard output to the file
				redirect_stdout(file) do
					#for j in 1:sweep
						energy, mps = dmrg(H, INMPS; nsweeps = sweep, mindim = 2, maxdim = maxbdim, eigsolve_krylovdim = kdim, cutoff = cutoff)
						x, y[i, :], nor = mps2f(mps)
						err[i,j] = ERROR(y[i,:],[1.0 / ((j - 1.0) / 2.0^N + shi[i]) for j in 1:2^N])
						INMPS = mps
					#end
					h5write("/workspaces/dif/mps.h5/x^-1_MPS($i)(N,sweep,shift,kdim,cutoff)=($(N),$(sweep),$(shift[i]),$(kdim),$(cutoff)).h5", "MPS", INMPS)
					#h5write("/workspaces/dif/mps.h5/x^-1_error($i)(N,sweep,shift,kdim,cutoff)=($(N),$(sweep),$(shift[i]),$(kdim),$(cutoff)).h5", "error", err)
				end
			end
			output_str = read("/workspaces/dif/x^-1_info($i)(N,sweep,shift,kdim,cutoff)=($(N),$(sweep),$(shift[i]),$(kdim),$(cutoff)).txt", String)
			lines = split(output_str, '\n')
			maxerr[i, :] = [parse(Float64, match(r"maxerr=([0-9.eE+-]+)", line).captures[1])
							for line in lines if occursin("maxerr=", line)]
			maxlinkdim[i, :] = [parse(Int, match(r"maxlinkdim=([0-9]+)", line).captures[1])
								for line in lines if occursin("maxerr=", line)]
			energy[i, :] = [parse(Float64, match(r"energy=([0-9.eE+-]+)", line).captures[1])
							for line in lines if occursin("maxerr=", line)]/factor
		end
	end
end
h5write("/workspaces/dif/mps.h5/x^-1_energy(N,sweep,shift,kdim,cutoff)=($(N),$(sweep),$(shift),$(kdim),$(cutoff)).h5", "energy", energy)
h5write("/workspaces/dif/mps.h5/x^-1_maxerr(N,sweep,shift,kdim,cutoff)=($(N),$(sweep),$(shift),$(kdim),$(cutoff)).h5", "error", maxerr)
h5write("/workspaces/dif/mps.h5/x^-1_maxlinkdim(N,sweep,shift,kdim,cutoff)=($(N),$(sweep),$(shift),$(kdim),$(cutoff)).h5", "bond_dim", maxlinkdim)
h5write("/workspaces/dif/mps.h5/x^-1_function(N,sweep,shift,kdim,cutoff)=($(N),$(sweep),$(shift),$(kdim),$(cutoff)).h5", "function", y)

@show energy



plot(
	1:sweep,
	[abs.(energy[i, 1:end]) for i ∈ 1:n],
	legend = true,
	seriestype = :path, marker = markers,
	xlabel = L"Sweep",
	ylabel = latexstring("|E|"),
	label = elab,
	yticks = [1e12, 1e-14, 1e-16, 1e-18, 1e-20],
	xticks = [2i for i in 1:sweep],
	xlim = (1-0.1, sweep+0.1),
	ylim = (1.0e-20, 1.0e-12),
	yscale = :log10,
	xguidefontsize = 14,
	yguidefontsize = 14,
	markersize = 3.0,
	linewidth = 2,
)
savefig("/workspaces/dif/x^-1_energy(N,sweep,shift,kdim,cutoff)=($(N),$(sweep),$(shift),$(kdim),$(cutoff)).svg")
=#