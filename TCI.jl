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
	using KrylovKit
	using JuliaFormatter
	using IterTools
	using NDTensors
	using Distributed
	using SpecialFunctions
	using SharedArrays

	import TensorCrossInterpolation as TCI

	# Convert the TensorTrain object to an ITensor MPS object

	using TCIITensorConversion

	import QuanticsGrids as QG
	using QuanticsTCI: quanticscrossinterpolate
	include(joinpath(@__DIR__, "functions.jl"))

	N = 20 # number of bits
	n = 6
	b = 10
	xmin = 0.0
	xmax = 2.0^(N) - 1.0
	tolerance = 1.0e-16
	str = "TCI"
	err2 = SharedArray{Float64}(n, b)
	shift = [(i - 1) * 2 for i in 1:n]
	shi = [Float64(2.0^(-shift[i])) for i in 1:n]
end

for i in 1:n
	err2[i, :] = h5read(joinpath(@__DIR__, str, str * "_error($i)(N,shift,dim,tolerance)=($(N),$(shift[i]),$b,$(tolerance)).h5"), "error")
end
#=
@sync begin
	@distributed for k ∈ 1:n
		function f(x)
			return 1.0 / (x + 2.0^(shift[k] + N))
		end

		z = Vector{Float64}(undef, 2^N)
		for i in 1:2^N
			z[i] = f(i - 1)
		end

		for j in 1:b
			qgrid = QG.DiscretizedGrid{1}(N, xmin, xmax; includeendpoint = false)
			ci, ranks, errors = quanticscrossinterpolate(Float64, f, qgrid; maxbonddim = j, tolerance = tolerance)

			# Construct a TensorTrain object from the TensorCI2 object

			tt = TCI.TensorTrain(ci.tci)

			sites = siteinds("S=1/2", N)
			M = ITensors.MPS(tt)
			M2 = ITensors.MPS(sites)
			for i in 0:N-1
				M2[N-i] = M[i+1]
			end

			x, y, nor = mps2f(M2)
			err2[k, j] = ERROR(y, z)
		end
		h5write(joinpath(@__DIR__,str, str * "_error($k)(N,shift,dim,tolerance)=($(N),$(shift[k]),$b,$(tolerance)).h5"), "error", err2[k,:])
	end
end
=#
#Plot
#flab = Matrix{String}(undef, 1, n)
flab = [latexstring("(x+2^{-$(shift[i])})^{-1}") for k in 1:1, i in 1:n] #~(error=$(@sprintf("%.2e", err2[i,end])))") for k in 1:1, i in 1:n]
colors = palette(:tab10)

plot([err2[i, :] for i in 1:n],
	yscale = :log10,
	#seriestype = :path, marker = :circle, color = :auto, markersize = 6,
	xlabel = L"Bond~dimension", ylabel = latexstring("Error"), label = flab,
	legend = :topright, legendfont = 14, legendtitle = L"TCI",
	xticks = [2i for i in 1:8], #yticks = [1e0, 1e-2, 1e-4, 1e-6, 1e-8],
	xlims = (0.9, 8.1), #ylim = (1.0e-8, 1.0e0),
	xguidefontsize = 18, yguidefontsize = 18,
	xtickfontsize = 14, ytickfontsize = 14,
	linewidth = 3, #aspect_ratio = 120
)

scatter!([bd[i, :] for i in 1:n], [err0[i, :] for i in 1:n],
	palette = palette(:auto)[1:n],
	marker = :star5, markersize = 10,
	label = "",
)
savefig(joinpath(@__DIR__, str, str *"&DMRG_error2(N,shift,dim,tolerance)=($(N),$(shift),$b,$(tolerance)).svg"))

