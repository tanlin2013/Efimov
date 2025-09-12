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
using Optim
using QuadGK
using LaTeXStrings

order = 100
n = 6
shi = Vector{Float64}(undef, n)
shift = Vector{Int}(undef, n)
shift = [(i - 1) * 2 for i in 1:n]
shi = [2.0^(-shift[i]) for i in 1:n]
min = Vector{Float64}(undef, n)
fac = Matrix{Float64}(undef, n, 2)
err2 = Matrix{Float64}(undef, n, order + 1)
str = "e^-x2x^-1"

#\int_0^1 (x+shi[i]-x[2]*exp(x[1]))^2
for i in 1:n
	f(x) = ((6.0 * shi[i]^2 + 6.0 * shi[i] + 2) * x[1]^2 
			+ 12.0 * x[2] * (shi[i] * x[1] - 1.0)
			- 12.0 * (shi[i] * x[1] + x[1] - 1.0) * exp(x[1]) * x[2] 
			- 3.0 * x[2]^2 * x[1]
			+ 3.0 * x[1] * x[2]^2 * exp(2.0 * x[1])) / (6.0 * x[1]^2)

	lower = [-10.0, -10.0]
	upper = [10.0, 10.0]

	# Run the optimization with box constraints
	result = optimize(f, lower, upper, [log(2.0), 1.0], Fminbox(BFGS()))
	println("Minimum value: ", result.minimum)
	println("Location of minimum: ", result.minimizer)
	d(x) = (exp(result.minimizer[1] * x) * result.minimizer[2] - (x + shi[i]))^2
	plot(d, 0, 1)
	min[i] = result.minimum
	fac[i, :] = result.minimizer
	#=
	f(x,y) = 1000.0*((6.0 * shi[i]^2 + 6.0 * shi[i] + 2)x^2 + 12.0 * y * (shi[i] * x - 1.0)
			- 12.0 * (shi[i] * x + x - 1.0) * exp(x)*y- 3.0 * y^2 * x
			+ 3.0 * x * y^2*exp(2.0 * x )) / (6.0 * x^2)
	contour(0.6:0.01:0.7, 1.0:0.01:1.1, f, fill=true, levels=40)
	=#
	rnor = Vector{Float64}(undef, order + 1)
	gnor = Vector{Float64}(undef, order + 1)

	for ord in 0:order
		g(x) = sum(exp(-result.minimizer[1] * x * (j + 1.0)) * result.minimizer[2]^(-j - 1.0) * (exp(result.minimizer[1] * x) * result.minimizer[2] - x - shi[i])^j for j in 0:ord)
		r(x) = 1.0 / (shi[i] + x)

		rnor[ord+1], rerror = quadgk(x -> r(x)^2, 0, 1)
		gnor[ord+1], gerror = quadgk(x -> g(x)^2, 0, 1)
		g2(x) = g(x) / gnor[ord+1]^0.5
		r2(x) = r(x) / rnor[ord+1] ^ 0.5
		errf(x) = (g2(x)-r2(x))^2
		inval, error = quadgk(x -> errf(x), 0, 1)
		err2[i, ord+1] = inval ^ 0.5
		if inval^0.5 < 1e-18
			break
		end
	end
end
h5write(joinpath(@__DIR__,str,str*".h5"), "approx", err2)

err2 = h5read(joinpath(@__DIR__,str,str*".h5"), "approx")

slab = Matrix{String}(undef, 1, n)
for i ∈ 1:n
	slab[1, i] = latexstring("1/(x+2^{-$(shift[i])})")
end

p = plot(
	yscale = :log10,
	xlabel = L"Order", ylabel = L"Error",
	xlims = (0, 100), ylims = (1e-18, 1),
	xticks = 0:10:100, yticks = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-8, 1e-12, 1e-16],
	xguidefontsize = 16, yguidefontsize = 16,
	xtickfontsize = 12, ytickfontsize = 12, 
	legend = (0.65, 0.49), legendfontsize = 14,
	)

for i in 1:n
	# Get the values from err2[i, :] and the corresponding indices
	y_values = err2[i, err2[i, :].> 1e-18]
	x_values = 0:length(y_values)-1  # Use the length of the current filtered series
	plot!(p, x_values, y_values, label = slab[1, i], linewidth = 2.5)
end
display(p)
savefig(joinpath(@__DIR__,str,str*".svg"))


