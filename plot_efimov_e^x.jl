using ITensors
using ITensorMPS
using ITensorGaussianMPS
using LinearAlgebra
using Plots, Colors
using Printf
using HDF5
using DataFrames
using LaTeXStrings
using DifferentialEquations


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
include(joinpath(@__DIR__, "functions.jl"))

s0 = 2.0
shift = 10
pos = zeros(Int, N)
x0 = 2.0^(-shift)
p = (2.0^(N) - 1.0) / log(2.0^(shift) + 1.0 - 2.0^(-N + shift))#Compare to the thesis k, k=2.0^N/p
d = -p * log(x0)#Compare to the thesis d', d'=d/2.0^N
dx2 = ((exp(-2.0 / p) * (exp(1.0 / p) + 1.0) * (exp(1.0 / p) - 1.0)^2) / 2.0)

scale = p * pi / s0
for i in 1:N
	num = sum(pos[j] * 2.0^(j - 1) for j in 1:N)
	pos[N-i+1] = floor((scale - num) / 2.0^(N - i))
end
energy = h5read(joinpath(@__DIR__, str, str * "_energy(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "energy")
y = h5read(joinpath(@__DIR__, str, str * "_function(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "function")
e = [energy[i, sweep] for i in 1:n]

x, g = let
	dx = Vector{Float64}(undef, 2^N)
	x = Vector{Float64}(undef, 2^N)
	z = Matrix{Float64}(undef, n, 2^N)
	z = copy(y)
	#x[1] = exp((0.0 / 2.0^N * 2.0^N - d) / p)
	x[1] = 2.0^(-shift)
	for i ∈ 1:2^N
		if 2^N > i
			#x[i+1] = exp((i / 2.0^N * 2.0^N - d) / p)
			x[i+1] = i / 2.0^N + 2.0^(-shift)
		end
		if 2^N > i
			dx[i] = x[i+1] - x[i]
		else
			dx[2^N] = 2.0^(-N)
			#dx[2^N] = exp(((2.0^N) / 2.0^N * 2.0^N - d) / p) - exp(((2.0^N - 1.0) / 2.0^N * 2.0^N - d) / p)
		end
		for j in 1:n
			z[j, i] = z[j, i] * exp(-1.0 * (i - 1.0) / p)
		end
	end

	for i in 1:n
		enor = (z[i, :] .* z[i, :])' * dx * 2.0^(N)
		z[i, :] = z[i, :] / enor^0.5
	end
	x, z
end

elab = [latexstring("E_{$(i)}=$(@sprintf("%.3e", e[i]))") for k in 1:1, i in 1:n]

g[4, :] = -g[4, :]
scale = Int(floor(2^N * pi / (2.0 * log(2^10 + 1 - 2^(-10)) / (1 - 2^(-20)))))

factor = 0.4
q = 2^(N-2)-scale
colors = [RGB(1.0, 1.0, 1.0) * factor + RGB(c) * (1 - factor) for c in palette(:auto)[1:n]]

plot([x[scale*(i-1)+1:2^14:scale*i] for i in 1:n], [g[i, scale*(i-1)+1:2^14:scale*i] for i in 1:n],
	xlabel = L"x", ylabel = L"\psi(e^{kx+d})", label="",
	xlims = (minimum(x), 1.0),
	xticks = [0.23, 0.45, 0.68, 0.91],
	xguidefontsize = 32, yguidefontsize = 32,
	xtickfontsize = 20, ytickfontsize = 20,
	legend = false, legendfont = 20, linewidth = 15,
	color = [colors[i] for k in 1:1, i in 1:n],
	#palette = palette(:auto)[1:n],
)

plot!(x[1:16:end], [g[i, 1:16:end] for i ∈ 1:n],
	linewidth = 6, palette = palette(:auto)[1:n],
	aspect_ratio = 160, 
)
savefig("C:/Users/User/Downloads/julia/MPS_in_Efimov_project/efimov_e^x/efimov_e^x_scaling(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).svg")

z = Matrix{Float64}(undef, n, 2^N)
for i in 1:n
	z[i,:] = g[i,:] / sum(g[i, scale*(i-1)+1:scale*i].^2)^0.5 
end

plot(x[scale+1:(2^4):2*scale], [z[i,scale*(i-1)+1:(2^4):scale*i] for i in 1:n],
	xlabel = L"x", ylabel = L"\psi(e^{kx+d})",label="",
	xlims = (2.0^(-N)*(-2.0)*scale, 2.0^(-N)*((2^N-1) + scale)),ylims=(-0.01,0.05),
	xticks = [0.23, 0.45], yticks=[],
	xguidefontsize = 32, yguidefontsize = 32,
	xtickfontsize = 20, ytickfontsize = 20,
	legendfont = 20, linewidth = 15, color = [colors[i] for k in 1:1, i in 1:n],
	#palette = palette(:auto)[1:n],
	aspect_ratio = 17.5, 
)

plot!([[ i/2.0^N for i in (2-i)*scale:2^4:(2^N-1) + (2-i)*scale] .+ 2.0^(-10) for i in 1:n], [z[i, 1:2^4:end] for i in 1:n],
	label = elab[:,1:n],xticks = [0.23, 0.45],
	linewidth = 6, 
	palette = palette(:auto)[1:n],
)

#xmax = [argmax(abs.(g[i, :])) for i in 1:n]
#vline!(x[xmax], label = "", color = :red, xticks = (x[xmax], [@sprintf("%.4f", xval) for xval in x[xmax]]))
savefig("C:/Users/User/Downloads/julia/MPS_in_Efimov_project/efimov_e^x/efimov_e^x_scaling_partial(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).svg")

#=
x0 = 1:n
df2 = DataFrame(X = 1:n, Y = log.(-e))
lin = lm(@formula(Y ~ X), df2)
c02, cx2 = coef(lin)
@show c02, cx2
=#

#=
rf = Matrix{Float64}(undef, n, Int(scale / 2) + 1)

for i in 1:n
	norq = sum(g[i, scale*(i-1)+1:scale*i] .^ 2)
	g[i, :] = g[i, :] / norq^0.5
	F = FFTW.rfft(g[i, scale*(i-1)+1:scale*i])
	rf[i, :] = real(abs.(F))
end

plot(collect(0:10), [rf[i, 1:11] for i ∈ 1:n],
	xlabel = L"\omega", ylabel = L"F[ψ](\omega)", label = elab,
	xlim=(-0.3,10.3), ylim=(10^0.68,10^3.1),
	xticks = [2i for i in 0:5],yticks = [10^1.0, 10^2.0, 10^3.0],
	yscale = :log10,
	xguidefontsize = 32, yguidefontsize = 32,
	xtickfontsize = 24, ytickfontsize = 24,
	markers = [:circle :rect :diamond :utriangle], markersize = 12,
	legend = true, legendfont = 8,
	linewidth = 3,
	aspect_ratio = 0.0055,
)
#savefig("C:/Users/User/Downloads/julia/MPS_in_Efimov_project/efimov_e^x/efimov_e^x_scaling_fft(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).svg")

=#