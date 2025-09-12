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
using GLM

#=
using LayeredLayouts
using Graphs
using CSV
using StatsModels
using LightGraphs
using IterTools
using FFTW
using SymPy
=#



function f(u, p, t)
	(u - 3.75)^2 / t / (1.0 - 0.0 / t^2) + u / t
end
#=
u0 = 0.0
tspan = (756.398,1000.0)

prob = ODEProblem(f, u0, tspan)
sol = solve(prob, Tsit5(), alg_hints = [:SSPRKMSVS43], dense = true, reltol = 1e-15, abstol = 1e-15)
=#


N = 20 #Qubit number
s0 = 2.0 #Scaling factor
#Coordinate exp(c(x-d)) with region xi to xf, where x from 0 to 1-2^(-N)
#####
shift = 10
xi = 2.0^(-shift)
scale0 = 1000.0
c = log(scale0 / (1.0 - 2.0^(-N)))#log(xf/xi)#/(1.0-2.0^(-N))
d = -log(xi) / c
cd = -log(xi)
scale = Int(floor(2^N * pi / (2.0 * log(2^10 + 1 - 2^(-10)) / (1 - 2^(-20)))))
#####

sites = siteinds("S=1/2", N) #Build the sites of two-level system
include(joinpath(@__DIR__, "functions.jl")) #Include most of the functions

NEXMPO = EX(sites, -1.5 * c)

#dmrg() set up 
#######
n = 5 #Number of states you want DMRG to search
maxbdim = 128 #maxdim in dmrg()
cutoff = 1.0e-16 #cutoff in dmrg()
sweep = 10 #nsweeps in dmrg()
kdim = 128 #eigsolve_krylovdim in d
weight = 2.0 #weight in dmrg()
factor = 1.0 #energy factor
str = "limit_cycle" #File name for saving


tmps = Vector{MPS}()
nor = Vector{Float64}(undef, n)
y = Matrix{Float64}(undef, 5, 2^N)
x = [2.0^(-20) * i for i in 0:(2^N-1)]

#tmps = [reading_mps(joinpath(@__DIR__,str,"mps.h5",str*"_MPS($i)(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"),sites) for i in 1:n]
energy = h5read(joinpath(@__DIR__, str, str * "_energy(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "energy")
e = [energy[i, sweep] for i in 1:n]
#=
for i in 1:n
	mps = noprime(NEXMPO * tmps[i]) #Back to the real state
	nor[i] = inner(mps, mps)
	x, y[i,:] = mps2f(mps)
	y[i,:] = y[i,:]./nor[i]^(0.5)
end
h5write(joinpath(@__DIR__, str, str * "_resacled_function(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "function", y)
=#

y = h5read(joinpath(@__DIR__, str, str * "_resacled_function(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "function")
elab = [latexstring("E=$(@sprintf("%.3e", e[i]))") for k in 1:1, i in 1:n]



factor = 0.4
q = 2^(N-2)-scale
colors = [RGB(1.0, 1.0, 1.0) * factor + RGB(c) * (1 - factor) for c in palette(:auto)[1:n]]
#=
plot([x[(2^N-scale*i):(2^4):(2^N-scale*(i-1)-1)] for i in 1:n-1], [y[i+1, (2^N-scale*i):(2^4):(2^N-scale*(i-1)-1)] for i in 1:n-1],
	xlabel = L"x", ylabel = L"\tilde{\psi}(e^{kx+d})", label="",
	xlims = (minimum(x), 1.0),
	xticks = [0.09, 0.32, 0.55, 0.77],
	xguidefontsize = 32, yguidefontsize = 32,
	xtickfontsize = 20, ytickfontsize = 20,
	legendfont = 20, linewidth = 15,
	color = [colors[i+1] for k in 1:1, i in 1:n-1],
	#palette = palette(:auto)[1:n],
)

plot!(x[1:16:end], [y[i, 1:16:end] for i in 1:n],
	label = elab,
	linewidth = 6,
	palette = palette(:auto)[2:n+1],
	aspect_ratio = 150,
)
#savefig("C:/Users/User/Downloads/julia/MPS_in_Efimov_project/limit_cycle/limit_cycle_scaling(N,scale,shift)=($(N),$(scale),$(shift)).svg")

z = Matrix{Float64}(undef, n-1, 2^N)
for i in 1:n-1
	z[i,:] = y[i+1,:] / sum(y[i+1, (2^N-scale*i):(2^N-scale*(i-1)-1)].^2)^0.5
end

plot(x[(2^N-scale*3):(2^4):(2^N-scale*(3-1)-1)], [z[i,(2^N-scale*i):(2^4):(2^N-scale*(i-1)-1)] for i in 1:n-1],
	xlabel = L"x", ylabel = L"\tilde{\psi}(e^{kx+d})",label="",
	xlims = (2.0^(-20)*(-2)*scale, 2.0^(-20)*((2^N-1) + scale)),ylims=(-0.01,0.05),
	xticks = [0.32, 0.55], yticks=[],
	xguidefontsize = 32, yguidefontsize = 32,
	xtickfontsize = 20, ytickfontsize = 20, 
	legendfont = 20, linewidth = 15, color = [colors[i] for k in 1:1, i in 2:n],
	palette = palette(:auto)[2:n],
	aspect_ratio = 17.5, 
)


plot!([[2.0^(-20) * i for i in (2-i)*scale:2^4:(2^N-1) + (2-i)*scale] for i in 1:(n-1)], [z[n-i, 1:2^4:end] for i in 1:(n-1)],
	label="",#label = elab[:,2:n],
	linewidth = 6, 
	palette = palette(:auto)[2:n],
)
#savefig("C:/Users/User/Downloads/julia/MPS_in_Efimov_project/limit_cycle/limit_cycle_scaling_partial(N,scale,shift)=($(N),$(scale),$(shift)).svg")



#=
rf = Matrix{Float64}(undef, n, Int(scale / 2) + 1)

for i in 1:n-1
	norq = sum(y[i+1, 2^N-scale*i:2^N-scale*(i-1)-1] .^ 2)
	y[i+1, :] = y[i+1, :] / norq^0.5
	F = FFTW.rfft(y[i+1, 2^N-scale*i:2^N-scale*(i-1)-1])
	rf[i+1, :] = real(abs.(F))
end

plot(collect(0:10), [rf[i, 1:11] for i ∈ 2:n], xlims = (0, 10),
	xlabel = L"\omega", ylabel = L"F[\tilde{\psi}_0](\omega)", label = elab[:, 2:n],
	xlim=(-0.3,10.3), ylim=(10^1.38,10^2.6),
	xticks = [2i for i in 0:5], yticks = [10^1.5,10^2,10^2.5],
	yscale = :log10,
	xguidefontsize = 32, yguidefontsize = 32,
	xtickfontsize = 24, ytickfontsize = 24,
	markers = [:circle :rect :diamond :utriangle], markersize = 12,
	legend = true, legendfont = 8,
	palette = palette(:auto)[2:n],
	linewidth = 3,
	aspect_ratio = 0.0207,
)
#savefig("C:/Users/User/Downloads/julia/MPS_in_Efimov_project/limit_cycle/limit_cycle_scaling_fft(N,scale,shift)=($(N),$(scale),$(shift)).svg")
=#

#=
x0 = 1:n-1
df2 = DataFrame(X = 1:n, Y = log.(-e[1:5]))
lin = lm(@formula(Y ~ X), df2)
c02, cx2 = coef(lin)
@show c02, cx2
y2 = 10.0 .^ (cx2 .* x0 .+ c02)
scatter(-e, label = "", color = :red, yscale = :log10, yticks = [10^(i+1) for i in 1:n-2],markersize = 4.0)
plot!(x0, y2, xlabel = L"Bound~state~number", ylabel = L"-Energy", label = latexstring("y=~$(@sprintf("%.4f", cx2))~x~+~$(@sprintf("%.4f", c02))"), color = :red, yscale = :log10, yticks = [10^(i+1) for i in 1:n-2],
xguidefontsize = 14,
yguidefontsize = 14,
legendfont = 14,
linewidth = 2,)
=#


gamma = h5read(joinpath(@__DIR__, str, str * "_rgflow(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "rgflow")
std2 = h5read(joinpath(@__DIR__, str, str * "_rgstd2(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "std2")
z = h5read(joinpath(@__DIR__, str, str * "_gamma_term(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "gamma_term")
yz = h5read(joinpath(@__DIR__, str, str * "_other_term(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "other_term")
rg_e = h5read("C:/Users/User/Downloads/julia/MPS_in_Efimov_project/limit_cycle/RG_E.h5", "Dataset1")
rg_x = h5read("C:/Users/User/Downloads/julia/MPS_in_Efimov_project/limit_cycle/RG_X.h5", "Dataset1")

a = (s0^2 + 0.25) - 0.5
b = s0
phi = atan(-a / b) - b * log(scale0) #f(scale0)=0
f(x) = b * tan(phi + b * log(x)) + a
plot([exp((i - 1.0) * c / 2.0^N) for i in 1:4:2^N], gamma[1:4:end], xscale = :log10,
	ylims = (-10, 15), xticks = [10^((i - 1) * 0.5) for i in 1:7],
	ylabel = L"f(\Lambda)", xlabel = L"\Lambda/\Lambda_0", label = "",
	xguidefontsize = 16, yguidefontsize = 16, xtickfontsize = 12, ytickfontsize = 12,
	legendfont = 14, linewidth = 2
)
plot!([exp((i - 1.0) * c / 2.0^N) for i in 1:4:2^N], f, xscale = :log10,
	label = "",
	color = :black,
	linewidth = 2,
)
threshold = exp((log(-energy[n, end]) / 2.0 + cd)) #\Lamda^2=E
vline!(
	[threshold],
	label = "",
	color = :black, linestyle = :dash,
	linewidth = 2,
)
plot!(rg_x[100:2^4:end], rg_e[1,100:2^4:end], ylim=(-20,20), xscale=:log10,linewidth = 2,)
#savefig(joinpath(@__DIR__, str, str * "_rgflow(N,scale,shift)=($(N),$(scale),$(shift)).svg"))


plot([exp((i - 1.0) * c / 2.0^N) for i in 5:2^4:2^N], z[5:2^4:end], xscale = :log10,yscale = :log10,
	xticks = [10^((i - 1) * 0.5) for i in 1:7],
	ylabel = L"f", xlabel = L"\Lambda/\Lambda_0", label = "YZ",
	xguidefontsize = 14, yguidefontsize = 14, legendfont = 14, linewidth = 3,
)
plot!([exp((i - 1.0) * c / 2.0^N) for i in 5:2^4:2^N], abs.(yz[5:2^4:end]), 
label = "Z", linewidth = 3,
)

plot([exp((i - 1.0) * c / 2.0^N) for i in 5:2^4:2^N], (yz[5:2^4:end]), 
label = "Z", linewidth = 3,xscale = :log10,
)
=#
