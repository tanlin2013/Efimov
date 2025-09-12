using ITensors
using ITensorMPS
using ITensorGaussianMPS
using LinearAlgebra
using Plots
using HDF5
using DataFrames
using LaTeXStrings
using DifferentialEquations
using IterTools

N = 4 #Qubit number
s0 = 2.0 #Scaling factor

#Coordinate exp(c(x-d)) with region xi to xf, where x from 0 to 1-2^(-N)
#####
shift = 10
xi = 2.0^(-shift)
scale = 1000.0
c = log(scale / (1.0 - 2.0^(-N)))#log(xf/xi)#/(1.0-2.0^(-N))
d = -log(xi) / c
cd = -log(xi)
#####

sites = siteinds("S=1/2", N) #Build the sites of two-level system
include(joinpath(@__DIR__, "functions.jl")) #Include most of the functions

"V_{xy} = u(x-y)e^{cx} + u(y-x)e^{cy}"
function EXORY(sites::Vector{Index{Int64}}, c::Float64)
	N = length(sites)
	EMPO = MPO(sites)
	#bond3[1] for prime(sites[i])>sites[i], bond3[2] for prime(sites[i])<sites[i]
	bond3 = Index[Index(3, "Link, l=$a") for a in 1:(N-1)]
	A = ITensor(sites[1], prime(sites[1]), bond3[1])
	A[sites[1]=>1, prime(sites[1])=>1, bond3[1]=>1] = 1.0
	A[sites[1]=>1, prime(sites[1])=>2, bond3[1]=>1] = exp(c / 2.0^N)
	A[sites[1]=>2, prime(sites[1])=>1, bond3[1]=>1] = 1.0
	A[sites[1]=>2, prime(sites[1])=>2, bond3[1]=>1] = exp(c / 2.0^N)
	A[sites[1]=>1, prime(sites[1])=>1, bond3[1]=>3] = 1.0
	A[sites[1]=>1, prime(sites[1])=>2, bond3[1]=>3] = 1.0
	A[sites[1]=>2, prime(sites[1])=>1, bond3[1]=>3] = exp(c / 2.0^N)
	A[sites[1]=>2, prime(sites[1])=>2, bond3[1]=>3] = exp(c / 2.0^N)

	A[sites[1]=>1, prime(sites[1])=>2, bond3[1]=>2] = exp(c / 2.0^N)
	A[sites[1]=>2, prime(sites[1])=>1, bond3[1]=>2] = exp(c / 2.0^N)
	EMPO[1] = A

	for a in 2:(N-1)
		A = ITensor(sites[a], prime(sites[a]), bond3[a-1], bond3[a])
		A[sites[a]=>1, prime(sites[a])=>1, bond3[a-1]=>1, bond3[a]=>1] = 1.0
		A[sites[a]=>1, prime(sites[a])=>2, bond3[a-1]=>1, bond3[a]=>1] = exp(c / 2.0^(N - a + 1))
		A[sites[a]=>2, prime(sites[a])=>1, bond3[a-1]=>1, bond3[a]=>1] = 1.0
		A[sites[a]=>2, prime(sites[a])=>2, bond3[a-1]=>1, bond3[a]=>1] = exp(c / 2.0^(N - a + 1))
		A[sites[a]=>1, prime(sites[a])=>1, bond3[a-1]=>3, bond3[a]=>3] = 1.0
		A[sites[a]=>1, prime(sites[a])=>2, bond3[a-1]=>3, bond3[a]=>3] = 1.0
		A[sites[a]=>2, prime(sites[a])=>1, bond3[a-1]=>3, bond3[a]=>3] = exp(c / 2.0^(N - a + 1))
		A[sites[a]=>2, prime(sites[a])=>2, bond3[a-1]=>3, bond3[a]=>3] = exp(c / 2.0^(N - a + 1))

		A[sites[a]=>1, prime(sites[a])=>2, bond3[a-1]=>1, bond3[a]=>2] = exp(c / 2.0^(N - a + 1))
		A[sites[a]=>2, prime(sites[a])=>1, bond3[a-1]=>3, bond3[a]=>2] = exp(c / 2.0^(N - a + 1))
		A[sites[a]=>1, prime(sites[a])=>1, bond3[a-1]=>2, bond3[a]=>2] = 1.0
		A[sites[a]=>2, prime(sites[a])=>2, bond3[a-1]=>2, bond3[a]=>2] = exp(c / 2.0^(N - a + 1))
		EMPO[a] = A
	end

	A = ITensor(sites[N], prime(sites[N]), bond3[N-1])
	A[sites[N]=>1, prime(sites[N])=>2, bond3[N-1]=>1] = exp(c / 2.0)
	A[sites[N]=>2, prime(sites[N])=>1, bond3[N-1]=>3] = exp(c / 2.0)

	A[sites[N]=>1, prime(sites[N])=>1, bond3[N-1]=>2] = 1.0
	A[sites[N]=>2, prime(sites[N])=>2, bond3[N-1]=>2] = exp(c / 2.0)
	EMPO[N] = A

	return EMPO
end

#Prepare MPSs, MPOs
#####
IMPS, IMPO = I(sites)
MOMPO = ADD(FONES(sites), -IMPO) #A matrix with all one elements, using in RG() to build gamma term. Remove digonal term. 
PEXMPO = EX(sites, 1.5 * c)
NEXMPO = EX(sites, -1.5 * c)
P3EXMPO = EX(sites, 3.0 * c)
P2EXMPO = EX(sites, 2.0 * c)
P3EXMPS = P3EXMPO * IMPS
VMPO = EXORY(sites, -c) #Potential term V(p,q)
dx = (exp(c / 2.0^N) - 1.0) #Integral dx'= exp(cx-cd)*dx
for i in 1:N
	P2EXMPO[i] = P2EXMPO[i] * exp(-2.0 * cd / N)
	#P3EXMPS[i] = P3EXMPS[i] * dx^(1 / N) * exp(-3.0 * cd / N)
	VMPO[i] = VMPO[i] * exp(1.0 * cd / N)
end

P3EXMPS = prime(IMPS)* dx^(1) * exp(-3.0 * cd)

HVT = [(VMPO[i] * delta(sites[i], prime(sites[i], 3), prime(sites[i], 2))) * delta(prime(sites[i], 3), sites[i]) for i in 1:N]
#=
C = [combiner(linkinds(VMPO)[i], linkinds(prime(IMPO))[i], linkinds(P3EXMPS)[i]) for i in 1:(N-1)]

VIMPO = MPO(sites)
VIMPO[1] = (HVT[1] * (prime(IMPO)[1] * P3EXMPS[1])) * C[1]
for i in 2:(N-1)
	VIMPO[i] = C[i-1] * (HVT[i] * (prime(IMPO)[i] * P3EXMPS[i])) * C[i]
end
VIMPO[N] = C[N-1] * (HVT[N] * (prime(IMPO)[N] * P3EXMPS[N]))

V = prime(PEXMPO) * (prime(VIMPO) * PEXMPO)
=#

VIMPO = MPO(sites)
for i in 1:N
    VIMPO[i]=HVT[i]*prime(P3EXMPS)[i]
end
V = prime(PEXMPO) * (prime(VIMPO) * PEXMPO)

H = ADD(P2EXMPO, -(s0^2 + 0.25) * V)
INMPS = randomMPS(sites, 2)
#=
#=
HM = mpo2m(H)
vals, vec = eigen(HM)
vec = permutedims(vec, (2, 1))
plot(vals[1:1:5],yscale=log10)
=#

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

include(joinpath(@__DIR__, "DIF2.jl")) #Solve first n states by dmrg() 

mps = tmps[n] #Pick highest bound state (0>E->0)
mps = noprime(NEXMPO * mps) #Back to the real state #NEXMPO * 
nor = inner(mps, mps)
for i in 1:N
	mps[i] = mps[i] / nor^(0.5 / N) #Renormalize mps
end
#=
y=Matrix{Float64}(undef, 5, 2^N)
for j in 1:5
	mps = tmps[j]
	mps = noprime(NEXMPO * mps)
	nor = inner(mps, mps)
	for i in 1:N
		mps[i] = mps[i] / nor^(0.5 / N) #Renormalize mps
	end
	x, y[j, :] = mps2f(mps)
end
=#

include(joinpath(@__DIR__, "RG.jl")) #Run RG flow and get the coupling constant gamma i different time(scale).
@show energy

#energy = h5read(joinpath(@__DIR__, str, str * "_energy(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "energy")
#gamma = h5read(joinpath(@__DIR__, str, str * "_rgflow(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "rgflow")
#std2 = h5read(joinpath(@__DIR__, str, str * "_rgstd2(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "std2")


plot([exp((i - 1.0) * c / 2.0^N) for i in 1:2^N], sqrt.(std2), xscale = :log10, yscale = :log10,
	xticks = [10^((i - 1) * 0.5) for i in 1:7],
	xlabel = L"\Lambda/\Lambda_0", ylabel = L"\sigma", label = "",
	xguidefontsize = 14, yguidefontsize = 14, legendfont = 14, linewidth = 2,
) # it's 2-norm error not std.
#savefig(joinpath(@__DIR__, str, str * "_rgstd2(N,scale,shift)=($(N),$(scale),$(shift)).svg"))


a = (s0^2 + 0.25) - 0.5
b = s0
phi = atan(-a / b) - b * log(scale)
f(x) = b * tan(phi + b * log(x)) + a
plot([exp((i - 1.0) * c / 2.0^N) for i in 1:2^N], gamma, xscale = :log10,
	ylims = (-100, 20), xticks = [10^((i - 1) * 0.5) for i in 1:7],
	ylabel = L"f", xlabel = L"\Lambda/\Lambda_0", label = "",
	xguidefontsize = 14, yguidefontsize = 14, legendfont = 14,# linewidth = 2,
)
plot!([exp((i - 1.0) * c / 2.0^N) for i in 1:2^N], f, xscale = :log10,
	label = "",
	color = :black,
	linewidth = 2,
)

threshold = exp((log(-energy[n, end]) / 2.0 + cd)) #\Lamda^2=E
vline!(>
	[threshold],
	label = "",
	color = :black, linestyle = :dash,
	linewidth = 2,
)
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
