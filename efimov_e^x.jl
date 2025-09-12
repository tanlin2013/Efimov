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

IMPS, IMPO = I(sites)
NEXMPO = EX(sites, -0.5 * (2.0^N / p))
NEXMPS = noprime(NEXMPO * IMPS)
N2EXMPO = EX(sites, -2.0 * (2.0^N / p))

function EDD(sites::Vector{Index{Int64}}, c::Float64)
	N = length(sites)
	bond3 = Index[Index(3, "link, l=$a") for a ∈ 1:N-1]
	DDMPO = MPO(sites)

	DD = ITensor(sites[1], prime(sites[1]), bond3[1])
	DD[sites[1]=>2, prime(sites[1])=>1, bond3[1]=>1] = 1.0 * exp(-2.0c * 2.0^(0) + 0.5c)
	DD[sites[1]=>1, prime(sites[1])=>2, bond3[1]=>3] = 1.0 * exp(0.5c)
	DD[sites[1]=>2, prime(sites[1])=>1, bond3[1]=>2] = 1.0 * exp(0.5c)
	DD[sites[1]=>1, prime(sites[1])=>2, bond3[1]=>1] = 1.0 * exp(-2.0c * 2.0^(0) + 0.5c)
	DD[sites[1]=>1, prime(sites[1])=>1, bond3[1]=>1] = -1.0 * (exp(-c) + 1.0)
	DD[sites[1]=>2, prime(sites[1])=>2, bond3[1]=>1] = -1.0 * (exp(-c) + 1.0) * exp(-2.0c * 2.0^(0))
	DDMPO[1] = DD

	for a ∈ 2:N-1
		DD = ITensor(sites[a], prime(sites[a]), bond3[a-1], bond3[a])
		DD[sites[a]=>2, prime(sites[a])=>1, bond3[a-1]=>3, bond3[a]=>1] = 1.0 * exp(-2.0c * 2.0^(a - 1))
		DD[sites[a]=>1, prime(sites[a])=>2, bond3[a-1]=>3, bond3[a]=>3] = 1.0
		DD[sites[a]=>2, prime(sites[a])=>1, bond3[a-1]=>2, bond3[a]=>2] = 1.0
		DD[sites[a]=>1, prime(sites[a])=>2, bond3[a-1]=>2, bond3[a]=>1] = 1.0 * exp(-2.0c * 2.0^(a - 1))
		DD[sites[a]=>1, prime(sites[a])=>1, bond3[a-1]=>1, bond3[a]=>1] = 1.0
		DD[sites[a]=>2, prime(sites[a])=>2, bond3[a-1]=>1, bond3[a]=>1] = 1.0 * exp(-2.0c * 2.0^(a - 1))
		DDMPO[a] = DD
	end

	DD = ITensor(sites[N], prime(sites[N]), bond3[N-1])
	DD[sites[N]=>2, prime(sites[N])=>1, bond3[N-1]=>3] = 1.0 * exp(-2.0c * 2.0^(N - 1))
	DD[sites[N]=>1, prime(sites[N])=>1, bond3[N-1]=>1] = 1.0
	DD[sites[N]=>2, prime(sites[N])=>2, bond3[N-1]=>1] = 1.0 * exp(-2.0c * 2.0^(N - 1))
	DD[sites[N]=>1, prime(sites[N])=>2, bond3[N-1]=>2] = 1.0 * exp(-2.0c * 2.0^(N - 1))
	DDMPO[N] = DD
	return DDMPO
end

EDDMPO = EDD(sites, 1.0 / p)
for i in 1:N
	EDDMPO[i] = EDDMPO[i] / (dx2^(0.0 / N) * exp(-2.0d / p / N))
	N2EXMPO[i] = N2EXMPO[i] * (dx2^(1.0 / N) / exp(-2.0d / p / N))
end

INMPS = noprime(NEXMPO * COS(sites, 1000.0)) #Initial MPS in dmrg()
H = ADD(-EDDMPO, -(s0^2 + 0.25) * N2EXMPO) #Hamiltonain MPO in dmrg()
factor = 1.0

#=
HM = zeros(2^N, 2^N)
f(i) = exp(-2.0 * (i - 1.0) / p)
V(i) = exp(-2.0 * (i - 1.0) / p)
for i in 1:2^N
	HM[i, i] = -(1.0 + exp(-1.0 / p)) * f(i)/ dx2 + (s0^2 + 0.25) * V(i)
	if i < 2^N
		HM[i, i+1] = exp(-1.5 / p) * f(i)/ dx2
		HM[i+1, i] = exp(-1.5 / p) * f(i)/ dx2
	end
end

val, vec = eigen(-HM)
vec = permutedims(vec, (2, 1))

val = val / exp(-2.0d/p)
nv = filter(x -> x < 0, val)
ni = findall(x -> x < 0, val)
df = DataFrame(X = ni, Y = -nv)
lin = lm(@formula(Y ~ X), df)
c0, cx = coef(lin)
@show c0, cx
scatter(df.X, df.Y, label="exact diagonalization", xlabel = "number of state", ylabel = "-energy", color = :blue, yscale=:log10)
=#

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
end

x, g = let
	dx = Vector{Float64}(undef, 2^N)
	x = Vector{Float64}(undef, 2^N)
	z = Matrix{Float64}(undef, n, 2^N)
	z = copy(y)
	#x[1] = exp((0.0 / 2.0^N * 2.0^N - d) / p)
	x[1] = 2.0^(-shift)
	for i ∈ 1:2^N
		if 2^N>i
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
			z[j, i] = z[j, i] * exp(-1.0*(i - 1.0) / p)
		end
	end

	for i in 1:n
		enor = (z[i, :].*z[i, :])' * dx *2.0^(N)
		z[i, :] = z[i, :] / enor^0.5
	end
	x, z
end

plot(
	x[1:2^8:end],
	[g[i, 1:2^8:end] for i ∈ 1:n],
	#legend = true,
	xlabel = L"z",#L"e^{kz+c}",
	ylabel = L"e^{-\frac{kz}{2}}ψ(e^{kz+c})",#L"ψ(e^{kz+c})",
	label = slab,
	#ylims=(-0.01,0.02),
	xlims = (minimum(x), 1.0),
	#yticks = -0.01:0.005:0.02,
	xguidefontsize = 14,
	yguidefontsize = 14,
	linewidth = 2,
)
xmax = [argmax(abs.(g[i, :])) for i in 1:n]
vline!(x[xmax], label = "", color = :red, xticks = (x[xmax], [@sprintf("%.4f", xval) for xval in x[xmax]]))
savefig("C:/Users/User/Downloads/julia/MPS_in_Efimov_project/efimov_e^x/efimov_e^x_scaling(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).svg")
savefig("C:/Users/User/Downloads/julia/MPS_in_Efimov_project/efimov_e^x/efimov_e^x_function(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).svg")


x0 = 1:n
e = Vector{Float64}(undef, n)
for i in 1:n
	e[i] = energy[i, sweep]
end
df2 = DataFrame(X = 1:n, Y = log10.(-e))
lin = lm(@formula(Y ~ X), df2)
c02, cx2 = coef(lin)
@show c02, cx2
y2 = 10.0 .^ (cx2 .* x0 .+ c02)
scatter(-e, label = "", color = :red, yscale = :log10, yticks = [10^(i+1) for i in 1:n],markersize = 4.0)
plot!(x0, y2, xlabel = L"Bound~state~number", ylabel = L"-Energy", label = latexstring("y=~$(@sprintf("%.4f", cx2))~x~+~$(@sprintf("%.4f", c02))"), color = :red, yscale = :log10, yticks = [10^(i+1) for i in 1:n],
xguidefontsize = 14,
yguidefontsize = 14,
legendfont=14,
linewidth = 2,)
savefig("C:/Users/User/Downloads/julia/MPS_in_Efimov_project/efimov_e^x/efimov_e^x_spectrum(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).svg")
=#######