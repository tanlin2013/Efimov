shi = SharedArray{Float64}(n)
shift = SharedArray{Int}(n)
for i in 1:n
	shift[i] = (i - 1) * 2
	shi[i] = Float64(2.0^(-shift[i]))
end
IMPS, IMPO = I(sites)

str = "x^-1"
function FD2F(F::MPO)
	N = length(F)
	sites = [noprime(siteinds(F)[i][1]) for i in 1:N]
	H = MPO(sites)
	IMPS, IMPO = I(sites)
	DDMPO = prime(DD(sites, :NBC))
	for j ∈ 1:N
		H[j] = prime(F[j]) * (DDMPO[j] * F[j])
	end
	H = -cfc(H)
	return H
end

#=Plot
slab = Matrix{String}(undef, 1, n)
elab = Matrix{String}(undef, 1, n)
flab = Matrix{String}(undef, 1, n)
for i ∈ 1:n
	slab[1, i] = latexstring("(x+2^{-$(shift[i])})^{-1}")
	#elab[1, i] = latexstring("(x+2^{-$(shift[i])})^{-1}~(energy=$(@sprintf("%.4e", energy[i,end])))")
	flab[1, i] = latexstring("(x+2^{-$(shift[i])})^{-1}~(error=$(@sprintf("%.2e", err[i,end])))")
end

markers = Matrix{Symbol}(undef, 1, n)
markers[1, :] = [:circle, :cross, :rect, :diamond,:utriangle, :star]

plot(
	[err[i, :] for i in 1:n],
	yscale = :log10,
	legend = true,
	seriestype = :path, marker = markers,
	xlabel = L"Sweep",
	ylabel = latexstring("Error"),
	label = flab,
	#yticks = [1e0, 1e-2, 1e-4, 1e-6, 1e-8],
	xticks = [2i for i in 1:sweep],
	xlim = (0.1, sweep+0.1),
	#ylim = (1.0e-8, 1.0e0),
	xguidefontsize = 14,
	yguidefontsize = 14,
	linewidth = 2,
)
savefig(joinpath(@__DIR__,"x^-1","x^-1_error(N,sweep,shift,kdim,cutoff)=($(N),$(sweep),$(shift),$(kdim),$(cutoff)).svg"))
=#