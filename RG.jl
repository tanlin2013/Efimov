gamma = Vector{Float64}(undef, 2^N)
std2 = Vector{Float64}(undef, 2^N)
yz = Vector{Float64}(undef, 2^N)
z = Vector{Float64}(undef, 2^N)

HVT = [(VMPO[i] * delta(sites[i], prime(sites[i], 3), prime(sites[i], 2))) * delta(prime(sites[i], 3), sites[i]) for i in 1:N]
HMOT = [(MOMPO[i] * delta(sites[i], prime(sites[i], 3), prime(sites[i], 2))) * delta(prime(sites[i], 3), sites[i]) for i in 1:N]


permutation = reverse(collect(IterTools.product(fill(0:1, N)...)))
#=
for (t,pos) in zip(1:2^N,permutation)
	ST = STEP(prime(sites), collect(pos))
	RST = prime(IMPO) - STEP(prime(sites), collect(pos))

	C = [combiner(linkinds(MOMPO)[i], linkinds(RST)[i], linkinds(P3EXMPS)[i]) for i in 1:N-1]    
	MOUV = MPO(sites)
	MOUV[1] = (HMOT[1] * (RST[1] * P3EXMPS[1])) * C[1]
	for i in 2:N-1
		MOUV[i] = C[i-1] * (HMOT[i] * (RST[i] * P3EXMPS[i])) * C[i]
	end
	MOUV[N] = C[N-1] * (HMOT[N] * (RST[N] * P3EXMPS[N]))
	MMPS = noprime(MOUV * mps) #(prime(PEXMPO) * (prime(MOUV) * NEXMPO))*mps
	#MT = inner(IMPS, MMPS)

	C = [combiner(linkinds(VMPO)[i], linkinds(ST)[i], linkinds(P3EXMPS)[i]) for i in 1:N-1]
	VUV = MPO(sites)
	VUV[1] = (HVT[1] * (ST[1] * P3EXMPS[1])) * C[1]
	for i in 2:N-1
		VUV[i] = C[i-1] * (HVT[i] * (ST[i] * P3EXMPS[i])) * C[i]
	end
	VUV[N] = C[N-1] * (HVT[N] * (ST[N] * P3EXMPS[N]))
	VMPS =  noprime(VUV * mps)#(prime(PEXMPO) * (prime(VUV) * NEXMPO))*mps
	#VT = inner(IMPS, VMPS)


	gamma[t] = - inner(VMPS, MMPS) / inner(MMPS, MMPS) #-VT/MT
	std2[t] = inner(MMPS, MMPS) + inner(VMPS, VMPS) - 2.0 * inner(VMPS, MMPS)
	@show t, gamma[t], std2[t]
	t += 1
end
=#

for (t, pos) in zip(1:(2^N), permutation)
	STUV = prime(IMPO) - STEP(prime(sites), collect(pos))
	smps = noprime(STUV * prime(mps))

	#Build gamma term with cutoff STUV
	C = [combiner(linkinds(MOMPO)[i], linkinds(STUV)[i], linkinds(P3EXMPS)[i]) for i in 1:(N-1)]
	MOUV = MPO(sites)
	MOUV[1] = (HMOT[1] * (STUV[1] * P3EXMPS[1])) * C[1]
	for i in 2:(N-1)
		MOUV[i] = C[i-1] * (HMOT[i] * (STUV[i] * P3EXMPS[i])) * C[i]
	end
	MOUV[N] = C[N-1] * (HMOT[N] * (STUV[N] * P3EXMPS[N]))
	#MOUV = prime(PEXMPO) * (prime(MOUV) * NEXMPO)#prime(IMPO) * (prime(PEXMPO) * MOUV)
	MOUV = prime(IMPO) * (STUV * MOUV)
	MOMPS = noprime(MOUV * smps)

	#Build other terms with cutoff STUV
	C = [combiner(linkinds(VMPO)[i], linkinds(STUV)[i], linkinds(P3EXMPS)[i]) for i in 1:(N-1)]
	VUV = MPO(sites)
	VUV[1] = (HVT[1] * (STUV[1] * P3EXMPS[1])) * C[1]
	for i in 2:(N-1)
		VUV[i] = C[i-1] * (HVT[i] * (STUV[i] * P3EXMPS[i])) * C[i]
	end
	VUV[N] = C[N-1] * (HVT[N] * (STUV[N] * P3EXMPS[N]))
	VUV = prime(IMPO) * (STUV * VUV)
	P2 = prime(IMPO) * (STUV * P2EXMPO)
	#VUV = prime(PEXMPO) * (prime(VUV) * NEXMPO)#prime(IMPO) * (prime(PEXMPO) * VUV)

	RGMPS = noprime(ADD(P2, -(s0^2 + 0.25) * VUV) * smps)

	#DMPS = RGMPS - energy[n, end] * mps
	Y = inner(RGMPS, RGMPS) + (energy[n, end]^2) * inner(smps, smps) - 2.0 * energy[n, end] * inner(RGMPS, smps)
	Z = inner(MOMPS, MOMPS)
	YZ = inner(MOMPS, RGMPS) - energy[n, end] * inner(MOMPS, smps)
	gamma[t] = - YZ / Z
	std2[t] = Y + gamma[t] .^ 2 * Z + 2.0 * gamma[t] * YZ
	yz[t] = YZ
	z[t] = Z
	#@show t, gamma[t]
	t += 1
end

z = [reverse(z)[i] for i in 1:(2^N)]
yz = [reverse(yz)[i] * exp((i-1.0) * c/2.0^N - cd) for i in 1:(2^N)]
gamma = [reverse(gamma)[i] * exp((i-1.0) * c/2.0^N - cd) for i in 1:(2^N)] #reverse and mutiply \Lambda
std2 = reverse(std2)

#h5write(joinpath(@__DIR__,str,str*"_rgflow(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "rgflow", gamma)
#h5write(joinpath(@__DIR__,str,str*"_rgstd2(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "std2", std2)
