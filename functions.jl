using ITensors
using LinearAlgebra
using DataFrames
using IterTools

"Build identity MPS and MPO for N qubits."
function I(sites::Vector{Index{Int64}})
	N = length(sites)
	d = dim(sites[1])
	IMPO = MPO(sites)
	IMPS = MPS(sites)
	bond1 = Index[Index(1, "Link, l=$a") for a ∈ 1:N-1]
	IO = ITensor(sites[1], prime(sites[1]), bond1[1])
	IS = ITensor(sites[1], bond1[1])
	for i in 1:d
		IO[sites[1]=>i, prime(sites[1])=>i, bond1[1]=>1] = 1.0
		IS[sites[1]=>i, bond1[1]=>1] = 1.0
	end
	IMPO[1] = IO
	IMPS[1] = IS
	for a ∈ 2:N-1
		IO = ITensor(sites[a], prime(sites[a]), bond1[a-1], bond1[a])
		IS = ITensor(sites[a], bond1[a-1], bond1[a])
		for i in 1:d
			IO[sites[a]=>i, prime(sites[a])=>i, bond1[a-1]=>1, bond1[a]=>1] = 1.0
			IS[sites[a]=>i, bond1[a-1]=>1, bond1[a]=>1] = 1.0
		end
		IMPO[a] = IO
		IMPS[a] = IS
	end
	IO = ITensor(sites[N], prime(sites[N]), bond1[N-1])
	IS = ITensor(sites[N], bond1[N-1])
	for i in 1:d
		IO[sites[N]=>i, prime(sites[N])=>i, bond1[N-1]=>1] = 1.0
		IS[sites[N]=>i, bond1[N-1]=>1] = 1.0
	end
	IMPO[N] = IO
	IMPS[N] = IS
	return IMPS, IMPO
end

"Build a full-ones matrix into MPO for N qubits."
function FONES(sites::Vector{Index{Int64}})
	N = length(sites)
	FMPO = MPO(sites)
	bond1 = Index[Index(1, "Link, l=$a") for a ∈ 1:N-1]

	A = ITensor(sites[1], prime(sites[1]), bond1[1])
	A[sites[1]=>1, prime(sites[1])=>1, bond1[1]=>1] = 1.0
	A[sites[1]=>2, prime(sites[1])=>1, bond1[1]=>1] = 1.0
	A[sites[1]=>1, prime(sites[1])=>2, bond1[1]=>1] = 1.0
	A[sites[1]=>2, prime(sites[1])=>2, bond1[1]=>1] = 1.0
	FMPO[1] = A

	for a ∈ 2:N-1
		A = ITensor(sites[a], prime(sites[a]), bond1[a-1], bond1[a])
		A[sites[a]=>1, prime(sites[a])=>1, bond1[a-1]=>1, bond1[a]=>1] = 1.0
		A[sites[a]=>1, prime(sites[a])=>2, bond1[a-1]=>1, bond1[a]=>1] = 1.0
		A[sites[a]=>2, prime(sites[a])=>1, bond1[a-1]=>1, bond1[a]=>1] = 1.0
		A[sites[a]=>2, prime(sites[a])=>2, bond1[a-1]=>1, bond1[a]=>1] = 1.0
		FMPO[a] = A
	end

	A = ITensor(sites[N], prime(sites[N]), bond1[N-1])
	A[sites[N]=>1, prime(sites[N])=>1, bond1[N-1]=>1] = 1.0
	A[sites[N]=>1, prime(sites[N])=>2, bond1[N-1]=>1] = 1.0
	A[sites[N]=>2, prime(sites[N])=>1, bond1[N-1]=>1] = 1.0
	A[sites[N]=>2, prime(sites[N])=>2, bond1[N-1]=>1] = 1.0
	FMPO[N] = A
	return FMPO
end

"Build f(x)=exp(cx) MPO for N qubits."
function EX(sites::Vector{Index{Int64}}, c::Float64)
	N = length(sites)
	EXMPO = MPO(sites)
	bond1 = Index[Index(1, "Link, l=$a") for a ∈ 1:N-1]

	A = ITensor(sites[1], prime(sites[1]), bond1[1])
	A[sites[1]=>1, prime(sites[1])=>1, bond1[1]=>1] = 1.0
	A[sites[1]=>2, prime(sites[1])=>2, bond1[1]=>1] = exp(c / 2.0^N)
	EXMPO[1] = A

	for a ∈ 2:N-1
		A = ITensor(sites[a], prime(sites[a]), bond1[a-1], bond1[a])
		A[sites[a]=>1, prime(sites[a])=>1, bond1[a-1]=>1, bond1[a]=>1] = 1.0
		A[sites[a]=>2, prime(sites[a])=>2, bond1[a-1]=>1, bond1[a]=>1] = exp(c / 2.0^(N - a + 1))
		EXMPO[a] = A
	end

	A = ITensor(sites[N], prime(sites[N]), bond1[N-1])
	A[sites[N]=>1, prime(sites[N])=>1, bond1[N-1]=>1] = 1.0
	A[sites[N]=>2, prime(sites[N])=>2, bond1[N-1]=>1] = exp(c / 2.0)
	EXMPO[N] = A
	return EXMPO
end

"Build f(x)=cos(kx) MPS for N qubits."
function COS(sites::Vector{Index{Int64}}, k::Float64)
	N = length(sites)
	CMPS = MPS(sites)
	bond = Index[Index(2, "Link, l=$a") for a ∈ 1:N-1]

	A = ITensor(sites[1], bond[1])
	A[sites[1]=>1, bond[1]=>1] = 1.0
	A[sites[1]=>1, bond[1]=>2] = 0.0
	A[sites[1]=>2, bond[1]=>1] = cos(k * 1.0 / 2.0^N)
	A[sites[1]=>2, bond[1]=>2] = -sin(k * 1.0 / 2.0^N)
	CMPS[1] = A

	for a in 2:N-1
		A = ITensor(sites[a], bond[a-1], bond[a])
		A[sites[a]=>1, bond[a-1]=>1, bond[a]=>1] = 1.0
		A[sites[a]=>1, bond[a-1]=>1, bond[a]=>2] = 0.0
		A[sites[a]=>1, bond[a-1]=>2, bond[a]=>1] = 0.0
		A[sites[a]=>1, bond[a-1]=>2, bond[a]=>2] = 1.0
		A[sites[a]=>2, bond[a-1]=>1, bond[a]=>1] = cos(k * 1.0 / 2.0^(N - a + 1))
		A[sites[a]=>2, bond[a-1]=>1, bond[a]=>2] = -sin(k * 1.0 / 2.0^(N - a + 1))
		A[sites[a]=>2, bond[a-1]=>2, bond[a]=>1] = sin(k * 1.0 / 2.0^(N - a + 1))
		A[sites[a]=>2, bond[a-1]=>2, bond[a]=>2] = cos(k * 1.0 / 2.0^(N - a + 1))
		CMPS[a] = A
	end

	A = ITensor(sites[N], bond[N-1])
	A[sites[N]=>1, bond[N-1]=>1] = 1.0
	A[sites[N]=>1, bond[N-1]=>2] = 0.0
	A[sites[N]=>2, bond[N-1]=>1] = cos(k * 1.0 / 2.0)
	A[sites[N]=>2, bond[N-1]=>2] = sin(k * 1.0 / 2.0)
	CMPS[N] = A

	return CMPS
end

"Build f(x)=x+shi MPS for N qubits."
function X(sites::Vector{Index{Int64}}, shi::Float64)
	N = length(sites)
	XMPO = MPO(sites)
	bond2 = Index[Index(2, "Link, l=$a") for a ∈ 1:N-1]
	A = ITensor(sites[1], prime(sites[1]), bond2[1])
	A[sites[1]=>2, prime(sites[1])=>2, bond2[1]=>1] = 1.0 / 2.0^(N) + shi / N
	A[sites[1]=>2, prime(sites[1])=>2, bond2[1]=>2] = 1.0
	A[sites[1]=>1, prime(sites[1])=>1, bond2[1]=>1] = 0.0 + shi / N
	A[sites[1]=>1, prime(sites[1])=>1, bond2[1]=>2] = 1.0
	XMPO[1] = A

	for a ∈ 2:N-1
		A = ITensor(sites[a], prime(sites[a]), bond2[a-1], bond2[a])
		A[sites[a]=>2, prime(sites[a])=>2, bond2[a-1]=>1, bond2[a]=>1] = 1.0
		A[sites[a]=>2, prime(sites[a])=>2, bond2[a-1]=>1, bond2[a]=>2] = 0.0
		A[sites[a]=>2, prime(sites[a])=>2, bond2[a-1]=>2, bond2[a]=>1] = 1.0 / 2.0^(N - a + 1) + shi / N
		A[sites[a]=>2, prime(sites[a])=>2, bond2[a-1]=>2, bond2[a]=>2] = 1.0
		A[sites[a]=>1, prime(sites[a])=>1, bond2[a-1]=>1, bond2[a]=>1] = 1.0
		A[sites[a]=>1, prime(sites[a])=>1, bond2[a-1]=>1, bond2[a]=>2] = 0.0
		A[sites[a]=>1, prime(sites[a])=>1, bond2[a-1]=>2, bond2[a]=>1] = 0.0 + shi / N
		A[sites[a]=>1, prime(sites[a])=>1, bond2[a-1]=>2, bond2[a]=>2] = 1.0
		XMPO[a] = A
	end

	A = ITensor(sites[N], prime(sites[N]), bond2[N-1])
	A[sites[N]=>2, prime(sites[N])=>2, bond2[N-1]=>1] = 1.0
	A[sites[N]=>2, prime(sites[N])=>2, bond2[N-1]=>2] = 1.0 / 2.0 + shi / N
	A[sites[N]=>1, prime(sites[N])=>1, bond2[N-1]=>1] = 1.0
	A[sites[N]=>1, prime(sites[N])=>1, bond2[N-1]=>2] = 0.0 + shi / N
	XMPO[N] = A

	return XMPO
end

"Build step f(x)=u(x-x') MPO where pos is the binary number of x'."
function STEP(sites::Vector{Index{Int64}}, pos::AbstractVector{Int})
	N = length(sites)
	SMPO = MPO(sites)
	bond2 = Index[Index(2, "Link, l=$a") for a ∈ 1:N-1]
	S = ITensor(sites[N], prime(sites[N]), bond2[N-1])
	if pos[N] == 1
		S[sites[N]=>2, prime(sites[N])=>2, bond2[N-1]=>2] = 1.0
	else
		S[sites[N]=>2, prime(sites[N])=>2, bond2[N-1]=>1] = 1.0
		S[sites[N]=>1, prime(sites[N])=>1, bond2[N-1]=>2] = 1.0
	end
	SMPO[N] = S

	for a ∈ 2:N-1
		S = ITensor(sites[a], prime(sites[a]), bond2[a-1], bond2[a])
		if pos[a] == 1
			S[sites[a]=>2, prime(sites[a])=>2, bond2[a-1]=>2, bond2[a]=>2] = 1.0
			S[sites[a]=>2, prime(sites[a])=>2, bond2[a-1]=>1, bond2[a]=>1] = 1.0
			S[sites[a]=>1, prime(sites[a])=>1, bond2[a-1]=>1, bond2[a]=>1] = 1.0
		else
			S[sites[a]=>2, prime(sites[a])=>2, bond2[a-1]=>1, bond2[a]=>2] = 1.0
			S[sites[a]=>1, prime(sites[a])=>1, bond2[a-1]=>2, bond2[a]=>2] = 1.0
			S[sites[a]=>2, prime(sites[a])=>2, bond2[a-1]=>1, bond2[a]=>1] = 1.0
			S[sites[a]=>1, prime(sites[a])=>1, bond2[a-1]=>1, bond2[a]=>1] = 1.0
		end
		SMPO[a] = S
	end

	S = ITensor(sites[1], prime(sites[1]), bond2[1])
	if pos[1] == 1
		S[sites[1]=>2, prime(sites[1])=>2, bond2[1]=>2] = 1.0
		S[sites[1]=>2, prime(sites[1])=>2, bond2[1]=>1] = 1.0
		S[sites[1]=>1, prime(sites[1])=>1, bond2[1]=>1] = 1.0
	else
		S[sites[1]=>2, prime(sites[1])=>2, bond2[1]=>2] = 1.0
		S[sites[1]=>1, prime(sites[1])=>1, bond2[1]=>2] = 1.0
		S[sites[1]=>2, prime(sites[1])=>2, bond2[1]=>1] = 1.0
		S[sites[1]=>1, prime(sites[1])=>1, bond2[1]=>1] = 1.0
	end
	SMPO[1] = S
	return SMPO
end

#not test it yet
"Build translation operator into MPO where pos is the translation steps."
function T(sites::Vector{Index{Int64}}, pos::AbstractVector{Int})
	N = length(sites)
	TMPO = MPO(sites)
	bond2 = Index[Index(2, "Link, l=$a") for a ∈ 1:N-1]

	A = ITensor(sites[1], prime(sites[1]), bond2[1])
	if pos[1] == 1
		A[sites[1]=>2, prime(sites[1])=>1, bond2[1]=>2] = 1.0
		A[sites[1]=>1, prime(sites[1])=>2, bond2[1]=>1] = 1.0
	else
		A[sites[1]=>2, prime(sites[1])=>2, bond2[1]=>1] = 1.0
		A[sites[1]=>1, prime(sites[1])=>1, bond2[1]=>1] = 1.0
	end
	TMPO[1] = A

	for a ∈ 2:N-1
		A = ITensor(sites[a], prime(sites[a]), bond2[a-1], bond2[a])
		if pos[a] == 1
			A[sites[a]=>2, prime(sites[a])=>1, bond2[a-1]=>1, bond2[a]=>2] = 1.0
			A[sites[a]=>1, prime(sites[a])=>2, bond2[a-1]=>1, bond2[a]=>1] = 1.0
			A[sites[a]=>2, prime(sites[a])=>2, bond2[a-1]=>2, bond2[a]=>2] = 1.0
			A[sites[a]=>1, prime(sites[a])=>1, bond2[a-1]=>2, bond2[a]=>2] = 1.0
		else
			A[sites[a]=>2, prime(sites[a])=>2, bond2[a-1]=>1, bond2[a]=>1] = 1.0
			A[sites[a]=>1, prime(sites[a])=>1, bond2[a-1]=>1, bond2[a]=>1] = 1.0
			A[sites[a]=>1, prime(sites[a])=>2, bond2[a-1]=>2, bond2[a]=>1] = 1.0
			A[sites[a]=>2, prime(sites[a])=>1, bond2[a-1]=>2, bond2[a]=>2] = 1.0
		end
		TMPO[a] = A
	end

	A = ITensor(sites[N], prime(sites[N]), bond2[N-1])
	if pos[N] == 1
		A[sites[N]=>1, prime(sites[N])=>2, bond2[N-1]=>1] = 1.0
	else
		A[sites[N]=>1, prime(sites[N])=>2, bond2[N-1]=>2] = 1.0
		A[sites[N]=>1, prime(sites[N])=>1, bond2[N-1]=>1] = 1.0
		A[sites[N]=>2, prime(sites[N])=>2, bond2[N-1]=>1] = 1.0
	end
	TMPO[N] = A
	return TMPO
end

"Build second-order differential operator into MPO for N qubits with overall factor dx2. 

NBC for Neumann B.C., DBC for Dirichlet B.C., and, PBC for periodic B.C."
function DD(sites::Vector{Index{Int64}}, which::Symbol = :NBC; dx2::Float64 = 1.0)
	N = length(sites)
	DDMPO = MPO(sites)
	if which == :NBC
		bond5 = Index[Index(5, "Link, l=$a") for a ∈ 1:N-1]

		DD = ITensor(sites[1], prime(sites[1]), bond5[1])
		DD[sites[1]=>2, prime(sites[1])=>1, bond5[1]=>1] = 1.0/dx2^(1/N)
		DD[sites[1]=>1, prime(sites[1])=>2, bond5[1]=>3] = 1.0/dx2^(1/N)
		DD[sites[1]=>2, prime(sites[1])=>1, bond5[1]=>2] = 1.0/dx2^(1/N)
		DD[sites[1]=>1, prime(sites[1])=>2, bond5[1]=>1] = 1.0/dx2^(1/N)
		DD[sites[1]=>1, prime(sites[1])=>1, bond5[1]=>1] = -2.0/dx2^(1/N)
		DD[sites[1]=>2, prime(sites[1])=>2, bond5[1]=>1] = -2.0/dx2^(1/N)
		DD[sites[1]=>1, prime(sites[1])=>1, bond5[1]=>4] = 1.0/dx2^(1/N)
		DD[sites[1]=>2, prime(sites[1])=>2, bond5[1]=>5] = 1.0/dx2^(1/N)
		DDMPO[1] = DD

		for a ∈ 2:N-1
			DD = ITensor(sites[a], prime(sites[a]), bond5[a-1], bond5[a])
			DD[sites[a]=>2, prime(sites[a])=>1, bond5[a-1]=>3, bond5[a]=>1] = 1.0/dx2^(1/N)
			DD[sites[a]=>1, prime(sites[a])=>2, bond5[a-1]=>3, bond5[a]=>3] = 1.0/dx2^(1/N)
			DD[sites[a]=>2, prime(sites[a])=>1, bond5[a-1]=>2, bond5[a]=>2] = 1.0/dx2^(1/N)
			DD[sites[a]=>1, prime(sites[a])=>2, bond5[a-1]=>2, bond5[a]=>1] = 1.0/dx2^(1/N)
			DD[sites[a]=>1, prime(sites[a])=>1, bond5[a-1]=>1, bond5[a]=>1] = 1.0/dx2^(1/N)
			DD[sites[a]=>2, prime(sites[a])=>2, bond5[a-1]=>1, bond5[a]=>1] = 1.0/dx2^(1/N)
			DD[sites[a]=>1, prime(sites[a])=>1, bond5[a-1]=>4, bond5[a]=>4] = 1.0/dx2^(1/N)
			DD[sites[a]=>2, prime(sites[a])=>2, bond5[a-1]=>5, bond5[a]=>5] = 1.0/dx2^(1/N)
			DDMPO[a] = DD
		end

		DD = ITensor(sites[N], prime(sites[N]), bond5[N-1])
		DD[sites[N]=>2, prime(sites[N])=>1, bond5[N-1]=>3] = 1.0/dx2^(1/N)
		DD[sites[N]=>1, prime(sites[N])=>1, bond5[N-1]=>1] = 1.0/dx2^(1/N)
		DD[sites[N]=>2, prime(sites[N])=>2, bond5[N-1]=>1] = 1.0/dx2^(1/N)
		DD[sites[N]=>1, prime(sites[N])=>2, bond5[N-1]=>2] = 1.0/dx2^(1/N)
		DD[sites[N]=>1, prime(sites[N])=>1, bond5[N-1]=>4] = 1.0/dx2^(1/N)
		DD[sites[N]=>2, prime(sites[N])=>2, bond5[N-1]=>5] = 1.0/dx2^(1/N)
		DDMPO[N] = DD
	elseif which == :DBC
		bond3 = Index[Index(3, "Link, l=$a") for a ∈ 1:N-1]

		DD = ITensor(sites[1], prime(sites[1]), bond3[1])
		DD[sites[1]=>2, prime(sites[1])=>1, bond3[1]=>1] = 1.0/dx2^(1/N)
		DD[sites[1]=>1, prime(sites[1])=>2, bond3[1]=>3] = 1.0/dx2^(1/N)
		DD[sites[1]=>2, prime(sites[1])=>1, bond3[1]=>2] = 1.0/dx2^(1/N)
		DD[sites[1]=>1, prime(sites[1])=>2, bond3[1]=>1] = 1.0/dx2^(1/N)
		DD[sites[1]=>1, prime(sites[1])=>1, bond3[1]=>1] = -2.0
		DD[sites[1]=>2, prime(sites[1])=>2, bond3[1]=>1] = -2.0
		DDMPO[1] = DD

		for a ∈ 2:N-1
			DD = ITensor(sites[a], prime(sites[a]), bond3[a-1], bond3[a])
			DD[sites[a]=>2, prime(sites[a])=>1, bond3[a-1]=>3, bond3[a]=>1] =
				1.0/dx2^(1/N)
			DD[sites[a]=>1, prime(sites[a])=>2, bond3[a-1]=>3, bond3[a]=>3] = 1.0/dx2^(1/N)
			DD[sites[a]=>2, prime(sites[a])=>1, bond3[a-1]=>2, bond3[a]=>2] = 1.0/dx2^(1/N)
			DD[sites[a]=>1, prime(sites[a])=>2, bond3[a-1]=>2, bond3[a]=>1] = 1.0/dx2^(1/N)
			DD[sites[a]=>1, prime(sites[a])=>1, bond3[a-1]=>1, bond3[a]=>1] = 1.0/dx2^(1/N)
			DD[sites[a]=>2, prime(sites[a])=>2, bond3[a-1]=>1, bond3[a]=>1] = 1.0/dx2^(1/N)
			DDMPO[a] = DD
		end

		DD = ITensor(sites[N], prime(sites[N]), bond3[N-1])
		DD[sites[N]=>2, prime(sites[N])=>1, bond3[N-1]=>3] = 1.0/dx2^(1/N)
		DD[sites[N]=>1, prime(sites[N])=>1, bond3[N-1]=>1] = 1.0/dx2^(1/N)
		DD[sites[N]=>2, prime(sites[N])=>2, bond3[N-1]=>1] = 1.0/dx2^(1/N)
		DD[sites[N]=>1, prime(sites[N])=>2, bond3[N-1]=>2] = 1.0/dx2^(1/N)
		DDMPO[N] = DD
	elseif which == :PBC
		bond3 = Index[Index(3, "Link, l=$a") for a ∈ 1:N-1]

		DD = ITensor(sites[1], prime(sites[1]), bond3[1])
		DD[sites[1]=>2, prime(sites[1])=>1, bond3[1]=>1] = 1.0/dx2^(1/N)
		DD[sites[1]=>1, prime(sites[1])=>2, bond3[1]=>3] = 1.0/dx2^(1/N)
		DD[sites[1]=>2, prime(sites[1])=>1, bond3[1]=>2] = 1.0/dx2^(1/N)
		DD[sites[1]=>1, prime(sites[1])=>2, bond3[1]=>1] = 1.0/dx2^(1/N)
		DD[sites[1]=>1, prime(sites[1])=>1, bond3[1]=>1] = -2.0/dx2^(1/N)
		DD[sites[1]=>2, prime(sites[1])=>2, bond3[1]=>1] = -2.0/dx2^(1/N)
		DDMPO[1] = DD

		for a ∈ 2:N-1
			DD = ITensor(sites[a], prime(sites[a]), bond3[a-1], bond3[a])
			DD[sites[a]=>2, prime(sites[a])=>1, bond3[a-1]=>3, bond3[a]=>1] = 1.0/dx2^(1/N)
			DD[sites[a]=>1, prime(sites[a])=>2, bond3[a-1]=>3, bond3[a]=>3] = 1.0/dx2^(1/N)
			DD[sites[a]=>2, prime(sites[a])=>1, bond3[a-1]=>2, bond3[a]=>2] = 1.0/dx2^(1/N)
			DD[sites[a]=>1, prime(sites[a])=>2, bond3[a-1]=>2, bond3[a]=>1] = 1.0/dx2^(1/N)
			DD[sites[a]=>1, prime(sites[a])=>1, bond3[a-1]=>1, bond3[a]=>1] = 1.0/dx2^(1/N)
			DD[sites[a]=>2, prime(sites[a])=>2, bond3[a-1]=>1, bond3[a]=>1] = 1.0/dx2^(1/N)
			DDMPO[a] = DD
		end

		DD = ITensor(sites[N], prime(sites[N]), bond3[N-1])
		DD[sites[N]=>2, prime(sites[N])=>1, bond3[N-1]=>3] = 1.0/dx2^(1/N)
		DD[sites[N]=>1, prime(sites[N])=>2, bond3[N-1]=>3] = 1.0/dx2^(1/N)
		DD[sites[N]=>1, prime(sites[N])=>1, bond3[N-1]=>1] = 1.0/dx2^(1/N)
		DD[sites[N]=>2, prime(sites[N])=>2, bond3[N-1]=>1] = 1.0/dx2^(1/N)
		DD[sites[N]=>1, prime(sites[N])=>2, bond3[N-1]=>2] = 1.0/dx2^(1/N)
		DD[sites[N]=>2, prime(sites[N])=>1, bond3[N-1]=>2] = 1.0/dx2^(1/N)
		DDMPO[N] = DD
	else
		error("Invalid which: $which. Must be :NBC, :DBC or :PBC.")
	end
	return DDMPO
end

"Build second-order differential operator in expponetial coordinate into MPO for N qubits with overall factor dx2. B.C.: f'(0)=0 and f(1)=0"
function EDD(sites::Vector{Index{Int64}}, c::Float64; dx2::Float64 = 1.0)
	N = length(sites)
	bond4 = Index[Index(4, "Link, l=$a") for a ∈ 1:N-1]
	DDMPO = MPO(sites)

	DD = ITensor(sites[1], prime(sites[1]), bond4[1])
	DD[sites[1]=>2, prime(sites[1])=>1, bond4[1]=>1] = 1.0 * exp(-2.0c * 2.0^(0) + 0.5c)/dx2^(1/N)
	DD[sites[1]=>1, prime(sites[1])=>2, bond4[1]=>3] = 1.0 * exp(0.5c)/dx2^(1/N)
	DD[sites[1]=>2, prime(sites[1])=>1, bond4[1]=>2] = 1.0 * exp(0.5c)/dx2^(1/N)
	DD[sites[1]=>1, prime(sites[1])=>2, bond4[1]=>1] = 1.0 * exp(-2.0c * 2.0^(0) + 0.5c)/dx2^(1/N)
	DD[sites[1]=>1, prime(sites[1])=>1, bond4[1]=>1] = -1.0 * (exp(-c) + 1.0)/dx2^(1/N)
	DD[sites[1]=>2, prime(sites[1])=>2, bond4[1]=>1] = -1.0 * (exp(-c) + 1.0) * exp(-2.0c * 2.0^(0))/dx2^(1/N)
	DD[sites[1]=>1, prime(sites[1])=>1, bond4[1]=>4] = 0.5 * (exp(-c) + 1.0)/dx2^(1/N)
	DDMPO[1] = DD

	for a ∈ 2:N-1
		DD = ITensor(sites[a], prime(sites[a]), bond4[a-1], bond4[a])
		DD[sites[a]=>2, prime(sites[a])=>1, bond4[a-1]=>3, bond4[a]=>1] = 1.0 * exp(-2.0c * 2.0^(a - 1))/dx2^(1/N)
		DD[sites[a]=>1, prime(sites[a])=>2, bond4[a-1]=>3, bond4[a]=>3] = 1.0/dx2^(1/N)
		DD[sites[a]=>2, prime(sites[a])=>1, bond4[a-1]=>2, bond4[a]=>2] = 1.0/dx2^(1/N)
		DD[sites[a]=>1, prime(sites[a])=>2, bond4[a-1]=>2, bond4[a]=>1] = 1.0 * exp(-2.0c * 2.0^(a - 1))/dx2^(1/N)
		DD[sites[a]=>1, prime(sites[a])=>1, bond4[a-1]=>1, bond4[a]=>1] = 1.0/dx2^(1/N)
		DD[sites[a]=>2, prime(sites[a])=>2, bond4[a-1]=>1, bond4[a]=>1] = 1.0 * exp(-2.0c * 2.0^(a - 1))/dx2^(1/N)
		DD[sites[a]=>1, prime(sites[a])=>1, bond4[a-1]=>4, bond4[a]=>4] = 1.0/dx2^(1/N)
		DDMPO[a] = DD
	end

	DD = ITensor(sites[N], prime(sites[N]), bond4[N-1])
	DD[sites[N]=>2, prime(sites[N])=>1, bond4[N-1]=>3] = 1.0 * exp(-2.0c * 2.0^(N - 1))/dx2^(1/N)
	DD[sites[N]=>1, prime(sites[N])=>1, bond4[N-1]=>1] = 1.0/dx2^(1/N)
	DD[sites[N]=>2, prime(sites[N])=>2, bond4[N-1]=>1] = 1.0 * exp(-2.0c * 2.0^(N - 1))/dx2^(1/N)
	DD[sites[N]=>1, prime(sites[N])=>2, bond4[N-1]=>2] = 1.0 * exp(-2.0c * 2.0^(N - 1))/dx2^(1/N)
	DD[sites[N]=>1, prime(sites[N])=>1, bond4[N-1]=>4] = 1.0/dx2^(1/N)
	DDMPO[N] = DD
	return DDMPO
end

"Build integral operator."
function IN(sites::Vector{Index{Int64}}; dx::Float64 = 1.0)
	N = length(sites)
	INMPO = MPO(sites)
	bond2 = Index[Index(2) for a in 1:N]
	A = ITensor(sites[1], prime(sites[1]), bond2[1])
	A[sites[1]=>1, prime(sites[1])=>1, bond2[1]=>1] = 1.0*dx^(1/N)
	A[sites[1]=>1, prime(sites[1])=>2, bond2[1]=>1] = 1.0*dx^(1/N)
	A[sites[1]=>2, prime(sites[1])=>1, bond2[1]=>1] = 1.0*dx^(1/N)
	A[sites[1]=>2, prime(sites[1])=>2, bond2[1]=>1] = 1.0*dx^(1/N) 
	A[sites[1]=>1, prime(sites[1])=>2, bond2[1]=>2] = 1.0*dx^(1/N) 
	A[sites[1]=>1, prime(sites[1])=>1, bond2[1]=>2] = 1.0*dx^(1/N)  
	A[sites[1]=>2, prime(sites[1])=>2, bond2[1]=>2] = 1.0*dx^(1/N)  
	INMPO[1] = A

	for a in 2:N-1
		A = ITensor(sites[a], prime(sites[a]), bond2[a-1], bond2[a])
		A[sites[a]=>1, prime(sites[a])=>1, bond2[a-1]=>1, bond2[a]=>1] = 1.0*dx^(1/N) 
		A[sites[a]=>1, prime(sites[a])=>2, bond2[a-1]=>1, bond2[a]=>1] = 1.0*dx^(1/N) 
		A[sites[a]=>2, prime(sites[a])=>1, bond2[a-1]=>1, bond2[a]=>1] = 1.0*dx^(1/N) 
		A[sites[a]=>2, prime(sites[a])=>2, bond2[a-1]=>1, bond2[a]=>1] = 1.0*dx^(1/N) 
		A[sites[a]=>1, prime(sites[a])=>2, bond2[a-1]=>1, bond2[a]=>2] = 1.0*dx^(1/N) 
		A[sites[a]=>1, prime(sites[a])=>1, bond2[a-1]=>2, bond2[a]=>2] = 1.0*dx^(1/N) 
		A[sites[a]=>2, prime(sites[a])=>2, bond2[a-1]=>2, bond2[a]=>2] = 1.0*dx^(1/N) 
		INMPO[a] = A
	end

	A = ITensor(sites[N], prime(sites[N]), bond2[N-1])
	A[sites[N]=>1, prime(sites[N])=>2, bond2[N-1]=>1] = 1.0*dx^(1/N) 
	A[sites[N]=>1, prime(sites[N])=>1, bond2[N-1]=>2] = 1.0*dx^(1/N) 
	A[sites[N]=>2, prime(sites[N])=>2, bond2[N-1]=>2] = 1.0*dx^(1/N) 
	INMPO[N] = A
	return INMPO
end

"Conbine the bond indices of MPO."
function cfc(mpo::MPO)
	N = length(mpo)
	sites = [noprime(siteinds(mpo)[i][1]) for i in 1:N]
	IMPS, IMPO = I(sites)
	C = []
	c = mpo[1] * IMPO[1]
	indices = ITensors.inds(c)
	push!(C, combiner(indices))
	mpo[1] = mpo[1] * C[1]
	for i ∈ 2:N-1
		c = mpo[i] * IMPO[i] * c
		indices = ITensors.inds(c)
		push!(C, combiner(indices))
		mpo[i] = C[i-1] * mpo[i] * C[i]
	end
	mpo[N] = C[N-1] * mpo[N]
	return mpo
end

"Conbine the bond indices of MPS."
function csc(mps::MPS)
	N = length(mps)
	sites = siteinds(mps)
	IMPS, IMPO = I(sites)
	C = []
	c = mps[1] * IMPS[1]
	indices = ITensors.inds(c)
	push!(C, combiner(indices))
	mps[1] = mps[1] * C[1]
	for i ∈ 2:N-1
		c = mps[i] * IMPS[i] * c
		indices = ITensors.inds(c)
		push!(C, combiner(indices))
		mps[i] = C[i-1] * mps[i] * C[i]
	end
	mps[N] = C[N-1] * mps[N]
	return mps
end

"Add two MPOs without contraction (julia will contract automatically if you write mpo1+mpo2)."
function ADD(mpo1::MPO, mpo2::MPO)
	N = length(mpo1)
	sites = [noprime(siteinds(mpo1)[i][1]) for i in 1:N]
	d = dim(sites[1])
	mpo = MPO(sites)
	b = []
	b1 = []
	b2 = []
	push!(b, Int(dim(mpo1[1]) / d^2 + dim(mpo2[1]) / d^2))
	push!(b1, Int(dim(mpo1[1]) / d^2))
	push!(b2, Int(dim(mpo2[1]) / d^2))
	for i ∈ 2:N-1
		push!(b, Int(dim(mpo1[i]) / (d^2 * b1[i-1]) + dim(mpo2[i]) / (d^2 * b2[i-1])))
		push!(b1, Int(dim(mpo1[i]) / (d^2 * b1[i-1])))
		push!(b2, Int(dim(mpo2[i]) / (d^2 * b2[i-1])))
	end
	nbond = Index[Index(b[i], "link, l=$i") for i ∈ 1:N-1]
	bond1 = []
	bond2 = []
	push!(bond1, linkind(mpo1, 1))
	push!(bond2, linkind(mpo2, 1))

	A = ITensor(sites[1], prime(sites[1]), nbond[1])
	for a ∈ 1:2, b ∈ 1:2
		for i ∈ 1:b1[1]
			A[sites[1]=>a, prime(sites[1])=>b, nbond[1]=>i] =
				mpo1[1][sites[1]=>a, prime(sites[1])=>b, bond1[1]=>i]
		end
		for j ∈ 1:b2[1]
			A[sites[1]=>a, prime(sites[1])=>b, nbond[1]=>(j+b1[1])] =
				mpo2[1][sites[1]=>a, prime(sites[1])=>b, bond2[1]=>j]
		end
	end
	mpo[1] = A

	for k ∈ 2:N-1
		push!(bond1, linkind(mpo1, k))
		push!(bond2, linkind(mpo2, k))
		A = ITensor(sites[k], prime(sites[k]), nbond[k-1], nbond[k])
		for a ∈ 1:2, b ∈ 1:2
			for i ∈ 1:b1[k], p ∈ 1:b1[k-1]
				A[sites[k]=>a, prime(sites[k])=>b, nbond[k-1]=>p, nbond[k]=>i] =
					mpo1[k][sites[k]=>a, bond1[k-1]=>p, prime(sites[k])=>b, bond1[k]=>i]
			end
			for j ∈ 1:b2[k], q ∈ 1:b2[k-1]
				A[sites[k]=>a, prime(sites[k])=>b, nbond[k-1]=>(q+b1[k-1]), nbond[k]=>(j+b1[k])] =
					mpo2[k][sites[k]=>a, bond2[k-1]=>q, prime(sites[k])=>b, bond2[k]=>j]
			end
		end
		mpo[k] = A
	end

	A = ITensor(sites[N], prime(sites[N]), nbond[N-1])
	for a ∈ 1:2, b ∈ 1:2
		for i ∈ 1:b1[N-1]
			A[sites[N]=>a, prime(sites[N])=>b, nbond[N-1]=>i] =
				mpo1[N][sites[N]=>a, prime(sites[N])=>b, bond1[N-1]=>i]
		end
		for j ∈ 1:b2[N-1]
			A[sites[N]=>a, prime(sites[N])=>b, nbond[N-1]=>(j+b1[N-1])] =
				mpo2[N][sites[N]=>a, prime(sites[N])=>b, bond2[N-1]=>j]
		end
	end
	mpo[N] = A
	return mpo
end

"Transform MPS into function."
function mps2f(mps::MPS)
	N = length(mps)
	sites = siteinds(mps)
	x = Vector{Float64}()
	y = Vector{Float64}()
	B = [ITensor(sites[i]) for i ∈ 1:N]

	nor = inner(mps, mps)
	permutation = collect(IterTools.product(fill(1:2, N)...))
	for p in permutation
		for i ∈ 1:N
			B[i][sites[i]=>p[i]] = 1.0
		end
		push!(x, sum(((p[i] - 1.0) / 2.0^(N - i + 1)) for i ∈ 1:N))
		tmps = prod((mps[i] * B[i]) for i ∈ 1:N) #can be improved
		for i ∈ 1:N
			B[i][sites[i]=>p[i]] = 0.0
		end
		push!(y, tmps[1])
	end
	return x, y, nor
end

"Transform MPO into matrix."
function mpo2m(mpo::MPO)
	N = length(mpo)
	sites = [noprime(siteinds(mpo)[i][1]) for i in 1:N]
	M = zeros(Float64, 2^N, 2^N)
	B = [ITensor(sites[i], prime(sites[i])) for i ∈ 1:N]

	permutation = collect(IterTools.product(fill(1:2, 2N)...))
	for p in permutation
		for i ∈ 1:N
			B[i][sites[i]=>p[i], prime(sites[i])=>p[N+i]] = 1.0
		end
		x = Int(sum(((p[i] - 1.0) * 2.0^(i - 1)) for i ∈ 1:N) + 1)
		y = Int(sum(((p[N+i] - 1.0) * 2.0^(i - 1)) for i ∈ 1:N) + 1)
		t = prod((mpo[i] * B[i]) for i ∈ 1:N)
		for i ∈ 1:N
			B[i][sites[i]=>p[i], prime(sites[i])=>p[N+i]] = 0.0
		end
		M[x, y] = t[1]
	end
	return M
end

"Calculate Frobenius norm error between two vectors."
function ERROR(v1::Vector{Float64}, v2::Vector{Float64})
	if length(v1) != length(v2)
        throw(ArgumentError("Vectors must have the same length"))
    end
    nor1 = norm(v1, 2)
    nor2 = norm(v2, 2)
	if nor1 == 0 || nor2 == 0
        throw(ArgumentError("Vectors must not a non-zero vectors"))
    end
	err = norm(v1/nor1-v2/nor2,2)
	return err
end

"Calculate the each bond dimension and totol elements of MPS."
function BDIM(mps::MPS)
	N = length(mps)
	bdim = Vector{Int64}()
	for i in 1:N-1
		push!(bdim, dim(linkind(mps, i)))
	end
	data = Vector{Int64}()
	push!(data, bdim[1] * 2)
	for i in 2:N-2
		push!(data, 2 * (bdim[i-1] * bdim[i]))
	end
	push!(data, 2 * bdim[N-1])
	ele = sum(data)
	return bdim, ele
end

"Read MPS in HDF5 file."
function reading_mps(file::String, sites::Vector{Index{Int64}})
	N = length(sites)
	bond = []
	link = zeros(Int, N - 1)
	rmps = h5read(file, "MPS")
	mps = MPS(sites)
	for n ∈ 1:1
		label = 1
		if get(rmps["MPS[$n]"]["inds"]["index_1"]["tags"], "tags", 0) == "S=1/2,Site,n=$n"
			sdim = rmps["MPS[$n]"]["inds"]["index_1"]["dim"]
			rdim = rmps["MPS[$n]"]["inds"]["index_2"]["dim"]
			link[n] = 2
			push!(bond, Index(rdim))
			A = ITensor(sites[n], bond[n])
			for r in 1:rdim, s in 1:sdim
				A[sites[n]=>s, bond[n]=>r] = rmps["MPS[$n]"]["storage"]["data"][label]
				label += 1
			end
		else
			rdim = rmps["MPS[$n]"]["inds"]["index_1"]["dim"]
			sdim = rmps["MPS[$n]"]["inds"]["index_2"]["dim"]
			link[n] = 1
			push!(bond, Index(rdim))
			A = ITensor(sites[n], bond[n])
			for s in 1:sdim, r in 1:rdim
				A[sites[n]=>s, bond[n]=>r] = rmps["MPS[$n]"]["storage"]["data"][label]
				label += 1
			end
		end
		mps[n] = A
	end

	for n ∈ 2:N-1
		label = 1
		if get(rmps["MPS[$n]"]["inds"]["index_1"]["tags"], "tags", 0) == "S=1/2,Site,n=$n"
			sdim = rmps["MPS[$n]"]["inds"]["index_1"]["dim"]
			if get(rmps["MPS[$n]"]["inds"]["index_2"]["tags"], "tags", 0) == get(rmps["MPS[$(n-1)]"]["inds"]["index_$(link[n-1])"]["tags"], "tags", 0)
				ldim = rmps["MPS[$n]"]["inds"]["index_2"]["dim"]
				rdim = rmps["MPS[$n]"]["inds"]["index_3"]["dim"]
				link[n] = 3
				push!(bond, Index(rdim))
				A = ITensor(sites[n], bond[n-1], bond[n])
				for r in 1:rdim, l in 1:ldim, s in 1:sdim
					A[sites[n]=>s, bond[n-1]=>l, bond[n]=>r] = rmps["MPS[$n]"]["storage"]["data"][label]
					label += 1
				end
			else
				rdim = rmps["MPS[$n]"]["inds"]["index_2"]["dim"]
				ldim = rmps["MPS[$n]"]["inds"]["index_3"]["dim"]
				link[n] = 2
				push!(bond, Index(rdim))
				A = ITensor(sites[n], bond[n-1], bond[n])
				for l in 1:ldim, r in 1:rdim, s in 1:sdim
					A[sites[n]=>s, bond[n-1]=>l, bond[n]=>r] = rmps["MPS[$n]"]["storage"]["data"][label]
					label += 1
				end
			end
		elseif get(rmps["MPS[$n]"]["inds"]["index_2"]["tags"], "tags", 0) == "S=1/2,Site,n=$n"
			sdim = rmps["MPS[$n]"]["inds"]["index_2"]["dim"]
			if get(rmps["MPS[$n]"]["inds"]["index_1"]["tags"], "tags", 0) == get(rmps["MPS[$(n-1)]"]["inds"]["index_$(link[n-1])"]["tags"], "tags", 1)
				ldim = rmps["MPS[$n]"]["inds"]["index_1"]["dim"]
				rdim = rmps["MPS[$n]"]["inds"]["index_3"]["dim"]
				link[n] = 3
				push!(bond, Index(rdim))
				A = ITensor(sites[n], bond[n-1], bond[n])
				for r in 1:rdim, s in 1:sdim, l in 1:ldim
					A[sites[n]=>s, bond[n-1]=>l, bond[n]=>r] = rmps["MPS[$n]"]["storage"]["data"][label]
					label += 1
				end
			else
				rdim = rmps["MPS[$n]"]["inds"]["index_1"]["dim"]
				ldim = rmps["MPS[$n]"]["inds"]["index_3"]["dim"]
				link[n] = 1
				push!(bond, Index(rdim))
				A = ITensor(sites[n], bond[n-1], bond[n])
				for l in 1:ldim, s in 1:sdim, r in 1:rdim
					A[sites[n]=>s, bond[n-1]=>l, bond[n]=>r] = rmps["MPS[$n]"]["storage"]["data"][label]
					label += 1
				end
			end
		else
			sdim = rmps["MPS[$n]"]["inds"]["index_3"]["dim"]
			if get(rmps["MPS[$n]"]["inds"]["index_1"]["tags"], "tags", 0) == get(rmps["MPS[$(n-1)]"]["inds"]["index_$(link[n-1])"]["tags"], "tags", 1)
				ldim = rmps["MPS[$n]"]["inds"]["index_1"]["dim"]
				rdim = rmps["MPS[$n]"]["inds"]["index_2"]["dim"]
				link[n] = 2
				push!(bond, Index(rdim))
				A = ITensor(sites[n], bond[n-1], bond[n])
				for s in 1:sdim, r in 1:rdim, l in 1:ldim
					A[sites[n]=>s, bond[n-1]=>l, bond[n]=>r] = rmps["MPS[$n]"]["storage"]["data"][label]
					label += 1
				end
			else
				rdim = rmps["MPS[$n]"]["inds"]["index_1"]["dim"]
				ldim = rmps["MPS[$n]"]["inds"]["index_2"]["dim"]
				link[n] = 1
				push!(bond, Index(rdim))
				A = ITensor(sites[n], bond[n-1], bond[n])
				for s in 1:sdim, l in 1:ldim, r in 1:rdim
					A[sites[n]=>s, bond[n-1]=>l, bond[n]=>r] = rmps["MPS[$n]"]["storage"]["data"][label]
					label += 1
				end
			end
		end
		mps[n] = A
	end

	for n ∈ N:N
		label = 1
		A = ITensor(sites[n], bond[n-1])
		if get(rmps["MPS[$n]"]["inds"]["index_1"]["tags"], "tags", 0) == "S=1/2,Site,n=$n"
			sdim = rmps["MPS[$n]"]["inds"]["index_1"]["dim"]
			ldim = rmps["MPS[$n]"]["inds"]["index_2"]["dim"]
			for l in 1:ldim, s in 1:sdim
				A[sites[n]=>s, bond[n-1]=>l] = rmps["MPS[$n]"]["storage"]["data"][label]
				label += 1
			end
		else
			ldim = rmps["MPS[$n]"]["inds"]["index_1"]["dim"]
			sdim = rmps["MPS[$n]"]["inds"]["index_2"]["dim"]
			for s in 1:sdim, l in 1:ldim
				A[sites[n]=>s, bond[n-1]=>l] = rmps["MPS[$n]"]["storage"]["data"][label]
				label += 1
			end
		end

		mps[n] = A
	end

	return mps
end

#Not necessary.
"Transform MPS into MPO, which means the operator of a diagonal matrix."
function mps2mpo(IRMPS::MPS)
	N = length(IRMPS)
	sites = siteinds(IRMPS)
	IRMPO = MPO(sites)
	bonds = []
	push!(bonds, linkind(IRMPS, 1))
	A = ITensor(sites[1], prime(sites[1]), bonds[1])
	for i in 1:2
		for j in 1:dim(linkind(IRMPS, 1))
			A[sites[1]=>i, prime(sites[1])=>i, bonds[1]=>j] = IRMPS[1][sites[1]=>i, linkind(IRMPS, 1)=>j]
		end
	end
	IRMPO[1] = A

	for n in 2:N-1
		push!(bonds, linkind(IRMPS, n))
		A = ITensor(sites[n], prime(sites[n]), bonds[n-1], bonds[n])
		for i in 1:2
			for j in 1:dim(linkind(IRMPS, n - 1))
				for k in 1:dim(linkind(IRMPS, n))
					A[sites[n]=>i, prime(sites[n])=>i, bonds[n-1]=>j, bonds[n]=>k] = IRMPS[n][sites[n]=>i, linkind(IRMPS, n - 1)=>j, linkind(IRMPS, n)=>k]
				end
			end
		end
		IRMPO[n] = A
	end

	A = ITensor(sites[N], prime(sites[N]), bonds[N-1])
	for i in 1:2
		for j in 1:dim(linkind(IRMPS, N - 1))
			A[sites[N]=>i, prime(sites[N])=>i, bonds[N-1]=>j] = IRMPS[N][sites[N]=>i, linkind(IRMPS, N - 1)=>j]
		end
	end
	IRMPO[N] = A
	return IRMPO
end


"Using dmrg() to solve the ground state or low excited stste.

tmps is the vector of projected stetes.

z is the vector you want calculate the Frobenius norm error."

#Abandon
function muti_rdmrg(mpo::MPO, mps0::MPS, sweep::Int, kdim::Int, maxbdim::Int, cutoff::Float64, noise::Union{Vector{Float64}, Float64};
	tmps::Union{Vector{MPS}, Nothing} = nothing, z::Union{Vector{Float64}, Nothing} = nothing)
	N = length(mpo)
	err = zeros(Float64, sweep + 1)
	bd = Array{Int}(undef, sweep + 1, N - 1) #each bond dimension for each state
	e = Vector{Float64}(undef, sweep + 1) #energy for each state
	ele = Array{Int}(undef)
	f = Vector{Float64}(undef, 2^N) #function for each state
	nmps = Vector{MPS}(undef, sweep + 1) #MPS for each state


	x, y = mps2f(mps0)
	energy = inner(prime(mps0), mpo * mps0)
	if z !== nothing
		err[1] = ERROR(y, z)
	end
	bd[1, :], el = BDIM(mps0)
	e[1] = energy
	nmps[1] = mps0

	for j in 1:sweep
		if tmps != nothing
			energy, mps = dmrg(
				mpo,
				tmps,
				nmps[j],
				cutoff = cutoff,
				maxdim = maxbdim,
				mindim = 2,
				noise = noise,
				nsweeps = 1,
				outputlevel = 1,
				eigsolve_krylovdim = kdim,
			)
		else
			energy, mps = dmrg(
				mpo,
				nmps[j],
				cutoff = cutoff,
				maxdim = maxbdim,
				mindim = 2,
				noise = noise,
				nsweeps = 1,
				outputlevel = 1,
				eigsolve_krylovdim = kdim,
			)
		end
		x, f = mps2f(mps)
		if z !== nothing
			err[j+1] = ERROR(f, z)
		end
		bd[j+1, :], ele = BDIM(mps)
		e[j+1] = energy
		nmps[j+1] = mps
	end
	return e, nmps, err, bd, ele, f
end

@info "Check before use(revised by 2025/03/01, update dx in EDD, DD IN). See old version backup in github"




