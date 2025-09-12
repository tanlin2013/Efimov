#Prepare H, INMPS, factor, str, and variables in dmrg()
maxerr = Matrix{Float64}(undef, n, sweep)
maxlinkdim = Matrix{Int}(undef, n, sweep)
energy = Matrix{Float64}(undef, n, sweep)
y = Matrix{Float64}(undef, n, 2^N)
tmps = Vector{MPS}()
inmps = Vector{MPS}(undef, n)

@time begin
	for i in 1:n
		open(joinpath(@__DIR__,str,"info.txt",str*"_info($i)(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).txt"), "w") do file
			# Redirect standard output to the file
			redirect_stdout(file) do
				energy, mps = dmrg(H, tmps, INMPS; nsweeps = sweep, mindim = 2, maxdim = maxbdim, eigsolve_krylovdim = kdim, cutoff = cutoff, weight = weight)
				push!(tmps, mps)
				x, y[i,:], nor = mps2f(mps)
				#h5write(joinpath(@__DIR__,str,"mps.h5",str*"_MPS($i)(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "MPS", mps)
			end
		end
		output_str = read(joinpath(@__DIR__,str,"info.txt",str*"_info($i)(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).txt"), String)
		lines = split(output_str, '\n')
		maxerr[i, :] = [parse(Float64, match(r"maxerr=([0-9.eE+-]+)", line).captures[1])
						for line in lines if occursin("maxerr=", line)]
		maxlinkdim[i, :] = [parse(Int, match(r"maxlinkdim=([0-9]+)", line).captures[1])
							for line in lines if occursin("maxerr=", line)]
		energy[i, :] = [parse(Float64, match(r"energy=([0-9.eE+-]+)", line).captures[1])
						for line in lines if occursin("maxerr=", line)]/factor
	end
end
@show energy

#h5write(joinpath(@__DIR__,str,str*"_energy(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "energy", energy)
#h5write(joinpath(@__DIR__,str,str*"_error(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "error", maxerr)
#h5write(joinpath(@__DIR__,str,str*"_bond_dim(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "bond_dim", maxlinkdim)
#h5write(joinpath(@__DIR__,str,str*"_function(N,sweep,shift,kdim,cutoff,weight)=($(N),$(sweep),$(shift),$(kdim),$(cutoff),$(weight)).h5"), "function", y)

