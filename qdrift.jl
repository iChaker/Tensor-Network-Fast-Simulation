using LinearAlgebra
using ITensors
using Random
using StatsBase
using JLD2
using ProgressMeter
include("states.jl")

function spectral_norm_hamiltonian_coefs(hamiltonian)

    if haskey(hamiltonian.parameters,"spectral_norm_coefs") 
        return hamiltonian.parameters["spectral_norm_coefs"]
    end

    coefs=zeros(Float64,length(hamiltonian.Hterms)) # sites x nbodies
    for (j,Hterm) in enumerate(hamiltonian.Hterms)
            unprimed_indices = [index for index in inds(Hterm) if plev(index) == 0]
            R=svd(Hterm,unprimed_indices)
            coefs[j]=maximum(R.S)           
    end
    hamiltonian.parameters["spectral_norm_coefs"]=coefs
    return coefs

end


# Function to save intermediate results to a common file
function save_intermediate_results(big_array, step, filename)
    lock(file_lock) do
        open(filename, "a") do f
            println(f, "Step $step:")
            for row in 1:nrows
                println(f, join(big_array[row, :], ","))
            end
            println(f, "")  # Newline for separation
        end
    end
end

function gen_unitaries_QDRIFT(hamiltonian,ttotal,nsteps)

    if !haskey(hamiltonian.parameters,"unitaries") 
        D = Dict{Tuple{Float64,Int64}, Vector{ITensor}}()
        hamiltonian.parameters["unitaries"]=D
    else 
        D=hamiltonian.parameters["unitaries"]
    end

    if haskey(D,(ttotal,nsteps)) return D[(ttotal,nsteps)] end

    coefs=spectral_norm_hamiltonian_coefs(hamiltonian) 
    unitaries = deepcopy(hamiltonian.Hterms)
    lambda = sum(coefs)
    qtau = lambda * ttotal / nsteps
    for (j,Hterm) in enumerate(hamiltonian.Hterms)
            unitaries[j]= exp(-im * Hterm * qtau / coefs[j])
    end
    
    D[(ttotal,nsteps)]=unitaries
    return unitaries
    
end

mutable struct QDRIFTEvolution 
    psis::Vector{MPS}
    hamiltonian::Hamiltonian
    ttotal::Float32 
    nsteps::Float32 
    unitaries
    measurements
    seed
    path
    

    function QDRIFTEvolution(mpsstate,hamiltonian,ttotal,nsteps;seed=0,svd_cutoff=1E-8,ntraj=1,save_percentage=0,saveall=true,outfolder=".")
       
        #IO bloat
        progIO=open("progress.log","a")
        if save_percentage>0
            name=string(hamiltonian.name)
            LhJ=string(length(hamiltonian.sites))*"-"*string(hamiltonian.parameters["h"])*"-"*string(hamiltonian.parameters["J"])
            qtau=string(ttotal)*"-"*string(nsteps)
            filename = name*"_"*LhJ*"_QDRIFT-"*string(ntraj)*"_"*qtau*".jld2"
            path=outfolder*"/"*filename
        end
        
        lastraj=1
        if saveall
            measurements = Dict(obs_name => zeros(Float64,ntraj, nsteps) for (obs_name,_) in mpsstate.observables)
        else
            measurements = Dict(obs_name => zeros(Float64,ntraj, 1) for (obs_name,_) in mpsstate.observables)
        end
        psis=Vector{MPS}(undef,ntraj)
        try @load path measurements lastraj catch e end
       
        #generate unitaries 
        unitaries=gen_unitaries_QDRIFT(hamiltonian,ttotal,nsteps)
        coefs=hamiltonian.parameters["spectral_norm_coefs"]
        P=coefs/sum(coefs)
        

        lk = Threads.SpinLock()
        meter = Progress(ntraj-lastraj+1;output=progIO)

        #run multiple qdrift evolutions
        Threads.@threads for traj in lastraj:ntraj

            if seed == 0
                rng=Random.seed!(Random.default_rng(), traj*nsteps) 
            else 
                rng=Random.seed!(Random.default_rng(), seed)
            end
            psitraj=deepcopy(mpsstate.psi)
            for step in 1:nsteps
                gate=StatsBase.sample(rng,unitaries,Weights(P))
                psitraj = apply(gate, psitraj; cutoff=svd_cutoff)        
                normalize!(psitraj)

                if saveall
                    for (obs_name, obs_func) in mpsstate.observables
                    measurements[obs_name][traj,step]=obs_func(psitraj)
                    end
                elseif step==nsteps
                    for (obs_name, obs_func) in mpsstate.observables
                        measurements[obs_name][traj,1]=obs_func(psitraj)
                    end
                end
            end
            psis[traj]=psitraj
            ProgressMeter.next!(meter; showvalues = [(:traj,traj)])

            if (save_percentage!=0) && (traj % ceil(save_percentage*ntraj/100) == 0)
                Threads.lock(lk) do 
                    lastraj=traj
                    @save path measurements lastraj
                end
            end

        end
        lastraj=ntraj
        @save path measurements lastraj


        new(psis,hamiltonian,ttotal,nsteps,unitaries,measurements,seed,path)
    end

end


