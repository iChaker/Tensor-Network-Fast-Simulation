using LinearAlgebra
using ITensors
using Random
using StatsBase
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
    return unitaries,coefs/lambda
    
end

mutable struct QDRIFTEvolution 
    mpsstate::MpsState
    hamiltonian::Hamiltonian
    ttotal::Float32 
    nsteps::Float32 
    unitaries
    measurements
    seed
    

    function QDRIFTEvolution(mpsstate,hamiltonian,ttotal,nsteps;seed=0,outfolder="",svd_cutoff=1E-8,unitaries="",P="")
        if isempty(unitaries) 
            unitaries, P =gen_unitaries_QDRIFT(hamiltonian,ttotal,nsteps)
        end
        measurements= Dict{String, Vector{Float32}}()
        for (obs_name, obs_func) in mpsstate.observables
            measurements[obs_name] = zeros(Float32, nsteps)
        end
        if seed > 0 rng=Random.seed!(Random.default_rng(), seed) else rng=Random.default_rng() end

        for step in 1:nsteps
            gate=StatsBase.sample(rng,unitaries,Weights(P))
            mpsstate.psi = apply(gate, mpsstate.psi; cutoff=svd_cutoff)        
            normalize!(mpsstate.psi)

            for (obs_name, obs_func) in mpsstate.observables
                measurements[obs_name][step]=obs_func(mpsstate.psi)
            end

        end

        if !isempty(outfolder)
            #oneshot fileoutout todo later
        end
        new(mpsstate,hamiltonian,ttotal,nsteps,unitaries,measurements,seed)
    end


end


function QDRIFTEvolutionSamples(L,ttotal,nstepsarray,nsamples;svd_cutoff=1E-8)

    O=[]
    for nsteps in nstepsarray
        ham=Hamiltonian("Ising1D",L)
        unitaries, P =gen_unitaries_QDRIFT(ham,ttotal,nsteps)
        nsteps_measurement = Vector{Any}(undef,nsamples)
        Threads.@threads for sampleid in 1:nsamples
            mpsstate=MpsState(ham.sites)
            qdriftrun= QDRIFTEvolution(mpsstate,ham,ttotal,nsteps;seed=sampleid*nsteps,unitaries=unitaries,P=P,svd_cutoff=svd_cutoff)
            nsteps_measurement[sampleid]=qdriftrun.measurements
        end
        push!(O,nsteps_measurement)
    end

    return O

end