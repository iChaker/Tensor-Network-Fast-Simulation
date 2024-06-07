using ITensors
using ProgressMeter
using JLD2




function gen_unitaries_TEBD_sequential(hamiltonian,tau)
    Hterms=hamiltonian.Hterms
    gates = ITensor[]
    for Hj in Hterms
        Gj = exp(-im * tau / 2 * Hj)
        push!(gates, Gj)
    end
    append!(gates, reverse(gates))

    return gates
end



function gen_unitaries_TEBD_alternating(hamiltonian,tau)
    sites=hamiltonian.sites
    Hterms=hamiltonian.Hterms
    interaction_order=hamiltonian.interaction_order
    L=length(sites)
    odd=1:interaction_order*2:L
    even=2:interaction_order*2:L
    gates = ITensor[]

    for j in odd
        for i in 1:interaction_order
            Gj = exp(-im * tau  * Hterms[j+i-1])
            push!(gates, Gj)

        end
    end
    
    for j in even
        for i in 1:interaction_order
            Gj = exp(-im * tau  * Hterms[j+i-1])
            push!(gates, Gj)

        end
    end

    return gates

end



mutable struct TEBDEvolution
    mpsstate::MpsState
    hamiltonian::Hamiltonian
    ttotal::Float32
    tau::Float32 
    unitaries 
    measurements
    path
    
    function TEBDEvolution(mpsstate,hamiltonian,ttotal,tau;outfolder=nothing,svd_cutoff=1E-8,algo="alt")
        if algo=="alt" unitaries=gen_unitaries_TEBD_alternating(hamiltonian,tau)
        elseif algo=="seq" unitaries=gen_unitaries_TEBD_sequential(hamiltonian,tau) 
        elseif algo=="seq2" unitaries=gen_unitaries_TEBD_sequential2(hamiltonian,tau) end
    

        measurements= Dict{String, Vector{Any}}()
        @showprogress for t in 0.0:tau:ttotal
            mpsstate.psi=apply(unitaries, mpsstate.psi; cutoff=svd_cutoff)
            normalize!(mpsstate.psi)
            for (obs_name, obs_func) in mpsstate.observables
                value= obs_func(mpsstate.psi)
                if !haskey(measurements,obs_name)
                    measurements[obs_name]=[value]
                else
                    push!(measurements[obs_name],value)
                end
            end
        end

        if !isnothing(outfolder)
            #format: Hname_L-h-J_TEBD_ttotal-tau
            name=string(hamiltonian.name)
            LhJ=string(length(hamiltonian.sites))*"-"*string(hamiltonian.parameters["h"])*"-"*string(hamiltonian.parameters["J"])
            ttau=string(ttotal)*"-"*string(tau)
            if algo=="alt"
                filename = name*"_"*LhJ*"_TEBDalt_"*ttau*".jld2"
            elseif algo=="seq"
                filename = name*"_"*LhJ*"_TEBDseq_"*ttau*".jld2"
            end
            
            path=outfolder*"/"*filename
            save_object(path,measurements)
        else 
            path=nothing
        end

        new(mpsstate,hamiltonian,ttotal,tau,unitaries,measurements,path)
    end

end
