using ITensors
using ProgressMeter
using JLD2


function gen_unitaries_TEBD_sequential(hamiltonian,tau)
    sites=hamiltonian.sites
    Hterms=hamiltonian.Hterms
    L=length(sites)
    gates = ITensor[]
    
    for j in 1:(L - 1)
        Gj = exp(-im * tau / 2 * Hterms[j][2])
        push!(gates, Gj)
    end
    # Include gates in reverse order too
    # (N,N-1),(N-1,N-2),...
    # include transverse field
    for j in 1:L
        Gj = exp(-im * tau / 2 * Hterms[j][1])
        push!(gates, Gj)
    end
    append!(gates, reverse(gates))

    return gates
end

function gen_unitaries_TEBD_alternating(hamiltonian,tau)
    sites=hamiltonian.sites
    Hterms=hamiltonian.Hterms
    L=length(sites)
    gates = ITensor[]
    for j in 1:2:L-1
        Gj = exp(-im * tau  * Hterms[j][2])
        push!(gates, Gj)
    end
    
    for j in 1:2:L
        Gj = exp(-im * tau  * Hterms[j][1])
        push!(gates, Gj)
    end


    for j in 2:2:L-1
        Gj = exp(-im * tau  * Hterms[j][2])
        push!(gates, Gj)
    end

    for j in 2:2:L
        Gj = exp(-im * tau  * Hterms[j][1])
        push!(gates, Gj)
    end

    Gj = exp(-im * tau  * Hterms[end][1])
    push!(gates, Gj)
    return gates

end



mutable struct TEBDEvolution
    mpsstate::MpsState
    hamiltonian::Hamiltonian
    ttotal::Float32
    tau::Float32 
    unitaries 
    measurements
    outfile
    
    function TEBDEvolution(mpsstate,hamiltonian,ttotal,tau;outfolder="",svd_cutoff=1E-8,algo="alt")
        if algo=="alt" unitaries=gen_unitaries_TEBD_alternating(hamiltonian,tau)
        elseif algo=="seq" unitaries=gen_unitaries_TEBD_sequential(hamiltonian,tau) end

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

        if !isempty(outfolder)
            #format: Hname_L-h-J_TEBD_ttotal-tau
            name=string(hamiltonian.name)
            LhJ=string(length(hamiltonian.sites))*"-"*string(hamiltonian.parameters["h"])*"-"*string(hamiltonian.parameters["J"])
            ttau=string(ttotal)*"-"*string(tau)
            if algo=="alt"
                filename = name*"_"*LhJ*"_TEBDalt_"*ttau
            elseif algo=="seq" 
                filename = name*"_"*LhJ*"_TEBDseq_"*ttau
            end
            
            outfile=outfolder*"/"*filename*".jld2"
            save_object(outfile,measurements)
        end

        new(mpsstate,hamiltonian,ttotal,tau,unitaries,measurements,outfile)
    end

end
