using ITensors

# Define hamiltonians here 
# gen_terms_XXXX(sites,params)
# add to hamlist in struct + default values

function gen_terms_Ising1D(sites,h,J)
    L=length(sites)
    Hj=[]
    for j in 1:L-1
      onesite = -h * op("Sx", sites[j])  #Negative sign??
      twosite = -J * op("Sz", sites[j]) * op("Sz", sites[j+1]) 
      push!(Hj,[onesite,twosite])
    end

    onesite = -h * op("Sx", sites[end])  #Negative sign??
    push!(Hj,[onesite])
    
    return vcat(Hj...),1,2
end




mutable struct Hamiltonian
    name::String
    sites::Vector{Index{Int64}}
    parameters::Dict{String, Any}
    Hterms
    dim
    interaction_order

    function Hamiltonian(name,sites;kwargs...)
        if isa(sites,Int)
            sites = siteinds("S=1/2", sites)
        end
        hamlist=["Ising1D"]
        print("\nHamiltonian list:",hamlist,"\n")
        if name == hamlist[1]
            if haskey(kwargs,:h ) h= kwargs[:h] else h=1 end
            if haskey(kwargs,:J ) J= kwargs[:J] else J=1 end
            Hterms,dim,interaction_order=gen_terms_Ising1D(sites,h,J)
            new(name,sites,Dict([("h",h), ("J",J)]),Hterms,dim,interaction_order)
        end
    end

end


