using ITensors

# define tensor networks and observalbes here 

function Szcenter(psi)
    return expect(psi, "Sz"; sites=Integer(round(length(psi)/2)))
end


mutable struct MpsState
    sites::Vector{Index{Int64}}
    psi
    observables #functions on psi
    function MpsState(sites,observables=Dict([("Szcenter",Szcenter), ("bdim",maxlinkdim)]))
        if isa(sites,Int)
            sites = siteinds("S=1/2", sites)
        end

        psi=MPS(sites, n -> isodd(n) ? "Up" : "Dn")
        
        new(sites,psi,observables)
    end

end

