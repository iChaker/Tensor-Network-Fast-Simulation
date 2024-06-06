using LinearAlgebra
# Define Hamiltonian
function TransverseIsing(N,hJ)
    I2 = [1+0.0im 0+0.0im ; 0+0.0im 1+0.0im]
    Sz = 0.5*[1+0.0im 0+0.0im ; 0+0.0im -1+0.0im]
    Sx = 0.5*[0+0.0im 1+0.0im ; 1+0.0im 0+0.0im]
    I2args= [I2 for j in 1:N] #list of N identity matrices                            
    H = zeros(2^N,2^N)              #initialize full hamiltonian matrix to 0
    # two site terms
    for j in 1:N-1
        Hargs=copy(I2args)
        Hargs[j] = Sz
        Hargs[j+1] = Sz 
        H= H + kron(Hargs...)
    end
    # single site terms
    for j in 1:N
        Hargs=copy(I2args)
        Hargs[j] = Sx * hJ
        H= H + kron(Hargs...)
    end
    H = (-1) * H 

    return H
end

# Define observable
function makeSzOp(N,site)    
    I2 =  [1+0.0im 0+0.0im ; 0+0.0im 1+0.0im]
    Sz= [I2 for j in 1:N] 
    Sz[site] = 0.5*[1+0.0im 0+0.0im ; 0+0.0im -1+0.0im]
    return kron(Sz...)
end

# Evolve and calculate <Sz>
function ExactQIsingCenterSz(N,tau,ttotal,h=1)

    @assert N<15 "system too big for exact diagonalization"

    H = TransverseIsing(N,h)
    vals, vecs = eigen(H)

    # intial wavefunction state  updownupdown..
    psi_z = zeros(ComplexF32,2^N)
    idx=parse(Int,join([ isodd(n) ? "0" : "1" for n in 1:N]),base=2)
    psi_z[idx] = 1

    Sz_measurements_ED = Float32[]
    Sz= makeSzOp(N,div(N,2))  
    for t in 0:tau:ttotal
        
        # wave function in energy basis
        psi_e = zeros(ComplexF32,2^N)
        for i in 1:2^N
        psi_e= psi_e +  ((transpose(conj(vecs[:,i])) * psi_z) * vecs[:,i]) * exp(-im * tau * vals[i]) 
        end
        
        # normalize
        psi_z = psi_e / ( transpose(conj(psi_e)) * psi_e )
        
        push!(Sz_measurements_ED,real(transpose(conj(psi_z)) * (Sz * psi_z) )) 
    end
    return Sz_measurements_ED

end

