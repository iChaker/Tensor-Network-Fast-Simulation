include("states.jl")
include("hamiltonians.jl")
include("TEBD.jl")
include("qdrift.jl")
include("exact.jl")
using Plots


function testTEBD(L,ttotal,tau=0.1,nsteps=500,svd_cutoff=1E-16)
    mpsstate=MpsState(L)
    ham=Hamiltonian("Ising1D",mpsstate.sites)
    TEBDevol=TEBDEvolution(mpsstate,ham,ttotal,tau;outfolder="trash",svd_cutoff=svd_cutoff,algo="seq")
    # trunc error (0) + trotter error (    )  


    mpsstate=MpsState(L)
    ham=Hamiltonian("Ising1D",mpsstate.sites)
    TEBDevol_alt=TEBDEvolution(mpsstate,ham,ttotal,tau;outfolder="trash",svd_cutoff=svd_cutoff,algo="alt")

    tebd = TEBDevol.measurements["Szcenter"]
    tebdalt = TEBDevol_alt.measurements["Szcenter"]
    exact=ExactQIsingCenterSz(L,tau,ttotal)
    et_X=range(0.0,ttotal,length(tebd))


    p=plot(et_X,exact,label="exact",xlabel="time",ylabel="Sz center")
    p=plot!(et_X,tebd,label="TEBD sequential")
    p=plot!(et_X,tebdalt,label="TEBD alternating")

    q=plot(et_X,abs.(exact-tebd),label="TEBD sequential error",xlabel="time",ylabel="Sz center L1 error")
    q=plot!(et_X,abs.(exact-tebdalt),label="TEBD alternating error ")


    return p, q, exact, tebd, tebdalt

end


function testQdriftsingle(L,ttotal,nsteps=500,cutoff=1E-16)
    mpsstate=MpsState(L)
    ham=Hamiltonian("Ising1D",mpsstate.sites)
    
    QDRIFTevol=QDRIFTEvolution(mpsstate,ham,ttotal,nsteps;svd_cutoff=cutoff)
    lambda= sum(ham.parameters["spectral_norm_coefs"]) 
    qdrift = QDRIFTevol.measurements["Szcenter"]
    exact=ExactQIsingCenterSz(L,mean(diff(range(0,ttotal,nsteps))),ttotal) #dumb
    q_X=range(0.0,ttotal,length(qdrift))

    p=plot(q_X,exact,label="exact",xlabel="time",ylabel="Sz center")
    p=plot!(q_X,transpose(qdrift),label="Qdrift "*string(nsteps))


    q=plot(q_X,abs.(exact-transpose(qdrift)),label="Qdrift "*string(nsteps)* " error")

    return p,q,qdrift,exact,lambda

end


function testQdriftmulti(L,ttotal,nsteps=300,cutoff=1E-8,ntraj=50,save_percentage=10,outfolder="trash",saveall=true)
    mpsstate=MpsState(L)
    ham=Hamiltonian("Ising1D",mpsstate.sites)
    
    QDRIFTevol=QDRIFTEvolution(mpsstate,ham,ttotal,nsteps;svd_cutoff=cutoff,ntraj=ntraj,saveall=saveall,save_percentage=save_percentage,outfolder=outfolder)
    qdrift = QDRIFTevol.measurements["Szcenter"]
    exact=ExactQIsingCenterSz(L,mean(diff(range(0,ttotal,nsteps))),ttotal) #dumb

    MEAN=mean(qdrift,dims=1)'
    STD=std(qdrift,dims=1)'
    display(qdrift)
    q_X=range(0.0,ttotal,length(MEAN))

    p=plot(q_X,exact,label="exact",xlabel="time",ylabel="Sz center")
    plot!(q_X,MEAN,grid=false,ribbon=STD,fillalpha=.5,label="Qdrift "*string(nsteps))


    q=plot(q_X,abs.(exact-MEAN),grid=false,ribbon=STD,fillalpha=.5,label="Qdrift "*string(nsteps)* "error")

    return p,q,exact,QDRIFTevol

end


function test(L,ttotal,tau=0.1,nsteps=500,svd_cutoff=1E-16)
    mpsstate=MpsState(L)
    ham=Hamiltonian("Ising1D",mpsstate.sites)
    TEBDevol=TEBDEvolution(mpsstate,ham,ttotal,tau;outfolder="trash",svd_cutoff=svd_cutoff,algo="seq")
    # trunc error (0) + trotter error (    )  


    mpsstate=MpsState(L)
    ham=Hamiltonian("Ising1D",mpsstate.sites)
    TEBDevol_alt=TEBDEvolution(mpsstate,ham,ttotal,tau;outfolder="trash",svd_cutoff=svd_cutoff,algo="alt")


    mpsstate=MpsState(L)
    ham=Hamiltonian("Ising1D",mpsstate.sites)
    QDRIFTevol=QDRIFTEvolution(mpsstate,ham,ttotal,nsteps)
    # 2 lambda^2 t^2 / nsteps = eps
    lambda= sum(vcat(ham.parameters["spectral_norm_coefs"]...)) 


    exact1=ExactQIsingCenterSz(L,tau,ttotal)
    exact2=ExactQIsingCenterSz(L,mean(diff(range(0,ttotal,nsteps))),ttotal)
    qdrift = QDRIFTevol.measurements["Szcenter"]
    tebd = TEBDevol.measurements["Szcenter"]
    tebdalt = TEBDevol_alt.measurements["Szcenter"]
    et_X=range(0.0,ttotal,length(tebd))
    q_X=range(0.0,ttotal,length(qdrift))

    p=plot(q_X,exact2,label="exact",xlabel="time",ylabel="Sz center")
    p=plot!(et_X,tebd,label="TEBD sequential")
    p=plot!(et_X,tebdalt,label="TEBD alternating")
    p=plot!(q_X,qdrift,label="Qdrift "*string(nsteps))

    q=plot(et_X,abs.(exact1-tebd),label="TEBD sequential error",xlabel="time",ylabel="Sz center L1 error")
    q=plot!(et_X,abs.(exact1-tebdalt),label="TEBD alternating error ")
    q=plot!(q_X,abs.(exact2-qdrift),label="Qdrift "*string(nsteps)* "error")

    estq=2*lambda*lambda*ttotal*ttotal / nsteps
    q=plot!(et_X,ones(Float32,length(et_X))*estq,label="estimated error for " * string(nsteps) * " qdrift gates")
    print("\nestimated error for "* string(nsteps) * "gates = " * string(estq))


    return p, q, estq, exact1, tebd, exact2, qdrift 

end
