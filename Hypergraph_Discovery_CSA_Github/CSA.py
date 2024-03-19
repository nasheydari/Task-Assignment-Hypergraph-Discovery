import timeit
import numpy as np
from numpy.random import rand
from Utils import *
import os
import logging
import json




def run_CSA(maxIter, file, I, I_ex, omega, gamma, n, m, a, t0, nswaps, nremove, E, B, Buds, sym, params):
    #initialization
    f = np.array([]);
    fb = np.array([]);
    pf = np.array([]);
    S_v = np.array([]);
    S_ex_v = np.array([]);
    Avtask = np.array([]);
    AvCoauth = np.array([]);
    Avauth = np.array([]);
    MinT = np.array([]);
    MaxT = np.array([]);
    VarT = np.array([]);
    Avtask_c = np.array([]);
    AvCoauth_c = np.array([]);
    Avauth_c = np.array([]);
    MinT_c = np.array([]);
    MaxT_c = np.array([]);
    VarT_c = np.array([]);
    chc = 0

    t = t0;  # temperature
    S = copy.deepcopy(I);  # Copying the incidence matrix of the hypergraph (solution)
    S_ex = copy.deepcopy(I_ex)
    it_swap = 0 #number of swaps per each removal
    gamma = gamma_generate(S_ex, S, B, n, m)
    # omega=sum(S_ex).reshape((1, m))

    # filter the nodes with zero degree just for calculating algeb con
    k = np.sum(S, axis=1);
    Sf = np.copy(S[k > 0, :]);
    gammaf = np.copy(gamma[k > 0, :]);
    nf = Sf.shape[0];

    if nf > 0:
        try:
            P = calculate_probability_matrix(Sf, gammaf, np.tile(omega, (nf, 1)));
            L = calculate_Laplacian_matrix(P);
            [e, ev] = calculate_algebraic_connectivity(L);
        except:
            e=0
        FlagC = (matrix_power(5 * P, n) > 0).all()
        print(FlagC)
        if sym:
            Pt = calculate_probability_matrix_alter(Sf, gammaf, np.tile(omega, (nf, 1)));
            Lt = calculate_Laplacian_matrix(Pt);
            [et, evt] = calculate_algebraic_connectivity(Lt);
            print(e,et)
            e1=e
            e=et*e*(10**6)
    else:
        e = 0;
        ev = np.ones([n, ])

    Lambda_n = 3 * np.ones([n, ]);
    Lambda_m = 3 * np.ones([m, ]);
    pen = params['pen']
    pen1 = params['pen1']
    pen2 = params['pen2']
    pen3 = params['pen3']
    [Cons, Conss, Cm, Cn, Cm_d, Cn_d] = calculate_Constraints_ex(S_ex, Lambda_m, Lambda_n, E, n, m, Buds, B);
    ConsBest = Cons;
    Avt = AverageTask(S, n)
    Ava= AverageAuth(S, m)
    Avco = AveragecoAuth(S, n, m)
    [Mint, Maxt, Vart] = MinMaxtask (S,n)

    Avtn=Avt
    Avan=Ava
    Avcon=Avco
    Mintn = Mint
    Maxtn = Maxt
    Vartn= Vart

    PF = (e - Cons) * pen * 1 / (pen1 * Avt / (2 * n + m) + pen2 * Avco / (2 * n + m) + pen3 * Ava / (2 * n + m));  # penalty function
    Best_S = np.copy(S);
    Best_S_ex = np.copy(S_ex);
    eBest = e;
    evBest = ev;
    etNew=0
    eNew1=0

    AvtBest = Avt
    AvcoBest = Avco
    AvaBest = Ava
    MintBest = Mint
    MaxtBest = Maxt
    VartBest = Vart

    PFBest = PF
    ExcessE = np.average(Cm_d)
    Nt = params['Nt'];
    StopCon = 0;
    j = 0;
    soli = 0
    niter = -1;
    ret = 0
    Stepwn = np.ones([n, ])
    Stepwm = np.ones([m, ])
    eVec=[]
    e1Vec=[]
    etVec=[]
    eVecB = []
    e1VecB = []
    etVecB = []


    skip=False

    while StopCon == 0:
        for i in range(Nt):
            niter = niter + 1;
            # this info is mainly for debuging purposes
            info = "%ld\t%ld\t%.6f\t%.6e\t%.6e\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%ld" % (
                niter, i, t, e, eBest, AvtBest, AvcoBest, (pen1 * AvtBest / (n + m) + pen2 * AvcoBest / (n + m)), Cons,
                ConsBest, ExcessE, ret);
            print(info);

            rn2 = random.random();
            # rn2=1;
            if rn2 > params['return']:
                ret = 0;
                # Si = perturbation(S, gamma, omega, nswaps, ev, E, B);
                if it_swap < 100: #this is for when we need to remove (after each removal, swap 100 times)
                    [Si_ex, Si] = perturbation_swap_ex(S_ex, S, nswaps, E, B, ev);
                    gamma = gamma_generate(Si_ex, Si, B, n, m)
                    omega = np.sum(np.multiply(Si, gamma), axis=0).reshape((1,m))
                    #it_swap += 1
                else:
                    [Si_ex, Si] = perturbation_remove_ex(S_ex, S, nremove, E, B, ev);
                    gamma = gamma_generate(Si_ex, Si, B, n, m)
                    omega = np.sum(np.multiply(Si, gamma), axis=0).reshape((1,m))
                    it_swap = 0
            else:
                ret = 1;
                it_swap = 0
                # Si = perturbation(Best_S, gamma, omega, nswaps, evBest, E, B);
                [Si_ex, Si] = perturbation_swap_ex(Best_S_ex, Best_S, nswaps, E, B, evBest);
                gamma = gamma_generate(Si_ex, Si, B, n, m)
                omega= np.sum(np.multiply(Si, gamma), axis=0).reshape((1,m))
                #it_swap +=1


            Sanity_check(Si, Si_ex, B)
            [Consn, Conssn, Cmn, Cnn, Cmn_d, Cnn_d] = calculate_Constraints_ex(Si_ex, Lambda_m, Lambda_n, E, n, m, Buds, B);
            if not Consn > 0:
                k = np.sum(Si, axis=1);
                Sif = np.copy(Si[k > 0, :]);
                gammaf = np.copy(gamma[k > 0, :]);
                nf = Sif.shape[0]
                #print(nf)
                if nf > 0:
                    FlagC = (matrix_power(5 * P, n) > 0).all()
                    print(FlagC)
                    #if FlagC:
                    if True:
                        try:
                            P = calculate_probability_matrix(Sif, gammaf, np.tile(omega, (nf, 1)));
                            L = calculate_Laplacian_matrix(P);
                            [eNew, evNew] = calculate_algebraic_connectivity(L);
                        except:
                            eNew=0

                    if sym:
                        try:
                            Pt = calculate_probability_matrix_alter(Sif, gammaf, np.tile(omega, (nf, 1)));
                            Lt = calculate_Laplacian_matrix(Pt);
                            [etNew, evtNew] = calculate_algebraic_connectivity(Lt);
                        except:
                            etNew=0
                        eNew1 = eNew
                        eNew = etNew * eNew * (10 ** 6)



                else:
                    eNew = 0;
                    eNew1 = 0
                    etNew= 0
                    evNew = np.ones([n, ]);
            else:
                eNew = e
                etNew = 0
                eNew1 = e
                evNew = ev

            Avtn = AverageTask(Si, n)
            Avcon = AveragecoAuth(Si, n, m)
            Avan = AverageAuth(Si, m)
            [Mintn, Maxtn, Vartn] = MinMaxtask (Si , n)

            PFn = (eNew - Consn) * pen* 1/( pen1 * Avtn / (2*n + m) + pen2 * Avcon / (2*n + m) + pen3 * Avan / (2*n + m) );

            # if Consn == 0:
            if PFn > PF or np.exp(- (PF - PFn) / t) > np.random.uniform(0, 1):
                # accept the new solution
                S = copy.deepcopy(Si);
                S_ex = copy.deepcopy(Si_ex)
                PF = PFn;
                e = eNew;
                ev = evNew;
                et = etNew
                e1= eNew1
                eVec.append(eNew)
                e1Vec.append(eNew1)
                etVec.append(etNew)
                Cons = Consn;
                Conss = Conssn;
                Cm = Cmn;
                Avt = Avtn
                Avco = Avcon
                Ava = Avan
                Mint = Mintn
                Maxt = Maxtn
                Vart = Vartn
                if PF > PFBest:
                    Best_S = copy.deepcopy(S);
                    Best_S_ex = copy.deepcopy(S_ex)
                    Best_Lambda_n = np.copy(Lambda_n);
                    Best_Lambda_m = np.copy(Lambda_m);
                    eBest = e;
                    etBest = et
                    e1Best = e1
                    evBest = ev;
                    eVecB.append(eBest)
                    e1VecB.append(e1Best)
                    etVecB.append(etBest)
                    ConsBest = Cons;
                    ConssBest = Conss;
                    PFBest = PF;
                    BCm = Cm;
                    AvcoBest = Avco
                    AvtBest = Avt
                    AvaBest = Ava
                    MaxtBest = Maxt
                    MintBest = Mint
                    VartBest = Vart


                    if niter >= maxIter - 4000:
                        if S_v.size == 0:
                            soli += soli
                            #S_v = mat_to_vec(Best_S)
                            S_v = np.reshape(Best_S,[1,n*m])
                            # S_ex_v=mat_to_vec_ex(Best_S_ex)
                        else:
                            soli += soli
                            S_v = np.append(S_v, np.reshape(Best_S,[1,n*m]), axis=0)
                            # S_ex_v = np.append(S_ex_v, mat_to_vec_ex(Best_S_ex), axis=0)

                Avtask = np.append(Avtask, AvtBest) #average degree of a node (average task)
                AvCoauth = np.append(AvCoauth, AvcoBest)
                Avauth = np.append(Avauth, AvaBest)
                MaxT = np.append(MaxT, MaxtBest) #maximum degree of a node (maximum task)
                MinT = np.append(MinT, MintBest) #minimum degree of a node (minimum task)
                VarT = np.append(VarT, VartBest) #variance of degree distribution

                Avtask_c = np.append(Avtask_c, Avt)
                AvCoauth_c = np.append(AvCoauth_c, Avco)
                Avauth_c = np.append(Avauth_c, Ava)
                MaxT_c = np.append(MaxT_c, Maxt)  # maximum degree of a node (maximum task)
                MinT_c = np.append(MinT_c, Mint)  # minimum degree of a node (minimum task)
                VarT_c = np.append(VarT_c, Vart)

                f = np.append(f, e);
                fb = np.append(fb, eBest);
                pf = np.append(pf, PF);

        j = j + 1;
        t = a * t;


        if t < params["tol"] or niter > maxIter or skip:
            StopCon = 1

    return [Best_S, Best_S_ex, PFBest, f, fb, pf, Avtask, AvCoauth, Avauth, MaxT, MinT, VarT, Avtask_c, AvCoauth_c, Avauth_c, MaxT_c, MinT_c, VarT_c, S_v, eVec, e1Vec, etVec, eVecB, e1VecB, etVecB];



def run_CSA_bipartite(maxIter, file, I, I_ex, omega, gamma, n, m, a, t0, nswaps, nremove, E, B, Buds, sym, params):
    #initialization
    f = np.array([]);
    fb = np.array([]);
    pf = np.array([]);
    S_v = np.array([]);
    S_ex_v = np.array([]);
    Avtask = np.array([]);
    AvCoauth = np.array([]);
    Avauth = np.array([]);
    MinT = np.array([]);
    MaxT = np.array([]);
    VarT = np.array([]);
    Avtask_c = np.array([]);
    AvCoauth_c = np.array([]);
    Avauth_c = np.array([]);
    MinT_c = np.array([]);
    MaxT_c = np.array([]);
    VarT_c = np.array([]);
    chc = 0

    t = params['temp'];  # temperature
    S = copy.deepcopy(I);  # Copying the incidence matrix of the hypergraph (solution)
    S_ex = copy.deepcopy(I_ex)
    it_swap = 0 #number of swaps per each removal
    gamma = gamma_generate(S_ex, S, B, n, m)
    # omega=sum(S_ex).reshape((1, m))

    # filter the nodes with zero degree just for calculating algeb con
    k = np.sum(S, axis=1);
    Sf = np.copy(S[k > 0, :]);
    gammaf = np.copy(gamma[k > 0, :]);
    nf = Sf.shape[0];

    if nf > 0:
        try:
            P = calculate_bipartite_probability_matrix(Sf, gammaf, np.tile(omega, (nf, 1)));
            L = calculate_Laplacian_matrix(P);
            [e, ev] = calculate_algebraic_connectivity(L);
        except:
            e=0
        if sym:
            try:
                Pt = calculate_probability_matrix_alter(Sf, gammaf, np.tile(omega, (nf, 1)));
                Lt = calculate_Laplacian_matrix(Pt);
                [et, evt] = calculate_algebraic_connectivity(Lt);
            except:
                et=0
            e1=e
            e=et*e*(10**6)
    else:
        e = 0;
        ev = np.ones([n, ])

    Lambda_n = 3 * np.ones([n, ]);
    Lambda_m = 3 * np.ones([m, ]);
    pen = params['pen']
    pen1 = params['pen1']
    pen2 = params['pen2']
    pen3 = params['pen3']
    [Cons, Conss, Cm, Cn, Cm_d, Cn_d] = calculate_Constraints_ex(S_ex, Lambda_m, Lambda_n, E, n, m, Buds, B);
    ConsBest = Cons;
    Avt = AverageTask(S, n)
    Ava= AverageAuth(S, m)
    Avco = AveragecoAuth(S, n, m)
    [Mint, Maxt, Vart] = MinMaxtask (S,n)

    Avtn=Avt
    Avan=Ava
    Avcon=Avco
    Mintn = Mint
    Maxtn = Maxt
    Vartn= Vart

    if params["data"]=="MAG":
        PF = (e - Cons) * (1 + pen / (
                pen1 * Avt / (2 * n + m) + pen2 * Avco / (2 * n + m) + pen3 * Ava / (
                2 * n + m)));  # penalty function
    else:
        PF = (e - Cons) * pen * 1 / (pen1 * Avt / (2 * n + m) + pen2 * Avco / (2 * n + m) + pen3 * Ava / (2 * n + m));  # penalty function

    Best_S = np.copy(S);
    Best_S_ex = np.copy(S_ex);
    eBest = e;
    evBest = ev;
    etNew=0
    eNew1=0

    AvtBest = Avt
    AvcoBest = Avco
    AvaBest = Ava
    MintBest = Mint
    MaxtBest = Maxt
    VartBest = Vart

    PFBest = PF
    ExcessE = np.average(Cm_d)
    Nt = params['Nt'];
    StopCon = 0;
    j = 0;
    soli = 0
    niter = -1;
    ret = 0
    Stepwn = np.ones([n, ])
    Stepwm = np.ones([m, ])
    eVec=[]
    e1Vec=[]
    etVec=[]
    eVecB = []
    e1VecB = []
    etVecB = []


    skip=False

    while StopCon == 0:
        for i in range(Nt):
            niter = niter + 1;
            # this info is mainly for debuging purposes
            info = "%ld\t%ld\t%.6f\t%.6e\t%.6e\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%ld" % (
                niter, i, t, e, eBest, AvtBest, AvcoBest, (pen1 * AvtBest / (n + m) + pen2 * AvcoBest / (n + m)), Cons,
                ConsBest, ExcessE, ret);
            print(info)
            if params['data']=='MAG':
                if niter < params["N_swapsb"]:
                    nswaps = params["nswapsb"]
                elif niter < params["N_swapsm"]:
                    nswaps = params["nswapsm"]
                else:
                    nswaps = params["nswaps"]

            rn2 = random.random()
            if rn2 > params["return"]:
                ret = 0;
                if it_swap < 100: #this is for when we need to remove (after each removal, swap 100 times)
                    [Si_ex, Si] = perturbation_swap_ex(S_ex, S, nswaps, E, B, ev);
                    gamma = gamma_generate(Si_ex, Si, B, n, m)
                    omega = np.sum(np.multiply(Si, gamma), axis=0).reshape((1,m))
                    #it_swap += 1
                else:
                    [Si_ex, Si] = perturbation_remove_ex(S_ex, S, nremove, E, B, ev);
                    gamma = gamma_generate(Si_ex, Si, B, n, m)
                    omega = np.sum(np.multiply(Si, gamma), axis=0).reshape((1,m))
                    it_swap = 0
            else:
                ret = 1;
                it_swap = 0
                [Si_ex, Si] = perturbation_swap_ex(Best_S_ex, Best_S, nswaps, E, B, evBest);
                gamma = gamma_generate(Si_ex, Si, B, n, m)
                omega= np.sum(np.multiply(Si, gamma), axis=0).reshape((1,m))



            Sanity_check(Si, Si_ex, B)
            [Consn, Conssn, Cmn, Cnn, Cmn_d, Cnn_d] = calculate_Constraints_ex(Si_ex, Lambda_m, Lambda_n, E, n, m, Buds, B);
            if not Consn > 0:

                k = np.sum(Si, axis=1);
                Sif = np.copy(Si[k > 0, :]);
                gammaf = np.copy(gamma[k > 0, :]);
                nf = Sif.shape[0]
                if nf > 0:
                    FlagC = (matrix_power(5 * P, n) > 0).all()
                    print(FlagC)
                    if True:
                        try:
                            P = calculate_bipartite_probability_matrix(Sif, gammaf, np.tile(omega, (nf, 1)));
                            L = calculate_Laplacian_matrix(P);
                            [eNew, evNew] = calculate_algebraic_connectivity(L);
                        except:
                            eNew=0

                    if sym:
                        Pt = calculate_probability_matrix_alter(Sif, gammaf, np.tile(omega, (nf, 1)));
                        Lt = calculate_Laplacian_matrix(Pt);
                        [etNew, evtNew] = calculate_algebraic_connectivity(Lt);
                        print(eNew, etNew)
                        eNew1=eNew
                        eNew = etNew * eNew*(10**6)


                else:
                    eNew = 0;
                    eNew1 = 0
                    etNew= 0
                    evNew = np.ones([n, ]);
            else:
                eNew = e
                etNew = 0
                eNew1 = e
                evNew = ev

            Avtn = AverageTask(Si, n)
            Avcon = AveragecoAuth(Si, n, m)
            Avan = AverageAuth(Si, m)
            [Mintn, Maxtn, Vartn] = MinMaxtask (Si , n)

            if params['data']=="MAG":
                PFn = (eNew - Consn) * (1 + pen / (
                        pen1 * Avtn / (2 * n + m) + pen2 * Avcon / (2 * n + m) + pen3 * Avan / (
                        2 * n + m)));  # penalty function

            else:
                PFn = (eNew - Consn) * pen* 1/( pen1 * Avtn / (2*n + m) + pen2 * Avcon / (2*n + m) + pen3 * Avan / (2*n + m) );

            # if Consn == 0:
            if PFn > PF or np.exp(- (PF - PFn) / t) > np.random.uniform(0, 1):
                # accept the new solution
                S = copy.deepcopy(Si);
                S_ex = copy.deepcopy(Si_ex)
                PF = PFn;
                e = eNew;
                ev = evNew;
                et = etNew
                e1= eNew1
                eVec.append(eNew)
                e1Vec.append(eNew1)
                etVec.append(etNew)
                Cons = Consn;
                Conss = Conssn;
                Cm = Cmn;
                Avt = Avtn
                Avco = Avcon
                Ava = Avan
                Mint = Mintn
                Maxt = Maxtn
                Vart = Vartn
                if PF > PFBest:
                    Best_S = copy.deepcopy(S);
                    Best_S_ex = copy.deepcopy(S_ex)
                    Best_Lambda_n = np.copy(Lambda_n);
                    Best_Lambda_m = np.copy(Lambda_m);
                    eBest = e;
                    etBest = et
                    e1Best = e1
                    evBest = ev;
                    eVecB.append(eBest)
                    e1VecB.append(e1Best)
                    etVecB.append(etBest)
                    ConsBest = Cons;
                    ConssBest = Conss;
                    PFBest = PF;
                    BCm = Cm;
                    AvcoBest = Avco
                    AvtBest = Avt
                    AvaBest = Ava
                    MaxtBest = Maxt
                    MintBest = Mint
                    VartBest = Vart



                    if niter >= maxIter - 4000:
                        if S_v.size == 0:
                            soli += soli
                            #S_v = mat_to_vec(Best_S)
                            S_v = np.reshape(Best_S,[1,n*m])
                            # S_ex_v=mat_to_vec_ex(Best_S_ex)
                        else:
                            soli += soli
                            S_v = np.append(S_v, np.reshape(Best_S,[1,n*m]), axis=0)
                            # S_ex_v = np.append(S_ex_v, mat_to_vec_ex(Best_S_ex), axis=0)

                Avtask = np.append(Avtask, AvtBest) #average degree of a node (average task)
                AvCoauth = np.append(AvCoauth, AvcoBest)
                Avauth = np.append(Avauth, AvaBest)
                MaxT = np.append(MaxT, MaxtBest) #maximum degree of a node (maximum task)
                MinT = np.append(MinT, MintBest) #minimum degree of a node (minimum task)
                VarT = np.append(VarT, VartBest) #variance of degree distribution

                Avtask_c = np.append(Avtask_c, Avt)
                AvCoauth_c = np.append(AvCoauth_c, Avco)
                Avauth_c = np.append(Avauth_c, Ava)
                MaxT_c = np.append(MaxT_c, Maxt)  # maximum degree of a node (maximum task)
                MinT_c = np.append(MinT_c, Mint)  # minimum degree of a node (minimum task)
                VarT_c = np.append(VarT_c, Vart)

                f = np.append(f, e);
                fb = np.append(fb, eBest);
                pf = np.append(pf, PF);

        j = j + 1;
        t = a * t;

        # if j > 1:
        # if fb[j-1] == fb[j - 2] or t < 10 ** (-6):
        if t < params["tol"] or niter > maxIter or skip:
            StopCon = 1

    return [Best_S, Best_S_ex, PFBest, f, fb, pf, Avtask, AvCoauth, Avauth, MaxT, MinT, VarT, Avtask_c, AvCoauth_c, Avauth_c, MaxT_c, MinT_c, VarT_c, S_v, eVec, e1Vec, etVec, eVecB, e1VecB, etVecB];

