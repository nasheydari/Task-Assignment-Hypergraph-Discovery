import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from numpy import linalg as LA

from numpy.random import rand
import random
import csv
import pickle
import copy
from numpy.linalg import matrix_power
from Utils import *
import os
import json


with open('./Config/Robustness.json') as f:
   params = json.load(f)


nsat_Best_all_av = 0
Costav_Best_all_av = 0
Conssav_Best_all_av = 0

nsat_initial_all_av = 0
Costav_initial_all_av = 0
Conssav_initial_all_av = 0




log={}





nr=28



if params['APS']:
    for it in range(1,nr+1):
        name_N='name_'+str(it)+'.txt'
        path_name=params['res_path_APS']+'names/'+name_N
        with open(path_name, 'r') as f1:
            Name = f1.readlines()
        Name=Name[0]
        path_file = params['folder_path_APS']+Name
        B_G=np.loadtxt(path_file)
        pack = 1
        B_G_pack = B_G / pack
        B_Gr_pack = np.ceil(B_G_pack)
        E = np.sum(B_Gr_pack, axis=0)
        B = np.sum(B_Gr_pack, axis=1)

        name_I = 'Initial_I_'+str(it)+'.txt'
        name_I_ex = 'Initial_I_ex_' + str(it) + '.txt'
        name_B = 'Best_I_'+str(it)+'.txt'
        name_B_ex = 'Best_I_ex_' + str(it) + '.txt'
        path_I= params['res_path_APS']+'Initial_I/'+name_I
        I_initial= np.loadtxt(path_I)
        path_I_ex = params['res_path_APS']+'Initial_I_ex/' + name_I_ex
        I_initial_ex = np.loadtxt(path_I_ex)
        path_B=params['res_path_APS']+'Best_I/'+name_B
        I_Best=np.loadtxt(path_B)
        path_B_ex = params['res_path_APS']+'Best_I_ex/' + name_B_ex
        I_Best_ex = np.loadtxt(path_B_ex)


        Buds = np.cumsum(B)
        Bud = int(sum(B))





        [n,m]=I_initial.shape


        [nex,mex] = I_initial_ex.shape



        Lambda_n = 3 * np.ones([n, ]);
        Lambda_m = 3 * np.ones([m, ]);

        Sanity_check(I_initial, I_initial_ex, B)
        Sanity_check(I_Best, I_Best_ex, B)






        path_I_b = params['res_path_APS_Bipartite'] + 'Initial_I/' + name_I
        I_initial = np.loadtxt(path_I_b)
        path_I_ex_b = params['res_path_APS_Bipartite'] + 'Initial_I_ex/' + name_I_ex
        path_B_b = params['res_path_APS_Bipartite'] + 'Best_I/' + name_B
        I_Best_b = np.loadtxt(path_B_b)
        path_B_ex_b = params['res_path_APS_Bipartite'] + 'Best_I_ex/' + name_B_ex
        I_Best_ex_b = np.loadtxt(path_B_ex_b)



        gamma_initial = gamma_generate(I_initial_ex, I_initial, B, n, m)
        omega_initial = np.reshape(E, (1, m))
        gamma_Best = gamma_generate(I_Best_ex, I_Best, B, n, m)
        omega_Best = np.reshape(E, (1, m))
        gamma_Best_b = gamma_generate(I_Best_ex_b, I_Best_b, B, n, m)
        omega_Best_b = np.reshape(E, (1, m))




        P_initial = calculate_probability_matrix(I_initial, gamma_initial, np.tile(omega_initial, (n, 1)));
        L_initial = calculate_Laplacian_matrix(P_initial);
        [e_initial, ev_initial] = calculate_algebraic_connectivity(L_initial);

        P_Best = calculate_probability_matrix(I_Best, gamma_Best, np.tile(omega_Best, (n, 1)));
        L_Best = calculate_Laplacian_matrix(P_Best);
        [e_Best, ev_Best] = calculate_algebraic_connectivity(L_Best);

        P_initial_b = calculate_bipartite_probability_matrix(I_initial, gamma_initial, np.tile(omega_initial, (n, 1)));
        L_initial_b = calculate_Laplacian_matrix(P_initial);
        [e_initial_b, ev_initial_b] = calculate_algebraic_connectivity(L_initial);

        P_Best_b = calculate_bipartite_probability_matrix(I_Best, gamma_Best, np.tile(omega_Best, (n, 1)));
        L_Best_b = calculate_Laplacian_matrix(P_Best);
        [e_Best_b, ev_Best_b] = calculate_algebraic_connectivity(L_Best);

        Avtask_initial = AverageTask(I_initial, n)
        Avtask_Best = AverageTask(I_Best, n)
        Avtask_Best_b = AverageTask(I_Best_b, n)

        Avauth_initial = AverageAuth(I_initial, m)
        Avauth_Best = AverageAuth(I_Best, m)
        Avauth_Best_b = AverageAuth(I_Best_b, m)

        Avcoauth_initial = AveragecoAuth(I_initial, n, m)
        Avcoauth_Best = AveragecoAuth(I_Best, n, m)
        Avcoauth_Best_b = AveragecoAuth(I_Best_b, n, m)







        Natt = params['Natt']

        nsat_Best = []
        Costav_Best = []
        Conssav_Best = []

        nsat_Best_b = []
        Costav_Best_b = []
        Conssav_Best_b = []

        nsat_initial = []
        Costav_initial = []
        Conssav_initial = []

        niter = params['niter']











        [Consn, Conssn, Cmn, Cnn, Cmn_d, Cnn_d] = calculate_Constraints_ex(I_Best_ex, Lambda_m, Lambda_n, E, n, m, Buds, B);




        if Consn>0:
            print("Solution is not feasible")


        if params["Best_APS"]:
            for i in range(niter):

                I_att, I_ex_att, B_att, E_att, Lambda_m_att, Lambda_n_att=Attack_node(I_Best, I_Best_ex, n, m, Natt, B, E, Lambda_m, Lambda_n)
                if np.sum(B_att)<np.sum(E_att):
                    print("unfeasible")

                Sanity_check(I_att,I_ex_att, B_att )
                [n_att,m_att]= I_att.shape

                omega = np.reshape(E_att, (1, m_att))

                gamma= gamma_generate(I_ex_att, I_att, B_att, n_att, m_att)
                k = np.sum(I_att, axis=1);
                I_att_f = np.copy(I_att[k > 0, :]);
                gammaf = np.copy(gamma[k > 0, :]);
                n_att_f = I_att_f.shape[0]
                P = calculate_probability_matrix(I_att, gammaf, np.tile(omega, (n_att_f, 1)));
                FlagC = (matrix_power(5 * P, n) > 0).all()





                [Cost, Conss, I_att_c, I_ex_att_c] =Restore_I_nodeat(I_att, I_ex_att, B_att, E_att, n_att, m_att, Lambda_m_att, Lambda_n_att)

                if Conss > 0:
                    nsat_Best.append(1)
                else:
                    nsat_Best.append(0)

                Conssav_Best.append(Conss)
                Costav_Best.append(Cost)


        if params["Initial_APS"]:
            for i in range(niter):
                I_initial_att, I_initial_ex_att, B_initial_att, E_initial_att, Lambda_initial_m_att, Lambda_initial_n_att=Attack_node(I_initial, I_initial_ex, n, m, Natt, B, E, Lambda_m, Lambda_n)

                if np.sum(B_initial_att)<np.sum(E_initial_att):
                    print("unfeasible")

                Sanity_check(I_initial_att,I_initial_ex_att, B_initial_att )
                [n_initial_att,m_initial_att]= I_initial_att.shape

                omega_initial = np.reshape(E_initial_att, (1, m_initial_att))

                gamma_initial= gamma_generate(I_initial_ex_att, I_initial_att, B_initial_att, n_initial_att, m_initial_att)
                k_initial = np.sum(I_initial_att, axis=1);
                I_initial_att_f = np.copy(I_initial_att[k_initial > 0, :]);
                gammaf_initial = np.copy(gamma_initial[k_initial > 0, :]);
                n_initial_att_f = I_initial_att_f.shape[0]
                P_initial = calculate_probability_matrix(I_initial_att_f, gammaf_initial, np.tile(omega_initial, (n_initial_att_f, 1)));
                FlagC = (matrix_power(5 * P_initial, n) > 0).all()


                [Cost_initial, Conss_initial, I_initial_att_c, I_initial_ex_att_c] =Restore_I_nodeat(I_initial_att, I_initial_ex_att, B_initial_att, E_initial_att, n_initial_att, m_initial_att, Lambda_initial_m_att, Lambda_initial_n_att)



                if Conss_initial > 0:
                    nsat_initial.append(1)
                else:
                    nsat_initial.append(0)

                Conssav_initial.append(Conss_initial)
                Costav_initial.append(Cost_initial)




        if params["Bipartite_APS"]:

            for i in range(niter):
                I_att_b, I_ex_att_b, B_att_b, E_att_b, Lambda_m_att_b, Lambda_n_att_b=Attack_node(I_Best_b, I_Best_ex_b, n, m, Natt, B, E, Lambda_m, Lambda_n)
                if np.sum(B_att_b)<np.sum(E_att_b):
                    print("unfeasible")

                Sanity_check(I_att_b,I_ex_att_b, B_att_b )
                [n_att_b,m_att_b]= I_att_b.shape

                omega_b = np.reshape(E_att_b, (1, m_att_b))

                gamma_b= gamma_generate(I_ex_att_b, I_att_b, B_att_b, n_att_b, m_att_b)
                k = np.sum(I_att_b, axis=1);
                I_att_f_b = np.copy(I_att_b[k > 0, :]);
                gammaf_b = np.copy(gamma_b[k > 0, :]);
                n_att_f_b = I_att_f_b.shape[0]
                P_b = calculate_probability_matrix(I_att_b, gammaf_b, np.tile(omega_b, (n_att_f_b, 1)));






                [Cost_b, Conss_b, I_att_c_b, I_ex_att_c_b] =Restore_I_nodeat(I_att_b, I_ex_att_b, B_att_b, E_att_b, n_att_b, m_att_b, Lambda_m_att_b, Lambda_n_att_b)

                if Conss > 0:
                    nsat_Best_b.append(1)
                else:
                    nsat_Best_b.append(0)

                Conssav_Best_b.append(Conss_b)
                Costav_Best_b.append(Cost_b)


        Costav_Best_av=np.average(Costav_Best)
        nsat_Best_av=np.average(nsat_Best)
        Conssav_Best_av=np.average(Conssav_Best)

        Costav_Best_std=np.std(Costav_Best)
        nsat_Best_std=np.std(nsat_Best)
        Conssav_Best_std=np.std(Conssav_Best)

        Costav_Best_b_av = np.average(Costav_Best_b)
        nsat_Best_b_av = np.average(nsat_Best_b)
        Conssav_Best_b_av = np.average(Conssav_Best_b)

        Costav_Best_b_std = np.std(Costav_Best_b)
        nsat_Best_b_std = np.std(nsat_Best_b)
        Conssav_Best_b_std = np.std(Conssav_Best_b)

        Costav_initial_av=np.average(Costav_initial)
        nsat_initial_av=np.average(nsat_initial)
        Conssav_initial_av=np.average(Conssav_initial)

        Costav_initial_std=np.std(Costav_initial)
        nsat_initial_std=np.std(nsat_initial)
        Conssav_initial_std=np.std(Conssav_initial)

        log[Name] = {}
        log[Name]['Optimized Sol Cost_av'] = Costav_Best_av
        log[Name]['Optimized Sol Unsat Cons_av'] = Conssav_Best_av
        log[Name]['Optimized Sol Unsat Iters_av'] = nsat_Best_av
        log[Name]['Bipartite Optimized Sol Cost_av'] = Costav_Best_b_av
        log[Name]['Bipartite Optimized Sol Unsat Cons_av'] = Conssav_Best_b_av
        log[Name]['Bipartite Optimized Sol Unsat Iters_av'] = nsat_Best_b_av
        log[Name]['Initial Sol Cost_av'] = Costav_initial_av
        log[Name]['Initial Sol Unsat Cons_av'] = Conssav_initial_av
        log[Name]['Initial Sol Unsat Iters_av'] = nsat_initial_av
        log[Name]['Optimized Sol Cost_std'] = Costav_Best_std
        log[Name]['Optimized Sol Unsat Cons_std'] = Conssav_Best_std
        log[Name]['Optimized Sol Unsat Iters_std'] = nsat_Best_std
        log[Name]['Bipartite Optimized Sol Cost_std'] = Costav_Best_b_std
        log[Name]['Bipartite Optimized Sol Unsat Cons_std'] = Conssav_Best_b_std
        log[Name]['Bipartite Optimized Sol Unsat Iters_std'] = nsat_Best_b_std
        log[Name]['Initial Sol Cost_std'] = Costav_initial_std
        log[Name]['Initial Sol Unsat Cons_std'] = Conssav_initial_std
        log[Name]['Initial Sol Unsat Iters_std'] = nsat_initial_std



        log[Name]['Optimized Algebraic Connectivity']= e_Best
        log[Name]['Optimized Bipartite Algebraic Connectivity'] = e_Best_b
        log[Name]['Initial Algebraic Connectivity'] = e_initial

        log[Name]['Optimized Average Assigned Papers'] = Avtask_Best
        log[Name]['Bipartite Optimized Average Assigned Papers'] = Avtask_Best_b
        log[Name]['Initial Assigned Papers'] = Avtask_initial

        log[Name]['Optimized Average Assigned Authors'] = Avauth_Best
        log[Name]['Bipartite Optimized Average Assigned Authors'] = Avauth_Best_b
        log[Name]['Initial Assigned Authors'] = Avauth_initial

        log[Name]['Optimized Average Co Authors'] = Avauth_Best
        log[Name]['Bipartite Optimized Average Co Authors'] = Avauth_Best_b
        log[Name]['Initial Average Co Authors'] = Avauth_initial

    if not os.path.exists(params["logging_path"]):
        # Create the folder if it doesn't exist
        os.makedirs(params["logging_path"])

    with open(params["logging_path"]+"log_attack"+str(params["Natt"])+".json", "w") as write_file:
        json.dump(log, write_file)


