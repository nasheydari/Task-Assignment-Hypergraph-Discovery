import timeit
import numpy as np
from numpy.random import rand
from Utils import *
from CSA import *
from Greedy import *
import os
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
import pickle





def Run_exp(params):
    if params['data']=='APS':
        file=0
        for files in os.listdir(params['folder_path']):
            temp_time = timeit.default_timer()
            file += 1

            print("Optimizing ", files)


            logging_path = params['logging_path']
            logging.basicConfig(filename=logging_path, filemode='w', level=logging.INFO)
            log = logging.getLogger('main')

            pth = params['folder_path'] + files
            B_G = np.loadtxt(pth)

            pack = params['pack']

            B_G_pack = B_G / pack
            B_Gr_pack = np.ceil(B_G_pack)
            E = np.ceil(np.sum(B_Gr_pack, axis=0))
            B = np.ceil(np.sum(B_Gr_pack, axis=1))

            [n, m] = B_G_pack.shape

            I_Gr_ex = I_ex_gen(B_Gr_pack, B, n, m)
            I_Gr = (B_Gr_pack > 0).astype(int)
            gamma_Gr = gamma_generate(I_Gr_ex, I_Gr, B, n, m)

            I = I_Gr
            I_ex = I_Gr_ex
            gamma = gamma_Gr



            Buds = np.cumsum(B)
            omega = np.reshape(E, (1, m))


            if np.sum(B) < np.sum(E):
                print("problem is infeasible")
                exit()



            # Run and save the optimization
            if params["Algorithm"]=="CSA":
                [Best, Best_ex, PFBest, AlgCon, AlgCon_best, penalty_fun, Avtask, AvCoauth, Avauth, MaxT, MinT, VarT, Avtask_c, AvCoauth_c,
                Avauth_c, MaxT_c, MinT_c, VarT_c, best_10, eVec, e1Vec, etVec, eVecB, e1VecB, etVecB] = run_CSA(params['epochs'], file,                                                                                            I, I_ex,
                                                                                                             omega,
                                                                                                             gamma, n,
                                                                                                             m,
                                                                                                             params['temp_decay'], params['temp'],
                                                                                                             params['nswaps'],
                                                                                                             params['nremove'], E,
                                                                                                             B, Buds,
                                                                                                             params['sym'], params);

            elif params["Algorithm"] == "CSA_Bipartite":
                [Best, Best_ex, PFBest, AlgCon, AlgCon_best, penalty_fun, Avtask, AvCoauth, Avauth, MaxT, MinT, VarT, Avtask_c, AvCoauth_c,
                 Avauth_c, MaxT_c, MinT_c, VarT_c, best_10, eVec, e1Vec, etVec, eVecB, e1VecB, etVecB] = run_CSA_bipartite(params["epochs"],
                                                                                                                 file,
                                                                                                                 I,
                                                                                                                 I_ex,
                                                                                                                 omega,
                                                                                                                 gamma,
                                                                                                                 n, m,
                                                                                                                 params['temp_decay'],
                                                                                                                 params['temp'],
                                                                                                                 params['nswaps'],
                                                                                                                 params['nremove'],
                                                                                                                 E, B,
                                                                                                                 Buds,
                                                                                                                 params["sym"], params);


            else:
                data_path = params['folder_path']

                # Number of processors for parallel processing
                n_proc = 80

                data = pd.read_csv(data_path, sep=' ', header=None)
                data_int = np.ceil(data)
                budget = np.array(data_int.sum(axis=1))
                energy = np.array(data_int.sum(axis=0))

                if params["Algorithm"] == 'Greedy':

                    projection = 'hypergraph'
                    start_time = time.time()
                    I0_connected, gamma0_connected = greedy_connected_initialisation(budget, energy)
                    I_phase1, OptimisingTrack_phase1, gamma_phase1 = Greedy_bestagent_parallel(budget, energy, I0_connected,
                                                                                               gamma0_connected, n_proc,
                                                                                               projection)
                    end_time = time.time()
                elif params["Algorithm"] == 'RandomGreedy':
                    Nagents_thres = 100
                    projection = 'hypergraph'
                    I0_connected, gamma0_connected = greedy_connected_initialisation(budget, energy)
                    I_phase1, OptimisingTrack_phase1, gamma_phase1 = RandomGreedy_bestagent_parallel(budget, energy,
                                                                                                     I0_connected,
                                                                                                     gamma0_connected,
                                                                                                     n_proc, Nagents_thres,
                                                                                                     projection)
                elif params["Algorithm"] == 'GreedyBipartite':
                    projection = 'bipartite'
                    I0_connected, gamma0_connected = greedy_connected_initialisation(budget, energy)
                    I_phase1, OptimisingTrack_phase1, gamma_phase1 = Greedy_bestagent_parallel(budget, energy, I0_connected,
                                                                                               gamma0_connected, n_proc,
                                                                                               projection)
                elif params["Algorithm"] == 'RandomGreedyBipartite':
                    Nagents_thres = 100
                    projection = 'bipartite'
                    I0_connected, gamma0_connected = greedy_connected_initialisation(budget, energy)
                    I_phase1, OptimisingTrack_phase1, gamma_phase1 = RandomGreedy_bestagent_parallel(budget, energy,
                                                                                                     I0_connected,
                                                                                                     gamma0_connected,
                                                                                                     n_proc, Nagents_thres,
                                                                                                     projection)
            if params["Algorithm"]=='CSA' or params["Algorithm"]=='CSA_Bipartite':
                time = timeit.default_timer() - temp_time

                log.info(f'{files}:, running time: {time}')

                path = params['res_path']
                name = './names/name_' + str(file) + '.txt'
                with open(path+name, 'w') as fe:
                    fe.write(files)

                name = './Initial_I/Initial_I' + '_' + str(file) + '.txt'
                np.savetxt(path+name, I)
                name = './Initial_I_ex/Initial_I_ex' + '_' + str(file) + '.txt'
                np.savetxt(path+name, I_ex)
                name = './Best_I/Best_I' + '_' + str(file) + '.txt'
                np.savetxt(path+name, Best)
                name = './Best_I_ex/Best_I_ex' + '_' + str(file) + '.txt'
                np.savetxt(path+name, Best_ex)
                name = './Avtask/Avtask' + '_' + str(file) + '.txt'
                np.savetxt(path+name, Avtask)
                name = './AvCoauth/AvCoauth' + '_' + str(file) + '.txt'
                np.savetxt(path+name, AvCoauth)
                name = './Avauth/Avauth' + '_' + str(file) + '.txt'
                np.savetxt(path+name, Avauth)
                name = './MaxT/MaxT' + '_' + str(file) + '.txt'
                np.savetxt(path+name, MaxT)
                name = './MinT/MinT' + '_' + str(file) + '.txt'
                np.savetxt(path+name, MinT)
                name = './VarT/VarT' + '_' + str(file) + '.txt'
                np.savetxt(path+name, VarT)

                name = './Best_AlgCon/Best_AlgCon' + '_' + str(file) + '.txt'
                np.savetxt(path+name, AlgCon)
                name = './Best_AlgCon_best/Best_AlgCon_best' + '_' + str(file) + '.txt'
                np.savetxt(path+name, AlgCon_best)
                name = './penalty_fun/penalty_fun' + '_' + str(file) + '.txt'
                np.savetxt(path+name, penalty_fun)

            else:
                os.chdir(params['res_path'])
                # plot of optimising trace of algebraic connectivity
                fig = plt.figure(figsize=(8, 6), dpi=400)
                plt.plot(OptimisingTrack_phase1['ac'])
                plt.xlabel("Optimization Round")
                plt.ylabel("Algebraic Connectivity")

                # plot of optimising trace of Average Number of Co-Authors
                fig = plt.figure(figsize=(8, 6), dpi=400)
                plt.plot(OptimisingTrack_phase1['N_avg_coauthors'])
                plt.ylabel("Average Number of Co-Authors")
                plt.xlabel("Optimization Round")

                # plot of optimising trace of Average Number of Authors per paper
                fig = plt.figure(figsize=(8, 6), dpi=400)
                plt.plot(OptimisingTrack_phase1['avg_Nauthors_per_paper'])
                plt.ylabel("Average Number of Authors per paper")
                plt.xlabel("Optimization Round")

                # plot of optimising trace of Average Number of Papers per Author
                fig = plt.figure(figsize=(8, 6), dpi=400)
                plt.plot(OptimisingTrack_phase1['avg_Npapers_per_author'])
                plt.ylabel("Average Number of Papers per Author")
                plt.xlabel("Optimization Round")

                file_name = f"results/APS/APS_2years/APS_{params['Algorithm']}_{data_path[7:-4]}_OptimisingTrack.pkl"
                print(file_name)
                with open(file_name, 'wb') as file:
                    pickle.dump(OptimisingTrack_phase1, file)

                file_name = f"results/APS/APS_2years/APS_{params['Algorithm']}_{data_path[7:-4]}_gamma.pkl"
                print(file_name)
                with open(file_name, 'wb') as file:
                    pickle.dump(gamma_phase1, file)

                file_name = f"results/APS/APS_2years/APS_{params['Algorithm']}_{data_path[7:-4]}_I.pkl"
                print(file_name)
                with open(file_name, 'wb') as file:
                    pickle.dump(I_phase1, file)







    elif params["data"]=="MAG":

        file = 'MAG_Hypergraph_Title_Only'
        logging_path = params["logging_path"] + file
        logging.basicConfig(filename=logging_path, filemode='w', level=logging.INFO)
        log = logging.getLogger('main')

        temp_time = timeit.default_timer()

        [n, m, H] = read_Hypergraph(params["folder_path"]+"MAG_Hypergraph_Title_Only.hyperedgelist");
        I = read_Hypergraph_as_Incidence(params["folder_path"]+"MAG_Hypergraph_Title_Only.hyperedgelist");
        gamma = np.loadtxt(params["folder_path"]+"MAG_Hypergraph_Title_Only.gamma");
        gamma = gamma.T

        print("Optimizing MAG_Hypergraph_Title_Only")

        B_G = gamma

        pack = 1

        B_G_pack = B_G / pack
        B_Gr_pack = np.ceil(B_G_pack)
        # Energy requirements of papers obtained from the original graph
        E = np.sum(B_Gr_pack, axis=0)
        # Budget requirements of papers obtained from the rounded up graph, to allow a bit of room for the budget vs energy
        B = np.sum(B_Gr_pack, axis=1)

        print(B_G_pack.shape)

        I_Gr_ex = I_ex_gen(B_Gr_pack, B, n, m)
        I_Gr = (B_Gr_pack > 0).astype(int)
        gamma_Gr = gamma_generate(I_Gr_ex, I_Gr, B, n, m)

        I = I_Gr
        I_ex = I_Gr_ex
        gamma = gamma_Gr


        Buds = np.cumsum(B)
        omega = np.reshape(E, (1, m))


        if np.sum(B) < np.sum(E):
            print("problem is infeasible")
            exit()


        # Run and save the optimization
        if params["Algorithm"]=="CSA":
            [Best, Best_ex, PFBest, AlgCon, AlgCon_best, penalty_fun, Avtask, AvCoauth, Avauth, MaxT, MinT, VarT, Avtask_c, AvCoauth_c,
             Avauth_c, MaxT_c, MinT_c, VarT_c, best_10, eVec, e1Vec, etVec, eVecB, e1VecB, etVecB] = run_CSA(
                params['epochs'], file,
                I, I_ex,
                omega,
                gamma, n,
                m,
                params['temp_decay'], params['temp'],
                params['nswaps'],
                params['nremove'], E,
                B, Buds,
                params['sym'], params);
        elif params["Algorithm"]=="CSA_Bipartite":
            [Best, Best_ex, PFBest, AlgCon, AlgCon_best, penalty_fun, Avtask, AvCoauth, Avauth, MaxT, MinT, VarT, Avtask_c, AvCoauth_c,
             Avauth_c, MaxT_c, MinT_c, VarT_c, best_10, eVec, e1Vec, etVec, eVecB, e1VecB, etVecB] = run_CSA(
                params['epochs'], file,
                I, I_ex,
                omega,
                gamma, n,
                m,
                params['temp_decay'], params['temp'],
                params['nswaps'],
                params['nremove'], E,
                B, Buds,
                params['sym'], params);

        time = timeit.default_timer() - temp_time


        log.info(f'{file}:, running time: {time}')




        path = params['res_path']

        name = 'Initial_I.txt'
        np.savetxt(path+name, I)
        name = 'Initial_I_ex.txt'
        np.savetxt(path+name, I_ex)
        name = 'Best_I.txt'
        np.savetxt(path+name, Best)
        name = 'Best_I_ex.txt'
        np.savetxt(path+name, Best_ex)
        name = 'Avtask.txt'
        np.savetxt(path+name, Avtask)
        name = 'AvCoauth.txt'
        np.savetxt(path+name, AvCoauth)
        name = 'Avauth.txt'
        np.savetxt(path+name, Avauth)
        name = 'MaxT.txt'
        np.savetxt(path+name, MaxT)
        name = 'MinT.txt'
        np.savetxt(path+name, MinT)
        name = 'VarT.txt'
        np.savetxt(path+name, VarT)


        name = 'Best_AlgCon.txt'
        np.savetxt(path+name, AlgCon)
        name = 'Best_AlgCon_best.txt'
        np.savetxt(path+name, AlgCon_best)
        name = 'Best_penalty_fun.txt'
        np.savetxt(path+name, penalty_fun)












with open('./Config/CSA_MAG_Bipartite.json') as f:
   params = json.load(f)
Run_exp(params)

















