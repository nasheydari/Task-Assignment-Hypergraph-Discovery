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






def calculate_probability_matrix(I, gamma, omega):
    [n,m]=I.shape

    R = (I * gamma).T;
    W = (I * omega);

    delta = np.sum(R, axis=1);
    d = np.sum(W, axis=1);
    delta= np.maximum(delta, np.ones(m,))
    d = np.maximum(d, np.ones(n,))

    P = np.diag(d ** -1) @ W @ np.diag(delta ** -1) @ R;
    P = np.nan_to_num(P, 0);

    return P;


def calculate_bipartite_probability_matrix(I, gamma, omega):
    [n, m] = I.shape

    R = (I * gamma).T;
    W = (I * omega);

    A = np.block([[np.zeros([n, n]), W], [R, np.zeros([m, m])]]);
    D = np.diag(np.sum(A, axis=1) ** (-1))
    D = np.nan_to_num(D, 0);
    P = D @ A;

    P = np.nan_to_num(P, 0);

    return P;

def calculate_Laplacian_matrix(P, k=1):
    eigenValues, eigenVectors = eigs(P.transpose(), k, which='LR');
    idx = eigenValues.argsort()[::-1];
    eigenValues = eigenValues[idx];
    eigenVectors = eigenVectors[:, idx];

    #Pi = np.diag(np.abs(eigenVectors[:, 0]));

    pi= np.abs(eigenVectors[:, 0]) / sum(np.abs(eigenVectors[:, 0]))

    Pi = np.diag(pi)

    L = Pi - (Pi @ P + P.transpose() @ Pi) / 2;

    return L;


def calculate_algebraic_connectivity(L, k=4):
    eigenValues, eigenVectors = eigsh(L, k, which='SM');
    if len(eigenValues[~ np.isclose(eigenValues, 0)])>0:
        ac = np.min(eigenValues[~ np.isclose(eigenValues, 0)]);
        Fv = eigenVectors[:, np.argwhere(eigenValues == ac).T[0]];
    else:
        ac=0
        Fv=np.ones([L.shape[0],])

    return [ac, Fv];


def calculate_algebraic_connectivity_b(L, k=10):
    converged = False;

    while not converged:
        converged = True;

        try:
            eigenValues, eigenVectors = eigsh(L, k, which='SM', maxiter=L.shape[0] * 100);  # , maxiter=L.shape[0]*100

        except (ArpackNoConvergence) as err:
            print(len(err.eigenvalues))

            if len(err.eigenvalues) >= 2:
                eigenValues = err.eigenvalues;
                eigenVectors = err.eigenvectors;

            else:
                k = 2 * k;
                converged = False;

                print(k)

    ac = np.min(eigenValues[~ np.isclose(eigenValues, 0)]);
    Fv = eigenVectors[:, np.argwhere(eigenValues == ac).T[0]]

    return [ac, Fv];



def calculate_Constraints_ex(I_ex, Lambda_m, Lambda_n, E,  n, m, Buds, B):
    C1 = np.maximum(np.zeros([m, ]), E - np.sum(I_ex, 0));
    C1_d = E - np.sum(I_ex, 0);
    C2 = np.zeros([n, ])
    C2_d = np.zeros([n, ])
    for j in range(n):
        if j > 0:
            C2[j] = np.maximum(0, np.sum(np.sum(I_ex[int(Buds[j - 1]):int(Buds[j]),:],0)) - B[j])
            C2_d[j] = np.sum(np.sum(I_ex[int(Buds[j - 1]):int(Buds[j]),:],0)) - B[j]
        else:
            C2[j] = np.maximum(0, np.sum(np.sum(I_ex[0:int(Buds[j]),:],0)) - B[j]);
            C2_d[j] = np.sum(np.sum(I_ex[0:int(Buds[j]),:],0)) - B[j];
    Cons = np.sum(Lambda_m * C1, axis=None) + np.sum(Lambda_n * C2, axis=None)
    Conss = np.sum(C1, axis=None) + np.sum(C2, axis=None)
    return [Cons, Conss, C1, C2, C1_d, C2_d]




def Extohyp(Si_ex, n, m, B):
    Si = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            if i > 0:
                if sum(Si_ex[int(sum(B[0:i])):int(sum(B[0:i + 1])), j]) > 0:
                    Si[i, j] = 1
            else:
                if sum(Si_ex[0:int(sum(B[0:i + 1])), j]) > 0:
                    Si[i, j] = 1
    return Si


def gamma_generate(Si_ex, Si, B, n, m):
    gamma = np.ones([n, m])
    for i in range(n):
        for j in range(m):
            if Si[i, j] > 0:
                if i > 0:
                    gamma[i, j] = np.sum(Si_ex[int(sum(B[0:i])):int(sum(B[0:i + 1])), j])
                else:
                    gamma[i, j] = np.sum(Si_ex[0:int(sum(B[0:i + 1])), j])
    return gamma





def read_Hypergraph(file_name):
    H = list();

    f = open(file_name, 'r');

    line = f.readline();
    vline = line.split();

    header = [int(numeric_string) for numeric_string in vline];
    n = header[0];
    m = header[1];

    for i in range(m):
        line = f.readline();
        vline = line.split();
        hyperedge = [int(numeric_string) for numeric_string in vline];

        H.append(hyperedge[1:]);

    H = np.array(H);

    return [n, m, H];


def read_Hypergraph_as_Incidence(file_name):
    f = open(file_name, 'r');

    line = f.readline();
    vline = line.split();

    header = [int(numeric_string) for numeric_string in vline];
    n = header[0];
    m = header[1];

    I = np.zeros([m, n]);

    for i in range(m):
        line = f.readline();
        vline = line.split();
        hyperedge = [int(numeric_string) for numeric_string in vline];
        I[i, hyperedge[1:]] = 1;

    return I;


# Read budget and energy constraints from txt files
def read_constraint(file_name):
    f = open(file_name, 'r');
    line = f.readline();
    vline = line.split();
    B = [int(bu) for bu in vline[1:]];
    line = f.readline();
    vline = line.split();
    E = [int(en) for en in vline[1:]];

    return [B, E];





def calculate_probability_matrix_alter(I, gamma, omega):
    It=I.T
    gammat= gamma.T
    omegat=omega.T

    Rt = (It * gammat)
    Wt = (It * omegat).T


    deltat = np.sum(Rt, axis=1)
    dt = np.sum(Wt, axis=1)
    deltat= np.maximum(deltat, np.ones(m,))
    dt = np.maximum(dt, np.ones(n,))


    P = np.diag(deltat ** -1) @ Rt @ np.diag(dt ** -1) @ Wt

    P = np.nan_to_num(P, 0);

    return P;









def calculate_Constraints(I, Lambda_n, Lambda_m, B, E, gamma, n, m):
    C1 = np.maximum(np.zeros([m, ]), E - np.sum(I * gamma, 0));
    C2 = np.maximum(np.zeros([n, ]), np.sum(I * gamma, 1) - B);
    Cons = np.sum(Lambda_n * C2, axis=None) + np.sum(Lambda_m * C1, axis=None);
    Conss = np.sum(C2, axis=None) + np.sum(C1, axis=None);
    return [Cons, Conss, C1, C2];







def constraints_diff(I, gamma, E, B):
    M = I * gamma;
    return [E - np.sum(M, axis=0), B - np.sum(M, axis=1)];






def check_constraints_bool(I, gamma, E, B):
    [Ee, Be] = constraints_diff(I, gamma, E, B);

    if (np.sum(Ee < 0) > 0 and np.sum(Be > 0) < 0):
        return False;

    else:
        return True;


def perturbation_swap_ex(I_ex, I, nswaps, E, B, ev):
    Si_ex = copy.deepcopy(I_ex);
    Si = copy.deepcopy(I)
    Bud = int(sum(B))
    Buds = np.cumsum(B).astype(np.int64)
    m = len(E)
    n= len(B)
    for i in range(nswaps):
        [Cons, Conss, Cm, Cn, Cm_d, Cn_d] = calculate_Constraints_ex(Si_ex, 3 * np.ones([m, ]), 3 * np.ones([n, ]), E, n, m, Buds, B);
        # print(Conss)
        # if Cons == 0:
        #     rnd = 1
        # else:
        rnd = np.random.uniform()
        if rnd < 0.35: #swap agents between tasks
            # author=np.random.choice(Bud)
            # hyperedges=np.argwhere(Si_ex[author,:]> 0).T[0]
            # print(np.sum(np.maximum(np.zeros([m,] ), Ee)))
            # for edge in  hyperedges:
            #     if Ee[edge]<-1:
            #         hyperedges_p= hyperedges_p+[int(edge)]
            hyperedges_p = np.argwhere(Cm_d <= -1).T[0]
            hyperedges_n = np.argwhere(Cm_d > 0).T[0]
            if len(hyperedges_p) > 0 and len(hyperedges_n) > 0:
                hyperedge0 = np.random.choice(hyperedges_p, 1).T[0]
                hyperedge1 = np.random.choice(hyperedges_n, 1).T[0]
                H0 = np.argwhere(Si_ex[:, hyperedge0] > 0).T[0];
                author = np.random.choice(H0, 1).T[0];
                Si_ex[author, hyperedge0] = 0
                Si_ex[author, hyperedge1] = 1
                author_r = np.argwhere(Buds > author)[0][0]
                if not (sum(Si_ex[int(Buds[author_r-1]):int(Buds[author_r]), hyperedge0]) > 0):
                    Si[author_r, hyperedge0] = 0
                Si[author_r, hyperedge1] = 1
        elif rnd<0.35: #swap tasks between agents
            # author=np.random.choice(Bud)
            # hyperedges=np.argwhere(Si_ex[author,:]> 0).T[0]
            # print(np.sum(np.maximum(np.zeros([m,] ), Ee)))
            # for edge in  hyperedges:
            #     if Ee[edge]<-1:
            #         hyperedges_p= hyperedges_p+[int(edge)]
            nodes_p = np.argwhere(Cn_d <= -1).T[0]
            nodes_n = np.argwhere(Cn_d > 0).T[0]
            if len(nodes_p) > 0 and len(nodes_n) > 0:
                node0 = np.random.choice(nodes_p, 1).T[0]
                node1 = np.random.choice(nodes_n, 1).T[0]
                H0 = np.argwhere(Si_ex[node1,:] > 0).T[0];
                tsk = np.random.choice(H0, 1).T[0];
                Si_ex[node0, tsk] = 1
                Si_ex[node1, tsk] = 0
                author_r0 = np.argwhere(Buds > node0)[0][0]
                author_r1 = np.argwhere(Buds > node1)[0][0]
                if author_r1 > 0:
                    atr1 = int(Buds[author_r1 - 1])
                else:
                    atr1 = 0
                if not (sum(Si_ex[atr1:int(Buds[author_r1]), tsk]) > 0):
                    Si[author_r1, tsk] = 0
                Si[author_r0, tsk] = 1
        else:
            # if ev.shape[0] == B.shape[0]:
            #     # choose 2 hyperedges
            #     d = np.abs(I * ev);
            #     p = np.sum(d, axis=0) / np.sum(d);
            #     p =(p+0.0001)/np.sum(p+0.0001)
            #     # choose 2 hyperedges
            #     hyperedges = np.random.choice(m, 2, replace=False, p=p);
            # else:
                # choose 2 hyperedges
            hyperedges = np.random.choice(m, 2, replace=False);
            # chose one node in each hyperedge
            H0 = np.argwhere(Si_ex[:, hyperedges[0]] > 0).T[0];
            H1 = np.argwhere(Si_ex[:, hyperedges[1]] > 0).T[0];

            if len(H0) != 0 and len(H1) != 0:
                node0 = np.random.choice(H0, 1).T[0];
                node1 = np.random.choice(H1, 1).T[0];


                Si_ex[node0, hyperedges[0]] = 0;
                Si_ex[node1, hyperedges[1]] = 0;
                Si_ex[node1, hyperedges[0]] = 1;
                Si_ex[node0, hyperedges[1]] = 1;
                author_r0 = np.argwhere(Buds > node0)[0][0]
                author_r1 = np.argwhere(Buds > node1)[0][0]
                if author_r1>0:
                    atr1=int(Buds[author_r1-1])
                else:
                    atr1 = 0
                if author_r0 > 0:
                    atr0 = int(Buds[author_r0 - 1])
                else:
                    atr0 = 0
                if not (np.sum(Si_ex[atr1:int(Buds[author_r1]), hyperedges[1]],axis=0) > 0):
                        Si[author_r1, hyperedges[1]] = 0
                if not (np.sum(Si_ex[atr0:int(Buds[author_r0]), hyperedges[0]],axis=0) > 0):
                    Si[author_r0, hyperedges[0]] = 0

                Si[author_r1, hyperedges[0]] = 1;
                Si[author_r0, hyperedges[1]] = 1;

                # [Ee, Be] = constraints_diff(Si_ex, gamma, E, B);
                # print(np.sum(np.maximum(np.zeros([m, ]), Ee)))

    return [Si_ex, Si];


def perturbation_remove_ex(I_ex, I, nremove, E, B, ev):
    Si_ex = copy.deepcopy(I_ex);
    Si = copy.deepcopy(I)
    Bud = int(sum(B))
    Buds = np.cumsum(B)
    m = len(E)
    for i in range(nremove):
        [Cons, Conss, Cm, Cn, Cm_d, Cn_d] = calculate_Constraints_ex(Si_ex, 3 * np.ones([m, ]), 3 * np.ones([n, ]), E, m, Buds, B);
        nodes_p = np.argwhere(Cn_d > 0).T[0]
        if len(nodes_p) > 0:
            node0 = np.random.choice(nodes_p, 1).T[0]
            H0 = np.argwhere(Si[node0, :] > 0).T[0];
            H0_p = np.argwhere(Cm_d < 0).T[0];
            if len(H0_p)>0:
                H0_pi=np.intersect1d(H0, H0_p)
            else:
                H0_pi=H0
            Hyperedge=np.random.choice(H0_pi, 1).T[0]
            if node0>0:
                atr0=int(Buds[node0 - 1])
            else:
                atr0=0
            if len(H0_pi)>0:
                sub_nodes=np.argwhere(Si_ex[atr0:int(Buds[node0]),Hyperedge] > 0).T[0];
                sub_node=np.random.choice(sub_nodes, 1).T[0]
                Si_ex[atr0+sub_node, Hyperedge] = 0
                if not (sum(Si_ex[atr0:int(Buds[node0]), Hyperedge]) > 0):
                    Si[node0, Hyperedge] = 0
    return [Si_ex, Si];




# WARNING: I need to better check this function. I am not trusting it 100%
def perturbation_add_node_task(I, gamma, E, B, Ee, Be):
    Si = copy.deepcopy(I);
    [n, m] = I.shape;

    # [Ee, Be] = constraints_diff(I, gamma, E, B);

    # chose a node wit a probability that is proportinal to its remaining budget
    nodeset = np.argwhere(Be > 0).T[0]
    if len(nodeset) != 0:
        pn = np.abs(Be[nodeset]) / np.sum(np.abs(Be[nodeset]));
        node = np.random.choice(nodeset, 1, replace=False, p=pn);

        # chose a task with a probability that is proportional to its extra required energy
        taskset = np.argwhere(I[node[0], :] == 0).T[0]
        Eeset = Ee[taskset]
        taskset1 = np.argwhere(Eeset > 0).T[0]
        pt = Ee[taskset[taskset1]] / np.sum(Ee[taskset[taskset1]]);
        task = np.random.choice(taskset[taskset1], 1, replace=False, p=pt);
        # task = np.random.choice(m, 1)

        # check the constraints
        if (Be[node[0]] - gamma[node[0], task[0]] >= 0):
            Si[node[0], task[0]] = 1;

    return Si;


# WARNING: I need to better check this function. I am not trusting it 100%
def perturbation_remove_node_task(I, gamma, E, B, Ee, Be):
    Si = copy.deepcopy(I);
    [n, m] = I.shape;

    # [Ee, Be] = constraints_diff(I, gamma, E, B);

    # chose a node with a probability that is proportinal to its extra allocated energy
    nodeset = np.argwhere(Be < 0).T[0]
    if len(nodeset) != 0:
        pn = -Be[nodeset] / np.sum(-Be[nodeset]);
        node = np.random.choice(nodeset, 1, replace=False, p=pn);

        # chose a task with a probability that is proportional to its extra allocated energy
        taskset = np.argwhere(I[node[0], :] > 0).T[0]
        Eeset = Ee[taskset]
        taskset1 = np.argwhere(Eeset < 0).T[0]
        pt = -Ee[taskset[taskset1]] / np.sum(-Ee[taskset[taskset1]]);
        task = np.random.choice(taskset[taskset1], 1, replace=False, p=pt)
        # task = np.random.choice(m, 1)

        Si[node[0], task[0]] = 0;

    return Si;


def perturbation_ex(I_ex, I, nswaps, E, B, ev):
    # adaptively choose between strategies
    # Ee = constraints_diff_ex(I_ex, E)

    [I_ex2, I2] = perturbation_swap_ex(I_ex, I, nswaps, E, B, ev);

    return [I_ex2, I2]





def mat_to_vec(S):
    [n, m] = S.shape
    S_v = np.zeros([1, n * m])
    for i in range(n):
        S_v[0, i * m:i * m + m] = S[i, :]
    return S_v














def AverageTask(S, n):
    taskav = 0
    for i in range(n):
        taskav += len(list(np.argwhere(S[i, :] > 0).T[0]))
    taskav2 = taskav / n
    return taskav2

def AverageAuth(S, m):
    Authav = 0
    for i in range(m):
        Authav += len(list(np.argwhere(S[:, i] > 0).T[0]))
    Authav2 = Authav / m
    return Authav2


def AveragecoAuth(S, n, m):
    coav = 0
    for i in range(n):
        ci = []
        for j in range(m):
            if S[i, j] > 0:
                ci.extend(list(np.argwhere(S[:, j] > 0).T[0]))
        ci = set(ci)
        coav += len(ci)
    coav2 = coav / n
    return coav2


def MinMaxtask(S, n):
    ci = np.array([]);
    for i in range(n):
        ci= np.append(ci , len(np.argwhere(S[i, :] > 0).T[0]))
    mint = np.min(ci)
    maxt = np.max(ci)
    vart = np.var(ci)
    return [mint, maxt, vart]




def coAuth_H(S, n, m):
    coauth = np.array([])
    for i in range(n):
        ci = []
        for j in range(m):
            if S[i, j] > 0:
                ci.extend(list(np.argwhere(S[:, j] > 0).T[0]))
        ci = set(ci)
        coauth = len(ci)
    return coauth






def Attack_node(I, I_ex, n, m, Natt, B, E, Lambda_m, Lambda_n):
    Bud=np.sum(B)
    Buds=np.cumsum(B)
    node_att = sorted(np.random.choice(n, n-Natt, replace=False));
    I_att= I[node_att, :]
    Buds_att=np.array([])
    for i in node_att:
        if i>0:
            Buds_att=np.append(Buds_att, range(int(Buds[i-1]), int(Buds[i])))
        else:
            Buds_att = np.append(Buds_att, range(0, int(Buds[i])))
    Buds_att=Buds_att.astype(np.int64)
    I_ex_att = I_ex[Buds_att,:]
    B_att= B[node_att]
    deficit=int(sum(B)-sum(B_att))
    E_red = E
    deficit2 = min(deficit, m)
    deficit3=deficit2
    while deficit2>0:
        E_red_p = sorted(np.random.choice(m, deficit2, replace=False))
        E_red[E_red_p] -= 1
        deficit2 = min(deficit - deficit3, m)
        if deficit2>0:
            deficit3 = deficit3+deficit2




    I_check=Extohyp(I_ex_att, n-Natt, m, B_att)
    if not (I_check==I_att).all():
        print('error')
    Sanity_check(I_att, I_ex_att, B_att)
    Lambda_n_att=Lambda_n[node_att]
    k = np.sum(I_att, axis=0)
    I_att_c = np.copy(I_att[:, k > 0])
    I_ex_att_c = np.copy(I_ex_att[:, k > 0])
    E_att= E_red[k>0]
    Lambda_m_att=Lambda_m[k>0]
    return [I_att_c, I_ex_att_c, B_att, E_att, Lambda_m_att, Lambda_n_att]

def findNH(H0, d, I_att):
    Hs = set(H0)
    hyps_c = Hs
    j = 1
    Ns = set([])
    Nodes = set([])
    Nodes_1 = set([])
    Hsd= Hs
    while j<=d:
        j += 1
        Nodes_c = Nodes
        for h in list(Hsd):
            n1 = np.argwhere(I_att[:, h] > 0).T[0]
            Ns = Ns.union(set(n1))
        Nodes = Nodes.union(Ns)
        Nodes_1 = Nodes.difference(Nodes_c)
        Hs=set([])
        Hyps_c_1 = hyps_c
        for n in list(Nodes_1):
            h1 = np.argwhere(I_att[n, :] > 0).T[0]
            Hs = Hs.union(set(h1))
        hyps_c = hyps_c.union(Hs)
        Hsd=hyps_c.difference(Hyps_c_1)
    return [Nodes_1 , Hsd]



def RankH0(H0, I_att, nodes):
    H0_dic = {}
    N0_dic = {}
    d=0
    n1={0}
    h1={0}
    while len(n1)>0 and len(h1)>0:
        d+=1
        [n1, h1] = findNH(H0, d, I_att)
        for h1n in h1:
            H0_dic[h1n] = d
        for n1n in n1:
            N0_dic[n1n] = d
    return [N0_dic,H0_dic]




def Restore_I_nodeat(I_att, I_ex_att, B_att, E_att, n_att, m_att, Lambda_m, Lambda_n):
    Penalty=5
    Cost=0
    Buds_att= np.cumsum(B_att)
    [Cons, Conss, Cm, Cn, Cm_d, Cn_d] = calculate_Constraints_ex(I_ex_att, Lambda_m, Lambda_n, E_att, n_att, m_att, Buds_att, B_att);
    I_ex_att_c=np.copy(I_ex_att)
    I_att_c = np.copy(I_att)
    k=0
    while Conss>0:
        k+=1
        print(k)
        hyperedges_p = np.argwhere(Cm_d > 0).T[0]
        Pn = Cm_d[hyperedges_p] / np.sum(Cm_d[hyperedges_p])
        H0= np.random.choice(hyperedges_p, 1, replace=False, p=Pn)[0];
        [Nn1, Hn1]=RankH0(np.array([H0]), I_att_c, set(range(n_att)))
        sortedNn1 = sorted(Nn1.items(), key=lambda x: x[1])
        for n11 in sortedNn1:
            hyperedges_n = np.argwhere(Cm_d < 0).T[0]
            n=n11[0]
            Hs = set(np.argwhere(I_att_c[n, :] > 0).T[0])
            if len(Hs.intersection(hyperedges_n))>0:
                H1 = np.random.choice(list(Hs.intersection(hyperedges_n)), 1)[0]
                if n>0:
                    n_ex=int(Buds_att[n-1])+np.argwhere(I_ex_att_c[int(Buds_att[n-1]):int(Buds_att[n]),H1] > 0).T[0][0]
                else:
                    n_ex = np.argwhere(I_ex_att_c[0:int(Buds_att[n]), H1] > 0).T[0][0]

                #author_r = np.argwhere(Buds > n)[0][0]
                I_ex_att_c[n_ex, H1] = 0
                I_ex_att_c[n_ex, H0] = 1
                Cm_d[H0] -= 1
                Cm[H0] -= 1
                Cm_d[H1] += 1
                Conss -= 1
                if not (np.sum(I_ex_att_c[int(Buds_att[n - 1]):int(Buds_att[n]), H1], axis=0) > 0):
                    I_att_c[n, H1] = 0
                I_att_c[n, H0] = 1
                Cost+=Hn1[H1]
                #print(Conss)
            if not (Cm_d[H0] > 0):
                break
        if k>n_att:
            Cost=5*Cost
            break

    return [Cost, Conss, I_att_c, I_ex_att_c]






def Sanity_check(I_att, I_ex_att, B_att ):
    Budss=np.cumsum(B_att).astype(np.int64)
    [nn,mm]=I_att.shape

    for i in range(nn):
        if i>0:
            ch=I_att[i, :] == np.minimum(np.ones([1,mm]), np.sum(I_ex_att[Budss[i - 1]:Budss[i], :],axis=0))
        else:
            ch = I_att[i, :] == np.minimum(np.ones([1,mm]), np.sum(I_ex_att[0:Budss[i], :],axis=0))

        if not ch.all():
            print('Error', i)




def I_ex_gen(B_G, B, n, m):
    I_G_ex = np.zeros([int(sum(B)),m])
    ind=0
    for i in range(n):
        for j in range(m):
            for k in range(int(ind),int(ind+B_G[i,j])):
                I_G_ex[k,j]=1
            ind+=B_G[i,j]
    return  I_G_ex







