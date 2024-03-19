

import pandas as pd
import numpy as np
from copy import deepcopy
import time
from scipy.sparse.linalg import eigs,eigsh
from multiprocessing import Pool
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import sys
import random




def H_Weight(I,omega):

    '''
    hyperedge weight matrix W is a |V| × |E| matrix with W(v,e) = ω(e) 
    if v ∈ e, and W(v,e) = 0 otherwise.
    '''
    I_copy=deepcopy(I)
    W=np.asarray([I_e*omega[i] for i,I_e in enumerate(I_copy)]).T
    return W

def Pi_guil(P, k=4):
    '''
    stationary distribution
    '''
    try:
        eigenValues, eigenVectors = eigs(P.transpose(), k, which='LR')

    except KeyboardInterrupt:
        sys.exit(1)
    except:
        k=len(P)
        eigenValues, eigenVectors = eigs(P.transpose(), k, which='LR')
    
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    pi = np.abs(eigenVectors[:,0])/sum(np.abs(eigenVectors[:,0]))

    return pi

def Laplacian(P,pi):
    Pi = np.diag(pi)
    laplacian = Pi - (Pi@P + P.T@Pi)/2
    return laplacian
    
def Vertex_Weight(I,gamma):

    '''
    The vertex-weight matrix R is an an |E| × |V | matrix
    where R[e,v]=contribution of vertex v in edge e
    '''
    I_copy=deepcopy(I)
    R=I*gamma
    return R

def Probability(I,gamma,omega):
    I_copy=deepcopy(I)
    de=np.sum(I_copy*gamma,axis=1)
    dv=[sum(I_v*omega) for I_v in I_copy.T]

    W=H_Weight(I_copy,omega)
    R=Vertex_Weight(I_copy,gamma)
    P = np.diag( np.array(dv)**-1 ) @ W @ np.diag( de**-1 ) @ R
    P = np.nan_to_num(P, 0)

    return P


def algebraic_connectivity(I,gamma,omega, k=4):
    '''
    the second smallest eigenvalue
    This eigenvalue is greater than 0 iff G is a connected graph.
    the number of zeros as eigenvalues of the Laplacian is the number of connected components
    Eigenvalues of a graph laplacian is no less than zero
    
    k is the number of eigenvalues needed to compute by the function
    if all k eigenvalues are too close to 0
    we instead compute all eigenvalues of the laplacian to find the smallest eigenvalue that is
    not close to zero, i.e. the algebraic connectivity
    '''

    

    I_copy=deepcopy(I)
    P=Probability(I_copy,gamma,omega)
    pi=Pi_guil(P,k)
    laplacian=Laplacian(P,pi)

    #print('check P normalised?',np.sum(P,axis=1))
    #print('\n')
    
    try:
        eigenValues, eigenVectors = eigsh(laplacian, k, which='SM');
        eigenValues_pos=eigenValues[ ~ np.isclose(eigenValues,0)]
        if len(eigenValues_pos)>0:
            lambda2_0 = np.min(eigenValues_pos)
        else:
            lambda2_0 =0   
        
        
        
        lambda2_0 = np.min(eigenValues[ ~ np.isclose(eigenValues,0)])

    except KeyboardInterrupt:
        sys.exit(1)
    except:
        k=len(laplacian)
        eigenValues, eigenVectors = eigsh(laplacian, k, which='SM')
        eigenValues_pos=eigenValues[ ~ np.isclose(eigenValues,0)]
        if len(eigenValues_pos)>0:
            lambda2_0 = np.min(eigenValues_pos)
        else:
            lambda2_0 =0   
                
    return lambda2_0    

def check_exist_viable_solution(sampled_author_budget,omega_APS_sample):
    if sum(sampled_author_budget)>=sum(omega_APS_sample):
        print('viable solution exists')
        return True
    else:
        print('no viable solution exists')
        return False      
    



# Calculating Laplacian matrices for assymetric matrices
def calculate_Laplacian_matrix ( P, k=1 ):
    eigenValues, eigenVectors = eigs(P.transpose(), k, which='LR');
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    Pi = np.diag(np.abs(eigenVectors[:,0]));

    L = Pi - (Pi@P + P.transpose()@Pi)/2
    
    return L;


def calculate_bipartite_probability_matrix ( I, gamma, omega ):
    R = I * gamma;
    W = I.T*omega;
    
    [m, n] = I.shape;
        
    A = np.block([[np.zeros([n, n]), W], [R, np.zeros([m, m])]]);
    D = np.diag(np.sum(A, axis=1)**(-1))
    D = np.nan_to_num(D, 0);
    P = D @ A;
    
    P = np.nan_to_num(P, 0);
    return P;




def norm_bipartite_algebraic_connectivity(I0,gamma0,omega0, k=4):
    '''
    the second smallest eigenvalue
    This eigenvalue is greater than 0 iff G is a connected graph.
    the number of zeros as eigenvalues of the Laplacian is the number of connected components
    Eigenvalues of a graph laplacian is no less than zero
    
    k is the number of eigenvalues needed to compute by the function
    if all k eigenvalues are too close to 0
    we instead compute all eigenvalues of the laplacian to find the smallest eigenvalue that is
    not close to zero, i.e. the algebraic connectivity
    '''

    I=deepcopy(I0)
    gamma=deepcopy(gamma0)
    omega=deepcopy(omega0)
    
    
    PB = calculate_bipartite_probability_matrix( I, gamma, omega );
    LB = calculate_Laplacian_matrix ( PB );
    
    try:
        eigenValues, eigenVectors = eigsh(LB, k=k, which='SM');
        eigenValues_pos=eigenValues[ ~ np.isclose(eigenValues,0)]
        if len(eigenValues_pos)>0:
            lambda2_0 = np.min(eigenValues_pos)
        else:
            lambda2_0 =0   
        
        
        
        lambda2_0 = np.min(eigenValues[ ~ np.isclose(eigenValues,0)])

    except KeyboardInterrupt:
        sys.exit(1)
    except:
        k=len(LB)
        eigenValues, eigenVectors = eigsh(LB, k=k, which='SM')
        eigenValues_pos=eigenValues[ ~ np.isclose(eigenValues,0)]
        if len(eigenValues_pos)>0:
            lambda2_0 = np.min(eigenValues_pos)
        else:
            lambda2_0 =0 
    return lambda2_0






def greedy_connected_initialisation(author_budget,omega):
    '''
    Use the highest budget agents to connect the graph
    the selected highest budget agent must has at least 2 units of budget
    
    Use the highest budget agents to span the graph first
    
    Then use the head2tail connection to connect the ed|ges
    
    Require budgets to be integers
    '''

    viable=check_exist_viable_solution(author_budget,omega)
    if not viable:
        print('not viable','len(author_budget)',len(author_budget),'len(energy)',len(omega))
        print('\n')
        return

    else:

        # Find the highest budget agents to span the edges
        n_tasks=len(omega)
        n_agents=len(author_budget)
        sorted_agents=np.argsort(-author_budget)
        sorted_budgets=sorted(author_budget,reverse=True)

        #sorted_budgets[2]=7
        #author_budget[sorted_agents[2]]=7



        # consider sorted_budgets_reserved that saves 1 unit of energy starting from the second spanning agent to later connect the graph
        sorted_budgets_reserved=[i-1 for i in sorted_budgets]
        sorted_budgets_reserved[0]+=1

        end_index=[i for i,j in enumerate(list(np.cumsum(sorted_budgets_reserved)>=n_tasks)) if j==True ][0]

        spanning_agents=sorted_agents[:end_index+1]
        size=(n_tasks,n_agents)
        I0=np.zeros(size)
        gamma=np.zeros(size)
        current_task_energy_requirement=deepcopy(omega)
        current_agent_energy_budget=deepcopy(author_budget)

        # Assign the agents to the edges to make sure there is no empty edge
        spanning_agent_index=0
        agent_j=spanning_agents[spanning_agent_index]
        #agent index in the spanning_agents list

        for task_i in range(n_tasks):

            if (spanning_agent_index==0) and (current_agent_energy_budget[agent_j])<1:
                # For the first spanning agent, use up all energy unless the left over energy is smaller than 1
                #print('current_agent_energy_budget[agent_j]',current_agent_energy_budget[agent_j])
                spanning_agent_index+=1
                agent_j=spanning_agents[spanning_agent_index]
                #print('spanning_agent_index,agent_j',spanning_agent_index,agent_j)

            if current_agent_energy_budget[agent_j]<2 and spanning_agent_index>0:
            # save one unit of budget starting from the second spanning agents
            # This unit of budget will later be used to connect the graph
                spanning_agent_index+=1
                agent_j=spanning_agents[spanning_agent_index]




            #print('-'*50)
            I0[task_i,agent_j]=1
            current_agent_energy_budget[agent_j]-=1
            current_task_energy_requirement[task_i]-=1
            gamma[task_i][agent_j]+=1
            #print('task_i,agent_j assigned',task_i,agent_j)
            #print('current_agent_energy_budget[agent_j]',current_agent_energy_budget[agent_j])
            #print('spanning_agent_index',spanning_agent_index)




        # connected the tasks by assigning tasks with more than 1 unit of energy requirement to the next spanning agent
        residual=sum(sorted_budgets_reserved[:end_index+1])-n_tasks # residual of sum of spanning agent reserved budgets- n tasks
        Nedges_spanned_by_each_agent=deepcopy(list(sorted_budgets_reserved[:end_index+1]))
        Nedges_spanned_by_each_agent[-1]-=residual
        Nedges_spanned_by_each_agent=[int(i) for i in Nedges_spanned_by_each_agent] #convert floats to integers
        if sum(np.array(Nedges_spanned_by_each_agent)<2)>0:
            print('Error! top agent budgets must all be at least 2')


        zero_head_edge_indices=np.cumsum(Nedges_spanned_by_each_agent).tolist()
        zero_head_edge_indices.insert(0,0)

        for current_spanning_agent_index,current_spanning_agent in enumerate(spanning_agents[:-1]):
             # choose a task spanned by the current_spanning_agent that still requires at least 1 units of energy after the spanning assignments

            viable_tasks=[task_j for task_j in list(range(n_tasks))[zero_head_edge_indices[current_spanning_agent_index]:zero_head_edge_indices[current_spanning_agent_index+1]] if current_task_energy_requirement[task_j]>=1]
            task_connect=random.choice(viable_tasks)
            # connect the edges by asking the next spanning agent to take on the chosen edge spanned by the previous agent
            next_spanning_agent=spanning_agents[current_spanning_agent_index+1]
            I0[task_connect,next_spanning_agent]=1 #assign the chosen task used to connecting the spanning agents
            current_agent_energy_budget[next_spanning_agent]-=1
            current_task_energy_requirement[task_connect]-=1
            gamma[task_connect][next_spanning_agent]+=1    

        return I0,gamma
   





def N_avg_coauthors(I):
    #for each author i, calculate the average number of the authors in the papers that author i 
    #is assigned to and then subtract 1 (to subtract author i). 
    #Then average this number over all authors. 
    K,N=np.shape(I)
    N_coauthor=[]
    for agent_j in range(N):
        tasks_agentj_is_in=[i for i,j in enumerate(I.T[agent_j]) if j>0]
        if tasks_agentj_is_in==[]:
            N_coauthor.append(0)
        else:
            N_coauthors_agent_j_across_tasks=[sum(I[task_i])-1 for task_i in tasks_agentj_is_in]
            N_coauthor.append(np.mean(N_coauthors_agent_j_across_tasks))
    return np.mean(N_coauthor)

def avg_Nauthors_per_paper(I):
    # average number of authors per paper
    # Or np.sum(I)/K
    Nauthors_across_papers=np.sum(I,axis=0)
    return np.mean(Nauthors_across_papers)

def avg_Npapers_per_author(I):
    #average number of papers assigned to an author
    # Or np.sum(I)/N
    Npapers_across_authors=np.sum(I,axis=1)
    return np.mean(Npapers_across_authors)


def Greedy_bestagent_parallel(author_budget,omega,I0,gamma,n_proc,projection='hypergraph',tol=1e-5,energy_packet=1):
    

    #For this function, assignment stops when all tasks are filled, meaning agent budgets are not used up

        
    #initialise gamma     
    n_tasks=len(omega)
    n_agents=len(author_budget)

    I0=deepcopy(I0)
    gamma=deepcopy(gamma)

    
    current_task_energy_requirement=np.copy(omega-np.sum(np.multiply(I0, gamma),axis=1))
    current_agent_energy_budget=np.copy(author_budget-np.sum(np.multiply(I0, gamma),axis=0))
    
    viable=check_exist_viable_solution(current_agent_energy_budget,current_task_energy_requirement)
    if not viable:
        return
    
    
    
    assigment_requirement_diff_task=np.sum(np.multiply(I0, gamma),axis=1)-omega
    task_energy_constraint=assigment_requirement_diff_task>-tol
    
    objective_track=[]
    N_avg_coauthors_track=[]
    avg_Nauthors_per_paper_track=[]
    avg_Npapers_per_author_track=[]
    
    while np.prod(task_energy_constraint)==0:

        non_filled_tasks=np.where(task_energy_constraint==False)[0].tolist()
    
        # check whether the current assignment satisfies the agent energy limit constraints
        assigment_budget_diff_agent=np.sum(np.multiply(I0, gamma),axis=0)-author_budget
        agent_energy_constraint=assigment_budget_diff_agent<tol
        available_agents=np.where(agent_energy_constraint==True)[0].tolist()

        I0,gamma,current_task_energy_requirement,current_agent_energy_budget,optimising_track_sub=Greedy_bestagent_sub_parallel(I0,gamma,current_task_energy_requirement,current_agent_energy_budget,non_filled_tasks,available_agents,energy_packet,n_proc,projection)
        
        
        #read results
        objective_track_sub=optimising_track_sub['ac']
        N_avg_coauthors_track_sub=optimising_track_sub['N_avg_coauthors']
        avg_Nauthors_per_paper_track_sub=optimising_track_sub['avg_Nauthors_per_paper']
        avg_Npapers_per_author_track_sub=optimising_track_sub['avg_Npapers_per_author']
        
        
        objective_track=objective_track+objective_track_sub
        N_avg_coauthors_track=N_avg_coauthors_track+N_avg_coauthors_track_sub
        avg_Nauthors_per_paper_track=avg_Nauthors_per_paper_track+avg_Nauthors_per_paper_track_sub
        avg_Npapers_per_author_track=avg_Npapers_per_author_track+avg_Npapers_per_author_track_sub

        
        #check task fullfillment 
        assigment_requirement_diff_task=np.sum(np.multiply(I0, gamma),axis=1)-omega
        task_energy_constraint=assigment_requirement_diff_task>-tol


    # multiplication by entry
    # check whether the current assignment satisfies the task energy requirement constraints    
    assigment_requirement_diff_task=np.sum(np.multiply(I0, gamma),axis=1)-omega
    task_energy_constraint=assigment_requirement_diff_task>-tol
    

    if np.prod(task_energy_constraint)==0:
        print('tasks not all filled, remaining tasks',list(np.where(task_energy_constraint==False)))
        
        
    else:
        print('all tasks filled')
    
    # check whether the current assignment satisfies the agent energy limit constraints

    
    assigment_budget_diff_agent=np.sum(np.multiply(I0, gamma),axis=0)-author_budget
    agent_energy_constraint=assigment_budget_diff_agent<tol
    
    
    #save results
    optimising_track=defaultdict()
    optimising_track['ac']=objective_track
    optimising_track['N_avg_coauthors']=N_avg_coauthors_track
    optimising_track['avg_Nauthors_per_paper']=avg_Nauthors_per_paper_track
    optimising_track['avg_Npapers_per_author']=avg_Npapers_per_author_track
        
    

    if np.prod(agent_energy_constraint)==1:
        print('agents energy limits not exceeded')
    else:
        print('agents energy limits exceeded, overworking agents',list(np.where(agent_energy_constraint==False)))
        return I0,optimising_track,gamma
    return I0,optimising_track,gamma



def RandomGreedy_bestagent_parallel(author_budget,omega,I0,gamma,n_proc,Nagents_thres,projection='hypergraph',tol=1e-5,energy_packet=1):
    

#For this function, assignment stops when all tasks are filled, meaning agent budgets are not used up

        
    #initialise gamma     
    n_tasks=len(omega)
    n_agents=len(author_budget)

    I0=deepcopy(I0)
    gamma=deepcopy(gamma)

    
    #I0=np.zeros((n_tasks,n_agents))
    #gamma=np.zeros((n_tasks,n_agents))
    
    
    current_task_energy_requirement=np.copy(omega-np.sum(np.multiply(I0, gamma),axis=1))
    current_agent_energy_budget=np.copy(author_budget-np.sum(np.multiply(I0, gamma),axis=0))
    
    viable=check_exist_viable_solution(current_agent_energy_budget,current_task_energy_requirement)
    if not viable:
        return
    
    
    
    assigment_requirement_diff_task=np.sum(np.multiply(I0, gamma),axis=1)-omega
    task_energy_constraint=assigment_requirement_diff_task>-tol
    
    objective_track=[]
    N_avg_coauthors_track=[]
    avg_Nauthors_per_paper_track=[]
    avg_Npapers_per_author_track=[]

    while np.prod(task_energy_constraint)==0:

        non_filled_tasks=np.where(task_energy_constraint==False)[0].tolist()
    
        # check whether the current assignment satisfies the agent energy limit constraints
        assigment_budget_diff_agent=np.sum(np.multiply(I0, gamma),axis=0)-author_budget
        agent_energy_constraint=assigment_budget_diff_agent<tol
        available_agents=np.where(agent_energy_constraint==True)[0].tolist()
        
        if len(available_agents)<Nagents_thres:
            I0,gamma,current_task_energy_requirement,current_agent_energy_budget,optimising_track_sub=Greedy_bestagent_sub_parallel(I0,gamma,current_task_energy_requirement,current_agent_energy_budget,non_filled_tasks,available_agents,energy_packet,n_proc,projection)
        else:
            I0,gamma,current_task_energy_requirement,current_agent_energy_budget,optimising_track_sub=RandomGreedy_bestagent_sub_parallel(I0,gamma,current_task_energy_requirement,current_agent_energy_budget,non_filled_tasks,available_agents,energy_packet,projection)
                                                                                                                                         
        
        #read results
        objective_track_sub=optimising_track_sub['ac']
        N_avg_coauthors_track_sub=optimising_track_sub['N_avg_coauthors']
        avg_Nauthors_per_paper_track_sub=optimising_track_sub['avg_Nauthors_per_paper']
        avg_Npapers_per_author_track_sub=optimising_track_sub['avg_Npapers_per_author']
        
        
        objective_track=objective_track+objective_track_sub
        N_avg_coauthors_track=N_avg_coauthors_track+N_avg_coauthors_track_sub
        avg_Nauthors_per_paper_track=avg_Nauthors_per_paper_track+avg_Nauthors_per_paper_track_sub
        avg_Npapers_per_author_track=avg_Npapers_per_author_track+avg_Npapers_per_author_track_sub

        
        #check task fullfillment 
        assigment_requirement_diff_task=np.sum(np.multiply(I0, gamma),axis=1)-omega
        task_energy_constraint=assigment_requirement_diff_task>-tol


    # multiplication by entry
    # check whether the current assignment satisfies the task energy requirement constraints    
    assigment_requirement_diff_task=np.sum(np.multiply(I0, gamma),axis=1)-omega
    task_energy_constraint=assigment_requirement_diff_task>-tol
    

    if np.prod(task_energy_constraint)==0:
        print('tasks not all filled, remaining tasks',list(np.where(task_energy_constraint==False)))
        
        
    else:
        print('all tasks filled')
    
    # check whether the current assignment satisfies the agent energy limit constraints

    
    assigment_budget_diff_agent=np.sum(np.multiply(I0, gamma),axis=0)-author_budget
    agent_energy_constraint=assigment_budget_diff_agent<tol
    
    
    #save results
    optimising_track=defaultdict()
    optimising_track['ac']=objective_track
    optimising_track['N_avg_coauthors']=N_avg_coauthors_track
    optimising_track['avg_Nauthors_per_paper']=avg_Nauthors_per_paper_track
    optimising_track['avg_Npapers_per_author']=avg_Npapers_per_author_track
        
    

    if np.prod(agent_energy_constraint)==1:
        print('agents energy limits not exceeded')
    else:
        print('agents energy limits exceeded, overworking agents',list(np.where(agent_energy_constraint==False)))
        return I0,optimising_track,gamma
    return I0,optimising_track,gamma




def RandomGreedy_bestagent_sub_parallel(I0,gamma0,current_task_energy_requirement,current_agent_energy_budget,non_filled_tasks,available_agents,energy_packet,projection='hypergraph'):

    

    objective_track_sub=[]
    N_avg_coauthors_track_sub=[]
    avg_Nauthors_per_paper_track_sub=[]
    avg_Npapers_per_author_track_sub=[]
    
        

    for task_i in non_filled_tasks:
        #print('task_i',task_i)

        if current_task_energy_requirement[task_i]<=0:
            pass

        else:
            # current_task_energy_requirement>0

############################################################################################################################################


            N_agents_taski=int(min(current_task_energy_requirement[task_i],len(available_agents)))
            # Number of agents we could assign in this round
            #print('available_agents',available_agents)
            #print('N_agents_taski',N_agents_taski)
            #print('sampled_agents',sampled_agents)
            sampled_agents=random.choices(available_agents,k=N_agents_taski)

            for agent_j in sampled_agents:  
                                    
                if current_agent_energy_budget[agent_j]>0 and current_task_energy_requirement[task_i]>0:

                    potential_energy_task_i_agent_j=min(current_agent_energy_budget[agent_j],current_task_energy_requirement[task_i],energy_packet)
                    current_agent_energy_budget[agent_j]-=potential_energy_task_i_agent_j
                    current_task_energy_requirement[task_i]-=potential_energy_task_i_agent_j
                    best_agent=agent_j
                    I0[task_i, best_agent] = 1
                    gamma0[task_i][best_agent]+=potential_energy_task_i_agent_j
                    
          
                    if projection=='hypergraph':
                        lambda2_0 =algebraic_connectivity(I0, gamma0, np.sum(np.multiply(I0, gamma0),axis=1), k=4)
                    if projection=='bipartite':
                        lambda2_0 =norm_bipartite_algebraic_connectivity(I0, gamma0, np.sum(np.multiply(I0, gamma0),axis=1), k=4)


                    objective_track_sub.append(lambda2_0)
                    N_avg_coauthors_track_sub.append(N_avg_coauthors(I0))
                    avg_Nauthors_per_paper_track_sub.append(avg_Nauthors_per_paper(I0))
                    avg_Npapers_per_author_track_sub.append(avg_Npapers_per_author(I0))

    


                    #print(sum(sum(I0)),task_i,best_agent,lambda2_0)

                elif current_task_energy_requirement[task_i]<=0:
                    #task filled, no need to check next agent, move on to the next task
                        break

                else:
                    #tasks not filled, current agent budget used up, check next agent
                    pass


############################################################################################################################################

    #save results
    optimising_track_sub=defaultdict()
    optimising_track_sub['ac']=objective_track_sub
    optimising_track_sub['N_avg_coauthors']=N_avg_coauthors_track_sub
    optimising_track_sub['avg_Nauthors_per_paper']=avg_Nauthors_per_paper_track_sub
    optimising_track_sub['avg_Npapers_per_author']=avg_Npapers_per_author_track_sub
    
    return I0,gamma0,current_task_energy_requirement,current_agent_energy_budget,optimising_track_sub



def Greedy_bestagent_sub_parallel(I0,gamma0,current_task_energy_requirement,current_agent_energy_budget,non_filled_tasks,available_agents,energy_packet,n_proc,projection='hypergraph'):
    
    if projection=='hypergraph':
        lambda2_0 =algebraic_connectivity(I0, gamma0, np.sum(np.multiply(I0, gamma0),axis=1), k=4)
    if projection=='bipartite':
        lambda2_0 =norm_bipartite_algebraic_connectivity(I0, gamma0, np.sum(np.multiply(I0, gamma0),axis=1), k=4)
        

    #objective_track_sub=[lambda2_0]
    #N_avg_coauthors_track_sub=[N_avg_coauthors(I0)]
    #avg_Nauthors_per_paper_track_sub=[avg_Nauthors_per_paper(I0)]
    #avg_Npapers_per_author_track_sub=[avg_Npapers_per_author(I0)]


    objective_track_sub=[]
    N_avg_coauthors_track_sub=[]
    avg_Nauthors_per_paper_track_sub=[]
    avg_Npapers_per_author_track_sub=[]
    
    

    
    #Since viable solution exists, assign tasks to agents untill all agent budgets are used up 
    

    for task_i in non_filled_tasks:
        #print('task_i',task_i)

        if current_task_energy_requirement[task_i]<=0:
            pass

        else:
            # current_task_energy_requirement>0


            inputs=[(current_agent_energy_budget,current_task_energy_requirement,task_i,agent_j,energy_packet,I0,gamma0,lambda2_0,projection) for agent_j in available_agents]

            if __name__ == "__main__":
                

                with Pool(n_proc) as p:
                    #results=dict(zip(list(range(n_rep)),p.map(repeat, list(range(n_rep)))))
                    obj_change=p.map(obj_change_by_availagents, inputs)




            #obj_change_per_unit_input=np.divide(objective_change,gamma_temp[task_i]) #objective change per unit input for each agent
            agents_ranked_by_indices=np.argsort(obj_change)[::-1] #agents ranked by objective change per unit input     

            for index in agents_ranked_by_indices:  
                agent_j=available_agents[index]
                    
                
                if current_agent_energy_budget[agent_j]>0 and current_task_energy_requirement[task_i]>0:

                    potential_energy_task_i_agent_j=min(current_agent_energy_budget[agent_j],current_task_energy_requirement[task_i],energy_packet)
                    current_agent_energy_budget[agent_j]-=potential_energy_task_i_agent_j
                    current_task_energy_requirement[task_i]-=potential_energy_task_i_agent_j
                    best_agent=agent_j
                    I0[task_i, best_agent] = 1
                    gamma0[task_i][best_agent]+=potential_energy_task_i_agent_j
                    
          
                    if projection=='hypergraph':
                        lambda2_0 =algebraic_connectivity(I0, gamma0, np.sum(np.multiply(I0, gamma0),axis=1), k=4)
                    if projection=='bipartite':
                        lambda2_0 =norm_bipartite_algebraic_connectivity(I0, gamma0, np.sum(np.multiply(I0, gamma0),axis=1), k=4)

            
                    objective_track_sub.append(lambda2_0)
                    N_avg_coauthors_track_sub.append(N_avg_coauthors(I0))
                    avg_Nauthors_per_paper_track_sub.append(avg_Nauthors_per_paper(I0))
                    avg_Npapers_per_author_track_sub.append(avg_Npapers_per_author(I0))

    


                    #print(sum(sum(I0)),task_i,best_agent,lambda2_0)

                elif current_task_energy_requirement[task_i]<=0:
                    #task filled, no need to check next agent, move on to the next task
                        break

                else:
                    #tasks not filled, current agent budget used up, check next agent
                    pass
                
    #save results
    optimising_track_sub=defaultdict()
    optimising_track_sub['ac']=objective_track_sub
    optimising_track_sub['N_avg_coauthors']=N_avg_coauthors_track_sub
    optimising_track_sub['avg_Nauthors_per_paper']=avg_Nauthors_per_paper_track_sub
    optimising_track_sub['avg_Npapers_per_author']=avg_Npapers_per_author_track_sub
    
    return I0,gamma0,current_task_energy_requirement,current_agent_energy_budget,optimising_track_sub




def obj_change_by_availagents(input_i):
    
    (current_agent_energy_budget,current_task_energy_requirement,task_i,agent_j,energy_packet,I0,gamma0,lambda2_0,projection)=input_i
    
    if current_agent_energy_budget[agent_j]<=0:
        return -99999.0

    else:

        #current_agent_energy_budget>0
        potential_energy_task_i_agent_j=min(current_agent_energy_budget[agent_j],current_task_energy_requirement[task_i],energy_packet)


        # agents dont need to spend energy more than required by a task
        # each agent assignment add one unit of energy towards completion of the task

        gamma_temp= np.copy(gamma0)
        I1 = np.copy(I0)#deep copy
        I1[task_i, agent_j] = 1;

        gamma_temp[task_i, agent_j]+=potential_energy_task_i_agent_j
        if projection=='hypergraph':
            lambda2_1 =algebraic_connectivity(I1, gamma_temp, np.sum(np.multiply(I1, gamma_temp),axis=1), k=4)
        if projection=='bipartite':
            lambda2_1 =norm_bipartite_algebraic_connectivity(I1, gamma_temp, np.sum(np.multiply(I1, gamma_temp),axis=1), k=4)
       

        return (lambda2_1-lambda2_0)/potential_energy_task_i_agent_j
####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################



'''
path='MAG/MAG_Hypergraph_Title_Only.gamma'
gamma = pd.read_csv(path,sep=' ',header = None)
gamma_int=np.ceil(gamma)
budget=np.array(gamma_int.sum(axis=1))
energy=np.array(gamma_int.sum(axis=0))
'''





data_path_all=['2years/LCC_PRE_1993_to_1994.txt','2years/LCC_PRE_1994_to_1995.txt','2years/LCC_PRE_1995_to_1996.txt','2years/LCC_PRE_1996_to_1997.txt','2years/LCC_PRE_1997_to_1998.txt','2years/LCC_PRE_1998_to_1999.txt','2years/LCC_PRE_1999_to_2000.txt','2years/LCC_PRE_2000_to_2001.txt','2years/LCC_PRE_2001_to_2002.txt','2years/LCC_PRE_2002_to_2003.txt','2years/LCC_PRE_2003_to_2004.txt','2years/LCC_PRE_2004_to_2005.txt','2years/LCC_PRE_2005_to_2006.txt','2years/LCC_PRE_2006_to_2007.txt','2years/LCC_PRE_2007_to_2008.txt','2years/LCC_PRE_2008_to_2009.txt','2years/LCC_PRE_2009_to_2010.txt','2years/LCC_PRE_2010_to_2011.txt','2years/LCC_PRE_2011_to_2012.txt','2years/LCC_PRE_2012_to_2013.txt','2years/LCC_PRE_2013_to_2014.txt','2years/LCC_PRE_2014_to_2015.txt','2years/LCC_PRE_2015_to_2016.txt','2years/LCC_PRE_2016_to_2017.txt','2years/LCC_PRE_2017_to_2018.txt','2years/LCC_PRE_2018_to_2019.txt','2years/LCC_PRE_2019_to_2020.txt','2years/LCC_PRE_2020_to_2021.txt']

data_path = data_path_all[0]

#Number of processors for parallel processing
n_proc=80

data = pd.read_csv(data_path,sep=' ',header = None)
data_int=np.ceil(data)
budget=np.array(data_int.sum(axis=1))
energy=np.array(data_int.sum(axis=0))

