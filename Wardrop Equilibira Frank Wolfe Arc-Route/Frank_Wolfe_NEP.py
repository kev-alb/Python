
"""
Created on Mon Jul 22 15:15:06 2019

@author: Kevin Albrechts
"""

#This code is part of my master's thesis.

#The Frank-Wolfe algorithm described here solves (NEP). Input is the network,
#described by the adjacency list, the demand vector, and the arc cost
#functions. Output is the (approximate) Wardrop equilibrium by an arc flow.

#-----------------------------------------------------------------------------

#Import necessary packages and modules.
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import csv

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

#Define auxiliary functions.
#-----------------------------------------------------------------------------

#Returns the value of the arc cost functions defined by the values in array t
#from input for a given fow value x. Be careful in the datasets that capacity
#(cap_a) (after popping in read_in, t[i][3], before that it was in t[i][5]) is
#never equal to 0. Otherwise, division by zero would occur!
def arc_cost_func(x, i):
    return t[i][0] + t[i][1]* t[i][2] * (x/t[i][3])**t[i][4]

#Returns the integral of an arc cost function within an interval.
def integrate(a, b, i):
    value =0
    value2=0
    #For accuracy of integral: Number of rectangles (N >> 1000 is taking
    #rather long). On my computer (Intel Core i5-7200U), N=1000 was a good
    #choice to see the changes of f^i and error values in the Anaconda console.
    N=1000
    for n in range(1, N+1):
        value += arc_cost_func(a+((n-(1/2))*((b-a)/N)), i)
    value2= ((b-a)/N)*value
    return value2

#Returns function value T(f).
def T(f):
    #lower bound of integral
    a=0
    T_value = 0
    for i in range(0, no_of_arcs):
        #The control variable i has to be passed to function integrate.
        T_value += integrate(a, f[i], i)
    return T_value

#Returns the gradient of T at f. It simply shows the arc costs induced by
#an arc flow f.
def Gradient(f):
    gradient = np.zeros((no_of_arcs,1))
    for i in range(0, no_of_arcs):
        gradient[i]= arc_cost_func(f[i], i)
    return gradient

#Returns function value T_sub(f), where T_sub is the linearized function of T.
def T_sub(y):
    T_linear_sub = T(f) + np.dot(Gradient(f).T, (y-f))
    T_linear_sub = np.asscalar(T_linear_sub)
    return T_linear_sub

#(Step 0 and 1, each:) To find a feasible arc flow. This function gets as
#input an arc flow. Based on this arc flow, the induced arc costs are
#calculated first, then the correpsonding route costs. Thereafter, the
#shortest routes can be identified. Based on these, the route flow h is
#calculated which then is transformed into the corresponding arc flow using
#the arc-route incidence matrix. Thus, output is this arc flow.
def arc_flow_shortest_routes(f):
    #Calculate the arc costs of each arc induced by the arc flow, i.e.,
    #t_induced is t(f).
    t_induced = Gradient(f)
    #Calculate the route costs of each route induced by the arc flow.
    c_induced = np.dot(Delta.T, t_induced)

    #Route all demand on the shortest routes.   
    #Define the route flow h from the shortest routes and demand rates.  
    #First initialize h and then calculate the entries (see next for-loop).
    h = np.zeros(no_of_routes)
    route_index = 0
    #Loop to identify the shortest routes of each commodity and to assign the
    #values to the entries of h, i.e., calculate h.
    for k in range(0, no_of_commodities):
        #Initialize the lowest cost ("shortest") with the cost of the first
        #route of commodity k.
        min_cost = c_induced[route_index]
        #The following list contains the indices of the shortest routes for
        #commodity k.
        shortest_routes_k = ([route_index])
        #Set the number of routes that correspond to commodity k.
        routes_in_R_k = no_of_routes_per_commodity[k]
        #r takes successively on the indices of those routes that belong to
        #commodity k, except the first one because it is the initial shortest
        #route.
        for r in range(route_index + 1, route_index + routes_in_R_k):
            #If a route is shorter, assign new min_cost and remember the index.
            if c_induced[r] < min_cost:
                min_cost = c_induced[r]
                shortest_routes_k = ([r])
            elif c_induced[r] == min_cost:
                #If more than one shortest route, collect the indices in the
                #array.
                shortest_routes_k.append(r)
        #Assign flow values to the entries of h.
        for r in range(route_index, route_index + routes_in_R_k):
            if r in shortest_routes_k:
                h[r] = d[k]/len(shortest_routes_k)
        route_index += routes_in_R_k

    #Calculate the feasible arc flow and return it.
    return np.dot(Delta, h)

#(Step 3:) Define objective function to minimize.
def objective(l):
    return T(f+l*p)

#(Step 3:) Define constraints for optimization: l \in [0,1]. constraint1 gives
#l<=1, constraint2 gives l >=0.
def constraint1(l):
    return 1 - l
def constraint2(l):
    return l

#(Step 3:) Load each constraint into a dictionary, respectively. Put them
#into a list and return this list.
def constraints(c1,c2):
    cons1 = ({'type': 'ineq', 'fun': c1})
    cons2 = ({'type': 'ineq', 'fun': c2})
    c12 = [cons1, cons2]
    return c12

#Reads in the data and returns it in the appropriate data structure.
def read_in(filename, which):
    #To create the correct name of the respective file stored in the same
    #folder. Take care of correct naming!
    if which == 1:
        ending = "_Network_Adjacency_List.csv"
    elif which == 2:
        ending = "_Demand.csv"
    elif which == 3:
        ending = "_ArcCostFunctions.csv"
    with open(filename + ending, 'r') as csvfile: 
        #Creating a csv reader object.
        csvreader = csv.reader(csvfile) 
        data=[]
        #Extracting each data row one by one.
        for row in csvreader: 
            data.append(row) 
    #Cast to int for adjacency list.
    if which == 1:
        for i in range(0, len(data)):
            for j in range(0, len(data[i])):
                data[i][j] = int(data[i][j])
    #Cast to float for demand and arc cost functions.
    elif which == 2 or which == 3:
        for i in range(0, len(data)):
            for j in range(0, len(data[i])):
                data[i][j] = float(data[i][j])
    for i in range(0, len(data)):
        #For adjacency list, pop only first element, i.e. the node to which
        #the others are adjacent.
        if which == 1:
            data[i].pop(0)
        #For demand vector and arc cost functions (vector), pop the first two
        #elements, i.e. the commodity/OD-pair specifying nodes and
        #respectively, the origin and destination nodes of an arc.
        elif which == 2 or which == 3:
            #Note the order of popping in a list.
            data[i].pop(1)
            data[i].pop(0)
    return data

#Starting from the adjacency list, the function calculates the arc-route
#incidence matrix Delta and further properties: number of arcs, routes, nodes,
#commodities; a list of all routes, of all arcs and of the number of routes
#per commodity.
def adj_list_to_Delta_etc (adj):
    
    #Recursive auxiliary function to print all routes.
    def get_all_routes_util(g, u, d, visited, route, s): 
        #Mark the current node u as visited and store it in route.
        visited[u] = True
        route.append(u)
        #If the current node is the same as the destination, then append
        #current route, which first needs to be shallow copied.
        if u == d:
            aux_route = route.copy()
            all_routes1[s].append(aux_route)
        #If the current node is not the destination, call the recursive
        #for all the nodes adjacent to this node.
        else:            
            for i in g[u]: 
                if visited[i] == False: 
                    get_all_routes_util(g, i, d, visited, route, s)                
        #Remove the current node from route and mark it as unvisited.
        route.pop() 
        visited[u]= False
           
    #Gives all routes from a node s to a node d.
    def get_all_routes(g, N, s, d):   
        #Mark all the nodes as not visited.
        visited = [False]*(N)
        #Create an array to store the routes.
        route = [] 
        #Call the recursive auxiliary function to get all routes from s to d.
        get_all_routes_util(g, s, d, visited, route, s) 

    #Start of adj_list_to_Delta_etc.
    
    #Calculate the number of nodes.    
    nodes = len(adj)
    
    #Create a list to store all cycle-free routes. This array is a list of
    #lists where these lists contain the routes originating at a node (in
    #increasing order).
    all_routes1=[[] for _ in range(len(adj))]
    for n in range(0, len(adj)):
        all_routes1.extend([])
        for m in range(0, len(adj)):
            if n == m:
                pass
            else:
                get_all_routes(adj, nodes, n, m)
    
    #Put all routes from the list of lists all_routes1 into a list that
    #contains all cycle-free routes. The only difference between the
    #all_routes1 list and this one is that this list has one dimension less.
    aux_all_routes = [item for sublist in all_routes1 for item in sublist]

    #Calculate the number of routes.
    routes = 0
    for route in range(0, len(aux_all_routes)):
        routes += 1

    #Calculate the number of arcs    
    arcs = 0
    for start_node in range(0, len(adj)):
        for end_node in adj[start_node]:
            arcs += 1
    
    #Create a list of all arcs, this is all_arcs1.
    aux_all_arcs = [None, None]*arcs
    i=0
    for start_node in range(0, len(adj)):
        for end_node in range(0, len(adj[start_node])):
            aux_all_arcs[i]= start_node
            aux_all_arcs[i+1] = adj[start_node][end_node]
            i +=2        
    all_arcs1 = [aux_all_arcs[i:i+2] for i in range(0, len(aux_all_arcs), 2)]
    
    #Create the arc-route incidence matrix Delta.
    #Create such an array of size arcs x routes filled with zeros first.
    arc_all_routes= np.zeros((arcs, routes), dtype=int)
    #Fill arc_all_routes with 1s if arc in route. The case that an arc is not
    #in a route does not have to be dealt with explicitly since in that case
    #the Delta matrix already has a zero as the respective entry.
    for each_arc in range(0, arcs):
        for each_route in range(0,routes):
            a=aux_all_routes[each_route]
            b=all_arcs1[each_arc]
            if b in [a[i:len(b)+i] for i in range(len(a))]:
                arc_all_routes[each_arc][each_route] = 1

    
    #Calculate number of commodities. This could also be done with simply the
    #length of the demand vector. However, doing it here enables one to use
    #this function for other programs as well.
    commodities = 0
    for start_node in range(0, len(all_routes1)):
        if len(all_routes1[start_node]) == 0:
            pass
        elif len(all_routes1[start_node]) == 1:
            commodities +=1
        #if length >1:
        else:
            commodities +=1
            #How many more have different end nodes and thus, are further
            #commodities:
            for route in range(0, len(all_routes1[start_node])-1):
                if all_routes1[start_node][route][-1] != \
                all_routes1[start_node][route+1][-1]:
                    commodities +=1
    
    #Calculate the number of routes per commodity.
    #Create an array of appropriate size first, initialized with 0s. These are
    #then incremented for every route correpsonding to k.
    no_of_routes_per_commodity1 = [0]*commodities
    #Start with the first commodity.
    k=0
    for start_node in range(0, len(all_routes1)):
        #print(len(all_routes1[start_node]))
        if len(all_routes1[start_node]) == 0:
            pass
        elif len(all_routes1[start_node]) == 1:
            no_of_routes_per_commodity1[k] =1
            k +=1
        elif len(all_routes1[start_node]) == 2:
            if all_routes1[start_node][0][-1] == \
            all_routes1[start_node][1][-1]:
                no_of_routes_per_commodity1[k] =1
            else:
                no_of_routes_per_commodity1[k] =2
            k +=1
        #if length >2:
        elif len(all_routes1[start_node]) > 2:
            no_of_routes_per_commodity1[k] =1
            for route in range(0, len(all_routes1[start_node])-1):
                if all_routes1[start_node][route][-1] == \
                all_routes1[start_node][route+1][-1]:
                    no_of_routes_per_commodity1[k] +=1
                else:
                    k +=1
                    no_of_routes_per_commodity1[k] +=1
            k +=1
    
    #return tuple of calculated results.
    return arcs, routes, aux_all_routes, all_arcs1, arc_all_routes, nodes, \
    commodities, no_of_routes_per_commodity1


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

#Input
#-----------------------------------------------------------------------------
print("The following networks are currently available (Sioux_Falls takes "
   "very long): ", "Pigou_Linear", "Braess_Paradox_Augmented", \
    "Five_City", "New_Five_City", "Sioux_Falls", sep='\n')
#The name of the network to be analysed is entered by the user.
network = input("Enter the name of one of the networks: ")

print("---------------------------------------------------------------")

#Network: Read in.

#Read in adjacency list.
adj_list = read_in(network, 1)
#Print out the adjacency list.
print("Adjacency list: ", adj_list)

#Transform adjacency list into arc-route incidence matrix Delta and calculate
#further characteristics.
no_of_arcs, no_of_routes, all_routes, all_arcs, Delta, no_of_nodes, \
no_of_commodities, no_of_routes_per_commodity = adj_list_to_Delta_etc(adj_list)

#Print out the number of nodes.
print("No. of nodes: ", no_of_nodes)
#Print out the number of arcs.        
print("No. of arcs: ", no_of_arcs)
#Print out the list of all arcs.
print("All arcs: ", all_arcs, sep = '\n')
#Print out the number of cycle-free routes.
print("No. of cycle-free routes: ", no_of_routes)
#Print out the list of all cycle-free routes. This list contains the same
#elements as the set R of all cycle-free routes described in my thesis where
#R is the union of disjoint sets R_k which contain those cycle-free routes
#that correspond to commodity k.
print("All cycle-free routes, i.e. \"set\" R: ", all_routes, sep = '\n')
#Print out the arc-route incidence matrix Delta.
print("Delta (arc-route incidence matrix): ", Delta, sep = '\n') 
#Print out the number of commodities, i.e. OD-pairs.
print("No. of commodities: ", no_of_commodities)
#Print out a list that contains the numbers of routes correpsonding to the
#commodities/OD-pairs.
print("No. of routes per commodity: ", no_of_routes_per_commodity, \
      sep = '\n')

#Demand: Read in.
d = read_in(network, 2)
d = np.reshape(d, -1)
print("Demand vector d: ", d, sep = '\n')

#Arc cost functions: Read in.
#Arc cost functions: Array t=(t_a(.))_{a\in \mathcal{A}}.
t = read_in(network, 3)
#After popping in readin the entries are:
#free-flow cost (alpha_a): t[i][0], beta_a: t[i][1], B: t[i][2], capacity
#(cap_a): t[i][3] and exponent (pow_a): t[i][4].
print ("Arc cost functions: free-flow cost (alpha_a), beta_a, B, capacity "
       "(cap_a), exponent (pow_a): ", t, sep = '\n')

print("---------------------------------------------------------------")

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

#Start of the algorithm: Step 0 to Step 5 are carried out.

#For displaying results (or I = [0], same).
I = []
LBD_values = []
UBD_values = []
#and additionally, the total costs incduced.
tot_costs = []

#-----------------------------------------------------------------------------

#Step 0: Initialization
#print("Step 0: Initialization")
    
#Set LBD = 0, iteration counter i = 0 and epsilon (for termination criteria)
#to an arbitrary value > 0.
LBD = 0
i = 0
#Information on the maximum number of iterations is displayed.
print("The maximum number of iterations is set to 50 to ensure that no "
      "endless loop occurs.")
#The value for epsilon is entered by the user.
epsilon = float(input("Enter the value of epsilon > 0 for termination "
                      "criteria, e.g. 0.0001: "))

print("---------------------------------------------------------------")

#Find first feasible flow based on free-flow cost. This is simply the gradient
#of T at an arc flow f whose components are all 0. Thus, it gives the "fixed"
#costs of each arc.
#Define such an arc flow of zeros, called "free-flow".
free_flow = np.zeros(no_of_arcs)
#Calculate f^0
f = arc_flow_shortest_routes(free_flow)
print("f^0 = ", f)


#-----------------------------------------------------------------------------

#Loop of Step 1 to Step 5, where epsilon defines the termination criteria in
#Step 2 and 5. This loop will only be stopped by one of the termination
#criteria in Step 2 and Step 5. However, to avoid an endless loop, an upper
#bound on the number of iterations is set, e.g. to 50 iterations.
#Nevertheless, this bound can be omitted or be very large. Change it if
#necessary.
while i < 50:
    
    #For display of results: Store iteration number i and total costs C(f^i)
    #in arrays.
    I = np.append(I, i)
    #Calculate the total costs induced by flow f^i and store in array.
    total_costs = np.asscalar(np.dot(f, Gradient(f)))
    tot_costs = np.append(tot_costs, total_costs)
    
    print("---------------------------------------------------------------")
    
    print("Iteration count i = ", i)
    print("f^i = ", f)


    #-------------------------------------------------------------------------

    #Step 1: Search direction generation.
    #print("Step 1: Search direction generation")

    #Find a solution to the linear subproblem. Let it be y^i.
    y = arc_flow_shortest_routes(f)
    #Calculate search direction p^i.
    p = y-f

    #-------------------------------------------------------------------------

    #Step 2: First convergence check.
    #print("Step 2: First convergence check")

    #Calculate LBD (lower bound).
    LBD = max(LBD, T_sub(y))
    LBD_values = np.append(LBD_values, LBD)
    #Calculate error in Step 2. Exclude the case of division by zero.
    if LBD != 0:
        error_step2 = ((T(f))-LBD)/LBD
        print ("Error in Step 2 = ", error_step2)
        #Termination criterion.
        if error_step2 < epsilon:
            #For display of results. "-1" to avoid displaying of that value.
            #See axes limits at "Displaying results".
            UBD_values = np.append(UBD_values, -1)
            print("Termination at Step 2.")
            #If true, terminate the algorithm and set the Wardrop equilibrium.
            break
        
    #Set Wardrop equilbrium f^{eq} to f^i (done after this while loop).

    #-------------------------------------------------------------------------

    #Step 3: Line Search.
    #print("Step 3: Line Search")

    #Set initial guess of step length l and load it into a numpy array.
    l = 0
    l0 = np.array([l])
    #Create the constraints.
    cons = constraints(constraint1, constraint2)
    #Call the solver to minimize objective function (T(f^i+l*p^i)) given the
    #constraints.
    sol = minimize(objective, l0, method = 'SLSQP', constraints = cons, \
                   options={'disp': False})
    #Retrieve optimal step length l_i.
    lmin = sol.x[0]

    #-------------------------------------------------------------------------

    #Step 4: Update.
    #print("Step 4: Update")

    #Calculate f^{i+1}.
    f = f + lmin*p

    #-------------------------------------------------------------------------

    #Step 5: Second convergence check.
    #print("Step 5: Second convergence check")

    #Calculate UBD (upper bound).
    UBD = T(f)
    UBD_values = np.append(UBD_values, UBD)
    #Calculate error in Step 5. Exclude the case of division by zero.
    if LBD != 0:
        error_step5 = (UBD-LBD)/LBD
        print ("Error in Step 5 = ", error_step5)
        #Termination criterion.
        if error_step5 < epsilon:
            #For display of results. "-1" to avoid displaying of that value.
            #See axes limits at "Displaying results".
            LBD_values = np.append(LBD_values, -1)
            print("Termination at Step 5.")
            #If true, terminate the algorithm and set the Wardrop equilibrium.
            break

    #Set Wardrop equilbrium f^{eq} to f^{i+1} (done after this while loop).
    
    #Increment iteration counter i (and go back to Step 1)
    i += 1
    
    #-------------------------------------------------------------------------

    #End of Step 1 to Step 5. Go back to Step 1 if expression of while is true.
    

#From Step 2 and 5: When while loop terminated, Wardrop equilibrium is set to
#either f^i (Step 2) or to the updated arc flow f^{i+1} (Step 5).
f_eq = f


#End of the algorithm.
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

#Additionally, calculate the total costs induced by the Wardrop equilibrium
#flow, C(f^{eq}).
Total_Costs = np.asscalar(np.dot(f_eq, Gradient(f_eq)))


#Displaying results.

print("---------------------------------------------------------------")
print("Results:", "", sep = '\n')

#Output (the main result): The Wardrop equilibrium f^{eq}.
print("Wardrop equilibrium f^eq = ", f_eq[:, None], "", sep = '\n')

#The total costs induced by the Wardrop equilibrium flow, C(f^{eq}).
print("Total costs induced by the Wardrop equilibrium, C(f^{eq}) = ", \
      Total_Costs, "\n")

#The number of iterations the algorithm needed/did.
#print("Iterations: ", i+1, "\n")

#A diagram showing the convergence of the upper bound UBD and the lower bound
#LBD over the iteration number i.
print("Convergence of the upper bound UBD and the lower bound LBD over "
      "iteration number i:")
fig1 = plt.figure()
plt.plot(I,UBD_values,'r*', LBD_values, 'bx')
plt.xlim((0, None))
plt.ylim((0, None))
plt.xlabel('Iteration number i')
plt.ylabel('UBD (red stars), LBD (blue xs)')
plt.show()

print("Total costs induced by the flow f^i, , C(f^i), "
      "over iteration number i:")
fig2 = plt.figure()
plt.plot(I,tot_costs,'r*')
plt.xlim((0, None))
plt.ylim((None, None))
plt.xlabel('Iteration number i')
plt.ylabel('C(f^i)')
plt.show()

#To save the plots.
#fig1.savefig("plot1.png", format='png', dpi=1000, bbox_inches='tight')
#fig2.savefig("plot2.png", format='png', dpi=1000, bbox_inches='tight')
