import numpy
import random
import math
import json
import functions
import optimizer

epsilon = 0.0000001

def population_docs_intial(population):    
   
    pop=[]
    print ("Popolation size=",len(population))
    x=random.sample(list(doc.keys()), len(population))
    for i in x:
        pop.append(data[str(i)])
    return pop

class GSA(optimizer.Optimizer):
    #Create an object that optimizes a given fitness function with GSA.
    def __init__(self, fitness_function, solution_size, lower_bounds, upper_bounds, population_size=2, max_iterations=1, G_initial=1.0, G_reduction_rate=0.5, **kwargs):

        optimizer.Optimizer.__init__(self, fitness_function, population_size,max_iterations, **kwargs)

        #set paramaters for users problem
        self.solution_size = solution_size
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # GSA variables
        self.G_initial = G_initial
        self.G_reduction_rate = G_reduction_rate
        self.velocities = [[0.0]*self.solution_size]*self.population_size

    def initialize(self):
        # Intialize GSA variables
        self.velocities = [[0.0]*self.solution_size]*self.population_size

    def create_initial_population(self, population_size):
        return create_initial_population(population_size, self.solution_size, self.lower_bounds, self.upper_bounds)

    def new_population(self, population, fitnesses):
        #print ("fitnesses=",fitnesses)
        new_pop, new_velocities = new_population(population, fitnesses, self.velocities,
                                                 self.G_initial, self.G_reduction_rate,
                                                 self.iteration, self.max_iterations)
        self.velocities = new_velocities

        h=[]
        return new_pop

#Create a random initial population of floating point values.    
def create_initial_population(population_size, solution_size,lower_bounds, upper_bounds):
    if len(lower_bounds) != solution_size or len(upper_bounds) != solution_size:
        raise ValueError("Lower and upper bounds much have a length equal to the problem size.")

    # Create random population
    population = []

    for i in range(population_size): #for every chromosome
        solution = []
        for j in range(solution_size): #for every bit in the chromosome
            solution.append(random.uniform(lower_bounds[j], upper_bounds[j])) #randomly add a 0 or a 1
        population.append(solution) #add the chromosome to the population


    pop=population_docs_intial(population)
    population=pop
    return population

def new_population(population, fitnesses, velocities, 
                    G_initial, G_reduction_rate, iteration, max_iterations):
    # Update the gravitational constant, and the best and worst of the population
    # Calulate the mass and acceleration for each solution
    # Update the velocity and position of each solution
    population_size = len(population)
    solution_size = len(population[0][0])
    G = G_gsa(G_initial, G_reduction_rate, iteration, max_iterations)
    masses = get_masses(fitnesses)
    # Create bundled solution with position and mass for the K best calculation
    solutions = [{'pos': pos, 'mass': mass} for pos, mass in zip(population, masses)]
    solutions.sort(key = lambda x: x['mass'], reverse=True)
    # Get the force on each solution
    # Only the best K solutions apply force
    # K linearly decreases to 1
    K = int(population_size-(population_size-1)*(iteration/float(max_iterations)))
    forces = []
    for i in range(population_size):
        force_vectors = []
        for j in range(K):
            # If it is not the same solution
            if population[i] != solutions[j]['pos']: #NOTE: this could use optimization
                force_vectors.append(gsa_force(G, masses[i], solutions[j]['mass'], 
                                                population[i], solutions[j]['pos']))
        forces.append(gsa_total_force(force_vectors, solution_size))

    # Get the accelearation of each solution
    accelerations = []
    for i in range(population_size):
        accelerations.append(gsa_acceleration(forces[i], masses[i]))

    # Update the velocity of each solution
    new_velocities = []
    for i in range(population_size):
        new_velocities.append(gsa_update_velocity(velocities[i], accelerations[i]))

    # Create the new population
    new_population = []
    for i in range(population_size):
        new_population.append(gsa_update_position(population[i], new_velocities[i]))

    test=solutions[:K]
    test=[d['pos'] for d in test]
    
    f=[];j=0   
    for i in new_population:
        features_old=f_data[str(population[j][1][1])]
        fe=features_old[1]
        features_old=(fe[0]+fe[1]+fe[2]+fe[3]+fe[4])*0.2

        if population[j] in test:
            index=numpy.argmax(i)
            #print ("index=",index)
            if index==0:            
                gm=data[str(population[j][1][0])]
                c=gm[1][1]
                features_new=f_data[str(c)]
                fe=features_new[1]
                features_new=(fe[0]+fe[1]+fe[2]+fe[3]+fe[4])*0.2
                prop= 0.9 * (features_new / mus) + 0.1
                dalta=features_new-features_old
                if dalta > 0:
                    f.append(gm)
                elif prop > random.random():
                    f.append(gm)
                
                else:
                    f.append(population[j])
                    
            else:
                gm=data[str(population[j][1][2])]
                c=gm[1][1]
                features_new=f_data[str(c)]
                fe=features_new[1]
                features_new=(fe[0]+fe[1]+fe[2]+fe[3]+fe[4])*0.2
                prop= 0.9 * (features_new / mus) + 0.1
                dalta=features_new-mus
                if dalta > 0:
                    f.append(gm)
                elif prop > random.random():
                    f.append(gm)
                
                else:
                    f.append(population[j])

        else:
            index=numpy.argmax(i)
            if index==0:            
                gm=data[str(population[j][1][0])]
                c=random.choice(gm[1])
                gm=data[str(c)]

                features_new=f_data[str(c)]
                fe=features_new[1]
                features_new=(fe[0]+fe[1]+fe[2]+fe[3]+fe[4])*0.2
                prop= 0.9 * (features_new / mus) + 0.1
                dalta=features_new-mus
                if dalta > 0:
                    f.append(gm)
                elif prop > random.random():
                    f.append(gm)
                
                else:
                    f.append(population[j])
                    
            else:
                gm=data[str(population[j][1][2])]
                c=random.choice(gm[1])
                gm=data[str(c)]

                features_new=f_data[str(c)]
                fe=features_new[1]
                features_new=(fe[0]+fe[1]+fe[2]+fe[3]+fe[4])*0.2
                prop= 0.9 * (features_new / mus) + 0.1
                dalta=features_new-mus
                if dalta > 0:
                    f.append(gm)
                elif prop > random.random():
                    f.append(gm)
                
                else:
                    f.append(population[j])

        j=j+1
    
    new_population=f

    lis=new_population

    flag=True
    while flag==True:
        duplicate=[idx for idx, val in enumerate(lis) if val in lis[:idx]]
        if duplicate !=[]:
            flag=True
            for t in duplicate:
                x=random.choice(lis[t][1])
                alter=data[str(x)]
                lis[t]=alter
        else:
            flag=False
            break
    return new_population, new_velocities

def G_physics(G_initial, t, G_reduction_rate):
    return G_initial*(1.0/t)**G_reduction_rate

def G_gsa(G_initial, G_reduction_rate, iteration, max_iterations):
    return G_initial*math.exp(-G_reduction_rate*iteration/float(max_iterations))

def get_masses(fitnesses):
    global mus
    best_fitness = max(fitnesses)
    worst_fitness = min(fitnesses)
    fitness_range = best_fitness-worst_fitness

    if mus < best_fitness:
        mus=best_fitness
        
    # Calculate raw masses for each solution
    m_vec = []
    for fitness in fitnesses:
        # Epsilon is added to prevent divide by zero errors
        m_vec.append((fitness-worst_fitness)/(fitness_range+epsilon)+epsilon)

    # Normalize to obtain final mass for each solution
    total_m = sum(m_vec)
    M_vec = []
    for m in m_vec:
        M_vec.append(m/total_m)
    return M_vec
  

def gsa_force(G, M_i, M_j, x_i, x_j):
    position_diff = numpy.subtract(x_j[0], x_i[0])
    position_diff=numpy.absolute(position_diff)
    distance = numpy.linalg.norm(position_diff) 
    # The first 3 terms give the magnitude of the force
    # The last term is a vector that provides the direction
    # Epsilon prevents divide by zero errors
    return G*(M_i*M_j)/(distance+epsilon)*position_diff

def gsa_total_force(force_vectors, vector_length):
    if len(force_vectors) == 0:
        return [0.0]*vector_length

    # To specify that the total force in each dimension 
    # is a random sum of the individual forces in that dimension.

    total_force = [0.0]*vector_length
    for force_vec in force_vectors:
        for d in range(vector_length):
            total_force[d] += random.uniform(0.0, 1.0)*force_vec[d]
    return total_force

def gsa_acceleration(total_force, M_i):

    return numpy.divide(total_force, M_i)

def gsa_update_velocity(v_i, a_i):
    # To specify that velocity is randomly weighted for each dimension.
    v = []
    #After finding, the optimal solution, the velocity becomes zero.
    for d in range(len(v_i)):
        v.append(random.uniform(0.0, 1.0)*v_i[d]+a_i[d])

    return v


def gsa_update_position(x_i, v_i):
    try:
        x_i=x_i[0]
        
    except:
        pass
    new_pos=list(numpy.add(x_i, v_i)) 

    return new_pos

if __name__ == '__main__':
    global f_data    
    global data
    mus=0.0000001
    global mus
    
    LANGUAGE="arabic"
    data_x={}
    for ii in range(1,11):
        f_data={}
        data={}
        with open(".../features"+str(ii)+".json", "r") as read_file:
            f_data = json.load(read_file)
        read_file.close()

        with open(".../nearest"+str(ii)+".json", "r") as read_file:
            data = json.load(read_file)
        read_file.close()

        datax={}
        for k,v in data.items():
            datax[k]=([v[0][0][1],v[0][0][1]],[int(k)]+[item[0] for item in v[0]])

        datay={}
        for k,v in datax.items():
            datay[k]=(v[0],[v[1][1],v[1][0],v[1][2]]+v[1][3:],f_data[str(k)][1])

        data=datay
        
        r_text=open(".../document"+str(ii)+".txt",'r', encoding='utf-8').read()
        parser= PlaintextParser(r_text, Tokenizer(LANGUAGE)).document.sentences

        li=[]
        for i in parser:
             li.append(str(i))

        doc=li
        doc = dict([(s_id, text) for s_id, text in enumerate(doc)])
        print ("Data size=",len(doc))
        DocSize=len(doc)
        persentage=30
        population_size=int(round((DocSize/100)*persentage))
        max_iterations=10
        print ("max_iterations=",max_iterations)
        c=float(len(doc)-1)
        
        my_gsa = GSA(functions.ackley, 2, [0.0]*2, [c]*2, population_size, max_iterations, decode_func=functions.ackley_real)
        best_solution = my_gsa.optimizer()

        best_solutionx=best_solution[0]
        popx=best_solution[1]


        

        popx.append(best_solutionx) if best_solutionx not in popx else popx
        popx=sorted(popx,key=lambda x: x[0][0], reverse=True)

        if len(popx)> population_size:
            popx=popx[:population_size]
            

        p=[]
        for i in popx:
            p.append(i[1][1])
        
        p.sort()#, reverse=True)

        with open(".../raw/raw"+str(ii)+".json", "r") as read_file:
            m = json.load(read_file)
        read_file.close()
        print ("\nSummary:\n")
        my_list=[]
        for i in p:
            sentence=m[str(i)]
            #print (i,":",sentence)
            my_list.append(sentence)

        print ("original-length=",sum([len(str(sentence).split()) for sentence in my_list]))
            
        my_list1=my_list
        k=[]
        e1=()
        e2=()

        for t in range(1,len(my_list)+1):
                s=sum([len(str(sentence).split()) for sentence in my_list1])
                #print (s)
                if s>250:
                    e1=(s,my_list1)
                    my_list1=my_list[:-t]
                    k=my_list1
                    
                else:
                    k=my_list1
                    e2=(s,k)
                    break
        if e1!=():            
            if ((e1[0]-250)<=(250-e2[0])) and (250-e2[0]>10):
                my_list=e1[1]
            else:
                my_list=e2[1]

        s="TASK"+str(ii)
        sum_s=sum([len(str(sentence).split()) for sentence in my_list])
        print (s,":",sum_s)        
        data_x[s]=sum_s
        
        s="task"+str(ii)+"_arabicSystem1.txt"        
        with open(".../system/"+s, 'w', encoding='utf-8') as f:

            for item in my_list:
                f.write("%s\n" % item)
        f.close()

    with open(".../data_file.json", "w") as write_file:
        json.dump(data_x, write_file)
    write_file.close()

