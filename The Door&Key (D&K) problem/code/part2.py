from utils import *

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


def get_agent_dir(given_dir):
    """
    input:
        the agent direction in the given form 1*2 vector
    output:
        the agent direction in our form (number) 
    """
    dir = np.copy(given_dir)
    if np.all(dir == np.array([1,0])): # right
        using_dir = 0
    elif np.all(dir == np.array([0,1])): # down
        using_dir = 1
    elif np.all(dir == np.array([-1,0])): # left
        using_dir = 2
    elif np.all(dir == np.array([0,-1])): # up
        using_dir = 3
    return using_dir

def get_front_pos(state):
    """
    input current state (kp,gp,id,x,y,dir,pk,ud1,ud2)
    return the pos that is infront of that
    """
    front_state = np.copy(state)
    dir = state[4]
    match dir:
        case 0: # right
            front_state[2] = front_state[2] + 1 
        case 1: # down
            front_state[3] = front_state[3] + 1 
        case 2: # left
            front_state[2] = front_state[2] - 1
        case 3: # up
            front_state[3] = front_state[3] - 1
    return front_state

            
def motion_model(state,motion):
    """
    input current state,and the motion
    return the result state and the cost of that motion
    """
    front = get_front_pos(state) 
     # find the pos that is in the front
   
    # initialize those bool

    
    match state[0]: # different key position
        case 0:
            key = np.array([1, 1]) + 1
        case 1:
            key = np.array([2, 3]) + 1 
        case 2:
            key = np.array([1, 6]) + 1
    
    match state[1]: # different goal position
        case 0:
            goal = np.array([5, 1]) + 1
        case 1:
            goal = np.array([6, 3]) + 1
        case 2:
            goal = np.array([5, 6]) + 1

    # door: (4, 2) and (4, 5)      
    motion_cost = 100000
    next_state = np.copy(state)
    if np.all(state[2:4] == goal): 
    # if next step is the goal position then stage cost is 0
        motion_cost = 0
    else:
        match motion:
            case 0: # MF
                if  (front[2] >= 9) or (front[3] >= 9) or \
                    (front[2] <= 0) or (front[3] <= 0) or \
                    (front[2] == 5 and \
                    ( (front[3] != 3 and front[3] !=6) or 
                      (front[6] == 0 or front[7] == 0) )  ):
                    # if front cell is out of the map
                    # if front cell is wall
                    # if front cell is a locked door
                    motion_cost = 100000
                    next_state = front
                else:
                    next_state = front
                    motion_cost = 1
            case 1: # TL
                next_state[4] = next_state[4] - 1
                if next_state[4] == -1:
                    next_state[4] = 3
                motion_cost = 1
            case 2: # TR
                next_state[4] = next_state[4] + 1
                if next_state[4] == 4:
                    next_state[4] = 0
                motion_cost = 1
            case 3: # PK
                if next_state[5] == 0 and np.all(front[2:4] == key): 
                    # only picked up the key if the key is in the front
                    next_state[5] = 1
                    motion_cost = 1
                else:
                    motion_cost = 100000
            case 4: # UD
                if state[5] == 1: # have key
                    if np.any(front == [5,3]) and state[6] == 0: # door is d1 and is locked
                        next_state[6] = 1 
                        motion_cost = 1
                    elif np.any(front == [5,6]) and state[7] == 0: # door is d2 and is locked
                        next_state[7] = 1
                        motion_cost = 1    
                else:
                    motion_cost = 100000    
    return next_state,motion_cost

def find_state_cost(state,last_v):

    cost = 100000
    control = 1
    # stage cost
    cost_list = np.zeros(5)
    for i in range(5):
        next,stage_cost = motion_model(state,i)   
        cost_list[i] = stage_cost + last_v[next[0],next[1],next[2],next[3],next[4],next[5],next[6],next[7]]
     
    control = np.argmin(cost_list)
    cost = cost_list[control]

    return cost,control

def initialization():
    """
    _summary_

    Args:
        env (_type_): enviroment
        info (_type_): information of the environment

    Returns:
        last_v: 
        current_v:
        control_space:
        T:
    """
    # 8*8 -> 10*10
    # horizion = 3*3*4*10*10*4*2*2*2 - 1
    horizion = 50
    
    # initialize terminal cost
    terminal_cost = np.full((3,3,10,10,4,2,2,2),100000)
    # goal position {(5, 1), (6, 3), (5, 6)}
    terminal_cost[:,0,6,2,:,:,:] = terminal_cost[:,1,7,4,:,:,:,:] = terminal_cost[:,2,6,7,:,:,:,:] = 0 
    
    last_v = np.copy(terminal_cost) # v function from last step
    current_v = np.copy(terminal_cost) # v function compute this step

    # create a space that keep on track of all the optimal control for every state at every step
    control_space = np.full((horizion,3,3,10,10,4,2,2,2),2)
    
    
    return last_v,current_v,control_space,horizion

def random_map():

    # example:
    # optim_act_seq = [TR, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF] # correted example
    
    last_v,current_v,control_space,horizion = initialization()
    T = horizion - 1     
    while T> 0:
        T -= 1
        
        for a in range(3): # kp
            for b in range(3): #dp
                for c in range(10): # x
                    for d in range(10): # y
                        if (c == 0 or d == 0 or c == 9 or d == 9) or (c == 5 and (d != 3 and d != 6)):
                                current_v[a,b,c,d,:,:,:,:] = 100000
                                # if the current block is wall we set the current_v infinite
                                # and just skip it
                                continue
                        for e in range(4): # direction 
                            for f in range(2): # whether pick up the key or not
                                for g in range(2): # door A is unlock or not
                                    for h in range(2): # door B is unlock or not
                                        
                                            
                                        state = np.array([a,b,c,d,e,f,g,h])
                                        current_v[a,b,c,d,e,f,g,h],control_space[T,a,b,c,d,e,f,g,h] = find_state_cost(state,last_v)
                                        
                                        # testing
                                        # if current_v[a,b,c,d,e,f,g,h] <= 1000 and (a == 2 and b == 0 ):
                                        #     print(T,state)
                                        #     print("v_func: ",current_v[a,b,c,d,e,f,g,h])
                                        #     print("control_space: ",control_space[T,a,b,c,d,e,f,g,h])
                                
        # when two conseccutive v function value equal we need to end the algorithm 
        if np.all(last_v == current_v):
            break                    
        last_v = np.copy(current_v) 
        
        # print(T)                        
    print("---------------------------------")
    
    steps = horizion - T - 1

    return control_space[T:,:,:,:,:,:,:,:,:],steps    


def find_initial_pos(info,env):
    goal = info['goal_pos']
    key = info['key_pos']
    init_dir = get_agent_dir(info['init_agent_dir'])
    door1 = env.unwrapped.grid.get(4,2)
    door2 = env.unwrapped.grid.get(4,5)
    
    # find door case
    if door1.is_open and door2.is_open:
        d1 = 1
        d2 = 1
    elif door1.is_open and door2.is_locked:
        d1 = 1
        d2 = 0
    elif door1.is_locked and door2.is_open:
        d1 = 0
        d2 = 1
    elif door1.is_locked and door2.is_locked:
        d1 = 0
        d2 = 0
          
    # different key position
    if np.all(key == [1, 1]): 
        kp = 0
    elif np.all(key == [2, 3]): 
        kp = 1
    elif np.all(key == [1, 6]): 
        kp = 2  
    
    # different goal position
    if np.all(goal == [5, 1]):
        gp = 0
    elif np.all(goal == [6, 3]):
        gp = 1
    elif np.all(goal == [5, 6]):
        gp = 2
        
    init_x,init_y = info['init_agent_pos']
    initial_pos = np.hstack((kp,gp,init_x+1,init_y+1,init_dir,0,d1,d2))
        
    return initial_pos

def get_optimal_control(control,steps,info,env):
    """
    control: A control space that contains the optimal control for every state at every time step
            (horizon*height*width*dir*pk*ud)
    initial_pos: the initial pos of the agent (x,y,dir,pk,ud) (1*5)
    goal_pos: the goal pos of the agent (x,y) (1*2)
    steps: how many steps has the while loop ran through (number)
    
    return: the optimal control from initial pos to the goal pos
    """
    
    initial_pos = find_initial_pos(info,env)
    goal = info['goal_pos'] + 1
    # put in the time variable 
    current_state = np.copy(initial_pos)
    control_seq = [] # initialize control sequnce as a list
    control_seq.append(control[0,initial_pos[0],initial_pos[1],initial_pos[2],initial_pos[3],initial_pos[4],initial_pos[5],initial_pos[6],initial_pos[7]]) 
    
    
    for i in range(steps):
        next, cost = motion_model(current_state,control_seq[i]) # move to next state
        # print(next)
        if np.all(next[2:4] == goal):
            break
        control_seq.append(control[i+1,next[0],next[1],next[2],next[3],next[4],next[5],next[6],next[7]]) # get the control of next step
        current_state = np.copy(next)
        
        
    
    return control_seq

