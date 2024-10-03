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
    input current state (x,y,dir,pk,ud)
    return the pos that is infront of that
    """
    front_state = np.copy(state)
    dir = state[2]
    match dir:
        case 0: # right
            front_state[0] = front_state[0] + 1 
        case 1: # down
            front_state[1] = front_state[1] + 1 
        case 2: # left
            front_state[0] = front_state[0] - 1
        case 3: # up
            front_state[1] = front_state[1] - 1
    return front_state

def motion_model(state,motion,info,env):
    """
    input current state,and the motion
    return the result state and the cost of that motion
    """
    next_state = np.copy(state)
    door = info['door_pos'] # get the door position
    key = info['key_pos'] # get the key position
    goal = info['goal_pos']
    wall = type(env.unwrapped.grid.get(0,0))
    
    if np.all(state[:2] == goal): 
    # if next step is the goal position then stage cost is 0
        motion_cost = 0
    else:
        match motion:
            case 0: # MF
                next_state = get_front_pos(state)
                if  (next_state[0] >= (info['height'] - 1 )) or \
                    (next_state[1] >= (info['width'] - 1 )) or \
                    (next_state[0] <= 0) or \
                    (next_state[1] <= 0) or \
                    (type(env.unwrapped.grid.get(next_state[0],next_state[1])) == wall) or\
                    (np.all(next_state[:2] == door) and next_state[4] == 0):
                    # if is ouside of the map
                    # if it is wall 
                    # if the door is unlocked
                    motion_cost = 200000
                else:
                    motion_cost = 1
            case 1: # TL
                next_state[2] = next_state[2] - 1
                if next_state[2] == -1:
                    next_state[2] = 3
                motion_cost = 1
            case 2: # TR
                next_state[2] = next_state[2] + 1
                if next_state[2] == 4:
                    next_state[2] = 0
                motion_cost = 1
            case 3: # PK
                front = get_front_pos(state)
                if next_state[3] == 0 and np.all(front[:2] == key): 
                    # only picked up the key if the key is in the front
                    next_state[3] = 1
                    motion_cost = 1
                else:
                    motion_cost = 200000
            case 4: # UD
                front = get_front_pos(state)
                if next_state[4] == 0 and next_state[3] == 1 and np.all(front[:2] == door):
                    # only unlock the door if, 
                    # door is in the front, door is locked, we have the key
                    next_state[4] = 1
                    motion_cost = 1
                else:
                    motion_cost = 200000     
    return next_state,motion_cost


def find_state_cost(state,last_v,info,env):

    cost = 100000
    control = 1
    # stage cost
    state_cost = np.zeros(5)

    for i in range(5):
        next,stage_cost = motion_model(state,i,info,env)        
        state_cost[i] = stage_cost + last_v[next[0],next[1],next[2],next[3],next[4]]
        
        
    control = np.argmin(state_cost)
    cost = state_cost[control]

    return cost,control


def get_optimal_control(control,steps,info,env):
    """
    control: A control space that contains the optimal control for every state at every time step
            (horizon*height*width*dir*pk*ud)
    initial_pos: the initial pos of the agent (x,y,dir,pk,ud) (1*5)
    goal_pos: the goal pos of the agent (x,y) (1*2)
    steps: how many steps has the while loop ran through (number)
    
    return: the optimal control from initial pos to the goal pos
    """
    init_dir = get_agent_dir(info['init_agent_dir'])
    initial_pos = np.hstack((info['init_agent_pos'],init_dir,0,0)) # x,y,dir,0,0
    # put in the time variable 
    current_state = np.copy(initial_pos)
    control_seq = [] # initialize control sequnce as a list
    control_seq.append(control[0,initial_pos[0],initial_pos[1],initial_pos[2],initial_pos[3],initial_pos[4]]) 
    
    for i in range(steps):
        next, cost = motion_model(current_state,control_seq[i],info,env) # move to next state
        if np.all(next[:2] == info['goal_pos']):
            break
        control_seq.append(control[i+1,next[0],next[1],next[2],next[3],next[4]]) # get the control of next step
        current_state = np.copy(next)
        
    
    return control_seq


def initialization(env,info):
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
    
    horizion = info['height']*info['width']*4*2*2 - 1
    
    # initialize terminal cost
    terminal_cost = np.full((info['height'],info['width'],4,2,2),100000)
    terminal_cost[info['goal_pos'][0],info['goal_pos'][1],:,:,:] = 0  
    
    last_v = np.copy(terminal_cost) # v function from last step
    current_v = np.copy(terminal_cost) # v function compute this step

    # create a space that keep on track of all the optimal control for every state at every step
    control_space = np.full((horizion,info['height'],info['width'],4,2,2),2)
    
     
    return last_v,current_v,control_space,horizion




def doorkey_problem(env,info):
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env
        
    Feel Free to modify this fuction
    """
    # example:
    # optim_act_seq = [TR, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF] # correted example
    
    last_v,current_v,control_space,horizion = initialization(env,info)
     
    T = horizion - 1     
    while T> 0:
        T -= 1
        for a in range(info['height']): # x
            for b in range(info['width']): # y
                if type(env.unwrapped.grid.get(a,b)) == type(env.unwrapped.grid.get(0,0)):
                    current_v[a,b,:,:,:] = 100000
                    # if the current block is wall we set the current_v infinite
                    # and just skip it
                    continue
                for c in range(4): # direction
                    for d in range(2): # whether pick up the key or not
                        for e in range(2): # whether the door is unlock or not
                            state = np.array([a,b,c,d,e])
                            current_v[a,b,c,d,e],control_space[T,a,b,c,d,e] = find_state_cost(state,last_v,info,env)
                            
                            # testing
                            # if current_v[a,b,c,d,e] <= 1000:
                            #     print("{},{},{},{},{},{}".format(T, a, b, c, d, e))
                            #     print("v_func: ",current_v[a,b,c,d,e])
                            #     print("control_space: ",control_space[T,a,b,c,d,e])
    
        # when two conseccutive v function value equal we need to end the algorithm 
        if np.all(last_v == current_v):
            break                    
        last_v = np.copy(current_v) 
        # print(T)                        
    # print("---------------------------------")
    
    steps = horizion - T - 1
    optim_act_seq = get_optimal_control(control_space[T:,:,:,:,:,:],steps,info,env)                 
    return optim_act_seq    
 
