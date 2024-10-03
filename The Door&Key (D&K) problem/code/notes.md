1. define state space
2. define motion model
    (MF,TR,TL,PK,UD)
    how they move btween states
3. define stage cost function
4. define terminal cost
5. find out time horizon
5. initialize 


MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

'height': num, 
'width': num, 
'init_agent_pos': 1*2tuple, 
'init_agent_dir': 1*2array, 
'key_pos': 1*2array, 
'door_pos': 1*2array, 
'goal_pos': 1*2array
     
RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3


## Part B

1. state space: (kp,gp,id,x,y,dir,pk,ud1,ud2) 
                ( 0, 1, 2,3,4,  5, 6,  7,  8)
    # kp: key position {(1, 1), (2, 3), (1, 6)} 
    # gp: goal position {(5, 1), (6, 3), (5, 6)}
    # id: initial door  0  ,  1  ,  2  ,  3
    #                 (1,1),(1,0),(0,1),(0,0)

2. two doors at (4, 2) and (4, 5)
   initial agent position: (3, 5) 



