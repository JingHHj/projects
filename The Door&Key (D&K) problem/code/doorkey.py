from utils import *
from example import example_use_of_gym_env
import part1
import part2
import gif2jpg

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Doors


def partA():
    env_path = "./envs/known_envs/example-8x8.env"
    
    name = "doorkey-5x5-normal"
    name = "known/" + name
    env_path = "./envs/known_envs/{}.env".format(name)
    env, info = load_env(env_path)  # load an environment
    seq = part1.doorkey_problem(env,info)  # find the optimal action sequence
    draw_gif_from_seq(seq,load_env(env_path)[0],name)  # draw a GIF & save
    gif2jpg.gif2jpg(name)
      
def partA_run_all():
    name_list = ["doorkey-5x5-normal",
            "doorkey-5x5-normal",
            "doorkey-6x6-normal",
            "doorkey-8x8-normal",
            "doorkey-6x6-direct",
            "doorkey-8x8-direct",
            "doorkey-6x6-shortcut",
            "doorkey-8x8-shortcut"]
    
    for name in name_list:
        env_path = "./envs/known_envs/{}.env".format(name)
        env, info = load_env(env_path)  # load an environment
        name = "known/" + name
        seq = part1.doorkey_problem(env,info)  # find the optimal action sequence
        draw_gif_from_seq(seq,load_env(env_path)[0],name)  # draw a GIF & save
        print("optimal control sequnce: ")
        print(seq)
        gif2jpg.gif2jpg(name)
        print("-----------------------------------")
    
def partB_run_all():
    
    control,steps = part2.random_map()

    name_list = []
    for i in range(36):
        name = "DoorKey-8x8-{}".format(i+1)
        name_list.append(name)
        env_path = "./envs/random_envs/{}.env".format(name)
        env, info = load_env(env_path)  # load an environment
        
        seq = part2.get_optimal_control(control,steps,info,env)
        print("control sequence of ",name)
        print(seq)
        name = "unknown/" + name
        draw_gif_from_seq(seq, load_env(env_path)[0],name)
        gif2jpg.gif2jpg(name)

# def testB():
    
    
    
#     name_list = ["DoorKey-8x8-32",
#                  "DoorKey-8x8-33",
#                  "DoorKey-8x8-34",
#                  "DoorKey-8x8-35",
#                  "DoorKey-8x8-36"]
#     for name in name_list:
#         env_path = "./envs/random_envs/{}.env".format(name)
#         env, info = load_env(env_path)  # load an environment
#         # print(info)
#         control,steps = part2.random_map(env)
#         seq = part2.get_optimal_control(control,steps,info,env)
#         print("control sequence of ",name)
#         print(seq)
#         name = "unknown/" + name
#         draw_gif_from_seq(seq, load_env(env_path)[0],name)
#         gif2jpg.gif2jpg(name)
           
  
  
def partB():
    control,steps = part2.random_map()
    
    name = "DoorKey-8x8-27"
    env_path = "./envs/random_envs/{}.env".format(name)
    env, info = load_env(env_path)  # load an environment
    # print(info)
    seq = part2.get_optimal_control(control,steps,info,env)
    print("control sequence of ",name)
    print(seq)
    name = "unknown/" + name
    draw_gif_from_seq(seq, load_env(env_path)[0],name)
      
# def partB():
#     env_folder = "./envs/random_envs"
#     env, info, env_path = load_random_env(env_folder)
#     control,steps = part2.random_map(env)
#     seq = part2.get_optimal_control(control,steps,info,env)
#     print(seq)
#     draw_gif_from_seq(seq, env)
    
if __name__ == "__main__":
    
    # example_use_of_gym_env()
    # partA_run_all()
    partB_run_all()
    # partA()
    # partB()
    # testB()



