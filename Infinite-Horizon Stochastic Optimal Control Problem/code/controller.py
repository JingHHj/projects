from time import time
import numpy as np
import utils
import casadi as cd
import matplotlib.pyplot as plt

# time_step = 0.5  # time between steps in seconds
# sim_time = 120  # simulation time (s)

class Controller:
    def __init__(self, iter_step, ref_traj):
        self.iter_step = iter_step
        self.states = np.tile(np.array([utils.x_init,utils.y_init,utils.theta_init]),
                              (self.iter_step,1)) # (iter_step, 3)
        self.control = np.tile(np.array([0.,0.]),
                               (self.iter_step - 1,1)) # (iter_step - 1, 2)
        self.ref_traj = ref_traj
        
        # info of the erros
        self.state_error = np.zeros((iter_step,3))
        self.p_error = np.zeros((iter_step,2))
        self.theta_error = np.zeros((iter_step,1))

        # info for optimization
        self.Ulb = np.array([utils.v_min,utils.w_min])
        self.Uub = np.array([utils.v_max,utils.w_max])
        self.q = 1
        self.Q = np.eye(2,dtype=float)
        self.R = np.eye(2,dtype=float)
    
    @staticmethod
    def get_noise():
        """
            Aim: get the noise matrix wt
        """
        sigma = np.array([0.04, 0.04, 0.004])

        w_xy = np.random.normal(0, sigma[0], 2)
        w_theta = np.random.normal(0, sigma[2], 1)
        w = np.concatenate((w_xy, w_theta))

        return w

    def renew(self, curr_step):
        """
        aim: renew the state with new control and motion model
        """
        # renew the state with given motion model
        next_state = utils.car_next_state(utils.time_step, self.states[curr_step], self.control[curr_step], noise=True)
        self.states[curr_step + 1,:] = next_state

        # renew the error
        # self.state_error = self.ref_traj - self.states

    def nlp(self,curr_step):
        """
        Aim:
            using Nlp solver from casadi, to do solve the optimization problem
        """
        T = self.iter_step - curr_step
        
        if T > 30:
            T = 30
        e = e1 = cd.SX(self.states[curr_step] - self.ref_traj[curr_step])
        # intialize the control
        u = cd.SX.sym('u',2,T-1)
        # turn Q and R into SX form
        Q = cd.SX(self.Q)
        R = cd.SX(self.R)
        # initialize value function f
        f = cd.SX.zeros(1,1)
        # intialize constrain g
        g = cd.SX.zeros(T-1,2)
        for i in range(T - 1):
            # define the g matrix
            gmi = cd.SX(np.array([
                    [utils.time_step * cd.cos(e[2] + self.ref_traj[curr_step + i,2]),0.],
                    [utils.time_step * cd.sin(e[2] + self.ref_traj[curr_step + i,2]),0.],
                    [0., utils.time_step]
            ]))
            # motion model
            e1 = e + cd.mtimes(gmi ,u[:,i]) + \
                           cd.SX(self.ref_traj[curr_step + i] - self.ref_traj[curr_step + i + 1] + self.get_noise())
            
            # define constrain g
            g[i,:] = e1[:2]
            # g = cd.vertcat(g,e1)
            # g = cd.vertcat(g,e1[1:])
            # g = cd.vertcat(g,e1[0])
            # g = cd.vertcat(g,e1[2])
            

            # define value function f with only stage cost
            stage_cost = cd.mtimes(e[:2].T,cd.mtimes(Q,e[:2])) + \
                         self.q * cd.sqrt( (1 - cd.cos(e[2])) ) + \
                         cd.mtimes(u[:,i].T,cd.mtimes(R,u[:,i]))
            f = f + stage_cost
            e = e1
         
        # define value function f with terminal cost
        terminal = cd.mtimes(e[:2].T,cd.mtimes(Q,e[:2])) + self.q * cd.sqrt( (1 - cd.cos(e[2])) )
        f = f + terminal
        # turn the g and f into the right shape
        g = cd.reshape(g,2*(T-1),1)
        u = cd.reshape(u,2*(T-1),1)
        # initialize solver
        solver = cd.nlpsol("solver", "ipopt", {'x':u,'f':f,'g':g})
        # turn the bound into the right shape
        ulb = np.tile(self.Ulb,T-1)
        uub = np.tile(self.Uub,T-1)
        # Solve the NLP
        result = solver(lbx = ulb,ubx = uub,lbg = 0,ubg = 0)     
        # renew the control
        self.control[curr_step,:] = result['x'][:2,:].T

    def nlp2(self,curr_step):
        """
        Aim:
            only using t and t+1 to do the optimization
            using Nlp solver from casadi, to do solve the optimization problem
        """

        
        e= self.states[curr_step] - self.ref_traj[curr_step]
        u = cd.SX.sym('u',2)

        gmatrix1 = cd.SX(np.array([
                    [utils.time_step * cd.cos(e[2] + self.ref_traj[curr_step,2]),0.],
                    [utils.time_step * cd.sin(e[2] + self.ref_traj[curr_step,2]),0.],
                    [0., utils.time_step] ]))
        
        e1 = cd.mtimes(gmatrix1 ,u) + \
            cd.SX(e.T + self.ref_traj[curr_step] - self.ref_traj[curr_step + 1] + get_noise())
        # define value function f
        
        termial =e1[0]**2 + e1[1]**2 + (1 - cd.cos(e1[2]))**2
        stage_state = e[0]**2 + e[1]**2 + (1 - cd.cos(e[2]))**2
        stage_control = u[0]**2 + u[1]**2
        f = termial + stage_state + stage_control
             
        # define constrain g
        g = cd.vertcat(e1[0],e1[2])
        nlp = {'x':u.T,'f':f,'g':g}

        # initialize solver
        solver = cd.nlpsol("solver", "ipopt", nlp)

        # Solve the NLP and define the bound
        result = solver(lbx = self.Ulb,ubx = self.Uub,lbg = 0,ubg = 0)     
        # renew the conrol

        self.control[curr_step] = result['x']

       