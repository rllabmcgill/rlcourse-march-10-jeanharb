import numpy as np
import seaborn, copy, sys, time
import matplotlib.pyplot as plt


class Env():
  def __init__(self):
    self.mdp = [[.5, .5, 0],
                [0, .5, .5],
                [.5, 0, .5]]
    self.s = np.array([1,0,0])
    self.gamma = 0.9

  def step(self):
    self.s = np.random.multinomial(1, self.mdp[self.s.argmax()])
    r = 0
    return r, self.s

class Agent(object):
  def __init__(self):
    self.V0 = np.array([100, -70, -30])

    self.a = copy.deepcopy(self.V0)
    self.b = np.array([23.094, -98.15, 75.056])

    self.l = 0.866
    self.eps = 0.05
    self.theta = 0.

  def v_s(self, s):
    return (np.dot(self.a, s)*np.cos(self.l*self.theta) - np.dot(self.b, s)*np.sin(self.l*self.theta)) * np.exp(self.eps*self.theta)

  def grad_s(self, s):
    left = (np.dot(self.a, s)*np.cos(self.l*self.theta) - np.dot(self.b, s)*np.sin(self.l*self.theta))*np.exp(self.eps*self.theta)*self.eps
    right = (np.dot(self.a, s)*(-self.l*np.sin(self.l*self.theta)) - np.dot(self.b, s)*(self.l*np.cos(self.l*self.theta)))*np.exp(self.eps*self.theta)
    grad = left + right
    return grad

  def grad_s2(self, s):
    left_left = (np.dot(self.a, s)*np.cos(self.l*self.theta) - np.dot(self.b, s)*np.sin(self.l*self.theta))*np.exp(self.eps*self.theta)*self.eps*self.eps
    left_right = (np.dot(self.a, s)*self.l*(-np.sin(self.l*self.theta)) - np.dot(self.b, s)*self.l*np.cos(self.l*self.theta))*np.exp(self.eps*self.theta)*self.eps
    right_left = (np.dot(self.a, s)*(-self.l*np.sin(self.l*self.theta)) - np.dot(self.b, s)*(self.l*np.cos(self.l*self.theta)))*np.exp(self.eps*self.theta)*self.eps
    right_right = (np.dot(self.a, s)*(-self.l*self.l*np.cos(self.l*self.theta)) + np.dot(self.b, s)*(self.l*self.l*np.sin(self.l*self.theta)))*np.exp(self.eps*self.theta)
    grad = left_left + left_right + right_left + right_right
    return grad    

  def update(self):
    pass

  def update_w(self, **kwargs):
    pass

def handle_close(evt):
  sys.exit()

class TD_agent(Agent):
  def __init__(self):
    super(self.__class__, self).__init__()
    self.lr = 0.002

  def update(self, td=0, s=0, s_prime=0, gamma=0):
    grad = self.grad_s(s)
    self.theta -= self.lr*grad

class TDC_agent(Agent):
  def __init__(self):
    super(self.__class__, self).__init__()
    self.alpha = 0.05
    self.beta = 0.5
    self.w = 0.

  def update(self, td=0, s=0, s_prime=0, gamma=0):
    _grad_s = self.grad_s(s)
    _grad_s_prime = self.grad_s(s_prime)
    h = (td - _grad_s*self.w)*self.grad_s2(s)*self.w
    grad = (td*_grad_s - (gamma*_grad_s_prime*_grad_s*self.w) - h)
    #print self.theta, _grad_s, _grad_s_prime, grad, s, s_prime, self.grad_s2(s)
    self.theta += self.alpha*grad
    #print "new_theta", self.theta

  def update_w(self, td=0, s=0):
    _grad_s = self.grad_s(s)
    self.w += self.beta*(td-(_grad_s * self.w))*_grad_s


def train(env, agents, draw_step=1000, refresh_rate=0.001):
  plt.ion()
  fig = plt.figure()
  plt.show(block=False)
  errs = []
  names = [a.__class__.__name__ for a in agents]
  for agent in agents:
    errs.append([])
    v = agent.V0
    s = env.s
    err = 0
    last_err = 1000
    breakit = False
    for step in range(200000):
      print "\b"*50, step, err, agent.theta, v
      sys.stdout.flush()

      r, s_prime = env.step()

      y_target = r + env.gamma*agent.v_s(s_prime)
      #print "->", y_target
      current_v = agent.v_s(s)
      #print "-->", current_v
      v[np.argmax(s)] = current_v
      td = y_target-current_v
      if (step+1)%1 == 0:
        agent.update(td=td, s=s, s_prime=s_prime, gamma=env.gamma)
      agent.update_w(td=td, s=s)

      s=s_prime

      err = (v**2).sum()
      counter = counter + 1 if np.abs(err-last_err) < 0.00000000001 else 0
      if counter > 100:
        print "counter passed"
        breakit = True
      last_err = err
      if err < 0.00000001: err = 0.0000001
      errs[-1].append(err)
      if err > 10000000000 or err < 0.000001:
        print v, err
        print "low error passed"
        breakit = True
      if step%draw_step == 0 or breakit:
        plt.clf()
        plt.yscale('log')
        fig.canvas.mpl_connect('close_event', handle_close)
        for e in errs: plt.plot(e)
        plt.legend(names[:len(errs)])
        plt.draw()
        plt.pause(refresh_rate)
      if breakit: break
  plt.pause(100)

if __name__ == "__main__":
  env = Env()
  agents = [TDC_agent(), TD_agent()]
  train(env, agents)

