#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Football gaming app
author: Maciej Mazur
"""

X=800
Y=400
GOALPOST = 400
FIELD = [150, 200]
RADIUS = 10
#LAYOUT = [X, Y, GOALPOST, FIELD, [20, 200], [780, 200], [330, 100]]
RESULT = [0, 0]
DIVISION = 30.0
KEYMAP = {111 : "Up", 113 : "Left", 116 : "Down", 114 : "Right",
          25 : "w", 38 : "a", 39 : "s", 40 : "d"}
import json
import copy
import sys
import math
import random
import numpy as np
from Tkinter import Tk, Canvas, Frame, BOTH, W

class Game():
  def __init__(self, control):
    self.control = control  
    self.init_new_state()

  def collision(self, p1, p2, shape1, shape2, b1, b2):
    [x1, y1, vx1, vy1] = p1
    [x2, y2, vx2, vy2] = p2
    [r1, w1] = shape1
    [r2, w2] = shape2
    dif = 0.005 
    if (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) > (r1 + r2) * (r1 + r2):
      return [p1, p2]
    [mx, my] = [(x1 + x2)/2.0, (y1 + y2)/2.0]
    [vnx, vny] = [x1 - mx, y1 - my]
    l = math.sqrt(vnx * vnx + vny * vny)
    if l == 0.0:
      self.init_new_state()
      return None
    [vnx, vny] = [vnx/l, vny/l]
    [vsx, vsy] = [vny, -vnx] 
    w = vnx * vsy - vsx * vny
    wa = vx1 * vsy - vsx * vy1
    wb = vnx * vx1 - vy1 * vny
    wa1 = wa/w
    wb1 = wb/w

    wa = vx2 * vsy - vsx * vy2
    wb = vnx * vx2 - vy2 * vny
    
    wa2 = wa/w
    wb2 = wb/w

    wap1 = (wa1 * (w1 - w2) + 2 * w2 * wa2)/(w1 + w2)
    wap2 = (wa2 * (w2 - w1) + 2 * w1 * wa1)/(w1 + w2)

    return [[x1, y1 , (wap1 + dif) * vnx + wb1 * vsx, (wap1 + dif)* vny + wb1 * vsy],[x2, y2, (wap2 - dif) * vnx + wb2 * vsx, (wap2 - dif)* vny + wb2 * vsy]]    
    
  def next_round(self):
    [b1, b2] = self.control.buttons()
    p1 = self.layout[4]
    p2 = self.layout[5]
    p1 = self.adjust_speed(p1, b1)
    p2 = self.adjust_speed(p2, b2)
    self.layout[4] = p1
    self.layout[5] = p2

    X = self.layout[0]
    Y = self.layout[1]
    [x, y, vx, vy] = self.layout[6]

    if x <= 0 or x >= X:
      if x <= 0 and vx < 0:
        vx = abs(vx)
      if x >=X and vx > 0:
        vx = -abs(vx)
      if True:
        if x <= 0:
          self.layout[7][1] = self.layout[7][1] + 1
        else:
          self.layout[7][0] = self.layout[7][0] + 1  
        x = X/2 
        y = Y/2
        vx = (random.random()-0.5)/DIVISION 
        vy = (random.random()-0.5)/DIVISION
    if y <= 0 or y >= Y:
      if y <= 0 and vy < 0:
        vy = abs(vy)
      if y >= Y and vy > 0:
        vy = -abs(vy)
    self.layout[6] = [x + X * vx, y + Y * vy, vx, vy]
    
    ball = self.layout[6]
    balls = self.collision(p1, ball, self.layout[8][0], self.layout[8][2], 0, 2)
    if balls == None :
      return
    [p1, ball] = balls
    balls = self.collision(p2, ball, self.layout[8][1], self.layout[8][2], 1, 2)
    if balls == None:
      return
    [p2, ball] = balls
    balls = self.collision(p1,   p2, self.layout[8][0], self.layout[8][1], 0, 1)
    if balls == None:
      return
    [p1, p2]   = balls

    self.layout[4] = self.move_player(p1, True)
    self.layout[5] = self.move_player(p2, False)
    self.layout[6] = ball

  def adjust_speed(self, p, buttons, ball= False):
    [x, y, vx, vy] = p
    [up, left, down, right] = buttons
    vx = vx * 0.5
    vy = vy * 0.5
    if right:
      vx = vx + 0.005
    if left:
      vx = vx - 0.005
    if up:
      vy = vy - 0.005
    if down:
      vy = vy + 0.005
    return [x, y, vx, vy]

  def move_player(self, p, side1):
    [x, y, vx, vy] = p
    x = x + X * vx
    y = y + Y * vy
    if x < 0: 
      x = 0
      vx = 0
    if x > X/2.0 and side1:
      x = X/2.0
      vx = 0

    if x < X/2.0 and not side1: 
      x = X/2.0
      vx = 0
    if x > X and not side1:
      x = X
      vx = 0

    if y < 0:
      y = 0
      vy = 0
    if y > Y:
      y = Y 
      vy = 0 
    return [x, y, vx, vy]
    
  def init_new_state(self):
    self.layout = [X, Y, GOALPOST, FIELD, [20, 200, 0.0, 0.0], [X-20, Y/2, 0.0, 0.0], [X/2.0, Y/2.0, (random.random()-0.5)/DIVISION, (random.random()-0.5)/DIVISION ], [0, 0], [[40, 5],[40, 5], [10, 5]]]

  def get_layout(self):
    return self.layout

class KeyControl():
  def __init__(self, root):
    print root
    self.pressed = {}
    self.root = root
    self._set_bindings()
       
  def _set_bindings(self):
    for x in KEYMAP:
      self.root.bind("<KeyPress-%s>" % KEYMAP[x], self._pressed)
      self.root.bind("<KeyRelease-%s>" % KEYMAP[x], self._released)
      self.pressed[x] = False
   
  def _pressed(self, event):
    self.pressed[event.keycode] = True

  def _released(self, event):
    self.pressed[event.keycode] = False
  
  def buttons(self):
    # UP DOWN LEFT RIGH
    return [[self.pressed[25], self.pressed[38], self.pressed[39], self.pressed[40]],
            [self.pressed[111], self.pressed[113], self.pressed[116], self.pressed[114]]]  

class Application(Frame):
  def __init__(self, parent, game):
    Frame.__init__(self, parent, background="green")
    self.parent = parent
    self.game = game
    self.initUI()
    self.canvas = Canvas(self)
    self.draw()

  def initUI(self):
    self.parent.title("Football")
    self.pack(fill=BOTH, expand=1)
 
  def draw(self):
    self.game.next_round()
    LAYOUT = self.game.get_layout()
    self.canvas.delete("all")
    self.canvas.create_text(300, 20, anchor=W, font="Purisa", text="RESULT " + str(LAYOUT[7][0]) + ":" + str(LAYOUT[7][1])) 
    self.canvas.create_text(50, 20, anchor=W, font="Purisa", text="Control WSAD" )
    self.canvas.create_text(650, 20, anchor=W, font="Purisa", text="Control ARROWS" )
    x0 = 10
    y0 = 40
    # WHOLE FIELD
    self.canvas.create_rectangle(x0, y0, x0 + LAYOUT[0], y0 + LAYOUT[1], outline="white", fill="green", width=2)
    self.canvas.create_rectangle(x0, y0, x0 + LAYOUT[0]/2, y0 + LAYOUT[1], outline="white", fill="green", width=1)
    self.canvas.create_oval(x0 + LAYOUT[0]/2 -1 , y0 + LAYOUT[1]/2 - 1, x0 + LAYOUT[0]/2, y0 + LAYOUT[1]/2, outline="white", 
        fill="white", width=4)
    # GOALKEEPER FIELDS
    self.canvas.create_rectangle(x0, y0 + LAYOUT[1]/2 - LAYOUT[3][1]/2, x0 + LAYOUT[3][0], y0 + LAYOUT[1]/2 + LAYOUT[3][1]/2, outline="white", fill="green", width=2)
    self.canvas.create_rectangle(x0 + LAYOUT[0] - LAYOUT[3][0], y0 + LAYOUT[1]/2 - LAYOUT[3][1]/2, x0 + LAYOUT[0], y0 + LAYOUT[1]/2 + LAYOUT[3][1]/2, outline="white", fill="green", width=2)

    # GOALPOSTS
    self.canvas.create_rectangle(x0 - 5, y0 + LAYOUT[1]/2 - LAYOUT[2]/2, x0, y0 + LAYOUT[1]/2 + LAYOUT[2]/2, fill="black")

    self.canvas.create_rectangle(x0 + LAYOUT[0]-1, y0 + LAYOUT[1]/2 - LAYOUT[2]/2, x0 + LAYOUT[0] + 5, y0 + LAYOUT[1]/2 + LAYOUT[2]/2, fill="black")

    # PLAYERS
    self.canvas.create_oval(x0 + LAYOUT[4][0]-LAYOUT[8][0][0], y0 + LAYOUT[4][1]-LAYOUT[8][0][0], x0 + LAYOUT[4][0] + LAYOUT[8][0][0], y0 + LAYOUT[4][1] + LAYOUT[8][0][0], fill="blue")
    self.canvas.create_oval(x0 + LAYOUT[5][0]-LAYOUT[8][1][0], y0 + LAYOUT[5][1]-LAYOUT[8][1][0], x0 + LAYOUT[5][0] + LAYOUT[8][1][0], y0 + LAYOUT[5][1] + LAYOUT[8][1][0], fill="gray")
    # BALL
    self.canvas.create_oval(x0 + LAYOUT[6][0]-LAYOUT[8][2][0], y0 + LAYOUT[6][1]-LAYOUT[8][2][0], x0 + LAYOUT[6][0] + LAYOUT[8][2][0], y0 + LAYOUT[6][1] + LAYOUT[8][2][0], fill="white")

    self.canvas.pack(fill=BOTH, expand=1)
    self.after(1, self.draw)    

class MockControl:
  def __init__(self):
    self.p1_buttons = [False, False, False, False]
    self.p2_buttons = [False, False, False, False]
  def buttons(self):
    return [self.p1_buttons, self.p2_buttons]

class Player:
  def __init__(self, features, output, recurrent, name, debug=False):
    self.debug = debug
    self.features = features
    self.output = output
    self.recurrent = recurrent
    self.name = name
    self.input = [0.0 for i in range(0, 1 + features + recurrent)]
    self.input[0] = 1
    self.out = [0.0 for i in range(0, output + recurrent)]
    self.edges = [[i, j, (random.random()-0.5)/(1+features+recurrent)] for j in range(0, output + recurrent) for i in range(0, 1 + features + recurrent)]
  def reset(self):
    self.input = [0.0 for i in range(0, 1 + self.features + self.recurrent)]
    self.input[0] = 1

  def next_round(self, feat):
    self.input[1:1 + self.features] = feat
    for i in xrange(len(self.out)):
      self.out[i] = 0.0
    for [i, j, weight] in self.edges:
      self.out[j] = self.out[j] + weight * self.input[i]

    for i in xrange(len(self.out)):
      self.out[i] = math.tanh(self.out[i])
    
    self.input[1 + self.features:] = self.out[3:] 
    if self.debug:
      print 'out', self.out[0:4], 'rec out', self.out[4:]
      print 'bias', self.input[0:1], 'features', self.input[1:1 + self.features],'recurrent', self.input[1+self.features:]

    return [x > 0.0 for x in self.out[0:4]]
  def serialize(self):
    with open(self.name + ".json", 'w') as outfile:
      json.dump({'features': self.features, 'output' : self.output, 'name' : self.name, 'recurrent' : self.recurrent, "edges" : self.edges}, outfile) 

  def load(self, filename):
    with open(filename) as data_file:
      data = json.load(data_file) 
    self.features = data['features']
    self.output = data['output']
    self.name = data['name']
    self.recurrent = data['recurrent']
    self.edges = data['edges']
    self.input = [1.0 for i in range(0, 1 + self.features + self.recurrent)]
    self.out = [0.0 for i in range(0, self.output + self.recurrent)]

class Player2:
  def __init__(self, features, output, recurrent, name, debug=False):
    self.debug = debug
    self.features = features
    self.output = output
    self.recurrent = recurrent
    self.name = name
    self.input = np.array([0.0 for i in range(0, 1 + features)])
    self.input[0] = 1.0
    self.mid = np.array([0.0 for i in range(0, 1 + recurrent)])
    self.mid[0] = 1.0
    self.out = np.array([0.0 for i in range(0, output)])
    self.edges = [np.random.normal(scale=0.01, size=(1+features, recurrent)),np.random.normal(scale=0.001, size=(1+recurrent, output)), np.random.normal(scale=0.001, size=(1+recurrent, recurrent))]

  def cp(self):
    p = Player2(self.features, self.output, self.recurrent, self.name)
    p.edges = [e.copy() for e in self.edges]    
    return p

  def reset(self):
    self.mid = np.array([0.0 for i in range(0, 1 + self.recurrent)])
    self.mid[0] = 1.0

  def next_round(self, feat):
    self.input[1:1 + self.features] = feat
    self.out = self.out * 0.0
    
    #self.mid = self.mid * 0.0
    self.mid[0] = 1.0

    self.mid[1:] = np.dot(self.input, self.edges[0])
    
    self.mid[1:] = np.tanh(self.mid[1:])

    self.out = np.dot(self.mid, self.edges[1])
    self.out = np.tanh(self.out)
    
    self.mid[1:] = np.tanh(np.dot(self.mid, self.edges[2]))

    if self.debug:
      print 'out', self.out
      print 'mid', self.mid
    return list(self.out > 0.0)

  def serialize(self):
    with open(self.name + ".json", 'w') as outfile:
      json.dump({'features': self.features, 'output' : self.output, 'name' : self.name, 'recurrent' : self.recurrent, "edges0" : self.edges[0].tolist(), "edges1" : self.edges[1].tolist(), "edges2" : self.edges[2].tolist()}, outfile) 

  def load(self, filename):
    with open(filename) as data_file:
      data = json.load(data_file) 
    self.features = data['features']
    self.output = data['output']
    self.name = data['name']
    self.recurrent = data['recurrent']
    self.edges =[0, 0, 0]
    self.edges[0] = np.array(data['edges0'])
    self.edges[1] = np.array(data['edges1'])
    self.edges[2] = np.array(data['edges2'])
    self.input = np.array([0.0 for i in range(0, 1 + self.features)])
    self.input[0] = 1.0
    self.mid = np.array([0.0 for i in range(0, 1 + self.recurrent)])
    self.mid[0] = 1.0
    self.out = np.array([0.0 for i in range(0, self.output)])
    
    
def start():
  root = Tk()
  root.geometry("820x510+200+200")
  control = KeyControl(root)
  game = Game(control)
  app = Application(root, game)
  root.mainloop()

class OneControl():
  def __init__(self, control, cpu):
    self.control = control
    self.cpu = cpu

  def buttons(self):
    [p1, p2] = self.control.buttons()
    [[x1, y1, vx1, vy1], [x2, y2, vx2, vy2], [bx, by, bvx, bvy]] = self.game.get_layout()[4:7]
    p1_w = [x1/X, y1/Y, vx1+0.5, vy1+0.5, x2/X, y2/Y, vx2+0.5, vy2+0.5, bx/X, by/Y, bvx+0.5, bvy+0.5]
    action_p1 = self.cpu.next_round(p1_w)
    return [action_p1, p2]

class TwoControl():
  def __init__(self, cpu1, cpu2):
    self.cpu1 = cpu1
    self.cpu2 = cpu2

  def buttons(self):
    [[x1, y1, vx1, vy1], [x2, y2, vx2, vy2], [bx, by, bvx, bvy]] = self.game.get_layout()[4:7]
    p1_w = [x1/X, y1/Y, vx1+0.5, vy1+0.5, x2/X, y2/Y, vx2+0.5, vy2+0.5, bx/X, by/Y, bvx+0.5, bvy+0.5]
    p2_w = [1 - x2/X, y2/Y, -vx2+0.5, vy2+0.5, 1 - x1/X, y1/Y, -vx1+0.5, vy1+0.5,  1 - bx/X, by/Y,  -bvx+0.5, bvy+0.5]
    action_p1 = self.cpu1.next_round(p1_w)
    action_p2 = self.cpu2.next_round(p2_w)
    action_p2 = [action_p2[0],action_p2[3],action_p2[2],action_p2[1]]
    return [action_p1, action_p2]

def play_cpus(cpu, cpu2):
  root = Tk()
  root.geometry("820x510+200+200")
  cpu_control = TwoControl(cpu, cpu2)
  game = Game(cpu_control)
  cpu_control.game = game
  app = Application(root, game)
  root.mainloop()

def play_cpu(cpu):
  root = Tk()
  root.geometry("820x510+200+200")
  control = KeyControl(root)
  cpu_control = OneControl(control, cpu)
  game = Game(cpu_control)
  cpu_control.game = game
  app = Application(root, game)
  root.mainloop()

def test_step(step, p1, p2, control, game, t, q, r, it = 2000):
  p1.edges[t][q][r] = p1.edges[t][q][r] + step 
  for i in xrange(it):
    [[x1, y1, vx1, vy1], [x2, y2, vx2, vy2], [bx, by, bvx, bvy]] = game.get_layout()[4:7]
    
    p1_w = [x1/X, y1/Y, vx1+0.5, vy1+0.5, x2/X, y2/Y, vx2+0.5, vy2+0.5, bx/X, by/Y, bvx+0.5, bvy+0.5]
    p2_w = [1 - x2/X, y2/Y, -vx2+0.5, vy2+0.5, 1 - x1/X, y1/Y, -vx1+0.5, vy1+0.5,  1 - bx/X, by/Y,  -bvx+0.5, bvy+0.5]
    action_p1 = p1.next_round(p1_w)
    control.p1_buttons = action_p1
    action_p2 = p2.next_round(p2_w)

    action_p3 = [action_p2[0],action_p2[3],action_p2[2],action_p2[1]]
    control.p2_buttons = action_p2
    #print action_p1
    game.next_round()
    if game.get_layout()[7][0] + game.get_layout()[7][1] > 20:
      break
        
  layout = game.get_layout()
  if layout[7][0] >= layout[7][1]:
    res = (0.1/(i+1) + layout[7][0])/(layout[7][0] + layout[7][1] + 1.0)
  else:
    res = (i/1000.0)*layout[7][0]/(layout[7][0] + layout[7][1] + 1.0)
  #if layout[7][0] + layout[7][1] < 20:
   # res = 0.0
  #print step, i, layout[7], res
  p1.edges[t][q][r] = p1.edges[t][q][r] - step
  
    
  return res

class Player3:
  def next_round(self, feat):
    [x,y, vx, vy, x1,y1, vx1, vy1, xb, yb, vxb, vyb] = feat
    
    return [y>=yb,False,y<yb,False]
def train(p1, p2, it=2000, eps=17, e=10000):
  control = MockControl()
  game = Game(control)
  res = 0.0
  dx = 0.001
  r = [0, 0]
  for epoch in xrange(e):
    r = [0.0, 0.0]
    edges = [[a,b,c] for a in xrange(len(p1.edges)) for b in xrange(p1.edges[a].shape[0]) for c in xrange(p1.edges[a].shape[1])]
    move = 0
    for i,[t,q,w] in enumerate(edges):
      best_step = [0.0, 0.0, 0, 0] 
      steps = [0.0, 0.0000001, -0.0000001, 0.000001, -0.000001, 0.00001, -0.00001, 0.0001, -0.0001, 0.001, -0.001, 0.01, -0.01, 0.1, -0.1, 0.5, -0.5][:eps]
      #[0.0, 0.0000001, -0.0000001, 0.000001, -0.000001, 0.00001, -0.00001, 0.0001, -0.0001, 0.001, -0.001, 0.01, -0.01, 0.1, -0.1]
      
      layout_begin = copy.deepcopy(game.get_layout())
      for step in steps:
        game.layout = copy.deepcopy(layout_begin)
        game.get_layout()[7] = [0.0, 0.0]
        random.seed((epoch+1)*(t+1)*(q+1)*(w+1)) #(epoch+1)*(t+1)*(q+1)*(w+1)
        p1.reset()
        #game.init_new_state()
        res = test_step(step, p1, p2, control, game, t,q,w, it)
        layout = game.get_layout()
        if res > best_step[0]:
          best_step = [res, step, layout[7][0], layout[7][1]]
      random.seed((epoch+1)*(t+1)*(q+1)*(w+1))
      p1.reset()
      #game.init_new_state()
      #game.layout = copy.deepcopy(layout_begin)
      game.get_layout()[7] = [0.0, 0.0]
      res = test_step(best_step[1], p1, p2, control, game, t,q,w, it)
      layout = game.get_layout()
      r[0] = r[0] + layout[7][0]
      r[1] = r[1] + layout[7][1]
      p1.edges[t][q][w] = p1.edges[t][q][w] + best_step[1] * 0.01
      if best_step[1] != 0.0:
        move = move + 1
      #if i % 100 == 99:
      #  print i,len(edges), r, r[0]/ (r[0] + r[1]+ 0.00001), layout[7], res, layout[7][0]/(layout[7][0]+layout[7][1]+0.001), best_step[1]
      #print str(epoch) + " " +  str(t) + " RESULT " + str(layout[7][0]) + ":" + str(layout[7][1])
    #p1.serialize()
    print "RESULT", epoch, r, move, r[0]/ (r[0] + r[1]+ 0.00001) 
  return [control, game, r]   
 
def family_train(cpu, enemy, size):
  cpu.name = cpu.name + 'best_' + str(size)
  best = [cpu.cp(), -1]
  res = [[cpu,0.0]]
  epochs = 3
  for i in xrange(1000):
    w = []
    for x, val in res:
      w.append([x.cp(), 1200, 17, epochs])
      w.append([x.cp(), 1200, 9, epochs])
      w.append([x.cp(), 1200, 5, epochs])
    #for x,val in res:
    #  for it in [100, 500, 2000]:
    #    for eps in [5, 9, 17]:
    #      w.append([x.cp(), it, eps, epochs])
    res = []
    for nr, x in enumerate(w):
      [c, it, eps, e] = x
      
      [control, game, rr] = train(cpu, enemy, it, eps, e)
 
      res.append([c, rr[0]/ (rr[0] + rr[1]+ 0.00001)])
      print 'test', nr, 'it, eps, epochs', [it, eps, e], rr
    res.sort(key=lambda tup: -tup[1])
    res = res[:3]
    
    if best[1] == -1:
      game.get_layout()[7] = [0.0, 0.0]
      test_step(0.0, best[0], enemy, control, game, 0,0,0, 1000000)
      layout = game.get_layout()
      win = layout[7][0]/ (layout[7][0] + layout[7][1]+ 0.00001)
      best[1] = win
    print "OLD_BEST", i, best[1]
    if best[1] < res[0][1]:
      best[0] = res[0][0].cp()
      best[1] = res[0][1]
    
    print "BEST", i, best[1]
    best[0].serialize()
    print [r for x,r in res]    

if __name__ == '__main__':
  command = sys.argv[1]
  if command == 'play':
    start()
  if command == 'train':
    p1 = Player2(12, 4, 10, "p0_10r")
    p1.load("p0_10r.json")
    p2 = Player3()
    train(p1, p2)
    p1.serialize()
  if command == 'train_family':
    p1 = Player2(12, 4, 10, "p0_10r")
    p1.load("p1_best_10.json")
    p1.name = 'p1_'
    p2 = Player3()
    family_train(p1, p2, 10)
    p1.serialize()
  if command == 'play_cpu':
    p1 = Player2(0,0,0,"", True)
    p1.load("p0_" + sys.argv[2] + "r.json")
    play_cpu(p1)
  if command == 'play_cpus':
    p1 = Player2(0,0,0,"", True)
    p1.load("p1_best_" + sys.argv[2] + ".json")
    play_cpus(p1, Player3())
