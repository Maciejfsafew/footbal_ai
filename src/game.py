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
DIVISION = 10.0
KEYMAP = {111 : "Up", 113 : "Left", 116 : "Down", 114 : "Right",
          25 : "w", 38 : "a", 39 : "s", 40 : "d"}
import json
import sys
import math
import random
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

    [p1, ball] = self.collision(p1, ball, self.layout[8][0], self.layout[8][2], 0, 2) 
    [p2, ball] = self.collision(p2, ball, self.layout[8][1], self.layout[8][2], 1, 2)
    [p1, p2]   = self.collision(p1,   p2, self.layout[8][0], self.layout[8][1], 0, 1)

    self.layout[4] = self.move_player(p1)
    self.layout[5] = self.move_player(p2)
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

  def move_player(self, p):
    [x, y, vx, vy] = p
    x = x + X * vx
    y = y + Y * vy
    if x < 0: 
      x = 0
      vx = 0
    if x > X:
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
    self.layout = [X, Y, GOALPOST, FIELD, [20, 200, 0.0, 0.0], [X-20, Y/2, 0.0, 0.0], [X/2.0, Y/2.0, (random.random()-0.5)/DIVISION, (random.random()-0.5)/DIVISION ], [0, 0], [[40, 5],[40, 5], [10, 1]]]

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
    self.input = [1.0 for i in range(0, 1 + features + recurrent)]
    self.out = [0.0 for i in range(0, output + recurrent)]
    self.edges = [[i, j, (random.random()-0.5)/(1+features+recurrent)] for j in range(0, output + recurrent) for i in range(0, 1 + features + recurrent)]
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
    p1_w = [x1/X, y1/Y, vx1, vy1, x2/X, y2/Y, vx2, vy2, bx/X, by/Y, bvx, bvy]
    action_p1 = self.cpu.next_round(p1_w)
    return [action_p1, p2]

def play_cpu(cpu):
  root = Tk()
  root.geometry("820x510+200+200")
  control = KeyControl(root)
  cpu_control = OneControl(control, cpu)
  game = Game(cpu_control)
  cpu_control.game = game
  app = Application(root, game)
  root.mainloop()

def test_step(step, p1, p2, control, game, t):
  p1.edges[t][2] = p1.edges[t][2] + step 
  for i in xrange(10000):
    [[x1, y1, vx1, vy1], [x2, y2, vx2, vy2], [bx, by, bvx, bvy]] = game.get_layout()[4:7]

    p1_w = [x1/X, y1/Y, vx1, vy1, x2/X, y2/Y, vx2, vy2, bx/X, by/Y, bvx, bvy]
    p2_w = [1 - x2/X, 1 - y2/Y, -vx2, -vy2, 1 - x1/X, 1 - y1/Y, -vx1, -vy1,  1 - bx/X, 1 - by/Y,  -bvx, -bvy]
    action_p1 = p1.next_round(p1_w)
    control.p1_buttons = action_p1
    action_p2 = p2.next_round(p2_w)

    action_p3 = [action_p2[0],action_p2[3],action_p2[2],action_p2[1]]
    control.p2_buttons = action_p2
    #print action_p1
    game.next_round()
          
  layout = game.get_layout()
 
  res = layout[7][0]/(layout[7][0] + layout[7][1] + 0.00001 )
  if layout[7][0] + layout[7][1]  < 20:
    res = 0.0
  p1.edges[t][2] = p1.edges[t][2] - step
  return res

def train(p1, p2):
  control = MockControl()
  game = Game(control)
  res = 0.0
  dx = 0.001
  r = [0, 0]
  for epoch in xrange(100):
    r = [0.0, 0.0]
    for t in xrange(len(p1.edges)):
      best_step = [0.0, 0.0, 0, 0] 
      for step in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.0, -0.1, -0.01, -0.001, -0.0001, -0.00001, -0.000001, -0.0000001]:
        res = test_step(step, p1, p2, control, game, t)
        layout = game.get_layout()
        if res > best_step[0]:
          best_step = [res, step, layout[7][0], layout[7][1]]
        game.init_new_state()

      test_step(step, p1, p2, control, game, t)
      layout = game.get_layout()
      r[0] = r[0] + layout[7][0]
      r[1] = r[1] + layout[7][1]
      p1.edges[t][2] = p1.edges[t][2] + best_step[1]
      print t,len(p1.edges), r, r[0]/ (r[0] + r[1]+ 0.00001), layout[7], layout[7][0]/(layout[7][0]+layout[7][1]+0.001) 
      #print str(epoch) + " " +  str(t) + " RESULT " + str(layout[7][0]) + ":" + str(layout[7][1])
    p1.serialize()
    print "RESULT", epoch, r    
 
def family_train(family):
  for it in xrange(20):
    for i in xrange(len(family)):
      for j in xrange(len(family)):
        if i != j:
          train(family[i], family[j]) 
if __name__ == '__main__':
  command = sys.argv[1]
  if command == 'play':
    start()
  if command == 'train':
    p1 = Player(12, 4, 5, "p3")
    #p1.load("p1.json")
    p2 = Player(12, 4, 5, "p2")
    train(p1, p2)
    p1.serialize()
  if command == 'play_cpu':
    p1 = Player(0,0,0,"", True)
    p1.load("p3.json")
    play_cpu(p1)
