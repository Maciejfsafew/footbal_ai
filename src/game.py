#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Football gaming app
author: Maciej Mazur
"""

X=800
Y=400
GOALPOST = 100
FIELD = [150, 200]
RADIUS = 10
#LAYOUT = [X, Y, GOALPOST, FIELD, [20, 200], [780, 200], [330, 100]]
RESULT = [0, 0]

KEYMAP = {111 : "Up", 113 : "Left", 116 : "Down", 114 : "Right",
          25 : "w", 38 : "a", 39 : "s", 40 : "d"}
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
      if abs(Y/2 - y) < GOALPOST/2:
        if x <= 0:
          self.layout[7][0] = self.layout[7][0] + 1
        else:
          self.layout[7][1] = self.layout[7][1] + 1  
        x = X/2 
        y = Y/2
        vx = random.random()/100.0
        vy = random.random()/100.0
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

  def adjust_speed(self, p, buttons):
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
    self.layout = [X, Y, GOALPOST, FIELD, [20, 200, 0.0, 0.0], [X-20, Y/2, 0.0, 0.0], [X/2.0, Y/2.0, random.random()/100.0, random.random()/100.0], [0, 0], [[40, 5],[40, 5], [10, 1]]]

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
    self.after(10, self.draw)    
    
def start():
  root = Tk()
  root.geometry("820x510+200+200")
  control = KeyControl(root)
  game = Game(control)
  app = Application(root, game)
  root.mainloop()
  

if __name__ == '__main__':
  start()
