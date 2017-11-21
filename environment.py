import gym
import numpy as np
from scipy.misc import imresize, toimage

class GymEnvironment():

  # For use with Open AI Gym Environment
  def __init__(self, env_id, screen_width, screen_height):
    self.env_id = env_id
    self.gym = gym.make(env_id)
    self.obs = None
    self.terminal = None

    self.screen_width = screen_width
    self.screen_height = screen_height

    # Define actions for games (gym-0.9.4 and ALE 0.5.1)
    if env_id == "Pong-v0":
      self.action_space = [1, 2, 3] # [NONE, UP, DOWN]
    elif env_id == "Breakout-v0":
      self.action_space = [1, 2, 3] # [FIRE, RIGHT, LEFT]
    else:
      self.action_space = [i for i in range(0, self.gym.action_space.n)] # 9 discrete actions are available

  def numActions(self):
    assert isinstance(self.gym.action_space, gym.spaces.Discrete)
    return len(self.action_space)

  def restart(self):
    self.obs = self.gym.reset()
    self.terminal = False

  def act(self, action):
    self.obs, reward, self.terminal, _ = self.gym.step(self.action_space[action])
    return reward, self.terminal

  def isTerminal(self):
    assert self.terminal is not None
    return self.terminal

  def render(self):
    self.gym.render()

  def getScreen(self):
    assert self.obs is not None

    new_bg_color = 0
    black_white = False

    if self.env_id == "MsPacman-v0":
        height_range = (0, 172)
        bg = (210, 164, 74)  # character

    elif self.env_id == "Pong-v0":
        height_range = (35, 193)
        bg = (144, 72, 17)  # bg
        black_white = True

    elif self.env_id == "Breakout-v0":
        height_range = (20, 198)
        bg = (142, 142, 142)  # walls

    elif self.env_id == "SpaceInvaders-v0":
        height_range = (20, 198)
        bg = (50, 132, 50)  # character
        new_bg_color = 255  # turn to white

    else:
        height_range = (0, 210)
        bg = (0, 0, 0)

    return pipeline(self.obs, (self.screen_height, self.screen_width),
                    height_range, bg, new_bg_color, black_white)


def pipeline(image, new_HW, height_range, bg, new_bg_color=0, black_white=False):
  """Returns a preprocessed image

  (1) Crop image (top and bottom)
  (2) Remove background & grayscale
  (3) Reszie to smaller image

  Args:
      image (3-D array): (H, W, C)
      new_HW (tuple): New image size (height, width)
      height_range (tuple): Height range (H_begin, H_end) else cropped
      bg (tuple): Background RGB Color (R, G, B)

  Returns:
      image (3-D array): (H, W, 1)
  """
  #toimage(image).show()
  image = crop_image(image, height_range)
  #toimage(image).show()
  image = resize_image(image, new_HW)
  #toimage(image).show()
  image = kill_background_grayscale(image, bg, new_bg_color, black_white)
  #toimage(image).show()
  image = np.expand_dims(image, axis=2)
  return image

def crop_image(image, height_range):
  """Crops top and bottom

  Args:
      image (3-D array): Numpy image (H, W, C)
      height_range (tuple): Height range between (min_height, max_height)
          will be kept

  Returns:
      image (3-D array): Numpy image (max_H - min_H, W, C)
  """
  h_beg, h_end = height_range
  return image[h_beg:h_end, ...]

def resize_image(image, new_HW):
  """Returns a resized image

  Args:
      image (3-D array): Numpy array (H, W, C)
      new_HW (tuple): Target size (height, width)

  Returns:
      image (3-D array): Resized image (height, width, C)
  """
  return imresize(image, new_HW, interp="nearest")

def kill_background_grayscale(image, bg, new_bg_color=0, black_white=False):
  """Make the background 0

  Args:
      image (3-D array): Numpy array (H, W, C)
      bg (tuple): RGB code of background (R, G, B)

  Returns:
      image (2-D array): Binarized image of shape (H, W)
          The background is new_color
  """
  H, W, _ = image.shape

  R = image[..., 0]
  G = image[..., 1]
  B = image[..., 2]
  cond = (R == bg[0]) & (G == bg[1]) & (B == bg[2])

  if black_white:
    image = np.zeros((H, W))
    image[~cond] = 1
  else:
    image = image.mean(axis=2)
    image[cond] = new_bg_color
    image = (image - 128) / 128  # normalize from -1. to 1.

  return image
