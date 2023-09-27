import pygame
import numpy as np

class Field:
    # class for compiling the array that the DQN will interpret
    def __init__(self, height=10, width=5):
        self.width = width
        self.height = height
        self.clear_field()

    def clear_field(self):
        self.body = np.zeros(shape=(self.height, self.width))

    def update_field(self, fruits, player):
        self.clear_field()

        # draw fruits
        for fruit in fruits:
            if not fruit.out_of_field:
                for y in range(fruit.y, min(fruit.y + fruit.height, self.height)):
                    for x in range(fruit.x, min(fruit.x + fruit.width, self.width-1)):
                        self.body[y][x] = 1
        
        # draw player
        for i in range(player.width):
            self.body[player.y][player.x + i] = 2

class Fruit:
    # class for the fruit
    def __init__(self, height=1, width=1, x=None, y=0, speed=1, field=None):
        self.field = field
        self.height = height
        self.width = width
        self.x = self.generate_x() if x == None else x
        self.y = y
        self.speed = speed
        self.out_of_field = False
        self.is_caught = 0

    def generate_x(self):
        return np.random.randint(0, self.field.width - self.width)

    def set_out_of_field(self):
        self.out_of_field = True if (self.y > self.field.height - 1) else False

    def move(self):
        self.y += self.speed
        self.set_out_of_field()

    def set_is_caught(self, player):
        if self.y != player.y:
            self.is_caught = 0
        else:
            if self.x + self.width > player.x and (self.x < player.x + player.width):
                self.is_caught = 1
            else:
                self.is_caught = -1

class Player:
    # class for the player
    def __init__(self, height=1, width=1, field=None):
        self.field = field
        self.height = height
        self.width = width
        self.x = int(self.field.width / 2 - width / 2)
        self.last_x = self.x
        self.y = self.field.height - 1
        self.dir = 0
        self.colour = "blue"

    def move(self):
        self.last_x = self.x
        self.x += self.dir
        self.dir = 0
        self.constrain()

    def action(self, action):
        if action == 1:
            self.dir = -1
        elif action == 2:
            self.dir = 1
        else:
            self.dir = 0

    def constrain(self):
        if self.x < 0:
            self.x = self.field.width - self.width
        elif (self.x + self.width) > self.field.width:
            self.x = 0

class Environment:
    # class for the environment
    F_HEIGHT = 12
    F_WIDTH = 12
    PLAYER_WIDTH = 2
    FRUIT_WIDTH = 1

    ENVIRONMENT_SHAPE = (F_HEIGHT, F_WIDTH, 1)
    ACTION_SPACE = [0, 1, 2]
    ACTION_SPACE_SIZE = len(ACTION_SPACE)
    ACTION_SHAPE = (ACTION_SPACE_SIZE,)
    PUNISHMENT = -1
    REWARD = 1
    score = 0
    MAX_VAL = 2

    LOSS_SCORE = -5
    WIN_SCORE = 5

    DRAW_MUL = 30
    WINDOW_HEIGHT = F_HEIGHT * DRAW_MUL
    WINDOW_WIDTH = F_WIDTH * DRAW_MUL

    game_tick = 0
    FPS = 20
    MOVE_FRUIT_EVERY = 1
    MOVE_PLAYER_EVERY = 1
    MAX_FRUIT = 1
    INCREASE_MAX_FRUIT_EVERY = 100
    SPAWN_FRUIT_EVERY_MIN = 2
    SPAWN_FRUIT_EVERY_MAX = 12
    next_spawn_tick = 0

    FRUIT_COLOURS = {-1: "red", 0: "black", 1: "green"}

    def __init__(self):
        self.reset()

    def get_state(self):
        return self.field.body / self.MAX_VAL

    def reset(self):
        self.game_tick = 0
        self.game_over = False
        self.game_won = False
        self.field = Field(height=self.F_HEIGHT, width=self.F_WIDTH)
        self.player = Player(field=self.field, width=self.PLAYER_WIDTH)
        self.score = 0
        self.fruits = []
        self.spawn_fruit()
        self.field.update_field(self.fruits, self.player)
        
        return self.get_state()

    def spawn_fruit(self):
        if len(self.fruits) < self.MAX_FRUIT:
            self.fruits.append(Fruit(field=self.field, height=self.FRUIT_WIDTH, width=self.FRUIT_WIDTH))
            self.set_next_spawn_tick()

    def set_next_spawn_tick(self):
        self.next_spawn_tick = self.game_tick + np.random.randint(self.SPAWN_FRUIT_EVERY_MIN, self.SPAWN_FRUIT_EVERY_MAX)

    def step(self, action=None):
        # this runs every step of the game
        # the QDN can pass an action to the game, and in return gets next game state, reward, etc.
        
        self.game_tick += 1

        if self.game_tick % self.INCREASE_MAX_FRUIT_EVERY == 0:
            self.MAX_FRUIT += 1

        if self.game_tick >= self.next_spawn_tick or len(self.fruits) == 0:
            self.spawn_fruit()

        if action != None:
            self.player.action(action)
        self.player.move()

        reward = 0

        if self.game_tick % self.MOVE_FRUIT_EVERY == 0:
            in_field_fruits = []
            for fruit in self.fruits:
                fruit.move()
                fruit.set_is_caught(self.player)
                if fruit.is_caught == 1:
                    self.update_score(self.REWARD)
                    reward = self.REWARD
                elif fruit.is_caught == -1:
                    self.update_score(self.PUNISHMENT)
                    reward = self.PUNISHMENT
                if not fruit.out_of_field:
                    in_field_fruits.append(fruit)
            self.fruits = in_field_fruits

        self.field.update_field(fruits=self.fruits, player=self.player)

        # print("SCORE:", self.score)
        if self.score <= self.LOSS_SCORE:
            self.game_over = True

        if self.score >= self.WIN_SCORE:
            self.game_won = True 

        return self.get_state(), reward, self.game_over or self.game_won, self.score
    
    def update_score(self, delta):
        self.score += delta

    def render(self, screen, solo=True, x_offset=0, y_offset=0):
        # for rendering the game
        if solo:
            screen.fill("white")
            pygame.display.set_caption(f"Score: {self.score}")

        # draw player
        pygame.draw.rect(
            screen,
            self.player.colour,
            ((self.player.x * self.DRAW_MUL + x_offset, self.player.y * self.DRAW_MUL + y_offset), (self.player.width * self.DRAW_MUL, self.player.height * self.DRAW_MUL))
        )

        # draw fruit
        for fruit in self.fruits:
            pygame.draw.rect(
                screen, 
                self.FRUIT_COLOURS[fruit.is_caught], 
                ((fruit.x * self.DRAW_MUL + x_offset, fruit.y * self.DRAW_MUL + y_offset), (fruit.width * self.DRAW_MUL, fruit.height * self.DRAW_MUL))
            )

def main():
    # if run as a script, the game is human playable at 15fps
    env = Environment()

    pygame.init()
    screen = pygame.display.set_mode((env.WINDOW_WIDTH, env.WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT]:
            env.player.dir = -1
        if keys[pygame.K_RIGHT]:
            env.player.dir = 1

        env.step()

        env.render(screen)

        pygame.display.flip()
        clock.tick(15)

    pygame.quit()

    return 0

if __name__ == "__main__":
    main()