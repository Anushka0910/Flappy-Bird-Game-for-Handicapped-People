import pygame
import random
from head_controls import HeadController

class FlappyBirdGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 512))
        self.clock = pygame.time.Clock()

        # Load game assets
        self.bird = pygame.image.load('assets/bird.png').convert_alpha()
        self.background = pygame.image.load('assets/background.jpg').convert_alpha()
        self.pipe_top = pygame.image.load('assets/pipe2.png').convert_alpha()
        self.pipe_bottom = pygame.image.load('assets/pipe1.png').convert_alpha()

        # Initial bird position
        self.bird_y = 250
        self.bird_movement = 0

        # Pipe settings
        self.pipe_gap = 170
        self.pipe_velocity = 3
        self.pipe_list = []
        self.spawn_pipe()

        # Head control setup
        self.head_controller = HeadController()

        # Game variables
        self.score = 0
        self.font = pygame.font.SysFont('Arial', 30)
        self.game_over = False
        self.paused = False

        # Buttons
        self.restart_button = pygame.Rect(300, 220, 200, 50)
        self.pause_button = pygame.Rect(660, 10, 120, 40)
        self.quit_button = pygame.Rect(300, 290, 200, 50)

    def spawn_pipe(self):
        pipe_height = random.randint(150, 300)
        pipe_top_y = pipe_height - self.pipe_top.get_height()
        pipe_bottom_y = pipe_height + self.pipe_gap
        pipe_x = 800
        self.pipe_list.append([pipe_x, pipe_top_y, pipe_bottom_y, False])

    def move_pipes(self):
        for pipe in self.pipe_list:
            pipe[0] -= self.pipe_velocity

        if self.pipe_list[0][0] < -self.pipe_top.get_width():
            self.pipe_list.pop(0)
            self.spawn_pipe()

    def check_collision(self):
        bird_rect = pygame.Rect(50, self.bird_y, self.bird.get_width(), self.bird.get_height())

        for pipe in self.pipe_list:
            pipe_top_rect = pygame.Rect(pipe[0], pipe[1], self.pipe_top.get_width(), self.pipe_top.get_height())
            pipe_bottom_rect = pygame.Rect(pipe[0], pipe[2], self.pipe_bottom.get_width(), self.pipe_bottom.get_height())

            if bird_rect.colliderect(pipe_top_rect) or bird_rect.colliderect(pipe_bottom_rect):
                return True

        if self.bird_y <= 0 or self.bird_y >= 512 - self.bird.get_height():
            return True

        return False

    def update_score(self):
        for pipe in self.pipe_list:
            if pipe[0] + self.pipe_top.get_width() < 50 and not pipe[3]:
                self.score += 10
                pipe[3] = True

    def draw_buttons(self):
        pygame.draw.rect(self.screen, (0, 255, 0), self.restart_button)
        pygame.draw.rect(self.screen, (0, 0, 255), self.pause_button)
        pygame.draw.rect(self.screen, (255, 0, 0), self.quit_button)
        restart_text = self.font.render("Restart", True, (255, 255, 255))
        self.screen.blit(restart_text, (self.restart_button.x + 40, self.restart_button.y + 10))
        pause_text = self.font.render("Pause", True, (255, 255, 255))
        self.screen.blit(pause_text, (self.pause_button.x + 20, self.pause_button.y + 10))
        quit_text = self.font.render("Quit", True, (255, 255, 255))
        self.screen.blit(quit_text, (self.quit_button.x + 50, self.quit_button.y + 10))

    def display_game_over(self):
        text = self.font.render(f"Game Over! Score: {self.score}", True, (255, 0, 0))
        self.screen.blit(text, (250, 150))
        self.draw_buttons()

    def display_score(self):
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

    def reset_game(self):
        self.bird_y = 250
        self.bird_movement = 0
        self.pipe_list = []
        self.spawn_pipe()
        self.score = 0
        self.game_over = False
        self.paused = False

    def pause_game(self):
        self.paused = not self.paused

    def run_game(self):
        while True:
            self.screen.blit(self.background, (0, 0))

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.game_over:
                        if self.restart_button.collidepoint(event.pos):
                            self.reset_game()
                        elif self.quit_button.collidepoint(event.pos):
                            pygame.quit()
                            return
                    if self.pause_button.collidepoint(event.pos) and not self.game_over:
                        self.pause_game()

            if not self.paused and not self.game_over:
                movement_direction = self.head_controller.get_head_movement()

                if movement_direction == 'up':
                    self.bird_movement = -5
                elif movement_direction == 'down':
                    self.bird_movement = 5
                else:
                    self.bird_movement = 1

                self.bird_y += self.bird_movement
                self.move_pipes()

                if self.check_collision():
                    self.game_over = True

                self.update_score()

            self.screen.blit(self.bird, (50, self.bird_y))
            for pipe in self.pipe_list:
                self.screen.blit(self.pipe_top, (pipe[0], pipe[1]))
                self.screen.blit(self.pipe_bottom, (pipe[0], pipe[2]))

            self.display_score()

            if self.game_over:
                self.display_game_over()

            pygame.display.update()
            self.clock.tick(30)

        self.head_controller.release()

# For testing purposes
if __name__ == "__main__":
    game = FlappyBirdGame()
    game.run_game()
