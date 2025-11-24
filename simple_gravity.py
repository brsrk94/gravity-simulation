"""
Simple 2D Gravity Simulation - Bouncing Ball
A basic example showing gravity acceleration (9.81 m/s²)
"""

import pygame
import numpy as np

# Constants
WIDTH, HEIGHT = 800, 600
GRAVITY = 9.81  # m/s² on Earth
FPS = 60
SCALE = 10  # pixels per meter

class Ball:
    def __init__(self, x, y, radius=20, color=(255, 100, 100)):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.acceleration = np.array([0.0, GRAVITY], dtype=float)
        self.radius = radius
        self.color = color
        self.bounce_damping = 0.8  # Energy loss on bounce
        
    def update(self, dt):
        """Update position using physics"""
        # Update velocity: v = v + a * dt
        self.velocity += self.acceleration * dt
        
        # Update position: p = p + v * dt
        self.position += self.velocity * dt * SCALE
        
        # Bounce off bottom
        if self.position[1] + self.radius >= HEIGHT:
            self.position[1] = HEIGHT - self.radius
            self.velocity[1] = -self.velocity[1] * self.bounce_damping
            
        # Bounce off sides
        if self.position[0] - self.radius <= 0:
            self.position[0] = self.radius
            self.velocity[0] = -self.velocity[0] * self.bounce_damping
        elif self.position[0] + self.radius >= WIDTH:
            self.position[0] = WIDTH - self.radius
            self.velocity[0] = -self.velocity[0] * self.bounce_damping
            
    def draw(self, screen):
        """Draw the ball"""
        pygame.draw.circle(screen, self.color, 
                         (int(self.position[0]), int(self.position[1])), 
                         self.radius)
        
        # Draw velocity vector
        if np.linalg.norm(self.velocity) > 0.1:
            end_pos = self.position + self.velocity * 5
            pygame.draw.line(screen, (100, 255, 100), 
                           self.position, end_pos, 2)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Gravity Simulation - Bouncing Ball")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    # Create balls
    balls = [
        Ball(100, 100, 20, (255, 100, 100)),
        Ball(300, 50, 15, (100, 100, 255)),
        Ball(500, 150, 25, (100, 255, 100)),
    ]
    
    # Give some initial horizontal velocity
    balls[1].velocity[0] = 5
    balls[2].velocity[0] = -3
    
    running = True
    paused = False
    
    print("╔═══════════════════════════════════════╗")
    print("║   2D GRAVITY SIMULATION - BALL DROP   ║")
    print("╚═══════════════════════════════════════╝")
    print("\n🎮 CONTROLS:")
    print("  SPACE - Pause/Resume")
    print("  CLICK - Add new ball")
    print("  R     - Reset")
    print("  ESC   - Exit")
    print("\n▶️  Starting simulation...\n")
    
    while running:
        dt = 1 / FPS
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("⏸️  Paused" if paused else "▶️  Resumed")
                elif event.key == pygame.K_r:
                    balls = [
                        Ball(100, 100, 20, (255, 100, 100)),
                        Ball(300, 50, 15, (100, 100, 255)),
                        Ball(500, 150, 25, (100, 255, 100)),
                    ]
                    balls[1].velocity[0] = 5
                    balls[2].velocity[0] = -3
                    print("🔄 Reset simulation")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Add ball at mouse position
                x, y = pygame.mouse.get_pos()
                color = (np.random.randint(100, 255), 
                        np.random.randint(100, 255), 
                        np.random.randint(100, 255))
                balls.append(Ball(x, y, np.random.randint(10, 30), color))
                print(f"➕ Added ball at ({x}, {y})")
        
        # Update
        if not paused:
            for ball in balls:
                ball.update(dt)
        
        # Draw
        screen.fill((20, 20, 40))
        
        # Draw ground
        pygame.draw.line(screen, (100, 100, 100), (0, HEIGHT-1), (WIDTH, HEIGHT-1), 3)
        
        # Draw balls
        for ball in balls:
            ball.draw(screen)
        
        # Draw UI
        status_text = "PAUSED" if paused else "RUNNING"
        status_color = (255, 100, 100) if paused else (100, 255, 100)
        text = font.render(status_text, True, status_color)
        screen.blit(text, (10, 10))
        
        info = font.render(f"Balls: {len(balls)} | Gravity: {GRAVITY} m/s²", True, (255, 255, 255))
        screen.blit(info, (10, 40))
        
        help_text = font.render("SPACE: Pause | CLICK: Add Ball | R: Reset", True, (200, 200, 200))
        screen.blit(help_text, (10, HEIGHT - 30))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    print("\n👋 Simulation ended!")
    pygame.quit()


if __name__ == "__main__":
    main()
