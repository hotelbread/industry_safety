import pygame

class AudioManager:
    def __init__(self):
        pygame.mixer.init()
        self.current_music = None
        # self.clock = pygame.time.Clock()

    def play_sound(self, sound_file):
        # print(f'[Debug][AudioManager] play_sound1')
        if pygame.mixer.music.get_busy():
            # print(f'[Debug][AudioManager] play_sound2')
            return False
        # print(f'[Debug][AudioManager] play_sound3')
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
        self.current_music = sound_file
        # while pygame.mixer.music.get_busy():
        #     pygame.time.Clock().tick(10)
        # print(f'[Debug][AudioManager] play_sound4')
        return True

    def play_sound_effect(self, sound_file):
        sound = pygame.mixer.Sound(sound_file)
        sound.play()
        
    def get_current(self): 
         return self.current_music
    
    def update(self):
        
        # self.clock.tick(10)
        
        if not pygame.mixer.music.get_busy():
            self.current_music = None