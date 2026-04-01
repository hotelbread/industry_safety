import pygame
try:
    pygame.mixer.init()
    print("mixer init OK")
    # pygame.mixer.music.load("/home/smh/Music/effectBad.mp3")
    pygame.mixer.music.load("action_2.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
except Exception as ex:
    print(f"error: {ex}")






# from gtts import gTTS
# text = "준비되셨나요. \n 그럼 3초 후 시작합니다! \n 각 동작을 5초간 취해보세요."
# tts = gTTS(text=text, lang='ko')
# tts.save("hello.mp3")

