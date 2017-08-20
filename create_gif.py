from moviepy.editor import *

my_clip = VideoFileClip("project_video_output.mp4").resize(0.2)
my_clip.subclip(30, 42).write_gif("sample.gif")