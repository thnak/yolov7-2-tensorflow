import os
import torch
import platform
import subprocess
import shlex
import math
try:
    from pyadl import *
except:
    pass


def getGPUtype():
    try:
        adv = ADLManager.getInstance().getDevices()
        ac = []
        for a in adv:
            ab = [str(a.adapterIndex), str(a.adapterName)]
            ac.append(ab)
    except:
        ac = None
    return ac

class FFMPEG_recorder():
    """Hardware Acceleration for video recording using FFMPEG
    Documents: https://trac.ffmpeg.org/wiki/Encode/H.265
    """

    def __init__(self, savePath=None, videoDimensions=(1280, 720), fps=30):
        """_FFMPEG recorder_
        Args:
            savePath (__str__, optional): _description_. Defaults to None.
            codec (_str_, optional): _description_. Defaults to None.
            videoDimensions (tuple, optional): _description_. Defaults to (1280, 720).
            fps (int, optional): _description_. Defaults to 30FPS.
        """
        self.savePath = savePath
        self.codec = None
        self.videoDementions = videoDimensions
        self.fps = fps
        mySys = platform.uname()
        osType = mySys.system
        if torch.cuda.is_available():
            self.codec = 'hevc_nvenc'
        elif osType == 'Windows' and 'AMD' in str(getGPUtype()):
            self.codec = 'hevc_amf'
        elif osType == 'Linux' and 'AMD' in str(getGPUtype()):
            self.codec = 'hevc_vaapi'
        else:
            self.codec = 'libx264'
        print(f'Using video codec: {self.codec}, os: {osType}')
        self.countFrame = 0
        self.startTime = 0.
        mpx = math.prod(self.videoDementions)
        self.bitRate = round(20 * (mpx/(3840*2160)) * (1 if round(self.fps/30,3) < 1 else round(self.fps/30,3)),3)
        self.subtitleContent = ''
        self.process = subprocess.Popen(shlex.split(f'ffmpeg -hide_banner -y -s {self.videoDementions[0]}x{self.videoDementions[1]} -pixel_format bgr24 -f rawvideo -r {self.fps} -i pipe: -vcodec {self.codec} -pix_fmt yuv420p -b:v {self.bitRate}M')+[self.savePath], stdin=subprocess.PIPE)
    def writeFrame(self, image=None):
        """Write frame by frame to video

        Args:
            image (_image_, require): the image will write to video
        """

        self.process.stdin.write(image.tobytes())
        
    def second_to_timecode(self,x=0) -> str:
        hour, x = divmod(x, 3600)
        minute, x = divmod(x, 60)
        second, x = divmod(x, 1)
        millisecond = int(x * 1000.)
        return '%.2d:%.2d:%.2d,%.3d' % (hour, minute, second, millisecond)
    
    def writeSubtitle(self, title='', fps=30):
        timeStep = 1/ fps
        timecode = self.second_to_timecode(self.startTime)
        timecode2 = self.second_to_timecode(self.startTime + timeStep)
        self.startTime += timeStep
           
        if title == '':
            title = f'UTC2'
        frame = f'{self.countFrame}\n'
        timeStamp = f'{timecode} --> {timecode2}\n'
        subtitile = f'{title}\n'
        self.subtitleContent += f'{frame}{timeStamp}{subtitile}\n'
        self.countFrame += 1
        
    def addSubtitle(self, hardSubtitle=False):
        save = self.savePath.replace('.mp4','withsub.mp4')
        subfile = save.replace('.mp4','.srt')
        with open(subfile, 'w') as f:
            f.write(self.subtitleContent)
        
        if hardSubtitle:
            process = subprocess.run(f"ffmpeg -hide_banner -i {self.savePath} -c:v copy -vf subtitles='{subfile}' {save}") #error
        else:
            process = subprocess.run(f"ffmpeg -hide_banner -i {self.savePath} -i {subfile} -c:v copy -c:s mov_text -metadata:s:s:0 language=eng {save}")
        
    def addAudio(self):
        pass
        
    def stopRecorder(self):
        """Stop record video"""
        self.process.stdin.close()
        self.process.wait()
        self.process.terminate()
