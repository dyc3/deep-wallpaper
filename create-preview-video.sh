#!/bin/bash
ffmpeg -r 60 -f image2 -s 160x90 -i preview/preview%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p -vf "scale=16:-2" preview.mp4
