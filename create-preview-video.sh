#!/bin/bash
ffmpeg -r 30 -f image2 -i visualization/preview/ACGAN/vis_epoch_%04d.png -s 1024x576 -sws_flags neighbor -vcodec libx264 -crf 25  -pix_fmt yuv420p preview.mp4
