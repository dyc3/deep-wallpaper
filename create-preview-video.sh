#!/bin/bash
ffmpeg -r 30 -f image2 -s 128x72 -i visualization/preview/ACGAN/vis_epoch_%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p -vf "scale=16:-2" preview.mp4
