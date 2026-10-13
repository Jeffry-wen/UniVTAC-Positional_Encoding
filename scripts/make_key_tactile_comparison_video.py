"""Synchronized RGB views + left/right Marker_RGB versus actual Depth_Deform.

Reads the online input recorder's HDF5. All panels use source_action_index;
queued actions repeat that same source observation. Playback is 10 fps, not a
claim about simulation wall-clock time. Raw camera/marker RGB uses sensor colors.
"""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def text(image,label,xy,size=.6,color=(235,235,235)):
    cv2.putText(image,label,xy,cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdf5',type=Path,default=ROOT/'.cache/key-online-deform/rerun/policy_inputs/1000000_failed.hdf5')
    parser.add_argument('--output',type=Path,default=ROOT/'.cache/key-online-deform/marker_rgb_vs_depth_deform_with_rgb.mp4')
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(args.hdf5) as f:
        actions=f['action_index'][:];sources=f['source_action_index'][:]
        process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo',
                                  '-pixel_format','rgb24','-video_size','1280x720','-framerate','10',
                                  '-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',
                                  '-movflags','+faststart',str(args.output)],stdin=subprocess.PIPE)
        cached_source=None
        try:
            for i,(action,source) in enumerate(zip(actions,sources)):
                if source!=cached_source:
                    base=np.full((720,1280,3),18,np.uint8)
                    for col,name in enumerate(('head','wrist')):
                        image=f[f'observation/{name}/rgb'][i]
                        base[56:416,col*640:(col+1)*640]=cv2.resize(image,(640,360),interpolation=cv2.INTER_LINEAR)
                        text(base,name.upper()+' RGB - same source observation',(col*640+12,49),.6)
                    for col,(side,kind) in enumerate((('left','marker'),('left','deform'),('right','marker'),('right','deform'))):
                        if kind=='marker':
                            image=f[f'tactile/{side}_tactile/rgb_marker'][i]
                            label=f'{side.upper()} Marker_RGB'
                        else:
                            image=f[f'policy_input/observation.images.tac_{side}'][i].transpose(1,2,0)
                            image=np.rint(np.clip(image,0,1)*255).astype(np.uint8)
                            label=f'{side.upper()} Depth_Deform'
                        image=cv2.resize(image,(256,256),interpolation=cv2.INTER_AREA)
                        base[448:704,col*320+32:col*320+288]=image
                        text(base,label,(col*320+12,440),.65)
                    text(base,'All panels share the inference-input timestamp; Depth_Deform is the actual policy input.',(12,716),.42)
                    cached_source=source
                frame=base.copy()
                text(frame,f'Pull Out Key | seed 1000000 | action {action:03d} | input source action {source:03d}',(12,24),.67)
                if action!=source:
                    text(frame,'QUEUED ACTION - INPUT HELD',(925,24),.55,(255,195,80))
                process.stdin.write(frame.tobytes())
                if action in (0,80,112):
                    cv2.imwrite(str(args.output.with_name(args.output.stem+f'_action_{action:03d}.png')),
                                cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        finally:
            process.stdin.close()
            code=process.wait()
        if code:
            raise RuntimeError(f'ffmpeg exited with {code}')
        metadata=dict(source_hdf5=str(args.hdf5.resolve()),video=str(args.output.resolve()),
                      frames=len(actions),fps=10,width=1280,height=720,
                      fresh_inputs=int((actions==sources).sum()),
                      layout=['head RGB','wrist RGB','left Marker_RGB','left Depth_Deform','right Marker_RGB','right Depth_Deform'],
                      note='All six views use the same source observation. Inputs held during queued actions. Sensor RGB colors for cameras and Marker_RGB; exact stored policy tensor for Depth_Deform, resized for presentation. Playback time is not physical time.')
        args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
        print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    main()
