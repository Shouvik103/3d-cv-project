from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
root=Path.cwd();out=root/'results/mesh';chair=out/'chair_sgbm_wide_roi/renders';temple=out/'temple_mono/renders'
font=ImageFont.load_default(size=26);small=ImageFont.load_default(size=20)
clips=[(6,chair/'colored_orbit.mp4','Chair: colored triangle mesh','57 fusion views; COLMAP poses + SGBM + TSDF'),(6,chair/'shaded_orbit.mp4','Chair: shaded surface','Thin and occluded regions can remain incomplete'),(4,chair/'wireframe_0.png','Triangle structure','Light mesh: approximately 80,000 triangles'),(6,temple/'colored_orbit.mp4','Temple: mesh from our SfM poses','Depth Anything V2 Small; robust inverse-depth alignment'),(4,out/'sample_mono/renders/colored_0.png','Generated tiles: our SfM + learned depth','Controlled sample; held-out fusion-view checks'),(4,out/'car_sgbm/renders/colored_0.png','Failure example: glossy toy car','Holes, surface noise and fused background remain limitations')]
with imageio.get_writer(out/'mesh_segment.mp4',fps=30,codec='libx264',quality=7,macro_block_size=2) as writer:
 for seconds,path,title,subtitle in clips:
  video=path.suffix=='.mp4';reader=imageio.get_reader(path) if video else None
  frames=reader.count_frames() if video else 1
  for i in range(seconds*30):
   a=reader.get_data(i%frames) if video else np.asarray(Image.open(path).convert('RGB'))
   im=Image.fromarray(a).resize((1280,720))
   if 'wireframe' in path.name:im=im.crop((320,130,960,620)).resize((1280,720))
   draw=ImageDraw.Draw(im);draw.rectangle((0,0,1280,78),fill=(245,245,245));draw.text((24,10),title,font=font,fill=(20,20,20));draw.text((24,44),subtitle,font=small,fill=(55,55,55));writer.append_data(np.asarray(im))
  if reader:reader.close()
(out/'demo_manifest.json').write_text(json.dumps(dict(duration_seconds=sum(c[0] for c in clips),fps=30,frames=900,resolution=[1280,720],scope='Mesh segment for integration into the course demo; captioned, not narrated',segments=[dict(seconds=s,source=str(p.relative_to(root)),title=t) for s,p,t,_ in clips]),indent=2))
print('Created mesh_segment.mp4')
