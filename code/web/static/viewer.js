/* Triangle surfaces and sampled point clouds share the same interaction model. */
class WorkbenchViewer {
  constructor(host) {
    if (!window.THREE || !THREE.OrbitControls) throw Error('The 3D viewer could not load. Check your internet connection and reload. You can still reconstruct and download outputs.');
    this.host = host;
    this.pointScale = 1;
    this.renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    host.append(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, .001, 1000);
    this.camera.up.set(0,0,1);
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.AmbientLight(0xffffff,.45));
    const key = new THREE.DirectionalLight(0xffffff,.7); key.position.set(3,-4,6); this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff,.2); fill.position.set(-3,4,2); this.scene.add(fill);
    this.axes = new THREE.Group();
    this.axes.add(new THREE.AxesHelper(1.35));
    for (const [label,position,color] of [['X',[1.47,0,0],'#ee9d92'],['Y',[0,1.47,0],'#a5d68c'],['Z',[0,0,1.47],'#94b8ef']]) {
      const canvas = document.createElement('canvas'); canvas.width=96; canvas.height=64;
      const ctx=canvas.getContext('2d'); ctx.font='500 40px system-ui'; ctx.fillStyle=color; ctx.textAlign='center'; ctx.fillText(label,48,46);
      const material=new THREE.SpriteMaterial({map:new THREE.CanvasTexture(canvas),depthTest:false});
      const sprite=new THREE.Sprite(material); sprite.position.set(...position); sprite.scale.set(.13,.09,1); this.axes.add(sprite);
    }
    this.scene.add(this.axes);
    const grid=new THREE.GridHelper(3,12,0x456456,0x304c40); grid.rotation.x=Math.PI/2; grid.position.z=-1.07;
    grid.material.transparent=true; grid.material.opacity=.35; this.axes.add(grid);
    this.raycaster=new THREE.Raycaster(); this.pointer=new THREE.Vector2();
    this.raycaster.params.Points.threshold=.018;
    this.marker=new THREE.Mesh(new THREE.SphereGeometry(.012,12,8),new THREE.MeshBasicMaterial({color:0xe8ecc0,depthTest:false}));
    this.marker.visible=false; this.marker.renderOrder=10; this.scene.add(this.marker);
    this.dragging=false;
    this.controls.addEventListener('start',()=>{this.dragging=true;this.hideHover();});
    this.controls.addEventListener('end',()=>{this.dragging=false;});
    this.renderer.domElement.addEventListener('pointermove',event=>this.hover(event));
    this.renderer.domElement.addEventListener('pointerleave',()=>this.hideHover());
    this.resizeObserver=new ResizeObserver(()=>this.resize()); this.resizeObserver.observe(host);
    this.resize(); this.fit();
    const animate=()=>{this.frame=requestAnimationFrame(animate);this.controls.update();this.renderer.render(this.scene,this.camera);}; animate();
  }
  resize(){const {width,height}=this.host.getBoundingClientRect();if(!width||!height)return;this.camera.aspect=width/height;this.camera.updateProjectionMatrix();this.renderer.setSize(width,height,false);}
  clear(){if(this.object){this.scene.remove(this.object);this.object.geometry.dispose();this.object.material.dispose();this.object=null;}this.hideHover();}
  load(data){
    this.clear(); this.data=data;
    const geometry=new THREE.BufferGeometry(); geometry.setAttribute('position',new THREE.Float32BufferAttribute(data.positions,3));
    geometry.setAttribute('color',new THREE.Uint8BufferAttribute(data.colors,3,true));
    if(data.kind==='mesh'){geometry.setIndex(data.indices);geometry.computeVertexNormals();this.object=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.DoubleSide}));}
    else this.object=new THREE.Points(geometry,new THREE.PointsMaterial({vertexColors:true,size:.008*this.pointScale,sizeAttenuation:true}));
    this.scene.add(this.object);geometry.computeBoundingSphere();this.sphere=geometry.boundingSphere.clone();this.fit();
    this.host.dataset.kind=data.kind;this.host.dataset.ready='true';
    this.host.dataset.count=String(data.kind==='mesh'?data.triangles:data.displayed_points);
  }
  fit(){
    const center=this.sphere?.center || new THREE.Vector3(); const radius=this.sphere?.radius || 1.4;
    const fov=Math.min(this.camera.fov*Math.PI/180,2*Math.atan(Math.tan(this.camera.fov*Math.PI/360)*this.camera.aspect));
    const distance=radius/Math.sin(fov/2)*1.18;
    this.camera.position.copy(center).add(new THREE.Vector3(1.1,-1.5,.85).normalize().multiplyScalar(distance));
    this.controls.target.copy(center);this.controls.minDistance=radius*.05;this.controls.maxDistance=distance*20;this.controls.update();this.hideHover();
  }
  appearance(mode){
    if(!this.object||this.data.kind!=='mesh')return;
    this.object.material.dispose();
    if(mode==='normals')this.object.material=new THREE.MeshNormalMaterial({side:THREE.DoubleSide});
    else if(mode==='shaded')this.object.material=new THREE.MeshLambertMaterial({color:0xaebcaf,side:THREE.DoubleSide});
    else if(mode==='wireframe')this.object.material=new THREE.MeshBasicMaterial({color:0xa5d6b8,wireframe:true,side:THREE.DoubleSide});
    else this.object.material=new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.DoubleSide});
  }
  pointSize(value){
    const size=Number(value);if(!Number.isFinite(size))return;
    this.pointScale=Math.max(.3,Math.min(3,size));
    this.raycaster.params.Points.threshold=.018*Math.sqrt(this.pointScale);
    if(this.object&&this.data.kind==='cloud')this.object.material.size=.008*this.pointScale;
  }
  hideHover(){document.querySelector('#hover-coordinates').hidden=true;this.marker.visible=false;}
  hover(event){
    if(!this.object||this.dragging||event.buttons)return;
    const now=performance.now();if(now-(this.lastHover||0)<60)return;this.lastHover=now;
    const rect=this.renderer.domElement.getBoundingClientRect();
    this.pointer.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);
    this.raycaster.setFromCamera(this.pointer,this.camera);
    const hit=this.raycaster.intersectObject(this.object,false)[0];if(!hit){this.hideHover();return;}
    // Point picking reports the actual selected vertex; mesh picking reports the
    // ray/triangle intersection. Both are in the normalized display frame.
    const point=this.data.kind==='cloud'?new THREE.Vector3().fromBufferAttribute(this.object.geometry.attributes.position,hit.index):hit.point;
    this.marker.position.copy(point);this.marker.visible=true;
    const tip=document.querySelector('#hover-coordinates');tip.textContent=`${this.data.kind==='mesh'?'Surface':'Point'} · arbitrary units\nX  ${point.x.toFixed(4)}   Y  ${point.y.toFixed(4)}   Z  ${point.z.toFixed(4)}`;tip.hidden=false;
    tip.dataset.x=point.x;tip.dataset.y=point.y;tip.dataset.z=point.z;
  }
}
window.WorkbenchViewer=WorkbenchViewer;
