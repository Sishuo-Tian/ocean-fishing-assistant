const el = id => document.getElementById(id);
let imageData = null, result = null, busy = false, filename = '';
const readImage = blob => new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=reject;r.readAsDataURL(blob)});
function switchView(showMask) {
  el('uploaded').src = showMask && result ? result.overlay : (result?.original || imageData || '');
  el('original').classList.toggle('active',!showMask);
  el('overlay').classList.toggle('active',showMask);
  el('original').setAttribute('aria-pressed',String(!showMask));
  el('overlay').setAttribute('aria-pressed',String(showMask));
}
function clearResult(){
  result=null;el('overlay').disabled=true;el('download').disabled=true;
  el('results').textContent='图片与目标就绪后，点击“开始分割”。';
  el('stamp').textContent='待分析';switchView(false);
}
function lock(value){
  busy=value;
  ['file','example','classes','run'].forEach(id=>el(id).disabled=value);
  el('run').textContent=value?'正在分割…':'开始分割';
}
async function loadPhoto(blob,name){
  const raw=await readImage(blob);
  await new Promise((resolve,reject)=>{const test=new Image();test.onload=resolve;test.onerror=()=>reject(Error('无法读取图片，请选择有效的 JPG/PNG。'));test.src=raw});
  imageData=raw;filename=name;clearResult();
  el('scene').style.display='none';el('uploaded').style.display='block';
  el('filename').textContent=name;
  el('caption').textContent='类别名称决定分割范围，请同时考虑背景及可能出现的其他生物。';
  el('message').textContent='图片已载入，可开始真实模型分割。';
}
el('file').onchange=async()=>{
  const f=el('file').files[0];if(!f||busy)return;
  try{
    if(!['image/jpeg','image/png'].includes(f.type)||f.size>15*1024*1024)throw Error('请选择不超过 15 MB 的 JPG/PNG。');
    await loadPhoto(f,f.name);
  }catch(e){el('message').textContent=e.message}
};
el('example').textContent='载入海龟测试照片';
el('example').onclick=async()=>{
  lock(true);
  try{const response=await fetch('/sample.jpg');
    if(!response.ok)throw Error((await response.json().catch(()=>({}))).error||'样例加载失败');
    await loadPhoto(await response.blob(),'Turtle_541.jpg · MAS3K 公开测试照片');
    el('classes').value='鱼类、海龟、水体背景';
    el('message').textContent='公开数据集样例，非远洋现场实拍。可用于验证接口。';
  }catch(e){el('message').textContent=e.message}finally{lock(false)}
};
el('classes').oninput=()=>{clearResult();el('message').textContent='类别已修改，请重新分割。'};
el('original').onclick=()=>switchView(false);
el('overlay').onclick=()=>switchView(true);
el('run').onclick=async()=>{
  if(busy)return;
  if(location.protocol==='file:'){el('message').textContent='请启动服务并通过 http://127.0.0.1:8765 打开页面。';return}
  if(!imageData){el('message').textContent='请先上传图片，或载入测试照片。';return}
  const classes=el('classes').value.split(/[,，、;；\n]+/).map(s=>s.trim()).filter(Boolean);
  if(!classes.length){el('message').textContent='请填写观察目标。';return}
  clearResult();lock(true);el('message').textContent='模型正在分析图片，请稍候…';
  try{
    const response=await fetch('/segment',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:imageData,classes})});
    const data=await response.json();if(!response.ok)throw Error(data.error||'推理失败');
    result=data;el('overlay').disabled=false;el('download').disabled=false;switchView(true);
    el('stamp').textContent='真实模型输出';
    el('results').replaceChildren();
    data.classes.forEach(c=>{
      const row=document.createElement('div');row.className='result';
      const dot=document.createElement('span');dot.className='dot';dot.style.background=c.color;
      row.append(dot,document.createTextNode(c.name+'：'+c.area_percent+'%'));el('results').append(row);
    });
    const note=document.createElement('p');note.className='hint';note.textContent='百分比为预测像素占比，不是置信度或数量。掩码需人工复核。';el('results').append(note);
    el('caption').textContent='输出 '+data.width+' × '+data.height+' 像素；本次处理 '+data.seconds+' 秒（不含上传）。';
    el('message').textContent='分割完成。可切换原图、下载类别掩码。';
  }catch(e){el('message').textContent='未生成结果：'+e.message;el('results').textContent='分析失败，请检查目标类别或图片后重试。'}
  finally{lock(false)}
};
function download(url,name){const a=document.createElement('a');a.href=url;a.download=name;a.click()}
const maskButton=document.createElement('button');maskButton.id='download';maskButton.className='secondary';maskButton.textContent='下载类别掩码 PNG';maskButton.disabled=true;
el('export').after(maskButton);
maskButton.onclick=()=>{if(result)download(result.mask,'segmentation-mask.png')};
el('export').onclick=()=>{
  const record={时间:new Date().toISOString(),图像:filename,状态:result?'真实模型输出':'尚未推理',类别:result?.classes||el('classes').value,模型:result?.model||null,备注:el('notes').value};
  const url=URL.createObjectURL(new Blob([JSON.stringify(record,null,2)],{type:'application/json;charset=utf-8'}));
  download(url,'水下观察记录.json');setTimeout(()=>URL.revokeObjectURL(url),1000);
};
el('run').textContent='开始分割';el('scene').style.display='none';
el('stamp').textContent='等待上传';el('filename').textContent='上传一张水下照片';
el('message').textContent='支持常见中文目标；其他类别请使用英文。自动加入水体背景。';
el('caption').textContent='上传图片或载入公开测试照片，开始真实分割。';
el('results').textContent='等待图片与目标类别。';
document.querySelector('.badge').textContent='本机模型推理';
document.querySelector('footer').textContent='Earth2Ocean ViT-B/16 · 水下模板 · 关闭逐图大模型描述。图片在本机处理。候选类别外的物体可能被误分；背景不是未知目标检测。';
document.querySelectorAll('.hint').forEach(e=>{if(e.textContent.includes('示例预设'))e.textContent='支持鱼类、金枪鱼、海龟、鲨鱼等常见中文类别，其他类别请用英文。'});
if(location.protocol!=='file:')fetch('/health').then(r=>r.json()).then(d=>{if(d.ready)el('message').textContent='模型已就绪。上传图片或载入测试照片。'}).catch(()=>el('message').textContent='无法连接本机模型，请检查服务是否启动。');
