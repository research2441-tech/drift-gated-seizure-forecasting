// Streaming EDF -> 4-s window features (18 bipolar derivations x 16 int16 features).
// Run in a browser page served from the PhysioNet S3 mirror (https://physionet-open.s3.amazonaws.com/).
// Usage: await runTasks2('feat_part1', [['chbmit/1.0.0/chb01/chb01_03.edf', 0, 3600], ...], 3);
window.fetch2 = window.fetch2 || window.fetch.bind(window);
(()=>{
// EDF -> compact 4 s window features (18 bipolar channels x 16 features), int16
window.X = window.X || {done:{}, err:{}, prog:{}, blobs:{}};
const BIP=['FP1-F7','F7-T7','T7-P7','P7-O1','FP1-F3','F3-C3','C3-P3','P3-O1','FP2-F4','F4-C4','C4-P4','P4-O2','FP2-F8','F8-T8','T8-P8','P8-O2','FZ-CZ','CZ-PZ'];
const ALIAS={T3:'T7',T4:'T8',T5:'P7',T6:'P8'};
const REFS=new Set(['REF','CS2','LE','AVG','A1','A2','M1','M2','REF1']);
const BANDS=[[0.5,2],[2,4],[4,6],[6,8],[8,10],[10,13],[13,16],[16,20],[20,25],[25,30],[30,38],[38,47],[65,95]];
const NF=16, WSEC=4;
function fftRe(re,im){const n=re.length;for(let i=1,j=0;i<n;i++){let b=n>>1;for(;j&b;b>>=1)j^=b;j^=b;if(i<j){let t=re[i];re[i]=re[j];re[j]=t;t=im[i];im[i]=im[j];im[j]=t;}}
 for(let len=2;len<=n;len<<=1){const a=-2*Math.PI/len,wr=Math.cos(a),wi=Math.sin(a);for(let i=0;i<n;i+=len){let cr=1,ci=0;for(let j=0;j<len/2;j++){const ur=re[i+j],ui=im[i+j],k=i+j+len/2;const vr=re[k]*cr-im[k]*ci,vi=re[k]*ci+im[k]*cr;re[i+j]=ur+vr;im[i+j]=ui+vi;re[k]=ur-vr;im[k]=ui-vi;const nr=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=nr;}}}}
function norm(l){l=l.toUpperCase().replace(/^EEG\s+/,'').trim();let p=l.split('-').filter(s=>s.length);if(p.length==3&&/^\d+$/.test(p[2]))p=p.slice(0,2);p=p.map(s=>ALIAS[s]||s);return p;}
async function processFile(key,r0,r1){
 const url='/'+key.split('/').map(encodeURIComponent).join('/');
 let hdrBytes=null, rec0=0;
 if(r0!==undefined){const h0=new Uint8Array(await (await fetch2(url,{headers:{Range:'bytes=0-255'}})).arrayBuffer());
  const ns0=parseInt(String.fromCharCode(...h0.subarray(252,256)).trim());const dur0=parseFloat(String.fromCharCode(...h0.subarray(244,252)).trim());
  const hb=new Uint8Array(await (await fetch2(url,{headers:{Range:'bytes=0-'+(256*(ns0+1)-1)}})).arrayBuffer());
  let recB=0;for(let i=0;i<ns0;i++){recB+=2*parseInt(String.fromCharCode(...hb.subarray(256+ns0*216+i*8,256+ns0*216+i*8+8)).trim());}
  rec0=Math.floor(r0/dur0);const rec1=Math.ceil(r1/dur0);const b0=256*(ns0+1)+rec0*recB, b1=256*(ns0+1)+rec1*recB-1;
  hdrBytes=hb; var rr=await fetch2(url,{headers:{Range:'bytes='+b0+'-'+b1}}); if(!rr.ok&&rr.status!=206) throw new Error('http '+rr.status);}
 const r=hdrBytes?rr:await fetch2(url); if(!r.ok) throw new Error('http '+r.status);
 const rd=r.body.getReader(); let q=hdrBytes?[hdrBytes]:[],qlen=hdrBytes?hdrBytes.length:0,qoff=0;
 const take=(n)=>{const out=new Uint8Array(n);let o=0;while(o<n){const c=q[0];const k=Math.min(n-o,c.length-qoff);out.set(c.subarray(qoff,qoff+k),o);o+=k;qoff+=k;if(qoff==c.length){q.shift();qoff=0;}}qlen-=n;return out;};
 const need=async(n)=>{while(qlen<n){const {done,value}=await rd.read();if(done)return false;q.push(value);qlen+=value.length;}return true;};
 const asc=(u,a,b)=>String.fromCharCode(...u.subarray(a,b)).trim();
 await need(256);const h=take(256);const ns=parseInt(asc(h,252,256)),ndr=parseInt(asc(h,236,244)),dur=parseFloat(asc(h,244,252));
 const meta={key,rec0,start:asc(h,168,176)+' '+asc(h,176,184),ndr,dur,ns};
 await need(ns*256);const sh=take(ns*256);const F=(o,w,i)=>asc(sh,ns*o+i*w,ns*o+(i+1)*w);
 const sig=[];for(let i=0;i<ns;i++){const s={label:F(0,16,i),pmin:parseFloat(F(104,8,i)),pmax:parseFloat(F(112,8,i)),dmin:parseFloat(F(120,8,i)),dmax:parseFloat(F(128,8,i)),n:parseInt(asc(sh,ns*216+i*8,ns*216+i*8+8))};s.g=(s.pmax-s.pmin)/(s.dmax-s.dmin);s.o=s.pmin-s.dmin*s.g;sig.push(s);}
 meta.labels=sig.map(s=>s.label);
 const nmax=Math.max(...sig.map(s=>s.n));const fs=Math.round(nmax/dur);meta.fs=fs;
 const bip={},ref={};sig.forEach((s,i)=>{if(s.n!=nmax)return;const p=norm(s.label);if(p.length==2&&!REFS.has(p[1])){const k=p[0]+'-'+p[1];if(!(k in bip))bip[k]=i;}else if(p.length>=1&&(p.length==1||REFS.has(p[1]))){if(!(p[0] in ref))ref[p[0]]=i;}});
 const plan=BIP.map(b=>{const [a,c]=b.split('-');if(b in bip)return [bip[b],-1];if(a in ref&&c in ref)return [ref[a],ref[c]];return null;});
 meta.avail=plan.map(p=>p?1:0);
 const W=fs*WSEC;let N=1;while(N<W)N<<=1; const df=fs/N;
 const hann=new Float32Array(W);for(let i=0;i<W;i++)hann[i]=0.5-0.5*Math.cos(2*Math.PI*i/(W-1));const U=hann.reduce((a,b)=>a+b*b,0);
 const bidx=BANDS.map(([lo,hi])=>[Math.ceil(lo/df),Math.min(N/2,Math.floor(hi/df))]);
 const used=[...new Set(plan.filter(p=>p).flat().filter(i=>i>=0))];
 const buf={};used.forEach(i=>buf[i]=new Float32Array(W+nmax*2));let fill=0;
 const offs=[];let o=0;sig.forEach(s=>{offs.push(o);o+=s.n*2;});const recB=o;
 const out=[];const re=new Float64Array(N),im=new Float64Array(N),x=new Float32Array(W);
 let rec=0;
 while(await need(recB)){const rb=take(recB);const dv=new DataView(rb.buffer);
  for(const i of used){const s=sig[i],b=buf[i];for(let k=0;k<s.n;k++)b[fill+k]=dv.getInt16(offs[i]+2*k,true)*s.g+s.o;}
  fill+=nmax;rec++;
  while(fill>=W){const v=new Int16Array(18*NF);
   for(let c=0;c<18;c++){const p=plan[c];if(!p){v.fill(-32768,c*NF,(c+1)*NF);continue;}
    const A=buf[p[0]],B=p[1]>=0?buf[p[1]]:null;let m=0;for(let k=0;k<W;k++){x[k]=B?A[k]-B[k]:A[k];m+=x[k];}m/=W;
    let ll=0,mx=0,fl=0;for(let k=0;k<W;k++){x[k]-=m;if(k){const d=x[k]-x[k-1];ll+=Math.abs(d);if(d===0)fl++;}const a=Math.abs(x[k]);if(a>mx)mx=a;}
    re.fill(0);im.fill(0);for(let k=0;k<W;k++)re[k]=x[k]*hann[k];fftRe(re,im);
    for(let j=0;j<13;j++){const [a,bb]=bidx[j];let s=0;for(let k=a;k<=bb;k++)s+=re[k]*re[k]+im[k]*im[k];s=2*s/(fs*U)/(bb-a+1);v[c*NF+j]=Math.max(-32000,Math.min(32000,Math.round(1000*Math.log10(s+1e-9))));}
    v[c*NF+13]=Math.round(1000*Math.log10(ll/W+1e-6));v[c*NF+14]=Math.round(1000*Math.log10(mx+1e-3));v[c*NF+15]=Math.round(10000*fl/(W-1));}
   out.push(v);
   for(const i of used){buf[i].copyWithin(0,W,fill);} fill-=W;}
  X.prog[key]=rec;X.recs=(X.recs||0)+1;
 }
 meta.nrec=rec;meta.nwin=out.length;
 const js=new TextEncoder().encode(JSON.stringify(meta));const hdr=new Uint32Array([js.length,out.length]);
 const data=new Int16Array(out.length*18*NF);out.forEach((v,i)=>data.set(v,i*18*NF));
 return new Blob([hdr,js,new Uint8Array(4-(js.length%4||4)),data]);
}
window.runGroupX=async function(name,keys,conc=3){X.blobs[name]=[];const parts=new Array(keys.length);let idx=0;X.prog[name]='running';
 async function worker(){while(idx<keys.length){const i=idx++;const k=keys[i];let tries=0;while(true){try{parts[i]=await processFile(k);X.done[k]=1;break;}catch(e){if(++tries>=3){X.err[k]=String(e);parts[i]=null;break;}}}}}
 await Promise.all(Array.from({length:conc},worker));
 const blob=new Blob(parts.filter(p=>p));X.blobs[name]=blob;
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name+'.bin';document.body.appendChild(a);a.click();
 X.prog[name]='saved '+blob.size;};

window.processFile2=processFile;
window.runTasks2=async function(name,tasks,conc=4){const parts=new Array(tasks.length);let idx=0;X.prog[name]='running 0/'+tasks.length;let nd=0;
 async function worker(){while(idx<tasks.length){const i=idx++;const [k,a,b]=tasks[i];let tries=0;while(true){try{parts[i]=await processFile(k,a,b);X.done[k+(a!==undefined?'@'+a:'')]=1;nd++;X.prog[name]='running '+nd+'/'+tasks.length;break;}catch(e){X.err[k]=String(e);if(++tries>=8){parts[i]=null;break;}await new Promise(r=>setTimeout(r,15000*tries));}}}}
 await Promise.all(Array.from({length:conc},worker));
 const blob=new Blob(parts.filter(p=>p));const a=document.createElementNS('http://www.w3.org/1999/xhtml','a');a.href=URL.createObjectURL(blob);a.download=name+'.bin';document.documentElement.appendChild(a);a.click();
 X.prog[name]='saved '+blob.size+' '+parts.filter(p=>p).length+'/'+tasks.length;};})();
