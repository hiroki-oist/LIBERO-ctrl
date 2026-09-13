"""The LIBERO env wrapper (Task) and the deterministic env seed.

Everything needed at run time lives inside the package, so that nothing depends on sys.path
manipulation.
"""
import os, numpy as np, torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
UP=np.array([0.,0.,1.])

def quat2mat(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def lookat_quat(p,t,up=UP):
    f=t-p; f/=np.linalg.norm(f); zc=-f
    xc=np.cross(up,zc); xc/=np.linalg.norm(xc); yc=np.cross(zc,xc)
    R=np.stack([xc,yc,zc],1); w=np.sqrt(max(1e-12,1+R[0,0]+R[1,1]+R[2,2]))/2
    return np.array([w,(R[2,1]-R[1,2])/(4*w),(R[0,2]-R[2,0])/(4*w),(R[1,0]-R[0,1])/(4*w)])
def rot_axis(ax,a):
    ax=ax/np.linalg.norm(ax); c,s=np.cos(a),np.sin(a)
    K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3)*c+s*K+(1-c)*np.outer(ax,ax)

def light_color(warm, tint):
    """Light colour, as an RGB gain that preserves Rec.709 luminance.

    warm : R x2^warm, B x2^-warm      (positive = warmer, negative = cooler)
    tint : G x2^tint, R,B x2^(-tint/2) (positive = green, negative = magenta)
    """
    c = np.array([2.0 ** warm, 1.0, 2.0 ** -warm], float)
    c = c * np.array([2.0 ** (-tint / 2), 2.0 ** tint, 2.0 ** (-tint / 2)])
    return c / (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])

ENV_SEED = 20260904      # pins the placement sampler; must match manifest.protocol.env_seed

def seed_env(seed=ENV_SEED):
    """Call immediately *before* constructing an env, to make fixture placement reproducible.

    LIBERO's 92-dimensional sim state does not contain the placement of static bodies. It holds
    qpos/qvel for the movable objects only; shelves, stoves and cabinets are positioned by
    robosuite's placement sampler when the env is constructed, and that sampler is not seeded.
    The same task with the same init_id therefore places the fixtures differently on every run,
    and set_init_state does not restore them, because they are not in the state.

    Nothing about this benchmark reproduces if this is not pinned.
    """
    import random
    random.seed(seed); np.random.seed(seed)
    return seed

class Task:
    def __init__(self, suite, ti, H=256, W=256, seed=ENV_SEED):
        if seed is not None: seed_env(seed)
        self.seed=seed
        bm=benchmark.get_benchmark_dict()[suite]()
        self.task=bm.get_task(ti); self.ti=ti; self.H,self.W=H,W
        self.suite_name=suite
        bddl=os.path.join(get_libero_path("bddl_files"),self.task.problem_folder,self.task.bddl_file)
        self.env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=H,camera_widths=W)
        self.env.reset(); self.sim=self.env.env.sim; m=self.m=self.sim.model
        self.S=np.asarray(torch.load(os.path.join(get_libero_path("init_states"),
            self.task.problem_folder,self.task.init_states_file),weights_only=False))
        self.CID=m.camera_name2id("agentview")
        self.P0,self.Q0=m.cam_pos[self.CID].copy(),m.cam_quat[self.CID].copy()
        self.LP0,self.LD0=m.light_pos.copy(),m.light_dir.copy()
        self.LDF0,self.LSP0=m.light_diffuse.copy(),m.light_specular.copy()
        self.AMB0=np.array(m.vis.headlight.ambient).copy()
        self.CS0=m.light_castshadow.copy()
        self._body_pos0=m.body_pos.copy()
        jn={m.joint_id2name(j) for j in range(m.njnt)}
        # every free object (for collision checks) and obj_of_interest (what gets perturbed)
        self.free_objs=[(m.joint_id2name(j).rsplit("_joint0",1)[0], m.jnt_qposadr[j]+1)
                        for j in range(m.njnt) if m.jnt_type[j]==0]
        self.tgt_objs=[(nm, m.jnt_qposadr[m.joint_name2id(f"{nm}_joint0")]+1)
                       for nm in self.env.obj_of_interest if f"{nm}_joint0" in jn]

        # Look-at point: the point on the optical axis closest to the centroid of the movable
        # objects. A fixed ground plane cannot be used instead, because table height differs
        # per arena and the intersection can fall behind the camera.
        f0=-quat2mat(self.Q0)[:,2]
        self.env.set_init_state(self.S[0]); self.sim.forward()
        cs=[]
        for nm,c in self.free_objs:
            bid=None
            for cand in (f"{nm}_main", nm):
                try: bid=m.body_name2id(cand); break
                except Exception: pass
            if bid is None:
                try: bid=int(m.jnt_bodyid[m.joint_name2id(f"{nm}_joint0")])
                except Exception: pass
            if bid is not None: cs.append(self.sim.data.xipos[bid].copy())
        ctr=np.mean(cs,axis=0) if cs else self.P0+f0
        d0=float(np.dot(ctr-self.P0,f0))
        if not np.isfinite(d0) or d0<0.2: d0=1.1          # fallback for degenerate geometry
        self.L0=self.P0+d0*f0
        u0=self.P0-self.L0; self.D0=np.linalg.norm(u0); u0=u0/self.D0
        self.AZ0,self.EL0=np.arctan2(u0[1],u0[0]),np.arcsin(u0[2])
        self.JR=m.jnt_range[:7].copy()

    def reset_model(self):
        m=self.m
        m.cam_pos[self.CID],m.cam_quat[self.CID]=self.P0,self.Q0
        m.light_pos[:],m.light_dir[:]=self.LP0,self.LD0
        m.light_diffuse[:],m.light_specular[:]=self.LDF0,self.LSP0
        m.vis.headlight.ambient[:]=self.AMB0
        m.light_castshadow[:]=self.CS0
        m.body_pos[:]=self._body_pos0

    def set_camera(self,d):
        az,el,dist,lx,ly=d; m=self.m
        L=self.L0+np.array([lx,ly,0.])
        a,e,dd=self.AZ0+np.radians(az),self.EL0+np.radians(el),self.D0+dist
        p=L+dd*np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])
        m.cam_pos[self.CID]=p; m.cam_quat[self.CID]=lookat_quat(p,L)

    def set_light5(self, d, shadow=False):
        """The lighting axis, five dimensions.

        d = (d_intensity_ev, d_log2_warm, d_tint_g, d_log2_ambient, d_elevation_deg)

        Shadows are not cast: LIBERO's default castshadow=false is kept. Enabling shadows makes
        MuJoCo stop applying the spotlight's cone falloff (cutoff 45 deg), and in scenes with a
        wide table that overflows the cone (libero_10 tasks 2, 3, 8, 9) exposure jumps by 1.28x
        and half the frame blows out. This is presumably why LIBERO disables them too.

        With no shadows the light's *azimuth* barely affects the image, so it is not part of the
        axis: spending one of five spherical dimensions on a parameter with no effect would
        dilute the others.

        Colour is parameterised as a symmetric logarithmic gain. Following the blackbody locus
        (6500K * 2^d) instead is asymmetric: pushed hard, the warm side saturates to pure red
        (1,0,0) while the cool side barely moves.
          warm : R x2^d,  B x2^-d          (R/B ratio = 2^(2d); 16x at L3)
          tint : G x2^d,  R,B x2^(-d/2)    (green / magenta; G x2.14 at L3)
        Both are normalised to preserve Rec.709 luminance, so only the colour changes.
        """
        ev, warm, tint, damb, el = d
        m = self.m
        for i in range(m.nlight):
            p = self.LP0[i].copy(); dv = self.LD0[i].copy()
            if abs(el) > 1e-12:
                rel = p - self.L0; n = np.linalg.norm(rel[:2])
                ax = np.array([-rel[1], rel[0], 0.]) / n if n > 1e-9 else np.array([0., 1., 0.])
                Re = rot_axis(ax, np.radians(el)); p = Re @ (p - self.L0) + self.L0; dv = Re @ dv
            m.light_pos[i] = p; m.light_dir[i] = dv / np.linalg.norm(dv)
            m.light_castshadow[i] = 1 if shadow else 0
        c = light_color(warm, tint)
        for i in range(m.nlight):
            m.light_diffuse[i] = self.LDF0[i] * (2. ** ev) * c
            m.light_specular[i] = self.LSP0[i] * (2. ** ev)
        m.vis.headlight.ambient[:] = np.clip(self.AMB0 * (2. ** damb), 0, 1)

    def close(self): self.env.close()
