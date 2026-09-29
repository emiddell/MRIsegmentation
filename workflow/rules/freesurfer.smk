"""FreeSurfer path: recon-all + mkheadsurf, exported into the NIfTI voxel grid.

Included by workflow/Snakefile when config freesurfer.enabled; uses its globals.
Runs in its own container image (config freesurfer.image), in parallel to the
MATLAB steps. Layout:
    results/<subject>/fs/<BASE>/          FreeSurfer subject dir (SUBJECTS_DIR = fs/)
    results/<subject>/fs/*.obj            surfaces in 1-based voxel indices of the
                                          staged NIfTI (see scripts/fs_surfaces_to_voxel.py)
"""

import struct
import sys

FS = config["freesurfer"]
FS_SD = f"{D}/fs"  # SUBJECTS_DIR
FS_SUBJ = f"{FS_SD}/{BASE}"
FS_EXPORTS = ["scalp", "lh.white", "rh.white", "lh.pial", "rh.pial", "cortex_white", "cortex_pial"]
FS_SMOOTH = int(FS.get("scalp_smooth_iterations", 20))
if FS_SMOOTH > 0:
    FS_EXPORTS.append("scalp_smoothed")


def fs_runner():
    lic = FS.get("license") or ""
    if not lic:
        raise ValueError("config freesurfer.license: path to the FreeSurfer license.txt is required")
    scripts = os.path.join(workflow.basedir, "scripts")
    binds = [(A(RES), A(RES)), (A(lic), A(lic)), (scripts, scripts)]
    env = [("FS_LICENSE", A(lic))]
    return runner(FS["image"], binds, env)


# recon-all & co. rely on the environment SetUpFreeSurfer.sh exports
FSRUN = fs_runner() + ' bash -c \'source "$FREESURFER_HOME/SetUpFreeSurfer.sh" >/dev/null 2>&1; '
FS_SCRIPT = os.path.join(workflow.basedir, "scripts", "fs_surfaces_to_voxel.py")


def nifti_fov_mm(path):
    """Largest field of view (mm) of a NIfTI-1 image, from the header only."""
    with open(path, "rb") as fh:
        hdr = fh.read(348)
    endian = "<" if struct.unpack("<i", hdr[:4])[0] == 348 else ">"
    dim = struct.unpack(endian + "8h", hdr[40:56])
    pixdim = struct.unpack(endian + "8f", hdr[76:108])
    return max(abs(dim[i] * pixdim[i]) for i in range(1, 4))


rule recon_all:
    """FreeSurfer recon-all (all stages), T2 for pial refinement when available."""
    input:
        img=IMG,
        t2=t2_input(IMG_T2),
    output:
        done=f"{FS_SUBJ}/scripts/recon-all.done",
        t1=f"{FS_SUBJ}/mri/T1.mgz",
        surfs=expand(f"{FS_SUBJ}/surf/{{h}}.{{s}}", h=["lh", "rh"], s=["white", "pial", "sphere.reg"], allow_missing=True),
    log:
        f"{LOGS}/recon_all.log",
    params:
        sd=lambda wc: A(FS_SD.format(subject=wc.subject)),
        subj=lambda wc: BASE.format(subject=wc.subject),
        t2=lambda wc, input: f"-T2 {A(input.t2[0])} -T2pial" if input.t2 and FS.get("use_t2", True) else "",
        # recon-all refuses fields of view above 256 mm unless told to crop
        cw256=lambda wc, input: "-cw256" if nifti_fov_mm(input.img) > 256 else "",
        extra=FS.get("recon_all_args", "") or "",
    threads: int(FS.get("threads", 8))
    resources:
        mem_mb=int(FS.get("mem_mb", 12000)),
    shell:
        # recon-all -i refuses to start if the subject directory already exists
        "rm -rf {params.sd}/{params.subj} && mkdir -p {params.sd} && "
        + FSRUN
        + "recon-all -sd {params.sd} -s {params.subj} -i $(realpath {input.img}) {params.t2} {params.cw256} "
        "-all -parallel -openmp {threads} {params.extra}' > {log} 2>&1"


rule head_surface:
    """Scalp surface from the conformed T1 (mkheadsurf), cleaned of thin artefact sheets, optionally smoothed."""
    input:
        done=f"{FS_SUBJ}/scripts/recon-all.done",
        t1=f"{FS_SUBJ}/mri/T1.mgz",
    output:
        f"{FS_SUBJ}/surf/lh.seghead",
        f"{FS_SUBJ}/surf/lh.smseghead" if FS_SMOOTH > 0 else [],
        mask=f"{FS_SUBJ}/mri/seghead.mgz",
        raw=f"{FS_SUBJ}/mri/seghead.orig.mgz",
    log:
        f"{LOGS}/head_surface.log",
    params:
        sd=lambda wc: A(FS_SD.format(subject=wc.subject)),
        subj=lambda wc: BASE.format(subject=wc.subject),
        clean=os.path.join(workflow.basedir, "scripts", "fs_clean_headmask.py"),
        radius=int(FS.get("scalp_open_radius", 1)),
        prior=lambda wc, output: (
            ""
            if FS.get("scalp_margin_mm") is None
            else " --talairach {0}/mri/transforms/talairach.xfm --t1 {0}/mri/T1.mgz --margin {1} --dim-below {2}".format(
                os.path.dirname(os.path.dirname(A(output.mask))),
                FS.get("scalp_margin_mm", 2),
                FS.get("scalp_dim_below", 70),
            )
        ),
        smooth=lambda wc, output: (
            f" && mris_smooth -nw -n {FS_SMOOTH} {A(output[0])} {A(output[1])}" if FS_SMOOTH > 0 else ""
        ),
    threads: 1
    resources:
        mem_mb=4000,
    shell:
        # mkheadsurf = mri_seghead + mri_mc + mris_smooth; the head mask is cleaned
        # in between and the surface re-tessellated the same way mkheadsurf does
        FSRUN
        + "export SUBJECTS_DIR={params.sd}; mkheadsurf -subjid {params.subj} -srcvol T1.mgz"
        " && cp {output.mask} {output.raw}"
        " && fspython {params.clean} {output.raw} {output.mask} --radius {params.radius}{params.prior}"
        " && mri_mc {output.mask} 1 {output[0]} && mris_smooth -n 10 -nw {output[0]} {output[0]}"
        "{params.smooth}' > {log} 2>&1"


rule fs_export:
    """Scalp and cortical surfaces as OBJ in 1-based voxel indices of the staged NIfTI."""
    input:
        img=IMG,
        t1=f"{FS_SUBJ}/mri/T1.mgz",
        surfs=expand(f"{FS_SUBJ}/surf/{{h}}.{{s}}", h=["lh", "rh"], s=["white", "pial"], allow_missing=True),
        scalp=f"{FS_SUBJ}/surf/lh.seghead",
        scalp_smoothed=f"{FS_SUBJ}/surf/lh.smseghead" if FS_SMOOTH > 0 else [],
    output:
        expand(f"{FS_SD}/{{name}}.obj", name=FS_EXPORTS, allow_missing=True),
    log:
        f"{LOGS}/fs_export.log",
    params:
        script=FS_SCRIPT,
        out=lambda wc: A(FS_SD.format(subject=wc.subject)),
        specs=lambda wc, input: " ".join(
            [f"scalp={A(input.scalp)}"]
            + ([f"scalp_smoothed={A(str(input.scalp_smoothed))}"] if input.scalp_smoothed else [])
            + [f"{os.path.basename(s)}={A(s)}" for s in input.surfs]
        ),
    threads: 1
    resources:
        mem_mb=2000,
    shell:
        # fspython ships numpy + nibabel, so the export runs in the same image
        fs_runner()
        + " fspython {params.script} $(realpath {input.img}) $(realpath {input.t1}) {params.out} {params.specs} > {log} 2>&1"


rule compare_surfaces:
    """QC: exact cross-sections + distances of MATLAB vs FreeSurfer scalp and cortex."""
    input:
        img=IMG,
        m_scalp=f"{D}/mask_scalp.obj",
        m_cortex=f"{D}/mask_brain.obj",
        f_scalp=f"{FS_SD}/scalp{'_smoothed' if FS_SMOOTH > 0 else ''}.obj",
        f_pial=f"{FS_SD}/cortex_pial.obj",
        f_white=f"{FS_SD}/cortex_white.obj",
    output:
        png=f"{RES}/qc/{{subject}}/surfaces_matlab_vs_freesurfer.png",
        json=f"{RES}/qc/{{subject}}/surfaces_matlab_vs_freesurfer.json",
    log:
        f"{LOGS}/compare_surfaces.log",
    params:
        script=os.path.join(workflow.basedir, "scripts", "compare_surfaces.py"),
        python=sys.executable,
    localrule: True
    shell:
        # runs with Snakemake's own interpreter: needs numpy, scipy, nibabel, matplotlib
        "{params.python} {params.script} {input.img} {output.png} {output.json} "
        "'MATLAB scalp=scalp:lime:{input.m_scalp}' 'MATLAB cortex (CAT12 central)=cortex:magenta:{input.m_cortex}' "
        "'FreeSurfer scalp=scalp:cyan:{input.f_scalp}' 'FreeSurfer pial=cortex:red:{input.f_pial}' "
        "'FreeSurfer white=cortex:yellow:{input.f_white}' > {log} 2>&1"
