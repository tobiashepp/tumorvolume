import argparse
import subprocess
import nibabel as nib
from pathlib import Path

def nifti2nora(directory=None, nii_filepath=None, array=None, affine=None, fname=None,
               dname = "SHOW_NII_0000000000", sid = "0000000000_20200101", project = "INBOX"):
    files_to_add = ":\n"

    inbox_dir_str = "/media/nora/imgdata/" + project + "/" + dname
    inbox_dir_name = Path(inbox_dir_str)
    inbox_dir_name.mkdir(exist_ok=True)
    inbox_dir_name_sid = inbox_dir_name/sid
    inbox_dir_name_sid.mkdir(exist_ok=True)

    if directory:
        msg = "copying all .nii from directory " + directory + " to the database"
        subprocess.call(["echo", msg])
        work_dir = Path(directory)
        for file in work_dir.glob("**/*.nii*"):
            files_to_add += str(file)
            files_to_add += "\n"
            dst = str(inbox_dir_name_sid/file.name)
            subprocess.call(["rsync", "-vau", file, dst])

    if nii_filepath:
        msg = "copying a .nii from file " + nii_filepath + " to the database"
        files_to_add += str(nii_filepath)
        files_to_add += "\n"
        subprocess.call(["echo", msg])
        dst = str(inbox_dir_name_sid/Path(nii_filepath).name)
        subprocess.call(["rsync", "-vau", nii_filepath, dst])

    if array and affine and fname:
        img = nib.Nifti1Image(array, affine)
        files_to_add += str(fname)
        dst = str(inbox_dir_name_sid/Path(fname))
        nib.save(img, dst)

    msg = "adding files to project" + files_to_add
    subprocess.call(["echo", msg])
    subprocess.call(["nora", "-p", project, "--add", inbox_dir_name_sid])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='directory containing .nii')
    parser.add_argument('--nii', help='.nii filepath')
    parser.add_argument('--arr', help='array to construct a .nii')
    parser.add_argument('--aff', help='affine to construct a .nii')
    parser.add_argument('--fname', help='filename to save constructed .nii')
    parser.add_argument('--dname', help='dummy name for nora: name_name_id. Default: KerberB_Test_0000000000')
    parser.add_argument('--sid', help='dummy study-id for nora: id_yyyymmdd. Default: 00000000000_20200101')
    parser.add_argument('--prj', help='nora-directory to upload to. Default: INBOX')


    args = parser.parse_args()
    if args.dir:
        directory = args.dir
    else:
        directory = None

    if args.nii:
        nii_filepath = args.nii
    else:
        nii_filepath= None

    if args.arr and args.aff and args.fname:
        array = args.arr
        affine=args.aff
        fname=args.fname
    else:
        array = None
        affine = None
        fname = None

    if args.dname:
        dname=args.dname
    else:
        dname = "SHOW_NII_0000000000"

    if args.sid:
        sid=args.sid
    else:
        sid = "0000000000_20200101"

    if args.prj:
        project=args.prj
    else:
        project = "INBOX"

    nifti2nora(directory=directory, nii_filepath=nii_filepath, array=array, affine=affine, fname=fname,
               dname=dname, sid=sid, project=project)

if __name__ == '__main__':
    main()