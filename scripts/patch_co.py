import os
p = os.environ["SCRATCH"]+"/SolarWM/src/solarwm/runtime/create_only.py"
s = open(p).read()
old = '''    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        _raise(error_type, f"{label} target appeared during publication: {destination}")
    _raise(error_type, f"{label} create-only publication failed: {os.strerror(error)}")'''
new = '''    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        _raise(error_type, f"{label} target appeared during publication: {destination}")
    if error == errno.EINVAL:
        # [hskim99 patch] Lustre ($SCRATCH on TACC Vista) does not support RENAME_NOREPLACE
        # (renameat2 returns EINVAL). Fall back to a check-then-rename. Single-writer use only.
        if os.path.lexists(destination):
            _raise(error_type, f"{label} target appeared during publication: {destination}")
        os.rename(source, destination)
        return
    _raise(error_type, f"{label} create-only publication failed: {os.strerror(error)}")'''
assert old in s; open(p, "w").write(s.replace(old, new))
import py_compile; py_compile.compile(p, doraise=True); print("create_only.py 패치 + 컴파일 OK")
