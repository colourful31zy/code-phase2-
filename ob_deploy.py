"""
OpenBayes 部署脚本：上传文件、创建软链接、启动训练
Usage: python code/phase2/ob_deploy.py [upload|run|check|fix]
"""
import paramiko, os, sys, time

HOST = 'ssh.openbayes.com'
PORT = 30172
USER = 'root'
PASS = '7jo0g0d9sys'

LOCAL_CODE = 'e:/实验/code/phase2'
LOCAL_SPLIT = 'e:/实验/data/02_processed/phase2/split'
LOCAL_RESULTS = 'e:/实验/data/02_processed/phase2/results/evaluation'

REMOTE_BASE = '/project'


def get_ssh():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
    return ssh


def run(ssh, cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    return out, err


def upload():
    """Upload code, split data, and results to OpenBayes."""
    transport = paramiko.Transport((HOST, PORT))
    transport.connect(username=USER, password=PASS)
    sftp = paramiko.SFTPClient.from_transport(transport)

    # Upload code files
    print("Uploading code...")
    sftp.mkdir(f'{REMOTE_BASE}/code', mode=0o755)
    for fn in ['05_stage2_ddi.py', 'config.py', 'utils.py']:
        local = os.path.join(LOCAL_CODE, fn)
        remote = f'{REMOTE_BASE}/code/{fn}'
        sftp.put(local, remote)
        print(f"  {fn} ({os.path.getsize(local)//1024}KB)")

    # Upload data
    print("Uploading data...")
    _mkdir_p(sftp, f'{REMOTE_BASE}/data/02_processed/phase2/split')
    for fn in os.listdir(LOCAL_SPLIT):
        local = os.path.join(LOCAL_SPLIT, fn)
        remote = f'{REMOTE_BASE}/data/02_processed/phase2/split/{fn}'
        size_mb = os.path.getsize(local) // 1024 // 1024
        print(f"  {fn} ({size_mb}MB)...", end=' ', flush=True)
        sftp.put(local, remote)
        print("OK")

    # Upload best_per_endpoint.csv
    _mkdir_p(sftp, f'{REMOTE_BASE}/data/02_processed/phase2/results/evaluation')
    src = os.path.join(LOCAL_RESULTS, 'best_per_endpoint.csv')
    dst = f'{REMOTE_BASE}/data/02_processed/phase2/results/evaluation/best_per_endpoint.csv'
    sftp.put(src, dst)
    print(f"  best_per_endpoint.csv OK")

    sftp.close()
    transport.close()
    print("Upload complete!")


def fix():
    """Fix symlinks on OpenBayes so config paths work."""
    ssh = get_ssh()
    out, err = run(ssh, 'ln -sf /project/data /data && ls -la /data | head -3')
    print(out[:500])
    if err: print('ERR:', err[:300])
    # Verify
    out2, err2 = run(ssh, 'ls /data/02_processed/phase2/split/split_meta.json')
    print("Verify:", out2[:200])
    if err2: print("ERR:", err2[:200])
    ssh.close()
    print("Symlink fix done!")


def run_training():
    """Start the training script in background."""
    ssh = get_ssh()
    # Kill any existing run
    run(ssh, 'pkill -f 05_stage2_ddi.py 2>/dev/null; sleep 1')
    # Start new
    cmd = f'cd {REMOTE_BASE} && nohup python code/05_stage2_ddi.py > stage2_output.log 2>&1 & echo PID=$!'
    out, err = run(ssh, cmd)
    print("Start output:", out)
    if err: print('ERR:', err[:300])
    time.sleep(3)
    # Check process
    out2, err2 = run(ssh, 'ps aux | grep stage2 | grep -v grep')
    print("Process:", out2[:200])
    # Show initial log
    out3, err3 = run(ssh, 'head -30 /project/stage2_output.log')
    print("=== Log ===")
    print(out3[:2000])
    if err3: print('ERR:', err3[:200])
    ssh.close()


def check():
    """Check training status."""
    ssh = get_ssh()
    out, err = run(ssh, 'ps aux | grep stage2 | grep -v grep')
    if out.strip():
        print("Running:", out[:200])
    else:
        print("Not running")
    out2, err2 = run(ssh, 'tail -30 /project/stage2_output.log 2>/dev/null || echo "No log"')
    print("=== Last output ===")
    print(out2[:1000])
    ssh.close()


def _mkdir_p(sftp, path):
    """Recursive mkdir via SFTP."""
    parts = path.strip('/').split('/')
    for i in range(1, len(parts) + 1):
        sub = '/' + '/'.join(parts[:i])
        try:
            sftp.stat(sub)
        except FileNotFoundError:
            sftp.mkdir(sub, mode=0o755)


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if action == 'upload':
        upload()
    elif action == 'fix':
        fix()
    elif action == 'run':
        run_training()
    elif action == 'check':
        check()
    elif action == 'all':
        upload()
        fix()
        run_training()
    else:
        print(f"Unknown action: {action}")
        print("Usage: python ob_deploy.py [upload|fix|run|check|all]")