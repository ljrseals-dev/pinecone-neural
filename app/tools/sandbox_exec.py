"""
Real sandbox via Docker + gVisor. Only use if you must execute
arbitrary model-generated code. Prefer the tool registry otherwise.
"""
import asyncio
import os
import tempfile


async def run_in_sandbox(
    code: str,
    timeout_s: float = 5.0,
    memory_mb: int = 256,
    image: str = "python:3.11-slim",
    runtime: str = "runsc",
) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        script = os.path.join(tmpdir, "script.py")
        with open(script, "w") as f:
            f.write(code)

        cmd = [
            "docker", "run", "--rm",
            "--runtime", runtime,
            "--network", "none",
            "--memory", f"{memory_mb}m",
            "--cpus", "0.5",
            "--pids-limit", "64",
            "--read-only",
            "-v", f"{script}:/script.py:ro",
            image,
            "python", "/script.py",
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(),
                                                  timeout=timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                return "SandboxError: timeout"
            if proc.returncode != 0:
                return f"SandboxError: exit {proc.returncode}: {err.decode()[:512]}"
            return out.decode()[:4096]
        except FileNotFoundError:
            return "SandboxError: docker not available"
