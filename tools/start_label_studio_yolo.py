#!/usr/bin/env python3
"""
Label Studio + YOLO ML Backend Launcher
Starts Label Studio and YOLO backend using current Poetry environment.
Run with: poetry run python tools/start_label_studio_yolo.py
"""

import os
import sys
import subprocess
import time
import requests
import shutil
from pathlib import Path

class LabelStudioLauncher:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.tools_dir = Path(__file__).parent
        self.ml_backend_path = self.tools_dir / "ml-backend"
        self.yolo_backend_path = self.ml_backend_path / "label_studio_ml" / "examples" / "yolo"
        self.backend_marker = self.ml_backend_path / ".installed"

    def run_command(self, command, cwd=None, check=True, quiet=True):
        """Execute shell command"""
        try:
            if quiet:
                result = subprocess.run(
                    command, shell=True, cwd=cwd, check=check,
                    text=True, capture_output=True
                )
                return result
            else:
                result = subprocess.run(command, shell=True, cwd=cwd, check=check)
                return result
        except subprocess.CalledProcessError as e:
            print(f"Error running: {command}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"Details: {e.stderr}")
            if check:
                sys.exit(1)
            return None
    
    def clone_ml_backend(self):
        """Clone ML backend repository"""
        if self.ml_backend_path.exists():
            print(f"ML backend already exists: {self.ml_backend_path}")
            return
        
        print("\nCloning ML backend...")
        self.run_command(
            "git clone --depth 1 https://github.com/HumanSignal/label-studio-ml-backend.git ml-backend",
            cwd=self.tools_dir, quiet=False
        )
        
        if not self.yolo_backend_path.exists():
            print("YOLO example not found in backend!")
            sys.exit(1)
        
        print("ML backend cloned")
    
    def install_ml_backend(self):
        """Install ML backend in development mode"""
        if self.backend_marker.exists():
            print("ML backend already installed")
            return
        
        print("\nInstalling ML backend...")
        self.run_command(f"pip install -e {self.ml_backend_path} --quiet")
        self.backend_marker.touch()
        print("ML backend installed")
    
    def verify_installation(self):
        """Verify all packages are installed"""
        print("\nVerifying installation...")
        
        checks = [
            ("Label Studio", "label_studio"),
            ("Label Studio ML", "label_studio_ml"),
            ("YOLO", "ultralytics"),
            ("PyTorch", "torch"),
        ]
        
        all_ok = True
        for name, module in checks:
            try:
                __import__(module)
                print(f"  {name}: OK")
            except ImportError:
                print(f"  {name}: MISSING")
                all_ok = False
        
        if not all_ok:
            print("\nSome dependencies are missing")
            print("Run: poetry install")
            sys.exit(1)
        
        print("All packages verified")
    
    def setup_pythonpath(self):
        """Set PYTHONPATH for ML backend"""
        import site
        site_packages = site.getsitepackages()[0]
        current_path = os.environ.get('PYTHONPATH', '')
        os.environ['PYTHONPATH'] = f"{site_packages}:{self.ml_backend_path}:{current_path}"
    
    def wait_for_service(self, url, name, timeout=60):
        """Wait for service to become available"""
        print(f"Waiting for {name}", end='', flush=True)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code < 500:
                    print(" OK")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            print(".", end='', flush=True)
            time.sleep(2)
        
        print(f" TIMEOUT ({timeout}s)")
        return False
    
    def start_label_studio(self):
        """Start Label Studio server"""
        print("\nStarting Label Studio...")
        print("URL: http://localhost:8080")
        
        process = subprocess.Popen(
            ["label-studio", "start"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=os.environ.copy()
        )
        
        if self.wait_for_service("http://localhost:8080", "Label Studio", 30):
            return process
        
        print("Failed to start Label Studio")
        try:
            process.kill()
        except:
            pass
        return None
    
    def start_ml_backend(self):
        """Start YOLO ML backend"""
        print("\nStarting YOLO ML Backend...")
        print("URL: http://localhost:9090")
        
        process = subprocess.Popen(
            ["label-studio-ml", "start", "."],
            cwd=self.yolo_backend_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=os.environ.copy()
        )
        
        if self.wait_for_service("http://localhost:9090/health", "ML Backend", 120):
            return process
        
        print("Failed to start ML Backend")
        try:
            process.kill()
        except:
            pass
        return None
    
    def print_success(self):
        """Print success message"""
        print(f"""
{'='*60}
All services running!
{'='*60}

Services:
  Label Studio:  http://localhost:8080
  ML Backend:    http://localhost:9090

Next steps:
  1. Open http://localhost:8080 in browser
  2. Create account (email + password)
  3. Create project: Object Detection
  4. Upload images/frames
  5. Settings > Machine Learning > Add Model
     URL: http://localhost:9090
  6. Enable "Use for interactive preannotation"
  7. Click "Predict" for auto-detection

Press Ctrl+C to stop
{'='*60}
        """)
    
    def cleanup(self, ls_proc, ml_proc):
        """Stop services"""
        print("\n\nStopping services...")
        
        for name, proc in [("Label Studio", ls_proc), ("ML Backend", ml_proc)]:
            if proc:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                    print(f"{name} stopped")
                except:
                    try:
                        proc.kill()
                    except:
                        pass
        
        print("Done")
    
    def run(self):
        """Main function"""
        print("Label Studio + YOLO ML Backend Launcher\n")
        
        # Setup
        self.clone_ml_backend()
        self.install_ml_backend()
        self.verify_installation()
        self.setup_pythonpath()
        
        # Start services
        ls_proc = self.start_label_studio()
        if not ls_proc:
            return 1
        
        ml_proc = self.start_ml_backend()
        if not ml_proc:
            self.cleanup(ls_proc, None)
            return 1
        
        # Success
        self.print_success()
        
        try:
            ls_proc.wait()
        except KeyboardInterrupt:
            self.cleanup(ls_proc, ml_proc)
        
        return 0


def main():
    """Entry point for poetry script"""
    launcher = LabelStudioLauncher()
    exit_code = launcher.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
