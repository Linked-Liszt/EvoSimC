# Docker SimC execution
from typing import List, Optional
import docker
import docker.errors
import tempfile
import os
import re
import json
import evosim.simc.simc_templates as simc_templates
from ..entities import SimCResult

class SimCRunner:
    """All-in-one SimC runner for APL fitness evaluation"""
    
    def __init__(self, image_name: str = "simulationcraftorg/simc", base_profile: str = None):
        """Initialize SimC runner with static character profile"""
        self.docker_client = docker.from_env()
        self.image_name = image_name
        self.base_profile = base_profile or self._default_profile()
        
        # Ensure Docker image is available
        try:
            self.docker_client.images.get(self.image_name)
        except docker.errors.ImageNotFound:
            print(f"Pulling Docker image: {self.image_name}")
            self.docker_client.images.pull(self.image_name)
    
    def evaluate_apl(self, apl: str, iterations: int = 1000, fight_length: int = 300) -> SimCResult:
        """Evaluate APL and return complete SimC results"""
        try:
            result = self._run_simulation(apl, iterations, fight_length)
            print(result.errors)
            return result
        except Exception as e:
            print(f"Simulation error: {e}")
            return SimCResult(
                dps=0.0,
                raw_output="",
                errors=[f"Simulation error: {str(e)}"],
                is_valid=False
            )
    
    def cleanup(self) -> None:
        """Clean up Docker resources"""
        try:
            self.docker_client.close()
        except Exception:
            pass
    
    def _run_simulation(self, apl: str, iterations: int, fight_length: int) -> SimCResult:
        """Internal method to execute SimC simulation"""
        # Create complete SimC profile
        full_profile = f"{self.base_profile}\n\n{apl}\n"
        full_profile += f"\niterations={iterations}\nfight_style=Patchwerk\nmax_time={fight_length}\n"
        full_profile += "json2=/app/SimulationCraft/output/report.json\n"
        
        # Create temporary file for profile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.simc', delete=False) as f:
            f.write(full_profile)
            profile_path = f.name
        
        # Create temporary directory for output files
        with tempfile.TemporaryDirectory() as temp_output_dir:
            try:
                # Mount specific file as profile.simc and output directory
                container = self.docker_client.containers.run(
                    self.image_name,
                    'profile.simc',  # Fixed filename that container expects
                    volumes={
                        profile_path: {'bind': '/app/SimulationCraft/profile.simc', 'mode': 'ro'},
                        temp_output_dir: {'bind': '/app/SimulationCraft/output', 'mode': 'rw'}
                    },
                    remove=True,
                    stdout=True,
                    stderr=True,
                    stream=False
                )
                
                # container is the output when remove=True and stream=False
                if isinstance(container, bytes):
                    stdout = container.decode('utf-8')
                    stderr = ""
                else:
                    stdout = container
                    stderr = ""
                
                # Try to read the JSON report file
                json_report_path = os.path.join(temp_output_dir, 'report.json')
                json_report_data = None
                if os.path.exists(json_report_path):
                    try:
                        with open(json_report_path, 'r') as json_file:
                            json_report_data = json.load(json_file)
                    except (IOError, json.JSONDecodeError):
                        pass
                
                # Parse results
                dps = self._parse_dps(json_report_data)
                errors = [stderr] if stderr.strip() else []
                
                return SimCResult(
                    dps=dps,
                    raw_output=stdout,
                    errors=errors,
                    is_valid=True
                )
                
            except docker.errors.ContainerError as e:
                # Container exited with non-zero code (SimC error)
                error_msg = f"SimC container failed with exit code {e.exit_status}"
                if hasattr(e, 'stderr') and e.stderr:
                    error_msg += f": {e.stderr.decode('utf-8')}"
                
                return SimCResult(
                    dps=0.0,
                    raw_output=e.stdout.decode('utf-8') if hasattr(e, 'stdout') and e.stdout else "",
                    errors=[error_msg],
                    is_valid=False
                )
            except Exception as e:
                # Other Docker/system errors
                return SimCResult(
                    dps=0.0,
                    raw_output="",
                    errors=[f"Docker error: {str(e)}"],
                    is_valid=False
                )
            finally:
                # Clean up temp file
                try:
                    os.unlink(profile_path)
                except Exception:
                    pass
    
    def _parse_dps(self, json_report_data: dict = None) -> float:
        """Extract DPS from SimC output or JSON report"""
        try:
            if json_report_data:
                # Navigate JSON structure to find DPS from report.json
                if 'sim' in json_report_data and 'players' in json_report_data['sim']:
                    players = json_report_data['sim']['players']
                    if players and 'collected_data' in players[0]:
                        return float(players[0]['collected_data']['dps']['mean'])
            
        except: 
            pass
        
        return 0.0
    
    
    def _default_profile(self) -> str:
        """Default static character profile"""
        return simc_templates.evoker_template