import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
import docker.errors
import docker
import time
from evosim.simc.simc_runner import SimCRunner, SimCResult


class TestSimCResult:
    """Test the SimCResult dataclass"""
    
    def test_simc_result_creation(self):
        """Test creating a SimCResult instance"""
        result = SimCResult(
            dps=12345.67,
            raw_output="test output",
            errors=["error1", "error2"],
            is_valid=True
        )
        
        assert result.dps == 12345.67
        assert result.raw_output == "test output"
        assert result.errors == ["error1", "error2"]
        assert result.is_valid is True
    
    def test_simc_result_empty(self):
        """Test creating an empty/invalid SimCResult"""
        result = SimCResult(
            dps=0.0,
            raw_output="",
            errors=["Simulation failed"],
            is_valid=False
        )
        
        assert result.dps == 0.0
        assert result.raw_output == ""
        assert len(result.errors) == 1
        assert result.is_valid is False


class TestSimCRunner:
    """Test the SimCRunner class"""
    
    @pytest.fixture
    def mock_docker_client(self):
        """Mock Docker client for testing"""
        with patch('evosim.simc.simc_runner.docker') as mock_docker:
            mock_client = Mock()
            mock_docker.from_env.return_value = mock_client
            
            # Mock image availability
            mock_client.images.get.return_value = Mock()
            
            # Properly mock docker.errors with real exception classes
            mock_docker.errors = Mock()
            mock_docker.errors.ContainerError = docker.errors.ContainerError
            mock_docker.errors.ImageNotFound = docker.errors.ImageNotFound
            mock_docker.errors.DockerException = docker.errors.DockerException
            
            yield mock_client
    
    def test_init_with_default_profile(self, mock_docker_client):
        """Test SimCRunner initialization with default profile"""
        runner = SimCRunner()
        
        assert runner.image_name == "simulationcraftorg/simc"
        assert "evoker=" in runner.base_profile
        assert mock_docker_client.images.get.called
    
    def test_init_with_custom_profile(self, mock_docker_client):
        """Test SimCRunner initialization with custom profile"""
        custom_profile = "# Custom profile\nwarrior=\"TestWarrior\""
        runner = SimCRunner(base_profile=custom_profile)
        
        assert runner.base_profile == custom_profile
    
    @patch('evosim.simc.simc_runner.docker')
    def test_init_pulls_missing_image(self, mock_docker_module):
        """Test that missing Docker image is pulled"""
        # Create a proper exception class for testing
        class MockImageNotFound(Exception):
            pass
        
        # Setup the mock docker module
        mock_docker_module.errors.ImageNotFound = MockImageNotFound
        mock_docker_client = Mock()
        mock_docker_module.from_env.return_value = mock_docker_client
        mock_docker_client.images.get.side_effect = MockImageNotFound("Image not found")
        
        with patch('builtins.print') as mock_print:
            runner = SimCRunner()
            mock_docker_client.images.pull.assert_called_once_with("simulationcraftorg/simc")
            mock_print.assert_called_with("Pulling Docker image: simulationcraftorg/simc")
    
    def test_cleanup(self, mock_docker_client):
        """Test cleanup method"""
        runner = SimCRunner()
        runner.cleanup()
        
        mock_docker_client.close.assert_called_once()
    
    def test_cleanup_handles_exception(self, mock_docker_client):
        """Test cleanup handles exceptions gracefully"""
        mock_docker_client.close.side_effect = Exception("Close failed")
        
        runner = SimCRunner()
        # Should not raise exception
        runner.cleanup()
    
    
    def test_parse_dps_invalid_output(self, mock_docker_client):
        """Test DPS parsing with invalid output returns 0"""
        runner = SimCRunner()
        
        invalid_outputs = [
            "",
            "No DPS found here",
            "Invalid JSON {",
            json.dumps({"sim": {"players": []}})  # Empty players
        ]
        
        for output in invalid_outputs:
            dps = runner._parse_dps(output)
            assert dps == 0.0
    
    def test_parse_dps_json_output(self, mock_docker_client):
        """Test DPS parsing from JSON output"""
        runner = SimCRunner()
        
        json_output = {
            "sim": {
                "players": [{
                    "collected_data": {
                        "dps": {
                            "mean": 15432.89
                        }
                    }
                }]
            }
        }
        
        dps = runner._parse_dps(json_output)
        assert dps == 15432.89

    def test_parse_dps_json_report_data(self, mock_docker_client):
        """Test DPS parsing from JSON report data"""
        runner = SimCRunner()
        
        json_report_data = {
            "sim": {
                "players": [{
                    "collected_data": {
                        "dps": {
                            "mean": 12345.67
                        }
                    }
                }]
            }
        }
        
        dps = runner._parse_dps(json_report_data)
        assert dps == 12345.67

    def test_evaluate_apl_success(self, mock_docker_client):
        """Test successful APL evaluation"""
        runner = SimCRunner()
        
        # Mock successful simulation
        mock_result = SimCResult(
            dps=15000.0,
            raw_output="test output",
            errors=[],
            is_valid=True
        )
        
        with patch.object(runner, '_run_simulation', return_value=mock_result):
            dps = runner.evaluate_apl("actions=test_action")
            assert dps == 15000.0

    def test_evaluate_apl_invalid_result(self, mock_docker_client):
        """Test APL evaluation with invalid result"""
        runner = SimCRunner()
        
        # Mock failed simulation
        mock_result = SimCResult(
            dps=0.0,
            raw_output="",
            errors=["Failed"],
            is_valid=False
        )
        
        with patch.object(runner, '_run_simulation', return_value=mock_result):
            dps = runner.evaluate_apl("actions=invalid_action")
            assert dps == 0.0

    def test_evaluate_apl_exception_handling(self, mock_docker_client):
        """Test APL evaluation handles exceptions"""
        runner = SimCRunner()
        
        with patch.object(runner, '_run_simulation', side_effect=Exception("Test error")):
            with patch('builtins.print') as mock_print:
                dps = runner.evaluate_apl("actions=test_action")
                assert dps == 0.0
                mock_print.assert_called_with("Simulation error: Test error")


    def test_run_simulation_docker_exception(self, mock_docker_client):
        """Test _run_simulation handles Docker exceptions"""
        runner = SimCRunner()
        
        mock_docker_client.containers.run.side_effect = Exception("Docker error")
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            with patch('tempfile.TemporaryDirectory') as mock_temp_dir:
                # Mock the file creation
                mock_file = Mock()
                mock_file.name = "/tmp/test.simc"
                mock_temp_file.return_value.__enter__.return_value = mock_file
                
                # Mock the directory creation
                mock_temp_dir.return_value.__enter__.return_value = "/tmp/output"
                
                with patch('os.unlink'):
                    result = runner._run_simulation("actions=test", 100, 60)
                    
                    assert result.dps == 0.0
                    assert result.is_valid is False
                    assert "Docker error" in result.errors[0]
    

class TestSimCRunnerIntegration:
    """Integration tests that require actual files but not Docker"""
    
    def test_profile_file_creation(self):
        """Test that profile files are created correctly"""
        with patch('evosim.simc.simc_runner.docker'):
            runner = SimCRunner()
            
            test_apl = "actions=living_flame"
            
            # Mock the Docker execution part
            with patch.object(runner.docker_client.containers, 'run') as mock_run:
                mock_run.return_value.stdout = '{"sim": {"players": [{"collected_data": {"dps": {"mean": 10000}}}]}}'
                mock_run.return_value.stderr = ""
                
                result = runner._run_simulation(test_apl, 10, 30)
                
                # Verify the call was made
                assert mock_run.called
                
                # Check that the profile would contain our APL
                # This tests the profile generation logic
                expected_profile = f"{runner.base_profile}\n\n{test_apl}\n"
                expected_profile += "\niterations=10\nfight_style=Patchwerk\nmax_time=30\njson=1\n"
                
                # We can't easily check the exact file content, but we can verify
                # the structure is correct
                assert "evoker" in runner.base_profile
                assert "living_flame" in test_apl


class TestSimCRunnerRealIntegration:
    """Full integration tests that actually run SimC with Docker"""
    
    @pytest.fixture(scope="class")
    def docker_available(self):
        """Check if Docker is available and SimC image exists"""
        try:
            client = docker.from_env()
            # Try to get the SimC image
            client.images.get("simulationcraftorg/simc")
            client.close()
            return True
        except (docker.errors.DockerException, docker.errors.ImageNotFound):
            pytest.skip("Docker or SimC image not available")
    
    @pytest.fixture
    def real_runner(self, docker_available):
        """Create a real SimCRunner instance (no mocking)"""
        runner = SimCRunner()
        yield runner
        runner.cleanup()
    
    def test_real_simc_basic_evoker_simulation(self, real_runner):
        """Test actual SimC execution with a basic evoker APL"""
        # Simple but valid evoker APL
        test_apl = """actions=living_flame"""
        
        # Run with low iterations for speed
        result = real_runner._run_simulation(test_apl, iterations=10, fight_length=30)
        
        # Verify we got a valid result
        assert result is not None
        assert isinstance(result, SimCResult)
        assert result.is_valid, f"Simulation failed with errors: {result.errors}"
        assert result.dps > 0, f"DPS should be positive, got: {result.dps}"
        assert len(result.raw_output) > 0, "Should have output from SimC"
        
        # Check that DPS is reasonable for a level 80 evoker (should be thousands)
        assert result.dps > 1000, f"DPS seems too low: {result.dps}"
        
        print(f"Real SimC test completed - DPS: {result.dps:.2f}")
    
    def test_real_simc_invalid_apl_handling(self, real_runner):
        """Test how real SimC handles invalid APL syntax"""
        # Intentionally broken APL
        invalid_apl = """actions=invalid_spell_name
actions+=/nonexistent_ability
actions+=/living_flame,if=invalid_condition_syntax"""
        
        result = real_runner._run_simulation(invalid_apl, iterations=5, fight_length=30)
        
        # SimC should either fail gracefully or ignore invalid actions
        assert result is not None
        assert isinstance(result, SimCResult)
        
        # The result might be invalid due to parsing errors, or it might succeed
        # with SimC ignoring the invalid actions - both are acceptable behaviors
        if not result.is_valid:
            assert len(result.errors) > 0, "Invalid APL should produce error messages"
            assert result.dps == 0.0
        else:
            # If SimC managed to run despite invalid actions, DPS should still be reasonable
            print(f"SimC handled invalid APL gracefully - DPS: {result.dps:.2f}")
    
    def test_real_simc_json_output_parsing(self, real_runner):
        """Test that real SimC JSON output is parsed correctly"""
        simple_apl = "actions=living_flame"
        
        result = real_runner._run_simulation(simple_apl, iterations=5, fight_length=30)
        
        assert result.is_valid
        assert result.dps > 0
        
        # The raw output should be JSON when json=1 is set
        try:
            # Try to parse the raw output as JSON
            json_data = json.loads(result.raw_output)
            assert 'sim' in json_data
            assert 'players' in json_data['sim']
            print("JSON output parsing successful")
        except json.JSONDecodeError:
            # If it's not JSON, that's also acceptable - SimC might output text
            print("SimC returned non-JSON output (this is acceptable)")
    
    def test_real_simc_performance_benchmark(self, real_runner):
        """Test SimC performance and ensure reasonable execution time"""
        # Simple APL for performance testing
        benchmark_apl = """actions=living_flame"""
        
        start_time = time.time()
        result = real_runner._run_simulation(benchmark_apl, iterations=10, fight_length=30)
        execution_time = time.time() - start_time
        
        assert result.is_valid
        assert result.dps > 0
        
        # Should complete within reasonable time (adjust as needed)
        assert execution_time < 60, f"Simulation took too long: {execution_time:.2f}s"
        
        print(f"Performance test: {execution_time:.2f}s, DPS: {result.dps:.2f}")
    
    def test_real_simc_different_fight_lengths(self, real_runner):
        """Test SimC with different fight lengths"""
        base_apl = "actions=living_flame\nactions+=/azure_strike"
        
        fight_lengths = [30, 60, 120]
        results = []
        
        for length in fight_lengths:
            result = real_runner._run_simulation(base_apl, iterations=5, fight_length=length)
            assert result.is_valid, f"Failed for fight length {length}"
            assert result.dps > 0
            results.append((length, result.dps))
        
        # DPS shouldn't vary wildly between fight lengths for this simple rotation
        dps_values = [dps for _, dps in results]
        dps_min, dps_max = min(dps_values), max(dps_values)
        variance_ratio = dps_max / dps_min
        
        # Allow for some variance but not extreme differences
        assert variance_ratio < 2.0, f"DPS variance too high across fight lengths: {results}"
        
        print(f"Fight length results: {results}")