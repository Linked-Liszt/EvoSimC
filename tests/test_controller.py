"""
Lightweight tests for the controller module.

These tests focus on core functionality and use mocking to avoid
external dependencies like MLflow, Docker, and LLM APIs.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from evosim.core.controller import (
    EvolutionController,
    EvolutionConfig,
    create_evolution_controller
)
from evosim.entities import APLProgram, SimCResult
from evosim.core.database import ProgramDatabase
from evosim.core.prompt_sampler import PromptSampler, DiffApplicationError
from evosim.llm import GeminiLLMClient, LLMGenerationError


class TestEvolutionConfig:
    """Test the evolution configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EvolutionConfig()
        
        assert config.max_generations == 100
        assert config.num_inspirations == 2
        assert config.simc_iterations == 1000
        assert config.simc_fight_length == 300
        assert config.verbose is True
        assert config.early_stopping_generations == 10
        assert config.experiment_name == "evosim_evolution"
        assert config.log_artifacts is True
        assert config.tracking_uri is None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = EvolutionConfig(
            max_generations=50,
            num_inspirations=3,
            verbose=False,
            experiment_name="test_experiment"
        )
        
        assert config.max_generations == 50
        assert config.num_inspirations == 3
        assert config.verbose is False
        assert config.experiment_name == "test_experiment"


class TestEvolutionController:
    """Test the main evolution controller functionality."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for the controller."""
        # Mock database
        database = Mock(spec=ProgramDatabase)
        database.get_statistics.return_value = {
            'count': 1,
            'best_dps': 50000.0,
            'avg_dps': 50000.0
        }
        
        # Mock prompt sampler
        prompt_sampler = Mock(spec=PromptSampler)
        prompt_sampler.build.return_value = "test prompt for LLM"
        prompt_sampler.apply_diff.return_value = "modified APL code"
        
        # Mock LLM client
        llm_client = Mock(spec=GeminiLLMClient)
        llm_client.generate.return_value = "LLM diff response"
        llm_client.get_usage_stats.return_value = {
            "total_requests": 5,
            "successful_requests": 4,
            "failed_requests": 1
        }
        
        # Mock SimC runner
        simc_runner = Mock()
        simc_runner.evaluate_apl.return_value = SimCResult(
            dps=51000.0,
            raw_output="SimC simulation output",
            errors=[],
            is_valid=True
        )
        
        return {
            'database': database,
            'prompt_sampler': prompt_sampler,
            'llm_client': llm_client,
            'simc_runner': simc_runner
        }
    
    @patch('evosim.core.controller.mlflow')
    def test_controller_initialization(self, mock_mlflow, mock_dependencies):
        """Test controller initialization with mocked MLflow."""
        config = EvolutionConfig(verbose=False)
        
        # Mock MLflow experiment
        mock_experiment = Mock()
        mock_experiment.experiment_id = "test_exp_id"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        
        controller = EvolutionController(
            database=mock_dependencies['database'],
            prompt_sampler=mock_dependencies['prompt_sampler'],
            llm_client=mock_dependencies['llm_client'],
            simc_runner=mock_dependencies['simc_runner'],
            config=config
        )
        
        # Verify initialization
        assert controller.config == config
        assert controller.database == mock_dependencies['database']
        assert controller.prompt_sampler == mock_dependencies['prompt_sampler']
        assert controller.llm_client == mock_dependencies['llm_client']
        assert controller.simc_runner == mock_dependencies['simc_runner']
        
        # Check initial stats
        expected_stats = {
            'generations_run': 0,
            'successful_mutations': 0,
            'failed_diff_applications': 0,
            'failed_evaluations': 0,
            'best_dps_seen': 0.0,
            'generations_without_improvement': 0
        }
        assert controller.stats == expected_stats
        
        # Verify MLflow setup was called
        mock_mlflow.set_experiment.assert_called_once()
    
    @patch('evosim.core.controller.mlflow')
    def test_initialize_with_baseline(self, mock_mlflow, mock_dependencies):
        """Test baseline initialization."""
        controller = EvolutionController(
            database=mock_dependencies['database'],
            prompt_sampler=mock_dependencies['prompt_sampler'],
            llm_client=mock_dependencies['llm_client'],
            simc_runner=mock_dependencies['simc_runner'],
            config=EvolutionConfig(verbose=False)
        )
        
        # Mock baseline program
        baseline_program = APLProgram(
            apl_code="actions=test_baseline",
            dps_score=50000.0,
            generation=0
        )
        mock_dependencies['database'].add.return_value = baseline_program
        
        result = controller.initialize_with_baseline("actions=test_baseline", "Test baseline")
        
        # Verify SimC evaluation was called
        mock_dependencies['simc_runner'].evaluate_apl.assert_called_once_with(
            "actions=test_baseline",
            iterations=1000,
            fight_length=300
        )
        
        # Verify database add was called
        mock_dependencies['database'].add.assert_called_once()
        
        # Verify stats were updated (using the mock simc_runner return value)
        assert controller.stats['best_dps_seen'] == 51000.0
        assert result == baseline_program
    
    def test_run_single_generation_success(self, mock_dependencies):
        """Test a successful single generation."""
        with patch('evosim.core.controller.mlflow'):
            controller = EvolutionController(
                database=mock_dependencies['database'],
                prompt_sampler=mock_dependencies['prompt_sampler'],
                llm_client=mock_dependencies['llm_client'],
                simc_runner=mock_dependencies['simc_runner'],
                config=EvolutionConfig(verbose=False)
            )
        
        # Setup mock data
        parent_program = APLProgram(
            apl_code="actions=parent",
            dps_score=50000.0,
            generation=1,
            program_id="parent_1"
        )
        
        inspiration_program = APLProgram(
            apl_code="actions=inspiration",
            dps_score=52000.0,
            generation=1,
            program_id="inspiration_1"
        )
        
        child_program = APLProgram(
            apl_code="modified APL code",
            dps_score=51000.0,
            generation=2,
            program_id="child_1"
        )
        
        # Configure mocks
        mock_dependencies['database'].sample.return_value = (parent_program, [inspiration_program])
        mock_dependencies['database'].add.return_value = child_program
        
        # Run single generation
        result = controller.run_single_generation()
        
        # Verify the evolutionary steps were called
        mock_dependencies['database'].sample.assert_called_once_with(num_inspirations=2)
        mock_dependencies['prompt_sampler'].build.assert_called_once_with(parent_program, [inspiration_program])
        mock_dependencies['llm_client'].generate.assert_called_once_with("test prompt for LLM")
        mock_dependencies['prompt_sampler'].apply_diff.assert_called_once_with("actions=parent", "LLM diff response")
        mock_dependencies['simc_runner'].evaluate_apl.assert_called_once_with(
            "modified APL code",
            iterations=1000,
            fight_length=300
        )
        mock_dependencies['database'].add.assert_called_once()
        
        # Verify result structure
        expected_result = {
            'parent_id': "parent_1",
            'parent_dps': 50000.0,
            'child_id': "child_1",
            'child_dps': 51000.0,
            'inspirations': ["inspiration_1"],
            'diff_applied': True,
            'evaluation_success': True
        }
        assert result == expected_result
        
        # Verify stats were updated
        assert controller.stats['successful_mutations'] == 1
    
    def test_run_single_generation_diff_error(self, mock_dependencies):
        """Test single generation with diff application error."""
        with patch('evosim.core.controller.mlflow'):
            controller = EvolutionController(
                database=mock_dependencies['database'],
                prompt_sampler=mock_dependencies['prompt_sampler'],
                llm_client=mock_dependencies['llm_client'],
                simc_runner=mock_dependencies['simc_runner'],
                config=EvolutionConfig(verbose=False)
            )
        
        # Setup mock data
        parent_program = APLProgram(
            apl_code="actions=parent",
            dps_score=50000.0,
            generation=1,
            program_id="parent_1"
        )
        
        child_program = APLProgram(
            apl_code="actions=parent",  # Same as parent due to diff failure
            dps_score=0.0,  # Failed evaluation
            generation=2,
            program_id="child_1"
        )
        
        # Configure mocks
        mock_dependencies['database'].sample.return_value = (parent_program, [])
        mock_dependencies['prompt_sampler'].apply_diff.side_effect = DiffApplicationError("Test diff error")
        mock_dependencies['database'].add.return_value = child_program
        
        # Run single generation
        result = controller.run_single_generation()
        
        # Verify diff application was attempted
        mock_dependencies['prompt_sampler'].apply_diff.assert_called_once()
        
        # Verify SimC evaluation was NOT called (due to diff error)
        mock_dependencies['simc_runner'].evaluate_apl.assert_not_called()
        
        # Verify stats were updated for failed diff
        assert controller.stats['failed_diff_applications'] == 1
        assert controller.stats['successful_mutations'] == 0
        
        # Verify result indicates failure
        assert result['evaluation_success'] is False
        assert result['child_dps'] == 0.0
    
    def test_run_single_generation_llm_error(self, mock_dependencies):
        """Test single generation with LLM generation error."""
        with patch('evosim.core.controller.mlflow'):
            controller = EvolutionController(
                database=mock_dependencies['database'],
                prompt_sampler=mock_dependencies['prompt_sampler'],
                llm_client=mock_dependencies['llm_client'],
                simc_runner=mock_dependencies['simc_runner'],
                config=EvolutionConfig(verbose=False)
            )
        
        # Setup mock data
        parent_program = APLProgram(apl_code="actions=parent", dps_score=50000.0, generation=1)
        mock_dependencies['database'].sample.return_value = (parent_program, [])
        mock_dependencies['llm_client'].generate.side_effect = LLMGenerationError("Test LLM error")
        
        # Run single generation and expect exception
        with pytest.raises(LLMGenerationError):
            controller.run_single_generation()
    
    def test_run_single_generation_simc_error(self, mock_dependencies):
        """Test single generation with SimC evaluation error."""
        with patch('evosim.core.controller.mlflow'):
            controller = EvolutionController(
                database=mock_dependencies['database'],
                prompt_sampler=mock_dependencies['prompt_sampler'],
                llm_client=mock_dependencies['llm_client'],
                simc_runner=mock_dependencies['simc_runner'],
                config=EvolutionConfig(verbose=False)
            )
        
        # Setup mock data
        parent_program = APLProgram(
            apl_code="actions=parent",
            dps_score=50000.0,
            generation=1,
            program_id="parent_1"
        )
        
        child_program = APLProgram(
            apl_code="modified APL code",
            dps_score=0.0,  # Failed evaluation
            generation=2,
            program_id="child_1"
        )
        
        # Configure mocks
        mock_dependencies['database'].sample.return_value = (parent_program, [])
        mock_dependencies['simc_runner'].evaluate_apl.side_effect = Exception("SimC error")
        mock_dependencies['database'].add.return_value = child_program
        
        # Run single generation
        result = controller.run_single_generation()
        
        # Verify stats were updated for failed evaluation
        assert controller.stats['failed_evaluations'] == 1
        assert controller.stats['successful_mutations'] == 0
        
        # Verify result indicates failure
        assert result['evaluation_success'] is False
        assert result['child_dps'] == 0.0
    
    def test_extract_reasoning(self, mock_dependencies):
        """Test reasoning extraction from LLM response."""
        with patch('evosim.core.controller.mlflow'):
            controller = EvolutionController(
                database=mock_dependencies['database'],
                prompt_sampler=mock_dependencies['prompt_sampler'],
                llm_client=mock_dependencies['llm_client'],
                simc_runner=mock_dependencies['simc_runner'],
                config=EvolutionConfig(verbose=False)
            )
        
        # Test with diff blocks
        llm_response = """I'll improve this APL by adding cooldown management.

This should increase DPS by better timing.

<<<<<<< SEARCH
actions=spell1
=======
actions=spell1,if=cooldown.ready
>>>>>>> REPLACE"""
        
        reasoning = controller._extract_reasoning(llm_response)
        expected = "I'll improve this APL by adding cooldown management.\n\nThis should increase DPS by better timing."
        assert reasoning == expected
        
        # Test without diff blocks
        llm_response_no_diff = "This is just reasoning without any diff blocks."
        reasoning = controller._extract_reasoning(llm_response_no_diff)
        assert reasoning == "This is just reasoning without any diff blocks."
        
        # Test with very long reasoning (should be truncated)
        long_response = "x" * 600 + "\n<<<<<<< SEARCH\ntest\n=======\ntest\n>>>>>>> REPLACE"
        reasoning = controller._extract_reasoning(long_response)
        assert len(reasoning) <= 503  # 500 + "..."
        assert reasoning.endswith("...")
    
    def test_get_best_apl(self, mock_dependencies):
        """Test getting the best APL."""
        with patch('evosim.core.controller.mlflow'):
            controller = EvolutionController(
                database=mock_dependencies['database'],
                prompt_sampler=mock_dependencies['prompt_sampler'],
                llm_client=mock_dependencies['llm_client'],
                simc_runner=mock_dependencies['simc_runner'],
                config=EvolutionConfig(verbose=False)
            )
        
        # Setup mock best program
        best_program = APLProgram(
            apl_code="actions=best_apl",
            dps_score=55000.0,
            generation=5
        )
        mock_dependencies['database'].get_best_programs.return_value = [best_program]
        
        program, apl_code = controller.get_best_apl()
        
        assert program == best_program
        assert apl_code == "actions=best_apl"
        mock_dependencies['database'].get_best_programs.assert_called_once_with(1)
    
    def test_get_best_apl_empty_database(self, mock_dependencies):
        """Test getting best APL from empty database."""
        with patch('evosim.core.controller.mlflow'):
            controller = EvolutionController(
                database=mock_dependencies['database'],
                prompt_sampler=mock_dependencies['prompt_sampler'],
                llm_client=mock_dependencies['llm_client'],
                simc_runner=mock_dependencies['simc_runner'],
                config=EvolutionConfig(verbose=False)
            )
        
        mock_dependencies['database'].get_best_programs.return_value = []
        
        with pytest.raises(ValueError, match="No programs in database"):
            controller.get_best_apl()
    
    def test_cleanup(self, mock_dependencies):
        """Test cleanup functionality."""
        with patch('evosim.core.controller.mlflow') as mock_mlflow:
            controller = EvolutionController(
                database=mock_dependencies['database'],
                prompt_sampler=mock_dependencies['prompt_sampler'],
                llm_client=mock_dependencies['llm_client'],
                simc_runner=mock_dependencies['simc_runner'],
                config=EvolutionConfig(verbose=False)
            )
            
            # Mock active run
            mock_mlflow.active_run.return_value = True
            
            controller.cleanup()
            
            # Verify cleanup was called
            mock_dependencies['simc_runner'].cleanup.assert_called_once()
            mock_mlflow.end_run.assert_called_once()
    
    def test_cleanup_with_errors(self, mock_dependencies):
        """Test cleanup handles errors gracefully."""
        with patch('evosim.core.controller.mlflow') as mock_mlflow:
            controller = EvolutionController(
                database=mock_dependencies['database'],
                prompt_sampler=mock_dependencies['prompt_sampler'],
                llm_client=mock_dependencies['llm_client'],
                simc_runner=mock_dependencies['simc_runner'],
                config=EvolutionConfig(verbose=False)
            )
            
            # Make cleanup methods raise exceptions
            mock_dependencies['simc_runner'].cleanup.side_effect = Exception("SimC cleanup error")
            mock_mlflow.active_run.side_effect = Exception("MLflow error")
            
            # Should not raise exceptions
            controller.cleanup()


class TestEvolutionControllerIntegration:
    """Test higher-level controller functionality with more complex scenarios."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for the integration tests."""
        # Mock database
        database = Mock(spec=ProgramDatabase)
        database.get_statistics.return_value = {
            'count': 1,
            'best_dps': 50000.0,
            'avg_dps': 50000.0
        }
        
        # Mock prompt sampler
        prompt_sampler = Mock(spec=PromptSampler)
        prompt_sampler.build.return_value = "test prompt for LLM"
        prompt_sampler.apply_diff.return_value = "modified APL code"
        
        # Mock LLM client
        llm_client = Mock(spec=GeminiLLMClient)
        llm_client.generate.return_value = "LLM diff response"
        llm_client.get_usage_stats.return_value = {
            "total_requests": 5,
            "successful_requests": 4,
            "failed_requests": 1
        }
        
        # Mock SimC runner
        simc_runner = Mock()
        simc_runner.evaluate_apl.return_value = SimCResult(
            dps=51000.0,
            raw_output="SimC simulation output",
            errors=[],
            is_valid=True
        )
        
        return {
            'database': database,
            'prompt_sampler': prompt_sampler,
            'llm_client': llm_client,
            'simc_runner': simc_runner
        }
    
    @patch('evosim.core.controller.mlflow')
    def test_run_evolution_early_stopping(self, mock_mlflow, mock_dependencies):
        """Test evolution with early stopping."""
        config = EvolutionConfig(
            max_generations=10,
            early_stopping_generations=3,
            verbose=False
        )
        
        controller = EvolutionController(
            database=mock_dependencies['database'],
            prompt_sampler=mock_dependencies['prompt_sampler'],
            llm_client=mock_dependencies['llm_client'],
            simc_runner=mock_dependencies['simc_runner'],
            config=config
        )
        
        # Mock parent and child programs
        parent_program = APLProgram(apl_code="actions=parent", dps_score=50000.0, generation=1, program_id="parent_1")
        child_program = APLProgram(apl_code="actions=child", dps_score=50000.0, generation=2, program_id="child_1")  # No improvement
        
        mock_dependencies['database'].sample.return_value = (parent_program, [])
        mock_dependencies['database'].add.return_value = child_program
        mock_dependencies['database'].get_best_programs.return_value = [child_program]
        
        # Mock MLflow run context
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = Mock(return_value=None)
        
        # Set initial best DPS to trigger early stopping logic
        controller.stats['best_dps_seen'] = 50000.0
        
        result = controller.run_evolution()
        
        # Should stop early due to no improvement
        assert result['evolution_stats']['generations_run'] <= 3
        assert controller.stats['generations_without_improvement'] >= 3
    
    @patch('evosim.core.controller.mlflow')
    def test_run_evolution_successful_improvement(self, mock_mlflow, mock_dependencies):
        """Test evolution with successful improvements."""
        config = EvolutionConfig(
            max_generations=3,
            verbose=False
        )
        
        controller = EvolutionController(
            database=mock_dependencies['database'],
            prompt_sampler=mock_dependencies['prompt_sampler'],
            llm_client=mock_dependencies['llm_client'],
            simc_runner=mock_dependencies['simc_runner'],
            config=config
        )
        
        # Mock improving generations
        improving_programs = [
            APLProgram(apl_code="actions=gen1", dps_score=51000.0, generation=2, program_id="gen1"),
            APLProgram(apl_code="actions=gen2", dps_score=52000.0, generation=3, program_id="gen2"),
            APLProgram(apl_code="actions=gen3", dps_score=53000.0, generation=4, program_id="gen3"),
        ]
        
        parent_program = APLProgram(apl_code="actions=parent", dps_score=50000.0, generation=1, program_id="parent")
        
        mock_dependencies['database'].sample.return_value = (parent_program, [])
        mock_dependencies['database'].add.side_effect = improving_programs
        mock_dependencies['database'].get_best_programs.return_value = [improving_programs[-1]]
        
        # Mock MLflow run context
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = Mock(return_value=None)
        
        result = controller.run_evolution()
        
        # Should complete all generations
        assert result['evolution_stats']['generations_run'] == 3
        assert result['evolution_stats']['best_dps_seen'] == 53000.0
        assert result['evolution_stats']['successful_mutations'] == 3


class TestCreateEvolutionController:
    """Test the factory function for creating evolution controllers."""
    
    @patch('evosim.core.database.create_database')
    @patch('evosim.core.prompt_sampler.create_prompt_sampler')
    @patch('evosim.llm.create_llm_client')
    @patch('evosim.core.controller.SimCRunner')
    @patch('evosim.core.controller.mlflow')
    def test_create_evolution_controller_defaults(self, mock_mlflow, mock_simc_runner, 
                                                  mock_create_llm, mock_create_prompt_sampler, 
                                                  mock_create_database):
        """Test factory function with default parameters."""
        # Setup mocks
        mock_database = Mock()
        mock_prompt_sampler = Mock()
        mock_llm_client = Mock()
        mock_simc_runner_instance = Mock()
        
        mock_create_database.return_value = mock_database
        mock_create_prompt_sampler.return_value = mock_prompt_sampler
        mock_create_llm.return_value = mock_llm_client
        mock_simc_runner.return_value = mock_simc_runner_instance
        
        # Mock baseline evaluation
        mock_simc_runner_instance.evaluate_apl.return_value = SimCResult(
            dps=50000.0,
            raw_output="Simulation completed successfully",
            errors=[],
            is_valid=True
        )
        baseline_program = APLProgram(apl_code="actions=baseline", dps_score=50000.0, generation=0)
        mock_database.add.return_value = baseline_program
        
        controller = create_evolution_controller("actions=baseline")
        
        # Verify components were created
        mock_create_database.assert_called_once_with("basic")
        mock_create_prompt_sampler.assert_called_once_with("basic")
        mock_create_llm.assert_called_once_with("gemini", api_key=None)
        mock_simc_runner.assert_called_once_with(image_name="simulationcraftorg/simc")
        
        # Verify controller was created
        assert isinstance(controller, EvolutionController)
        assert controller.database == mock_database
        assert controller.prompt_sampler == mock_prompt_sampler
        assert controller.llm_client == mock_llm_client
        assert controller.simc_runner == mock_simc_runner_instance
    
    @patch('evosim.core.database.create_database')
    @patch('evosim.core.prompt_sampler.create_prompt_sampler')
    @patch('evosim.llm.create_llm_client')
    @patch('evosim.core.controller.SimCRunner')
    @patch('evosim.core.controller.mlflow')
    def test_create_evolution_controller_custom_config(self, mock_mlflow, mock_simc_runner,
                                                       mock_create_llm, mock_create_prompt_sampler,
                                                       mock_create_database):
        """Test factory function with custom configuration."""
        # Setup mocks
        mock_database = Mock()
        mock_prompt_sampler = Mock()
        mock_llm_client = Mock()
        mock_simc_runner_instance = Mock()
        
        mock_create_database.return_value = mock_database
        mock_create_prompt_sampler.return_value = mock_prompt_sampler
        mock_create_llm.return_value = mock_llm_client
        mock_simc_runner.return_value = mock_simc_runner_instance
        
        # Mock baseline evaluation
        mock_simc_runner_instance.evaluate_apl.return_value = SimCResult(
            dps=50000.0,
            raw_output="Simulation completed successfully",
            errors=[],
            is_valid=True
        )
        baseline_program = APLProgram(apl_code="actions=baseline", dps_score=50000.0, generation=0)
        mock_database.add.return_value = baseline_program
        
        custom_config = EvolutionConfig(
            max_generations=50,
            experiment_name="custom_experiment"
        )
        
        controller = create_evolution_controller(
            baseline_apl="actions=baseline",
            llm_api_key="test_key",
            simc_image="custom/simc:latest",
            config=custom_config,
            experiment_name="override_experiment",
            mlflow_tracking_uri="http://localhost:5000"
        )
        
        # Verify custom parameters were used
        mock_create_llm.assert_called_once_with("gemini", api_key="test_key")
        mock_simc_runner.assert_called_once_with(image_name="custom/simc:latest")
        
        # Verify config was updated with MLflow settings
        assert controller.config.experiment_name == "override_experiment"
        assert controller.config.tracking_uri == "http://localhost:5000"
        assert controller.config.max_generations == 50  # Original custom value preserved


class TestControllerStatistics:
    """Test statistics tracking and reporting."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for the statistics tests."""
        # Mock database
        database = Mock(spec=ProgramDatabase)
        database.get_statistics.return_value = {
            'count': 1,
            'best_dps': 50000.0,
            'avg_dps': 50000.0
        }
        
        # Mock prompt sampler
        prompt_sampler = Mock(spec=PromptSampler)
        prompt_sampler.build.return_value = "test prompt for LLM"
        prompt_sampler.apply_diff.return_value = "modified APL code"
        
        # Mock LLM client
        llm_client = Mock(spec=GeminiLLMClient)
        llm_client.generate.return_value = "LLM diff response"
        llm_client.get_usage_stats.return_value = {
            "total_requests": 5,
            "successful_requests": 4,
            "failed_requests": 1
        }
        
        # Mock SimC runner
        simc_runner = Mock()
        simc_runner.evaluate_apl.return_value = SimCResult(
            dps=51000.0,
            raw_output="SimC simulation output",
            errors=[],
            is_valid=True
        )
        
        return {
            'database': database,
            'prompt_sampler': prompt_sampler,
            'llm_client': llm_client,
            'simc_runner': simc_runner
        }
    
    def test_get_final_results(self, mock_dependencies):
        """Test final results compilation."""
        with patch('evosim.core.controller.mlflow'):
            controller = EvolutionController(
                database=mock_dependencies['database'],
                prompt_sampler=mock_dependencies['prompt_sampler'],
                llm_client=mock_dependencies['llm_client'],
                simc_runner=mock_dependencies['simc_runner'],
                config=EvolutionConfig(verbose=False)
            )
        
        # Set up some stats
        controller.stats['generations_run'] = 5
        controller.stats['successful_mutations'] = 3
        controller.stats['failed_diff_applications'] = 1
        controller.stats['best_dps_seen'] = 55000.0
        
        # Mock best programs
        best_programs = [
            APLProgram(apl_code="actions=best", dps_score=55000.0, generation=5, program_id="best", reasoning="Best reasoning"),
            APLProgram(apl_code="actions=second", dps_score=54000.0, generation=4, program_id="second", reasoning="x" * 200)
        ]
        mock_dependencies['database'].get_best_programs.return_value = best_programs
        
        # Mock final database stats
        mock_dependencies['database'].get_statistics.return_value = {
            'count': 10,
            'best_dps': 55000.0,
            'avg_dps': 52000.0
        }
        
        results = controller._get_final_results()
        
        # Verify structure
        assert 'evolution_stats' in results
        assert 'database_stats' in results
        assert 'llm_stats' in results
        assert 'best_programs' in results
        
        # Check evolution stats
        assert results['evolution_stats']['generations_run'] == 5
        assert results['evolution_stats']['successful_mutations'] == 3
        assert results['evolution_stats']['best_dps_seen'] == 55000.0
        
        # Check best programs formatting
        assert len(results['best_programs']) == 2
        assert results['best_programs'][0]['id'] == "best"
        assert results['best_programs'][0]['dps'] == 55000.0
        assert results['best_programs'][0]['reasoning'] == "Best reasoning"
        
        # Check reasoning truncation
        assert results['best_programs'][1]['reasoning'].endswith("...")
        assert len(results['best_programs'][1]['reasoning']) <= 103  # 100 + "..."
