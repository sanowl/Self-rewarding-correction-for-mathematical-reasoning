import random
import re
import json
import math
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
VERIFY_CORRECT = "[VERIFY] correct"
VERIFY_WRONG = "[VERIFY] wrong"

# Add custom exceptions
class MathReasoningError(Exception):
    """Base exception for math reasoning errors."""
    pass

class ValidationError(MathReasoningError):
    """Raised when input validation fails."""
    pass

class EvaluationError(MathReasoningError):
    """Raised when evaluation fails."""
    pass

@dataclass
class Problem:
    """Represents a mathematical problem with its solution."""
    prompt: str
    correct_answer: str
    difficulty: str


@dataclass
class Attempt:
    """Represents a solution attempt for a problem."""
    reasoning: str
    final_answer: str
    is_correct: bool = False


@dataclass
class SelfRewardingResponse:
    """Represents a self-evaluation of a solution attempt."""
    evaluation: str  # VERIFY_CORRECT or VERIFY_WRONG
    reasoning: str   # Reasoning for the evaluation
    is_accurate: bool = False  # Whether the evaluation matches the ground truth


@dataclass
class Trajectory:
    """Represents a full trajectory of problem-solving including attempts and evaluations."""
    problem: Problem
    attempts: List[Attempt] = None
    evaluations: List[SelfRewardingResponse] = None
    final_answer: str = None
    is_correct: bool = False
    
    def __post_init__(self):
        if self.attempts is None:
            self.attempts = []
        if self.evaluations is None:
            self.evaluations = []


class MathVerifier:
    """Simulates ground-truth verification of mathematical solutions."""
    
    @staticmethod
    def verify(problem: Problem, attempt: Attempt) -> bool:
        """
        Verify if an attempt's answer matches the correct answer.
        
        This is a simplified version. In a real-world scenario, this would use
        symbolic math libraries like SymPy to verify mathematical equivalence.
        """
        # Extract the final answer from the attempt
        # This normalizes spacing and removes potential formatting differences
        normalized_attempt = attempt.final_answer.strip().lower()
        normalized_correct = problem.correct_answer.strip().lower()
        
        return normalized_attempt == normalized_correct


class SelfRewardingReasoner:
    """
    Simulates a LLM with self-rewarding reasoning capabilities.
    
    This class represents the core implementation of the paper, where a model can:
    1. Generate initial mathematical reasoning (first attempt)
    2. Self-evaluate the correctness of its solution
    3. Generate an improved solution based on its evaluation
    """
    
    def __init__(self, 
                 rewarding_accuracy: Dict[str, float] = None,
                 correction_accuracy: float = 0.7,
                 max_attempts: int = 2):
        """
        Initialize the reasoner with specified accuracy parameters.
        
        Args:
            rewarding_accuracy: Dict with 'correct' and 'incorrect' recognition rates
            correction_accuracy: Probability of successfully correcting an incorrect solution
            max_attempts: Maximum number of correction attempts allowed
        """
        # Default accuracy parameters from the paper's results
        self.rewarding_accuracy = rewarding_accuracy or {
            'correct': 0.92,    # Accuracy in recognizing correct solutions
            'incorrect': 0.50   # Accuracy in recognizing incorrect solutions
        }
        self.correction_accuracy = correction_accuracy
        self.max_attempts = max_attempts
    
    def generate_initial_attempt(self, problem: Problem) -> Attempt:
        """
        Generate the initial reasoning attempt for a problem.
        
        This simulates the model generating a first solution to the problem.
        In a real implementation, this would call the actual LLM.
        """
        # Simulate a model that solves problems with varying accuracy based on difficulty
        difficulty_accuracy = {
            'easy': 0.8,
            'medium': 0.6,
            'hard': 0.4
        }
        
        accuracy = difficulty_accuracy.get(problem.difficulty, 0.5)
        is_correct = random.random() < accuracy
        
        # Generate mock reasoning
        if is_correct:
            reasoning = f"To solve the problem '{problem.prompt}', I'll work through it step by step... [reasoning steps] ... Therefore, the answer is {problem.correct_answer}."
            final_answer = problem.correct_answer
        else:
            # Generate a plausibly wrong answer
            wrong_answer = self._generate_wrong_answer(problem.correct_answer)
            reasoning = f"To solve the problem '{problem.prompt}', I'll approach it as follows... [reasoning with error] ... Therefore, the answer is {wrong_answer}."
            final_answer = wrong_answer
        
        return Attempt(reasoning=reasoning, final_answer=final_answer, is_correct=is_correct)
    
    def self_evaluate(self, problem: Problem, attempt: Attempt) -> SelfRewardingResponse:
        """
        Generate a self-evaluation of an attempt.
        
        This simulates the model evaluating its own solution using the self-rewarding approach.
        """
        # Determine if the model correctly identifies the attempt's correctness
        actual_correct = attempt.is_correct
        
        # Use different accuracy rates for correct vs incorrect attempts
        if actual_correct:
            recognition_accuracy = self.rewarding_accuracy['correct']
        else:
            recognition_accuracy = self.rewarding_accuracy['incorrect']
        
        # Determine if the evaluation is accurate based on the recognition accuracy
        evaluation_is_accurate = random.random() < recognition_accuracy
        
        # Based on accuracy, determine the evaluation result
        if evaluation_is_accurate:
            # Model correctly evaluates the attempt
            evaluation = VERIFY_CORRECT if actual_correct else VERIFY_WRONG
        else:
            # Model incorrectly evaluates the attempt
            evaluation = VERIFY_WRONG if actual_correct else VERIFY_CORRECT
        
        # Generate explanation for the evaluation
        if evaluation == VERIFY_CORRECT:
            reasoning = f"I've verified my solution by checking each step and the final answer. The approach is sound and the calculations are correct. The final answer {attempt.final_answer} is correct."
        else:
            reasoning = f"Upon reviewing my solution, I've found an error in my reasoning. The approach has a mistake that led to an incorrect answer {attempt.final_answer}."
        
        return SelfRewardingResponse(
            evaluation=evaluation,
            reasoning=reasoning,
            is_accurate=evaluation_is_accurate
        )
    
    def generate_correction(self, problem: Problem, attempt: Attempt, evaluation: SelfRewardingResponse) -> Attempt:
        """
        Generate a corrected attempt based on self-evaluation.
        
        This simulates the model refining its solution after determining its previous attempt was incorrect.
        """
        if evaluation.evaluation == VERIFY_CORRECT:
            # If the model believes its solution is correct, it simply returns the same attempt
            return attempt
        
        # Model attempts to correct its solution
        correction_succeeds = random.random() < self.correction_accuracy
        
        if correction_succeeds:
            # The model successfully corrects its mistake
            reasoning = f"Let me correct my previous solution to the problem '{problem.prompt}'... [improved reasoning] ... Therefore, the correct answer is {problem.correct_answer}."
            return Attempt(
                reasoning=reasoning,
                final_answer=problem.correct_answer,
                is_correct=True
            )
        else:
            # The model fails to correct and produces another wrong answer
            another_wrong_answer = self._generate_wrong_answer(problem.correct_answer)
            reasoning = f"Let me reconsider my approach to the problem '{problem.prompt}'... [reasoning with different error] ... Therefore, the answer is {another_wrong_answer}."
            return Attempt(
                reasoning=reasoning,
                final_answer=another_wrong_answer,
                is_correct=False
            )
    
    def solve_problem(self, problem: Problem) -> Trajectory:
        """
        Solve a problem using the self-rewarding framework.
        
        This implements the full pipeline described in the paper:
        1. Generate initial solution
        2. Self-evaluate the solution
        3. If deemed incorrect, attempt to correct it
        4. Repeat until satisfaction or max attempts reached
        """
        trajectory = Trajectory(problem=problem)
        
        # Generate the initial attempt
        current_attempt = self.generate_initial_attempt(problem)
        trajectory.attempts.append(current_attempt)
        
        attempt_count = 1
        while attempt_count <= self.max_attempts:
            # Evaluate the current attempt
            evaluation = self.self_evaluate(problem, current_attempt)
            trajectory.evaluations.append(evaluation)
            
            # If the model believes its solution is correct, stop
            if evaluation.evaluation == VERIFY_CORRECT:
                break
            
            # Otherwise, try to correct the solution
            if attempt_count < self.max_attempts:
                current_attempt = self.generate_correction(problem, current_attempt, evaluation)
                trajectory.attempts.append(current_attempt)
            
            attempt_count += 1
        
        # Record the final answer and its correctness
        trajectory.final_answer = trajectory.attempts[-1].final_answer
        trajectory.is_correct = trajectory.attempts[-1].is_correct
        
        return trajectory
    
    def _generate_wrong_answer(self, correct_answer: str) -> str:
        """
        Generate a plausibly wrong answer different from the correct one.
        
        In real models, this would be an emergent behavior from the model making reasoning errors.
        """
        try:
            # For numeric answers, perturb the correct value
            value = float(correct_answer)
            perturbation = random.choice([0.9, 1.1, 0.5, 2.0, -1.0])
            wrong_value = value * perturbation
            
            # Format similarly to the correct answer
            if correct_answer.isdigit():
                return str(int(wrong_value))
            else:
                return f"{wrong_value:.2f}"
        except ValueError:
            # For non-numeric answers (e.g., fractions, expressions)
            options = [
                f"{correct_answer}0",  # Append 0
                f"2{correct_answer}",  # Prepend 2
                f"-{correct_answer}",  # Negate
                "0"  # Default fallback
            ]
            return random.choice(options)


class Evaluator:
    """
    Evaluates the performance of a self-rewarding reasoner.
    
    This class implements the metrics used in the paper to evaluate the effectiveness
    of the self-rewarding correction approach.
    """
    
    @staticmethod
    def calculate_metrics(trajectories: List[Trajectory]) -> Dict[str, float]:
        """
        Calculate metrics from a set of solution trajectories.
        
        Metrics follow those defined in the paper:
        - Turn 1: Accuracy of the first attempt
        - Final accuracy: Accuracy of the final answer
        - Delta(t1, t2): Improvement in accuracy from first to final
        - Delta_i->c: Fraction of problems changed from incorrect to correct
        - Delta_c->i: Fraction of problems changed from correct to incorrect
        - RM Accuracy: Reward model accuracy in evaluating correctness
        """
        total = len(trajectories)
        if total == 0:
            return {
                "turn_1_accuracy": 0,
                "final_accuracy": 0,
                "delta_t1_t2": 0,
                "delta_i_to_c": 0,
                "delta_c_to_i": 0,
                "rm_accuracy_correct": 0,
                "rm_accuracy_incorrect": 0
            }
        
        # Calculate basic accuracy metrics
        turn_1_correct = sum(1 for t in trajectories if t.attempts[0].is_correct)
        final_correct = sum(1 for t in trajectories if t.is_correct)
        
        # Calculate transition metrics
        incorrect_to_correct = sum(1 for t in trajectories 
                                 if not t.attempts[0].is_correct and t.is_correct)
        correct_to_incorrect = sum(1 for t in trajectories 
                                 if t.attempts[0].is_correct and not t.is_correct)
        
        # Calculate reward model accuracy
        rm_correct_evals = 0
        rm_total_correct_attempts = 0
        rm_incorrect_evals = 0
        rm_total_incorrect_attempts = 0
        
        for trajectory in trajectories:
            for i, attempt in enumerate(trajectory.attempts):
                if i < len(trajectory.evaluations):  # Make sure there's an evaluation for this attempt
                    if attempt.is_correct:
                        rm_total_correct_attempts += 1
                        if trajectory.evaluations[i].evaluation == VERIFY_CORRECT:
                            rm_correct_evals += 1
                    else:
                        rm_total_incorrect_attempts += 1
                        if trajectory.evaluations[i].evaluation == VERIFY_WRONG:
                            rm_incorrect_evals += 1
        
        # Calculate final metrics
        metrics = {
            "turn_1_accuracy": turn_1_correct / total * 100,
            "final_accuracy": final_correct / total * 100,
            "delta_t1_t2": (final_correct - turn_1_correct) / total * 100,
            "delta_i_to_c": incorrect_to_correct / total * 100,
            "delta_c_to_i": correct_to_incorrect / total * 100,
            "rm_accuracy_correct": rm_correct_evals / max(1, rm_total_correct_attempts) * 100,
            "rm_accuracy_incorrect": rm_incorrect_evals / max(1, rm_total_incorrect_attempts) * 100
        }
        
        return metrics


def sample_math_problems() -> List[Problem]:
    """Generate a sample set of mathematical problems for testing."""
    return [
        Problem(
            prompt="What is the value of x in the equation 2x + 5 = 13?",
            correct_answer="4",
            difficulty="easy"
        ),
        Problem(
            prompt="Find the area of a circle with radius 5 cm.",
            correct_answer="78.54",
            difficulty="easy"
        ),
        Problem(
            prompt="Solve for x: log(x) + log(x+3) = log(4x)",
            correct_answer="1",
            difficulty="medium"
        ),
        Problem(
            prompt="If f(x) = x² - 3x + 2 and g(x) = 2x + 1, find f(g(3)).",
            correct_answer="30",
            difficulty="medium"
        ),
        Problem(
            prompt="The set of points (x, y, z) that satisfy 2x = 3y = -z is a line. The set of points (x, y, z) that satisfy 6x = -y = -4z is another line. Find the angle between these lines, in degrees.",
            correct_answer="90",
            difficulty="hard"
        ),
        Problem(
            prompt="A particular star has an absolute magnitude M = -7. If this star is observed in a galaxy that is at a distance of 3Mpc, what will its apparent magnitude be?",
            correct_answer="20.39",
            difficulty="hard"
        ),
        Problem(
            prompt="How many 3-letter words can we make from the letters A, B, C, D, and E, if we are allowed to repeat letters, and we must use the letters A and B at least once each in every word?",
            correct_answer="24",
            difficulty="hard"
        ),
        Problem(
            prompt="Find the sum of all positive integers less than 1000 that are divisible by either 3 or 5.",
            correct_answer="233168",
            difficulty="medium"
        ),
        Problem(
            prompt="A fair six-sided die is rolled twice. What is the probability that the sum of the two rolls is at least 10?",
            correct_answer="1/6",
            difficulty="medium"
        ),
        Problem(
            prompt="Find all values of x such that |2x - 3| < 4.",
            correct_answer="-0.5 < x < 3.5",
            difficulty="medium"
        )
    ]


def run_experiment(num_problems=50, 
                   reward_accuracy=None, 
                   correction_accuracy=0.7,
                   seed=None):
    """
    Run an experiment with the self-rewarding correction framework.
    
    Args:
        num_problems: Number of problems to solve
        reward_accuracy: Dict with 'correct' and 'incorrect' recognition rates
        correction_accuracy: Probability of successfully correcting an incorrect solution
        seed: Random seed for reproducibility
    
    Returns:
        Dict containing the evaluation metrics
    """
    try:
        if num_problems <= 0:
            raise ValidationError("Number of problems must be positive")
        
        if correction_accuracy < 0 or correction_accuracy > 1:
            raise ValidationError("Correction accuracy must be between 0 and 1")
        
        if reward_accuracy:
            for key in ['correct', 'incorrect']:
                if key not in reward_accuracy:
                    raise ValidationError(f"Missing '{key}' in reward_accuracy")
                if not 0 <= reward_accuracy[key] <= 1:
                    raise ValidationError(f"Reward accuracy '{key}' must be between 0 and 1")
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Sample or generate problems
        problems = sample_math_problems()
        
        # If we need more problems than we have in our sample, duplicate with variations
        if num_problems > len(problems):
            duplicated_problems = []
            for i in range(num_problems):
                problem_idx = i % len(problems)
                problem = problems[problem_idx]
                # Create slight variations for duplicated problems
                if i >= len(problems):
                    prompt_variation = f"{problem.prompt} (variation {i//len(problems)})"
                    duplicated_problems.append(Problem(
                        prompt=prompt_variation,
                        correct_answer=problem.correct_answer,
                        difficulty=problem.difficulty
                    ))
                else:
                    duplicated_problems.append(problem)
            problems = duplicated_problems
        else:
            problems = problems[:num_problems]
        
        # Initialize the reasoner
        reasoner = SelfRewardingReasoner(
            rewarding_accuracy=reward_accuracy,
            correction_accuracy=correction_accuracy
        )
        
        # Solve all problems and collect trajectories
        trajectories = []
        for problem in problems:
            trajectory = reasoner.solve_problem(problem)
            trajectories.append(trajectory)
        
        # Evaluate the results
        metrics = Evaluator.calculate_metrics(trajectories)
        
        return {
            "metrics": metrics,
            "trajectories": trajectories
        }
    
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in experiment: {e}")
        raise MathReasoningError(f"Experiment failed: {str(e)}")


def compare_models():
    """
    Compare different configurations of the self-rewarding framework.
    
    This simulates the different ablation studies from the paper.
    """
    experiment_configs = [
        {
            "name": "Intrinsic self-correction",
            "params": {
                "reward_accuracy": {"correct": 0.5, "incorrect": 0.5},
                "correction_accuracy": 0.2
            }
        },
        {
            "name": "Self-rewarding IFT",
            "params": {
                "reward_accuracy": {"correct": 0.7, "incorrect": 0.5},
                "correction_accuracy": 0.5
            }
        },
        {
            "name": "Self-rewarding IFT + DPO",
            "params": {
                "reward_accuracy": {"correct": 0.85, "incorrect": 0.55},
                "correction_accuracy": 0.6
            }
        },
        {
            "name": "Self-rewarding IFT + PPO",
            "params": {
                "reward_accuracy": {"correct": 0.92, "incorrect": 0.60},
                "correction_accuracy": 0.7
            }
        },
        {
            "name": "Gold Reward Model (Oracle)",
            "params": {
                "reward_accuracy": {"correct": 1.0, "incorrect": 1.0},
                "correction_accuracy": 0.8
            }
        }
    ]
    
    results = {}
    
    for config in experiment_configs:
        logger.info(f"Running experiment: {config['name']}")
        experiment_result = run_experiment(
            num_problems=50,
            reward_accuracy=config["params"]["reward_accuracy"],
            correction_accuracy=config["params"]["correction_accuracy"],
            seed=42  # For reproducibility
        )
        
        results[config["name"]] = experiment_result["metrics"]
    
    return results


def display_results(results):
    """Display comparison results in a formatted table."""
    headers = [
        "Model", "Turn 1 Acc", "Final Acc", "Δ(t1, t2)", 
        "Δi→c", "Δc→i", "RM Acc (C)", "RM Acc (I)"
    ]
    
    print("\n" + "="*100)
    print("Self-Rewarding Correction for Mathematical Reasoning - Experiment Results")
    print("="*100)
    
    # Print headers
    header_str = f"{headers[0]:<25} "
    for h in headers[1:]:
        header_str += f"{h:<12} "
    print(header_str)
    print("-"*100)
    
    # Print results for each model
    for model_name, metrics in results.items():
        row = f"{model_name:<25} "
        row += f"{metrics['turn_1_accuracy']:<12.2f} "
        row += f"{metrics['final_accuracy']:<12.2f} "
        row += f"{metrics['delta_t1_t2']:<12.2f} "
        row += f"{metrics['delta_i_to_c']:<12.2f} "
        row += f"{metrics['delta_c_to_i']:<12.2f} "
        row += f"{metrics['rm_accuracy_correct']:<12.2f} "
        row += f"{metrics['rm_accuracy_incorrect']:<12.2f} "
        print(row)
    
    print("="*100)
    print("All values except model names are percentages (%)")
    print("="*100)


def demonstrate_trajectory():
    """Demonstrate a full problem-solving trajectory with the self-rewarding framework."""
    try:
        # Create a challenging math problem
        problem = Problem(
            prompt="A particular star has an absolute magnitude M = −7. If this star is observed " +
                   "in a galaxy that is at a distance of 3Mpc, what will its apparent magnitude be?",
            correct_answer="20.39",
            difficulty="hard"
        )
        
        # Create a reasoner with specific parameters that will demonstrate self-correction
        reasoner = SelfRewardingReasoner(
            rewarding_accuracy={"correct": 0.9, "incorrect": 0.9},
            correction_accuracy=0.9,
            max_attempts=2
        )
        
        # Force the initial attempt to be incorrect
        def mock_initial_attempt(problem):
            reasoning = (
                f"To determine the apparent magnitude of the star, I can use the distance modulus formula:\n"
                f"m - M = 5 log10(d) - 5\n"
                f"where:\n"
                f"- m is the apparent magnitude\n"
                f"- M is the absolute magnitude\n"
                f"- d is the distance to the star in parsecs\n\n"
                f"Given:\n"
                f"- M = -7\n"
                f"- d = 3 Mpc = 3 × 10^6 pc\n\n"
                f"Substituting these values:\n"
                f"m - (-7) = 5 log10(3 × 10^6) - 5\n"
                f"m + 7 = 5 log10(3 × 10^6) - 5\n"
                f"m + 7 = 5 × (log10(3) + log10(10^6)) - 5\n"
                f"m + 7 = 5 × (0.4771 + 6) - 5\n"
                f"m + 7 = 5 × 6.4771 - 5\n"
                f"m + 7 = 32.3855 - 5\n"
                f"m + 7 = 14.5855\n"
                f"m = 7.5855\n\n"
                f"Therefore, the apparent magnitude of the star is approximately 7.58."
            )
            return Attempt(reasoning=reasoning, final_answer="7.58", is_correct=False)
        
        # Replace the original method with our mock
        original_method = reasoner.generate_initial_attempt
        reasoner.generate_initial_attempt = lambda p: mock_initial_attempt(p)
        
        # Force the self-evaluation to correctly identify the error
        def mock_self_evaluate(problem, attempt):
            if attempt.final_answer == "7.58":
                reasoning = (
                    f"Upon reviewing my solution, I need to check my calculations:\n"
                    f"m - (-7) = 5 log10(3 × 10^6) - 5\n"
                    f"m + 7 = 5 log10(3 × 10^6) - 5\n\n"
                    f"Let's calculate 5 log10(3 × 10^6) - 5 step by step:\n"
                    f"log10(3 × 10^6) = log10(3) + log10(10^6) ≈ 0.4771 + 6 = 6.4771\n"
                    f"5 × 6.4771 = 32.3855\n"
                    f"32.3855 - 5 = 27.3855\n\n"
                    f"So, m + 7 = 27.3855, not 14.5855 as I incorrectly calculated.\n"
                    f"This means m = 27.3855 - 7 = 20.3855, which rounds to 20.39.\n\n"
                    f"I made an error in my arithmetic. The correct result should be approximately 20.39."
                )
                return SelfRewardingResponse(
                    evaluation=VERIFY_WRONG,
                    reasoning=reasoning,
                    is_accurate=True
                )
            else:
                return SelfRewardingResponse(
                    evaluation=VERIFY_CORRECT if attempt.is_correct else VERIFY_WRONG,
                    reasoning="The solution appears correct." if attempt.is_correct else "The solution contains errors.",
                    is_accurate=True
                )
        
        # Replace the self-evaluation method
        original_evaluate = reasoner.self_evaluate
        reasoner.self_evaluate = lambda p, a: mock_self_evaluate(p, a)
        
        # Force the correction to produce the right answer
        def mock_correction(problem, attempt, evaluation):
            if attempt.final_answer == "7.58" and evaluation.evaluation == VERIFY_WRONG:
                reasoning = (
                    f"Let me correct my previous solution to the problem.\n\n"
                    f"Using the distance modulus formula:\n"
                    f"m - M = 5 log10(d) - 5\n"
                    f"where:\n"
                    f"- m is the apparent magnitude\n"
                    f"- M is the absolute magnitude (= -7)\n"
                    f"- d is the distance to the star (= 3 × 10^6 parsecs)\n\n"
                    f"Substituting these values:\n"
                    f"m - (-7) = 5 log10(3 × 10^6) - 5\n"
                    f"m + 7 = 5 log10(3 × 10^6) - 5\n"
                    f"m + 7 = 5 × (log10(3) + log10(10^6)) - 5\n"
                    f"m + 7 = 5 × (0.4771 + 6) - 5\n"
                    f"m + 7 = 5 × 6.4771 - 5\n"
                    f"m + 7 = 32.3855 - 5\n"
                    f"m + 7 = 27.3855\n"
                    f"m = 27.3855 - 7\n"
                    f"m = 20.3855\n\n"
                    f"Therefore, the apparent magnitude of the star is approximately 20.39."
                )
                return Attempt(reasoning=reasoning, final_answer="20.39", is_correct=True)
            return attempt
        
        # Replace the correction method
        original_correction = reasoner.generate_correction
        reasoner.generate_correction = lambda p, a, e: mock_correction(p, a, e)
        
        # Now solve the problem and demonstrate the trajectory
        trajectory = reasoner.solve_problem(problem)
        
        # Restore original methods
        reasoner.generate_initial_attempt = original_method
        reasoner.self_evaluate = original_evaluate
        reasoner.generate_correction = original_correction
        
        print("\n" + "="*100)
        print("Demonstration of Self-Rewarding Correction")
        print("="*100)
        print(f"Problem: {problem.prompt}")
        print(f"Correct Answer: {problem.correct_answer}")
        print("="*100)
        
        # Print the trajectory
        for i, attempt in enumerate(trajectory.attempts):
            print(f"\nAttempt {i+1}:")
            print(f"Answer: {attempt.final_answer}")
            print(f"Correct: {attempt.is_correct}")
            print("\nReasoning:")
            print(attempt.reasoning)
            
            if i < len(trajectory.evaluations):
                evaluation = trajectory.evaluations[i]
                print("\nSelf-Evaluation:")
                print(f"Verdict: {evaluation.evaluation}")
                print("\nEvaluation Reasoning:")
                print(evaluation.reasoning)
                print(f"Evaluation Accuracy: {evaluation.is_accurate}")
            
            print("-"*100)
        
        print(f"\nFinal Answer: {trajectory.final_answer}")
        print(f"Correct: {trajectory.is_correct}")
        print("="*100)
    
    except Exception as e:
        logger.error(f"Error during trajectory demonstration: {e}")
        print("\nError occurred during demonstration. Please check logs for details.")
        return None


if __name__ == "__main__":
    """
    Main execution script to demonstrate the self-rewarding correction framework.
    """
    try:
        # Run a full demonstration of a single trajectory
        demonstrate_trajectory()
        
        # Run comparison of different model configurations
        print("\nRunning model comparisons...")
        results = compare_models()
        display_results(results)
        
        # Additional experiment to show the effect of varying reward model accuracy
        print("\nRunning accuracy sensitivity analysis...")
        sensitivity_results = {}
        
        reward_accuracy_configs = [
            {"correct": 0.6, "incorrect": 0.4},
            {"correct": 0.7, "incorrect": 0.5},
            {"correct": 0.8, "incorrect": 0.6},
            {"correct": 0.9, "incorrect": 0.7},
            {"correct": 1.0, "incorrect": 1.0}
        ]
        
        for i, reward_accuracy in enumerate(reward_accuracy_configs):
            try:
                if not all(0 <= v <= 1 for v in reward_accuracy.values()):
                    raise ValidationError("Reward accuracy values must be between 0 and 1")
                    
                model_name = f"Model with RM Acc (C={reward_accuracy['correct']:.1f}, I={reward_accuracy['incorrect']:.1f})"
                experiment_result = run_experiment(
                    num_problems=30,
                    reward_accuracy=reward_accuracy,
                    correction_accuracy=0.7,
                    seed=42
                )
                sensitivity_results[model_name] = experiment_result["metrics"]
                
            except (ValidationError, MathReasoningError) as e:
                logger.error(f"Error in sensitivity analysis for config {i}: {e}")
                continue
        
        if sensitivity_results:
            print("\nSensitivity Analysis Results:")
            display_results(sensitivity_results)
        else:
            print("\nNo valid results from sensitivity analysis")
            
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        print("\nExecution failed. Please check logs for details.")