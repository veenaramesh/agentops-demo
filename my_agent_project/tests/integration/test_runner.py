"""
Databricks Integration Test Runner

Custom test runner for Databricks notebooks that provides pytest-like functionality
without requiring pytest installation in the Databricks environment.
"""

import traceback
import time
from typing import List, Callable, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum


class TestStatus(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    duration: float
    error: Optional[str] = None
    error_traceback: Optional[str] = None


class DatabricksTestRunner:
    """
    A custom test runner for Databricks notebooks that provides pytest-like functionality
    """
    
    def __init__(self, fail_fast: bool = False, verbose: bool = True):
        """
        Initialize the test runner
        
        Args:
            fail_fast: Stop execution on first failure
            verbose: Print detailed output during test execution
        """
        self.tests = []
        self.results: List[TestResult] = []
        self.fail_fast = fail_fast
        self.verbose = verbose
    
    def test(self, name: str, skip: bool = False):
        """
        Decorator to mark test functions
        
        Args:
            name: Display name for the test
            skip: Whether to skip this test
        """
        def decorator(func: Callable):
            self.tests.append({
                'name': name,
                'func': func,
                'skip': skip
            })
            return func
        return decorator
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Alias for run_all for consistency"""
        return self.run_all()
    
    def run_all(self) -> Dict[str, Any]:
        """
        Run all registered tests
        
        Returns:
            Dictionary containing test results and summary
        """
        passed = 0
        failed = 0
        skipped = 0
        total_start_time = time.time()
        
        if self.verbose:
            print("Databricks Agent Integration Tests")
            print("=" * 80)
            print(f"Running {len(self.tests)} tests...")
            print()
        
        for test_info in self.tests:
            test_name = test_info['name']
            test_func = test_info['func']
            skip = test_info['skip']
            
            if skip:
                if self.verbose:
                    print(f"  SKIPPED: {test_name}")
                skipped += 1
                self.results.append(TestResult(
                    name=test_name,
                    status=TestStatus.SKIPPED,
                    duration=0.0
                ))
                continue
            
            if self.verbose:
                print(f"Running: {test_name}")
            
            start_time = time.time()
            
            try:
                test_func()
                duration = time.time() - start_time
                
                if self.verbose:
                    print(f"  PASSED: {test_name} ({duration:.2f}s)")
                
                passed += 1
                self.results.append(TestResult(
                    name=test_name,
                    status=TestStatus.PASSED,
                    duration=duration
                ))
                
            except Exception as e:
                duration = time.time() - start_time
                error_traceback = traceback.format_exc()
                
                if self.verbose:
                    print(f" ! FAILED: {test_name} ({duration:.2f}s)")
                    print(f"   Error: {str(e)}")
                    print()
                
                failed += 1
                self.results.append(TestResult(
                    name=test_name,
                    status=TestStatus.FAILED,
                    duration=duration,
                    error=str(e),
                    error_traceback=error_traceback
                ))
                
                if self.fail_fast:
                    print("  Fail-fast enabled - stopping on first failure")
                    break
        
        total_duration = time.time() - total_start_time
        
        if self.verbose:
            print("=" * 80)
            print("  Test Results Summary")
            print(f"      Passed:  {passed}")
            print(f"      Failed:  {failed}")
            print(f"      Skipped: {skipped}")
            print(f"      Total Time: {total_duration:.2f}s")
            print(f"      Success Rate: {(passed/(passed+failed)*100) if (passed+failed) > 0 else 0:.1f}%")
            print()
        
        summary = {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": len(self.tests),
            "duration": total_duration,
            "success_rate": (passed/(passed+failed)*100) if (passed+failed) > 0 else 0,
            "results": [self._result_to_dict(r) for r in self.results]
        }
        
        if failed > 0:
            error_msg = f"Integration tests failed: {failed} test(s) failed out of {passed + failed}"
            #raise Exception(error_msg)
        
        if self.verbose and failed == 0:
            print("All tests passed successfully!")
        
        return summary
    
    def _result_to_dict(self, result: TestResult) -> Dict[str, Any]:
        """Convert TestResult to dictionary for serialization"""
        return {
            "name": result.name,
            "status": result.status.value,
            "duration": result.duration,
            "error": result.error
        }


def create_test_runner(fail_fast: bool = False, verbose: bool = True) -> DatabricksTestRunner:
    """
    Create a new test runner instance
    
    Args:
        fail_fast: Stop on first failure
        verbose: Print detailed output
    
    Returns:
        DatabricksTestRunner instance
    """
    return DatabricksTestRunner(fail_fast=fail_fast, verbose=verbose)