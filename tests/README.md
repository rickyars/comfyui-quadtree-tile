# Tests

This directory contains all testing, debugging, and analysis tools for the ComfyUI Quadtree Tile project.

## Directory Structure

```
tests/
├── README.md                              # This file
├── analysis/                              # Analysis and profiling tools
├── debug/                                 # Debugging and tracing scripts
├── diagnostics/                           # Diagnostic tools for specific features
├── integration/                           # Integration and QA tests
└── unit/                                  # Unit tests
```

## Quick Start

### Running Unit Tests

```bash
# Run all unit tests
python -m pytest unit/

# Run specific test
python -m pytest unit/test_quadtree.py

# Run with verbose output
python -m pytest unit/ -v
```

### Running Analysis Tools

Analysis tools help understand performance, behavior, and correctness:

```bash
# Analyze Gaussian weight distribution
python analysis/analyze_gaussian_weights.py

# Analyze variance metrics
python analysis/analyze_variance_metrics.py

# Analyze root size calculation
python analysis/analyze_root_size.py
```

### Using Debug Tools

Debug scripts trace execution and help diagnose issues:

```bash
# Trace recursive subdivision
python debug/trace_recursive_subdivision.py

# Debug specific pixel behavior
python debug/debug_00_pixel.py
```

### Running Diagnostics

Diagnostic tools test specific features in isolation:

```bash
# Diagnose Gaussian blending
python diagnostics/diagnostic_gaussian_blending.py
```

## Test Categories

### Unit Tests (`unit/`)

Test individual functions and components in isolation. These tests should be fast and not require external dependencies.

**Files**:
- `test_bug_reproduction.py` - Reproduces historical bugs to prevent regression
- `test_coverage_filter.py` - Tests coverage filtering logic
- `test_current_quadtree.py` - Tests current quadtree implementation
- `test_edge_cases.py` - Tests edge cases and boundary conditions
- `test_filter_math_verification.py` - Verifies mathematical correctness of filters
- `test_gaussian_fix.py` - Tests Gaussian weight fixes
- `test_gradient_metric.py` - Tests gradient-based metrics
- `test_leaf_filtering.py` - Tests leaf node filtering
- `test_rectangular_edge_tiles.py` - Tests rectangular edge tile handling
- `test_square_logic.py` - Tests square tile logic
- `test_square_quadtree.py` - Tests square quadtree implementation
- `test_square_root_fix.py` - Tests square root calculation fixes
- `test_subdivision_issue.py` - Tests subdivision edge cases
- `test_variance_fix.py` - Tests variance calculation fixes
- `test_weight_accumulation.py` - Tests weight accumulation logic

### Integration Tests (`integration/`)

Test how components work together and interact with external systems (GPU, ComfyUI, etc.).

**Files**:
- `qa_test_filtering.py` - QA tests for filtering logic
- `test_gpu_device_compatibility.py` - Tests GPU device compatibility

### Analysis Tools (`analysis/`)

Scripts that analyze behavior, performance, and correctness. Not pass/fail tests, but tools for understanding.

**Files**:
- `analyze_gaussian_weights.py` - Analyzes Gaussian weight distribution and blending
- `analyze_variance_metrics.py` - Analyzes variance metric behavior across images
- `analyze_root_size.py` - Analyzes root size calculation for different input dimensions

### Debug Tools (`debug/`)

Scripts for tracing execution, debugging issues, and understanding runtime behavior.

**Files**:
- `trace_recursive_subdivision.py` - Traces recursive subdivision process
- `debug_00_pixel.py` - Debugs behavior at specific pixel coordinates
- `debug_00_pixel_no_torch.py` - Debug without PyTorch dependencies
- `debug_00_pure.py` - Pure Python debug implementation

### Diagnostics (`diagnostics/`)

Comprehensive diagnostic tools that test specific features in depth.

**Files**:
- `diagnostic_gaussian_blending.py` - Comprehensive Gaussian blending diagnostics

## Writing New Tests

### Unit Tests

Place in `unit/` directory with `test_` prefix:

```python
# unit/test_my_feature.py
import pytest
from tiled_diffusion import my_function

def test_my_function_basic():
    result = my_function(input_data)
    assert result == expected_output

def test_my_function_edge_case():
    result = my_function(edge_case_input)
    assert result == expected_edge_output
```

### Analysis Tools

Place in `analysis/` directory:

```python
# analysis/analyze_my_feature.py
"""
Analyzes the behavior of my_feature across various inputs.
"""

def analyze_feature():
    # Run analysis
    # Print results
    # Generate visualizations if needed
    pass

if __name__ == '__main__':
    analyze_feature()
```

### Debug Scripts

Place in `debug/` directory:

```python
# debug/debug_my_issue.py
"""
Debugs specific issue with detailed tracing.
"""

def debug_issue():
    # Set up minimal reproduction
    # Add verbose logging
    # Trace execution
    pass

if __name__ == '__main__':
    debug_issue()
```

## Test Organization Principles

1. **Unit tests** - Fast, isolated, no external dependencies
2. **Integration tests** - Slower, test component interactions
3. **Analysis tools** - Exploratory, generate insights
4. **Debug scripts** - Reproduce and diagnose specific issues
5. **Diagnostics** - Comprehensive feature testing

## Running All Tests

```bash
# Run all unit tests
python -m pytest unit/

# Run all integration tests
python -m pytest integration/

# Run all tests (unit + integration)
python -m pytest unit/ integration/
```

## Coverage

To run tests with coverage:

```bash
# Install coverage
pip install pytest-cov

# Run with coverage report
python -m pytest unit/ --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Contributing Tests

When adding new tests:

1. **Choose the right category**: unit, integration, analysis, debug, or diagnostics
2. **Name files descriptively**: `test_feature_name.py` or `analyze_feature.py`
3. **Add docstrings**: Explain what the test/tool does
4. **Update this README**: Add your test to the appropriate section
5. **Keep tests focused**: One test should test one thing
6. **Make tests reproducible**: Use fixed seeds for randomness
7. **Clean up resources**: Close files, free memory, etc.

## Continuous Integration

Tests are run automatically on:
- Pull requests
- Commits to main branch
- Nightly builds

Ensure all unit tests pass before submitting PRs.
