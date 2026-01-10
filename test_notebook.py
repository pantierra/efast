#!/usr/bin/env python3
"""
Execute the efast_interactive notebook programmatically for testing.

Usage:
    python test_notebook.py
    python test_notebook.py --sitename innsbruck --season 2024
"""

import argparse
import sys
from pathlib import Path

try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except ImportError:
    print("❌ nbconvert not installed. Install with: pip install nbconvert")
    sys.exit(1)


def execute_notebook(notebook_path: Path, sitename: str = None, season_year: str = None):
    """Execute notebook programmatically."""
    print(f"📓 Executing notebook: {notebook_path}")
    
    # Read notebook
    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)
    
    # If site/season provided, inject them into the notebook
    if sitename and season_year:
        # Find the cell with site selection and modify it
        for cell in nb.cells:
            if cell.cell_type == 'code' and 'site_dropdown' in cell.source:
                # Add code to set site/season after widget creation
                cell.source += f"\n\n# Set site/season for automated execution\n"
                cell.source += f"site_dropdown.value = '{sitename}'\n"
                cell.source += f"season_dropdown.value = '{season_year}'\n"
                break
    
    # Execute notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        ep.preprocess(nb, {'metadata': {'path': notebook_path.parent}})
        print("✅ Notebook executed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Notebook execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description="Execute efast_interactive notebook")
    parser.add_argument(
        '--notebook', default='efast_interactive.ipynb',
        help='Notebook file (default: efast_interactive.ipynb)'
    )
    parser.add_argument(
        '--sitename',
        help='Site name to use (optional, will use widget default if not provided)'
    )
    parser.add_argument(
        '--season',
        help='Season year to use (optional, will use widget default if not provided)'
    )
    
    args = parser.parse_args()
    
    notebook_path = Path(args.notebook)
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return 1
    
    if (args.sitename and not args.season) or (args.season and not args.sitename):
        print("❌ Both --sitename and --season must be provided together")
        return 1
    
    return execute_notebook(notebook_path, args.sitename, args.season)


if __name__ == '__main__':
    sys.exit(main())


