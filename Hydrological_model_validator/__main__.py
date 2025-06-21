#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
from pathlib import Path

# --- Color codes for styling ---
RESET = '\033[0m'
GREEN = '\033[38;5;40m'
RED = '\033[38;5;196m'
ORANGE = '\033[38;5;208m'
VIOLET = '\033[38;5;141m'
CYAN = '\033[38;5;51m'
YELLOW = '\033[38;5;226m'

def print_banner():
    banner = f"""{CYAN}                                                                                                                                                                                                                                                                                
    __  __          __              _    __      ___     __      __            
   / / / /_  ______/ /________     | |  / /___ _/ (_)___/ /___ _/ /_____  _____
  / /_/ / / / / __  / ___/ __ \    | | / / __ `/ / / __  / __ `/ __/ __ \/ ___/
 / __  / /_/ / /_/ / /  / /_/ /    | |/ / /_/ / / / /_/ / /_/ / /_/ /_/ / /    
/_/ /_/\__, /\__,_/_/   \____/     |___/\__,_/_/_/\__,_/\__,_/\__/\____/_/     
      /____/                                                                   

                                                                    
    {YELLOW}
    Data Report Generator for Bio-Geo-Hydrological simulations
    {RESET}
    """
    
    print(banner)
    
def main():
    import argparse
    import sys
    
    from Hydrological_model_validator.Report_generator import generate_full_report
    from Hydrological_model_validator.Processing.time_utils import Timer
    
    parser = argparse.ArgumentParser(
        description="Generate full report with spatial and timeseries data."
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Input data folder path or dict of file paths (JSON string or Python dict string)."
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output base directory. If not specified, prompts or defaults to REPORT folder."
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check input files and exit (no report generation)."
    )

    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Do not generate PDF report (generate plots and dataframes only)."
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output."
    )

    parser.add_argument(
        "--open-report",
        action="store_true",
        help="Open the PDF report automatically after generation."
    )

    parser.add_argument(
        "--variable",
        type=str,
        required=False,
        help="Variable name for plots and report labeling."
    )

    parser.add_argument(
        "--unit",
        type=str,
        required=False,
        help="Unit string for the variable (e.g., 'm3/s', 'mg/L')."
    )
    
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress the ASCII art/banner if running in batch mode."
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Hydrological_model_validator v4.9.1"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show detailed program information and exit."
        )

    args, _ = parser.parse_known_args()
    
    if args.info:
        print("""
Hydrological_model_validator v4.9.1 

###################################################################
#  Tools for the analysis and validation of Bio-Geo-Hydrological  #
#           simulations and other climatological data             #
###################################################################

Author: Alessandro Gozzoli - Alma Mater Studiorum - Università di Bologna  
Role: Student - Physics of the Earth System
Email: alessandro.gozzoli4@studio.unibo.it  
GitHub: https://github.com/AlessandroGozzoli
""")
        sys.exit(0)
    
    args = parser.parse_args()
    
    if args.input is None:
        parser.error("the following arguments are required: input")

    # Try to interpret input as dict if possible (optional)
    data_folder = args.input
    if data_folder.strip().startswith("{") and data_folder.strip().endswith("}"):
        import ast
        try:
            data_folder = ast.literal_eval(data_folder)
        except Exception as e:
            print(f"Failed to parse input dict: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        data_folder = data_folder.strip()
        
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("REPORT")  # or any default path you'd like
    
    if not args.no_banner:
        print_banner()
    
    # Start timer for overall process
    tic = time.time()
    
    if args.verbose:
        print(f"{ORANGE}Starting report generation...{RESET}")

    # Call your report generation function
    with Timer("Beginning report generation process"):
        generate_full_report(
            data_folder=data_folder,
            output_dir=args.output_dir,
            check_only=args.check,
            generate_pdf=not args.no_pdf,
            verbose=args.verbose,
            variable=args.variable,
            unit=args.unit,
            open_report=args.open_report
            )
    
    # End the timer, tell the time
    toc = time.time()
    elapsed = toc - tic
    
    if args.verbose:
        print(f"{GREEN}[DONE]{RESET} Report generated in {elapsed:.2f} seconds.", args.verbose)
    print(f"{GREEN}Output saved in: {output_dir.resolve()}{RESET}")

if __name__ == "__main__":
    main()