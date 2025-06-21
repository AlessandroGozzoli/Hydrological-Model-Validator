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
    import textwrap
    
    from Hydrological_model_validator.Report_generator import generate_full_report
    from Hydrological_model_validator.Processing.time_utils import Timer
    
    parser = argparse.ArgumentParser(
        description="Generate a comprehensive evaluation report from observed and simulated Bio-Geo-Hydrological datasets.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "input",
        nargs="?",
        help=textwrap.dedent("""\
            Path to the input data directory or a dictionary of file paths.
            You can pass:
              - a folder containing: obs_spatial, sim_spatial, obs_ts, sim_ts, and mask
              - or a stringified dictionary (JSON or Python format) mapping these keys to files:
                {
                    "obs_spatial": "obs_spatial.nc",
                    "sim_spatial": "sim_spatial.nc",
                    "obs_ts": "obs_timeseries.csv",
                    "sim_ts": "sim_timeseries.csv",
                    "mask": "mask.nc"
                }
        """)
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        metavar='path',
        help=textwrap.dedent("""\
            Path to the output folder where the report and plots will be saved.
            If not provided, you'll be prompted to accept the default './REPORT' folder.
        """)
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Only validate input files and structure, then exit without generating any report."
    )

    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF generation. Only generate plots and dataframes (saved as .json)."
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging and console output during processing."
    )

    parser.add_argument(
        "--open-report",
        action="store_true",
        help="Automatically open the PDF report after generation (if created)."
    )

    parser.add_argument(
        "--variable",
        type=str,
        metavar='var_name',
        help=textwrap.dedent("""\
            Name of the target variable to label plots and outputs (e.g., 'Chlorophyll-a').
            If not provided the code will prompt the user to add it while in the routine.
        """)
    )

    parser.add_argument(
        "--unit",
        type=str,
        metavar='unit_str',
        help=textwrap.dedent("""\
            Unit of the variable (e.g., 'mg/L', 'm3/s') shown in labels and legends.
            The provided unit will be converted into latex format when added to plots.
            If not provided the code will prompt the user to add it while in the routine.
        """)
    )

    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress the ASCII banner shown at program start. Useful in batch scripts."
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Display detailed program information and exit."
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Hydrological_model_validator v4.9.1",
        help="Display the current version and exit."
    )

    args, _ = parser.parse_known_args()
    
    if args.info:
        print(textwrap.dedent("""\
            Hydrological Model Validator - Preliminary Report Generator
            -----------------------------------------------------
            ###################################################################
            #  Tools for the analysis and validation of Bio-Geo-Hydrological  #
            #           simulations and other climatological data             #
            ###################################################################

            INPUT REQUIREMENTS:
              • 2 spatial datasets (observed & simulated) as NetCDF or GeoTIFF
              • 2 basin-average timeseries (observed & simulated) as CSV, Excel, or NetCDF
              • A spatial mask file (e.g., from a model grid)

            OUTPUT:
              • Time series, scatter, seasonal plots
              • Taylor, target, violin, and whisker-box diagrams
              • Efficiency metrics by month, year, and total
              • PDF report (unless --no-pdf is specified)
              
            Author: Alessandro Gozzoli - Alma Mater Studiorum - Università di Bologna  
            Role: Student - Physics of the Earth System
            Email: alessandro.gozzoli4@studio.unibo.it  
            GitHub: https://github.com/AlessandroGozzoli

            Run with --help for full CLI options.
        """))
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